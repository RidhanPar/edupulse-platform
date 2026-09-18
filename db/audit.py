"""The audit log: append-only enforcement and its one sanctioned exception, writing
events, and reading them back one bounded page at a time.

Every statement SQLAlchemy sends to the database passes _enforce_append_only, so
ORM flushes, bulk ORM statements, Core statements and text() SQL are all covered.
The only write it lets through to an existing row is the redaction performed by
redact_expired_audit_personal_data.

TODO(section 7): add a PostgreSQL trigger refusing UPDATE and DELETE on
audit_events, for writes that do not come through this application (psql, other
services). It must still allow the redaction update below. SQLite has no
equivalent, so it lands with the rest of the Postgres-specific migration work.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from flask import has_request_context, request
from sqlalchemy import Select, Uuid, event, literal, null, tuple_, update
from sqlalchemy.engine import Engine

from db import db
from db.models import AUDIT_PERSONAL_DATA_RETENTION_DAYS, AuditEvent, utcnow
from db.tenancy import scoped_select

# Every action the application records. The audit page offers exactly these as filters;
# tests/test_audit_events.py fails if code records an action missing from this list.
AUDIT_ACTIONS = (
    "login_success",
    "login_failure",
    "logout",
    "password_changed",
    "invite_accepted",
    "dataset_uploaded",
    "model_trained",
    "prediction_run",
    "results_exported",
)

# Fixed server-side page size. The table only grows, so no caller may ask for more.
AUDIT_PAGE_SIZE = 50


class AppendOnlyViolation(RuntimeError):
    """Raised when code tries to modify or remove an audit record."""


_TABLE = r'(?:[\w"]+\.)?"?audit_events"?(?!\w)'
_AUDIT_MUTATION = re.compile(
    rf"\b(?:UPDATE|DELETE\s+FROM|TRUNCATE(?:\s+TABLE)?(?:\s+ONLY)?|(?:INSERT\s+OR\s+)?REPLACE\s+INTO)\s+{_TABLE}"
    rf"|\bINSERT\s+INTO\s+{_TABLE}.*\bON\s+CONFLICT\b.*\bDO\s+UPDATE\b",
    re.IGNORECASE | re.DOTALL,
)
# The exact statement the redaction function generates: both personal fields set to
# NULL and nothing else. Any other SET list is refused, even with the grant.
_REDACTION_STATEMENT = re.compile(
    r'\s*UPDATE\s+"?audit_events"?\s+SET\s+ip_address\s*=\s*NULL\s*,\s*user_agent\s*=\s*NULL\s+WHERE\s.*',
    re.IGNORECASE | re.DOTALL,
)
# Checked by identity, so it cannot be forged by passing a truthy value.
_REDACTION_GRANT = object()
_GRANT_OPTION = "audit_personal_data_redaction"


@event.listens_for(Engine, "before_cursor_execute")
def _enforce_append_only(conn, cursor, statement, parameters, context, executemany):
    if not _AUDIT_MUTATION.search(statement):
        return
    options = context.execution_options if context is not None else {}
    if options.get(_GRANT_OPTION) is _REDACTION_GRANT and _REDACTION_STATEMENT.fullmatch(statement):
        return
    raise AppendOnlyViolation("audit_events is append-only: rows cannot be updated, replaced or deleted")


def redact_expired_audit_personal_data(now: datetime | None = None) -> int:
    """Null ip_address and user_agent on audit rows older than the personal-data retention period.

    This is the only permitted change to an existing audit row. Nothing schedules it
    yet; the retention job will. The caller commits. Returns the number of rows redacted.
    """
    cutoff = (now or utcnow()) - timedelta(days=AUDIT_PERSONAL_DATA_RETENTION_DAYS)
    table = AuditEvent.__table__
    statement = (
        update(table)
        .where(table.c.created_at < cutoff)
        .where(table.c.ip_address.is_not(None) | table.c.user_agent.is_not(None))
        .values(ip_address=null(), user_agent=null())
        .execution_options(**{_GRANT_OPTION: _REDACTION_GRANT})
    )
    return db.session.execute(statement).rowcount


def record_audit_event(
    action: str,
    *,
    user=None,
    organisation_id=None,
    entity_type: str | None = None,
    entity_id=None,
    details: dict | None = None,
) -> AuditEvent:
    """Add an audit row to the current session; the caller commits.

    Passing `user` records both their user_id and organisation_id. The request's IP
    address and user agent are captured when there is a request.
    """
    ip_address = user_agent = None
    if has_request_context():
        ip_address = request.remote_addr
        user_agent = (request.user_agent.string or None) and request.user_agent.string[:512]
    event = AuditEvent(
        organisation_id=user.organisation_id if user is not None else organisation_id,
        user_id=user.id if user is not None else None,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        ip_address=ip_address,
        user_agent=user_agent,
        details=details,
    )
    db.session.add(event)
    return event


@dataclass(frozen=True)
class AuditFilters:
    user_id: uuid.UUID | None = None
    action: str | None = None
    start: datetime | None = None  # inclusive
    end: datetime | None = None  # exclusive


@dataclass(frozen=True)
class AuditPage:
    events: list[AuditEvent]
    newer_cursor: str | None
    older_cursor: str | None


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def encode_cursor(event: AuditEvent) -> str:
    """An opaque page position: the event's created_at (as UTC microseconds) and id."""
    created_at = event.created_at if event.created_at.tzinfo else event.created_at.replace(tzinfo=timezone.utc)
    return f"{(created_at - _EPOCH) // timedelta(microseconds=1)}.{event.id.hex}"


def decode_cursor(cursor: str | None) -> tuple[datetime, uuid.UUID] | None:
    """The position in a cursor, or None for anything malformed (which means the first page)."""
    try:
        micros, event_id = cursor.split(".")
        return _EPOCH + timedelta(microseconds=int(micros)), uuid.UUID(hex=event_id)
    except (AttributeError, ValueError, OverflowError):
        return None


def audit_query(organisation_id, filters: AuditFilters, *, before=None, after=None, limit: int) -> Select:
    """Keyset-paginated SELECT over one organisation's events.

    Rows are ordered by (created_at, id); id breaks ties between events recorded in the
    same microsecond. Newest first, or oldest first when paging towards newer rows.
    """
    statement = scoped_select(AuditEvent, organisation_id)
    if filters.user_id is not None:
        statement = statement.where(AuditEvent.user_id == filters.user_id)
    if filters.action is not None:
        statement = statement.where(AuditEvent.action == filters.action)
    if filters.start is not None:
        statement = statement.where(AuditEvent.created_at >= filters.start)
    if filters.end is not None:
        statement = statement.where(AuditEvent.created_at < filters.end)

    position = tuple_(AuditEvent.created_at, AuditEvent.id)

    def bound(cursor):
        return tuple_(literal(cursor[0], AuditEvent.created_at.type), literal(cursor[1], Uuid()))

    if after is not None:
        return (
            statement.where(position > bound(after))
            .order_by(AuditEvent.created_at.asc(), AuditEvent.id.asc())
            .limit(limit)
        )
    if before is not None:
        statement = statement.where(position < bound(before))
    return statement.order_by(AuditEvent.created_at.desc(), AuditEvent.id.desc()).limit(limit)


def audit_page(organisation_id, filters: AuditFilters, *, before=None, after=None, page_size: int = AUDIT_PAGE_SIZE) -> AuditPage:
    """One bounded page of an organisation's audit log, newest first.

    Never uses OFFSET or COUNT(*): both get slower as the table grows, while a keyset
    position costs the same on the first page and the millionth row.
    """
    page_size = max(1, min(page_size, AUDIT_PAGE_SIZE))
    rows = list(
        db.session.scalars(audit_query(organisation_id, filters, before=before, after=after, limit=page_size + 1))
    )
    has_more = len(rows) > page_size
    rows = rows[:page_size]
    if after is not None:
        rows.reverse()
        newer = encode_cursor(rows[0]) if has_more and rows else None
        older = encode_cursor(rows[-1]) if rows else None
    else:
        newer = encode_cursor(rows[0]) if before is not None and rows else None
        older = encode_cursor(rows[-1]) if has_more else None
    return AuditPage(events=rows, newer_cursor=newer, older_cursor=older)
