"""Append-only enforcement for audit_events, and its one sanctioned exception.

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
from datetime import datetime, timedelta

from flask import has_request_context, request
from sqlalchemy import event, null, update
from sqlalchemy.engine import Engine

from db import db
from db.models import AUDIT_PERSONAL_DATA_RETENTION_DAYS, AuditEvent, utcnow


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
