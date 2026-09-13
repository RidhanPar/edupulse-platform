import re
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import delete, null, select, text, update

from db import db
from db.audit import _GRANT_OPTION, _REDACTION_GRANT, AppendOnlyViolation, redact_expired_audit_personal_data
from db.models import AUDIT_PERSONAL_DATA_RETENTION_DAYS, AuditEvent, utcnow

ROOT = Path(__file__).resolve().parents[1]
TABLE = AuditEvent.__table__
PERSONAL_FIELDS = {"ip_address", "user_agent"}


def _event(organisation, user, age_days: int, **fields) -> AuditEvent:
    return AuditEvent(
        organisation_id=organisation.id,
        user_id=user.id,
        action="login_success",
        ip_address="203.0.113.7",
        user_agent="Mozilla/5.0",
        created_at=utcnow() - timedelta(days=age_days),
        **fields,
    )


def _row(event_id) -> dict:
    return db.session.execute(select(TABLE).where(TABLE.c.id == event_id)).one()._asdict()


@pytest.fixture()
def audit_event(make_org):
    organisation, user = make_org("alpha")
    event = _event(organisation, user, age_days=AUDIT_PERSONAL_DATA_RETENTION_DAYS + 30)
    db.session.add(event)
    db.session.commit()  # inserting is allowed
    return event


# Every way this application could send an UPDATE, DELETE or overwrite to audit_events.
def _orm_attribute_update(event):
    event.action = "tampered"
    db.session.commit()


def _orm_delete(event):
    db.session.delete(event)
    db.session.commit()


def _orm_bulk_update(event):
    db.session.execute(update(AuditEvent).values(action="tampered"))


def _orm_bulk_delete(event):
    db.session.execute(delete(AuditEvent))


def _core_update_via_session(event):
    db.session.execute(update(TABLE).values(action="tampered"))


def _core_delete_via_engine(event):
    with db.engine.begin() as connection:
        connection.execute(delete(TABLE))


def _text_update(event):
    db.session.execute(text("UPDATE audit_events SET action = 'tampered'"))


def _text_delete_via_driver(event):
    with db.engine.begin() as connection:
        connection.exec_driver_sql('DELETE FROM "audit_events"')


def _insert_or_replace(event):
    db.session.execute(
        text("INSERT OR REPLACE INTO audit_events (id, action, created_at) VALUES (:id, 'tampered', CURRENT_TIMESTAMP)"),
        {"id": event.id.hex},
    )


def _upsert(event):
    db.session.execute(
        text("INSERT INTO audit_events (id, action, created_at) VALUES (:id, 'x', CURRENT_TIMESTAMP) "
             "ON CONFLICT (id) DO UPDATE SET action = 'tampered'"),
        {"id": event.id.hex},
    )


def _truncate(event):
    db.session.execute(text("TRUNCATE TABLE audit_events"))


def _redaction_statement_without_the_grant(event):
    db.session.execute(update(TABLE).values(ip_address=null(), user_agent=null()))


def _forged_grant(event):
    db.session.execute(
        update(TABLE).values(ip_address=null(), user_agent=null()).execution_options(**{_GRANT_OPTION: True})
    )


def _real_grant_on_another_column(event):
    db.session.execute(
        update(TABLE).values(action="tampered").execution_options(**{_GRANT_OPTION: _REDACTION_GRANT})
    )


def _real_grant_redaction_plus_another_column(event):
    db.session.execute(
        update(TABLE)
        .values(ip_address=null(), user_agent=null(), created_at=utcnow())
        .execution_options(**{_GRANT_OPTION: _REDACTION_GRANT})
    )


TAMPERING_ATTEMPTS = [
    _orm_attribute_update,
    _orm_delete,
    _orm_bulk_update,
    _orm_bulk_delete,
    _core_update_via_session,
    _core_delete_via_engine,
    _text_update,
    _text_delete_via_driver,
    _insert_or_replace,
    _upsert,
    _truncate,
    _redaction_statement_without_the_grant,
    _forged_grant,
    _real_grant_on_another_column,
    _real_grant_redaction_plus_another_column,
]


@pytest.mark.parametrize("attempt", TAMPERING_ATTEMPTS, ids=lambda attempt: attempt.__name__.lstrip("_"))
def test_no_path_but_the_redaction_function_can_change_an_audit_row(audit_event, attempt):
    before = _row(audit_event.id)

    with pytest.raises(AppendOnlyViolation):
        attempt(audit_event)
    db.session.rollback()

    assert _row(audit_event.id) == before


def test_redaction_nulls_only_the_personal_fields_of_expired_rows(make_org):
    organisation, user = make_org("alpha")
    expired = _event(organisation, user, AUDIT_PERSONAL_DATA_RETENTION_DAYS + 1,
                     entity_type="dataset", entity_id="d-1", details={"rows": 3})
    recent = _event(organisation, user, AUDIT_PERSONAL_DATA_RETENTION_DAYS - 1)
    db.session.add_all([expired, recent])
    db.session.commit()
    expired_before, recent_before = _row(expired.id), _row(recent.id)

    assert redact_expired_audit_personal_data() == 1
    db.session.commit()

    expired_after = _row(expired.id)
    assert expired_after["ip_address"] is None and expired_after["user_agent"] is None
    assert {k: v for k, v in expired_after.items() if k not in PERSONAL_FIELDS} == \
        {k: v for k, v in expired_before.items() if k not in PERSONAL_FIELDS}
    assert _row(recent.id) == recent_before
    assert redact_expired_audit_personal_data() == 0


def test_only_the_redaction_function_holds_the_grant():
    holders = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.py")
        if not {".venv", "tests"} & set(path.relative_to(ROOT).parts)
        and re.search(r"\b_REDACTION_GRANT\b", path.read_text(encoding="utf-8"))
    )

    assert holders == ["db/audit.py"]
