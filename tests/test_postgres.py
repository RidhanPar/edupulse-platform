"""Behaviour that only PostgreSQL can show: migrations, enum types, constraints and the
append-only trigger. SQLite passes tests these would fail (the dataset_kind enum bug in
migration 5596dac8a7b4 was only caught here).

Runs when TEST_POSTGRES_URL points at a disposable database (CI provides one). The
schema is dropped and recreated around every test, so never point it at real data.
"""
import os
import uuid
from datetime import datetime, timedelta, timezone

import psycopg2
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from flask_migrate import downgrade, upgrade
from sqlalchemy import inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError

from app import create_app
from db import db
from db.audit import redact_expired_audit_personal_data
from db.models import AuditEvent, Dataset, DatasetKind, ModelArtifact, Organisation, PlanTier, Role, User

POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="set TEST_POSTGRES_URL to run the PostgreSQL tests")

MIGRATIONS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "migrations")
FIRST_REVISION = "cb67422390af"
BEFORE_FEATURES = "c3f1a9d2e7b4"
LEGACY_FEATURES = ["attendance", "assignment_score", "quiz_score", "study_time",
                   "lms_activity", "previous_grade", "missed_submissions"]


def _reset_schema():
    with db.engine.begin() as connection:
        connection.exec_driver_sql("DROP SCHEMA public CASCADE")
        connection.exec_driver_sql("CREATE SCHEMA public")


@pytest.fixture()
def pg_app(tmp_path):
    app = create_app(
        {"SQLALCHEMY_DATABASE_URI": POSTGRES_URL, "STORAGE_LOCAL_ROOT": str(tmp_path), "RATELIMIT_ENABLED": False},
        env="testing",
    )
    with app.app_context():
        _reset_schema()
        yield app
        db.session.remove()
        _reset_schema()
        db.engine.dispose()


@pytest.fixture()
def migrated(pg_app):
    upgrade(directory=MIGRATIONS)
    return pg_app


def _raw(sql: str, params=None) -> None:
    """Run SQL on a raw driver connection, bypassing the application's own audit guard,
    so that only the database itself can refuse the statement."""
    connection = db.engine.raw_connection()
    try:
        connection.cursor().execute(sql, params)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _enum_types() -> list[str]:
    with db.engine.connect() as connection:
        return sorted(row[0] for row in connection.execute(text("SELECT typname FROM pg_type WHERE typtype = 'e'")))


def _seed_tenant():
    organisation = Organisation(name="Alpha", slug="alpha")
    owner = User(organisation=organisation, email="owner@alpha.test", role=Role.OWNER)
    db.session.add_all([organisation, owner])
    db.session.commit()
    return organisation, owner


def test_head_matches_the_models_and_downgrades_cleanly(migrated):
    with db.engine.connect() as connection:
        assert compare_metadata(MigrationContext.configure(connection), db.metadata) == []
        functions = connection.execute(
            text("SELECT count(*) FROM pg_proc WHERE proname = 'audit_events_append_only'")
        ).scalar()
    assert _enum_types() == ["dataset_kind", "plan_tier", "user_role"]
    assert functions == 1

    downgrade(directory=MIGRATIONS, revision="base")

    with db.engine.connect() as connection:
        assert inspect(connection).get_table_names() == ["alembic_version"]
        assert connection.execute(
            text("SELECT count(*) FROM pg_proc WHERE proname = 'audit_events_append_only'")
        ).scalar() == 0
    assert _enum_types() == []
    upgrade(directory=MIGRATIONS)


def test_a_populated_database_upgrades_from_the_first_revision(pg_app):
    upgrade(directory=MIGRATIONS, revision=FIRST_REVISION)
    org_id, user_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    with db.engine.begin() as connection:
        connection.execute(text("INSERT INTO organisations (id, name, slug, created_at, plan_tier) "
                                "VALUES (:id, 'Legacy', 'legacy', now(), 'standard')"), {"id": org_id})
        connection.execute(text("INSERT INTO users (id, organisation_id, email, role, created_at, is_active) "
                                "VALUES (:id, :org, 'owner@legacy.test', 'owner', now(), true)"),
                           {"id": user_id, "org": org_id})
        connection.execute(text("INSERT INTO datasets (id, organisation_id, kind, original_filename, storage_key, "
                                "row_count, column_names, uploaded_by, uploaded_at) VALUES (:id, :org, 'training', "
                                "'t.csv', 'legacy-key', 1, '[]', :user, '2024-02-29 09:30:00+00')"),
                           {"id": dataset_id, "org": org_id, "user": user_id})
    upgrade(directory=MIGRATIONS, revision=BEFORE_FEATURES)
    artifact_id = uuid.uuid4()
    with db.engine.begin() as connection:
        connection.execute(text("INSERT INTO model_artifacts (id, organisation_id, algorithm_name, metrics, "
                                "feature_importances, model_comparison, storage_key, trained_by, trained_at, is_active) "
                                "VALUES (:id, :org, 'Random Forest', '{}', '{}', '[]', 'legacy-model', :user, now(), true)"),
                           {"id": artifact_id, "org": org_id, "user": user_id})

    upgrade(directory=MIGRATIONS)

    organisation = db.session.get(Organisation, org_id)
    assert (organisation.plan_tier, organisation.is_active) == (PlanTier.PILOT, True)
    assert db.session.get(Dataset, dataset_id).retention_expires_at == datetime(2026, 2, 28, 9, 30, tzinfo=timezone.utc)
    assert db.session.get(ModelArtifact, artifact_id).features == LEGACY_FEATURES


def test_the_database_refuses_to_change_or_remove_audit_rows(migrated):
    organisation, owner = _seed_tenant()
    event = AuditEvent(organisation_id=organisation.id, user_id=owner.id, action="login_success",
                       ip_address="203.0.113.7", user_agent="Mozilla/5.0")
    db.session.add(event)
    db.session.commit()  # inserting is allowed
    attempts = [
        ("UPDATE audit_events SET action = 'tampered'", None),
        ("UPDATE audit_events SET ip_address = NULL, user_agent = NULL, action = 'tampered'", None),
        ("UPDATE audit_events SET created_at = now()", None),
        ("DELETE FROM audit_events", None),
        ("TRUNCATE audit_events", None),
    ]

    for sql, params in attempts:
        with pytest.raises(psycopg2.Error) as refused:
            _raw(sql, params)
        assert "append-only" in str(refused.value), sql

    db.session.expire_all()
    stored = db.session.get(AuditEvent, event.id)
    assert (stored.action, stored.ip_address) == ("login_success", "203.0.113.7")


def test_the_database_allows_only_the_personal_data_redaction(migrated):
    organisation, owner = _seed_tenant()
    old = AuditEvent(organisation_id=organisation.id, user_id=owner.id, action="login_success",
                     ip_address="203.0.113.7", user_agent="Mozilla/5.0",
                     created_at=datetime.now(timezone.utc) - timedelta(days=91))
    db.session.add(old)
    db.session.commit()

    assert redact_expired_audit_personal_data() == 1
    db.session.commit()

    db.session.expire_all()
    stored = db.session.get(AuditEvent, old.id)
    assert (stored.ip_address, stored.user_agent, stored.action) == (None, None, "login_success")


def test_a_user_with_datasets_cannot_be_deleted(migrated):
    organisation, owner = _seed_tenant()
    db.session.add(Dataset(organisation_id=organisation.id, kind=DatasetKind.TRAINING, original_filename="t.csv",
                           storage_key=f"org/{organisation.id}/datasets/{uuid.uuid4()}.csv", row_count=1,
                           column_names=["student_id"], uploaded_by=owner.id))
    db.session.commit()

    db.session.delete(owner)
    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()


def test_only_one_active_model_per_organisation(migrated):
    organisation, owner = _seed_tenant()

    def artifact(active):
        return ModelArtifact(organisation_id=organisation.id, algorithm_name="Random Forest", features=["attendance"],
                             metrics={}, feature_importances={}, model_comparison=[], trained_by=owner.id,
                             storage_key=f"org/{organisation.id}/models/{uuid.uuid4()}.pkl", is_active=active)

    db.session.add_all([artifact(True), artifact(False), artifact(False)])
    db.session.commit()
    db.session.add(artifact(True))
    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()


def test_unknown_enum_values_are_rejected(migrated):
    with pytest.raises(DBAPIError):
        db.session.execute(text("INSERT INTO organisations (id, name, slug, created_at, plan_tier) "
                                "VALUES (:id, 'Bad', 'bad', now(), 'standard')"), {"id": uuid.uuid4()})
    db.session.rollback()
