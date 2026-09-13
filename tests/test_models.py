import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from flask_migrate import downgrade, upgrade
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from app import create_app
from db import db
from db.models import (
    DATASET_RETENTION_MONTHS,
    Dataset,
    DatasetKind,
    ModelArtifact,
    Organisation,
    PlanTier,
    Role,
    User,
    add_months,
)

ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = str(ROOT / "migrations")
INITIAL_REVISION = "cb67422390af"


def _artifact(organisation, user, active: bool) -> ModelArtifact:
    return ModelArtifact(
        organisation_id=organisation.id,
        algorithm_name="Decision Tree",
        metrics={"f1": 0.8},
        feature_importances={"attendance": 0.4},
        model_comparison=[{"model": "Decision Tree", "f1": 0.8}],
        storage_key=f"org/{organisation.id}/models/{uuid.uuid4()}.pkl",
        trained_by=user.id,
        is_active=active,
    )


def _dataset(organisation, user, **overrides) -> Dataset:
    fields = {
        "organisation_id": organisation.id,
        "kind": DatasetKind.TRAINING,
        "original_filename": "training.csv",
        "storage_key": f"org/{organisation.id}/datasets/{uuid.uuid4()}.csv",
        "row_count": 10,
        "column_names": ["student_id", "target"],
        "uploaded_by": user.id,
    }
    fields.update(overrides)
    return Dataset(**fields)


def test_enums_are_stored_as_lowercase_values(make_org):
    _, user = make_org("alpha", role=Role.STAFF)

    stored = db.session.execute(text("SELECT role FROM users WHERE email = :email"), {"email": user.email})

    assert stored.scalar_one() == "staff"
    db.session.expire_all()
    assert db.session.get(User, user.id).role is Role.STAFF


def test_json_columns_round_trip(make_org):
    organisation, user = make_org("alpha")
    artifact = _artifact(organisation, user, active=True)
    db.session.add(artifact)
    db.session.commit()
    db.session.expire_all()

    loaded = db.session.get(ModelArtifact, artifact.id)

    assert loaded.metrics == {"f1": 0.8}
    assert loaded.model_comparison == [{"model": "Decision Tree", "f1": 0.8}]


def test_email_is_unique_across_organisations(make_org):
    _, user_a = make_org("alpha")
    org_b, _ = make_org("beta")
    db.session.add(User(organisation_id=org_b.id, email=user_a.email, role=Role.VIEWER))

    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()


def test_only_one_active_model_per_organisation(make_org):
    org_a, user_a = make_org("alpha")
    org_b, user_b = make_org("beta")
    db.session.add_all([
        _artifact(org_a, user_a, active=True),
        _artifact(org_a, user_a, active=False),
        _artifact(org_a, user_a, active=False),
        _artifact(org_b, user_b, active=True),
    ])
    db.session.commit()

    db.session.add(_artifact(org_a, user_a, active=True))
    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()


def test_foreign_keys_are_enforced_on_sqlite(app):
    db.session.add(User(organisation_id=uuid.uuid4(), email="orphan@nowhere.test", role=Role.VIEWER))

    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()


def test_new_organisations_are_active_pilots(app):
    organisation = Organisation(name="Alpha", slug="alpha")
    db.session.add(organisation)
    db.session.commit()
    db.session.expire_all()

    loaded = db.session.get(Organisation, organisation.id)

    assert loaded.is_active is True
    assert loaded.plan_tier is PlanTier.PILOT


def test_unknown_plan_tier_is_rejected_by_the_database(app):
    with pytest.raises(IntegrityError):
        db.session.execute(
            text(
                "INSERT INTO organisations (id, name, slug, created_at, plan_tier, is_active) "
                "VALUES (:id, 'Alpha', 'alpha', CURRENT_TIMESTAMP, 'standard', 1)"
            ),
            {"id": uuid.uuid4().hex},
        )
    db.session.rollback()


def test_dataset_retention_defaults_to_24_months_after_upload(make_org):
    organisation, user = make_org("alpha")
    dataset = _dataset(organisation, user)
    db.session.add(dataset)
    db.session.commit()

    assert DATASET_RETENTION_MONTHS == 24
    assert dataset.retention_expires_at == add_months(dataset.uploaded_at, 24)
    assert dataset.deleted_at is None


def test_dataset_retention_is_counted_in_calendar_months(make_org):
    organisation, user = make_org("alpha")
    dataset = _dataset(organisation, user, uploaded_at=datetime(2024, 2, 29, 9, 30, tzinfo=timezone.utc))
    db.session.add(dataset)
    db.session.flush()

    assert dataset.retention_expires_at == datetime(2026, 2, 28, 9, 30, tzinfo=timezone.utc)


def test_explicit_dataset_retention_is_kept(make_org):
    organisation, user = make_org("alpha")
    expiry = datetime(2030, 1, 1, tzinfo=timezone.utc)
    dataset = _dataset(organisation, user, retention_expires_at=expiry)
    db.session.add(dataset)
    db.session.flush()

    assert dataset.retention_expires_at == expiry


def test_user_with_datasets_cannot_be_deleted(make_org):
    organisation, user = make_org("alpha")
    db.session.add(_dataset(organisation, user))
    db.session.commit()

    db.session.delete(user)
    with pytest.raises(IntegrityError):
        db.session.commit()
    db.session.rollback()

    assert db.session.get(User, user.id) is not None


def _file_backed_app(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
    return create_app(
        {"SQLALCHEMY_DATABASE_URI": uri, "STORAGE_LOCAL_ROOT": str(tmp_path / "storage")},
        env="testing",
    )


def test_migrations_match_models_and_reverse_cleanly(tmp_path):
    app = _file_backed_app(tmp_path)

    with app.app_context():
        upgrade(directory=MIGRATIONS)
        with db.engine.connect() as connection:
            diff = compare_metadata(MigrationContext.configure(connection), db.metadata)
            inspector = inspect(connection)
            user_fk_ondelete = {
                (table, fk["constrained_columns"][0]): fk["options"].get("ondelete")
                for table in ("datasets", "model_artifacts", "audit_events")
                for fk in inspector.get_foreign_keys(table)
                if fk["referred_table"] == "users"
            }
        downgrade(directory=MIGRATIONS, revision="base")
        upgrade(directory=MIGRATIONS)
        db.engine.dispose()

    assert diff == []
    assert user_fk_ondelete == {
        ("datasets", "uploaded_by"): "RESTRICT",
        ("model_artifacts", "trained_by"): "RESTRICT",
        ("audit_events", "user_id"): "RESTRICT",
    }


def test_follow_up_migration_preserves_existing_rows(tmp_path):
    app = _file_backed_app(tmp_path)
    org_id, user_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    uploaded_at = "2025-03-31 12:00:00.000000"

    with app.app_context():
        upgrade(directory=MIGRATIONS, revision=INITIAL_REVISION)
        with db.engine.begin() as connection:
            connection.execute(
                text("INSERT INTO organisations (id, name, slug, created_at, plan_tier) "
                     "VALUES (:id, 'Alpha', 'alpha', :at, 'standard')"),
                {"id": org_id.hex, "at": uploaded_at},
            )
            connection.execute(
                text("INSERT INTO users (id, organisation_id, email, role, created_at, is_active) "
                     "VALUES (:id, :org, 'owner@alpha.test', 'owner', :at, 1)"),
                {"id": user_id.hex, "org": org_id.hex, "at": uploaded_at},
            )
            connection.execute(
                text("INSERT INTO datasets (id, organisation_id, kind, original_filename, storage_key, "
                     "row_count, column_names, uploaded_by, uploaded_at) "
                     "VALUES (:id, :org, 'training', 'training.csv', 'org/a/datasets/b.csv', 10, '[]', :user, :at)"),
                {"id": dataset_id.hex, "org": org_id.hex, "user": user_id.hex, "at": uploaded_at},
            )

        upgrade(directory=MIGRATIONS)

        organisation = db.session.get(Organisation, org_id)
        dataset = db.session.get(Dataset, dataset_id)
        assert organisation.plan_tier is PlanTier.PILOT  # legacy "standard" is mapped to pilot
        assert organisation.is_active is True
        assert dataset.uploaded_by == user_id
        assert dataset.retention_expires_at == datetime(2027, 3, 31, 12, 0)
        assert dataset.deleted_at is None
        with pytest.raises(IntegrityError), db.engine.begin() as connection:
            connection.execute(
                text("INSERT INTO organisations (id, name, slug, created_at, plan_tier, is_active) "
                     "VALUES (:id, 'Beta', 'beta', :at, 'standard', 1)"),
                {"id": uuid.uuid4().hex, "at": uploaded_at},
            )
        db.session.remove()
        db.engine.dispose()
