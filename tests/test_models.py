import uuid
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from flask_migrate import upgrade
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app import create_app
from db import db
from db.models import ModelArtifact, Role, User

ROOT = Path(__file__).resolve().parents[1]


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


def test_migrations_match_models(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
    app = create_app({"SQLALCHEMY_DATABASE_URI": uri}, env="testing")

    with app.app_context():
        upgrade(directory=str(ROOT / "migrations"))
        with db.engine.connect() as connection:
            diff = compare_metadata(MigrationContext.configure(connection), db.metadata)
        db.engine.dispose()

    assert diff == []
