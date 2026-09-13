import io
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import create_app
from db import db
from db.models import ModelArtifact, Organisation, Role, User
from utils.storage import model_key, storage_for

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def app(tmp_path):
    app = create_app(
        {
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "STORAGE_BACKEND": "local",
            "STORAGE_LOCAL_ROOT": str(tmp_path / "storage"),
        },
        env="testing",
    )
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    """A client with no logged-in user."""
    return app.test_client()


@pytest.fixture()
def make_org(app):
    """Create an organisation with one user of the given role. Returns (organisation, user)."""

    def _make(slug: str, role: Role = Role.OWNER):
        organisation = Organisation(name=slug.title(), slug=slug)
        user = User(organisation=organisation, email=f"{role.value}@{slug}.test", role=role)
        db.session.add_all([organisation, user])
        db.session.commit()
        return organisation, user

    return _make


@pytest.fixture()
def tenant(app, make_org):
    """Create an organisation and user, returning a test client logged in as that user."""

    def _tenant(slug: str, role: Role = Role.OWNER) -> SimpleNamespace:
        organisation, user = make_org(slug, role)
        client = app.test_client()
        with client.session_transaction() as session:
            # The key Flask-Login uses, so these tests keep working once it arrives.
            session["_user_id"] = str(user.id)
        return SimpleNamespace(client=client, organisation=organisation, user=user)

    return _tenant


@pytest.fixture(scope="session")
def csv_fixtures() -> dict[str, bytes]:
    return {name: (FIXTURES / f"{name}.csv").read_bytes() for name in ("training", "prediction", "actual")}


@pytest.fixture()
def upload():
    def _upload(client, path: str, data: bytes, filename: str = "data.csv"):
        return client.post(path, data={"file": (io.BytesIO(data), filename)}, content_type="multipart/form-data")

    return _upload


@pytest.fixture(scope="session")
def trained(csv_fixtures):
    """One real training run on the fixture data, shared by every test that needs a model."""
    import pandas as pd

    from utils.train_model import train_and_select_best

    return train_and_select_best(pd.read_csv(io.BytesIO(csv_fixtures["training"])))


@pytest.fixture()
def ready_tenant(tenant, upload, csv_fixtures, trained):
    """A tenant with all three datasets uploaded and an active model."""
    from utils.train_model import serialize_model

    def _ready(slug: str, role: Role = Role.OWNER) -> SimpleNamespace:
        ready = tenant(slug, role)
        for path, name in (("/upload-train", "training"), ("/upload-predict", "prediction"), ("/upload-actual", "actual")):
            assert upload(ready.client, path, csv_fixtures[name]).status_code == 302
        artifact_id = uuid.uuid4()
        artifact = ModelArtifact(
            id=artifact_id,
            organisation_id=ready.organisation.id,
            algorithm_name=trained.model_name,
            metrics=trained.metrics,
            feature_importances=trained.feature_importances,
            model_comparison=trained.model_comparison,
            storage_key=model_key(ready.organisation.id, artifact_id),
            trained_by=ready.user.id,
            is_active=True,
        )
        storage_for(ready.organisation.id).put(artifact.storage_key, serialize_model(trained.model))
        db.session.add(artifact)
        db.session.commit()
        ready.artifact = artifact
        return ready

    return _ready
