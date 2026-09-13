import pytest

from app import create_app
from db import db
from db.models import Organisation, Role, User


@pytest.fixture()
def app():
    app = create_app({"SQLALCHEMY_DATABASE_URI": "sqlite://"}, env="testing")
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
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
