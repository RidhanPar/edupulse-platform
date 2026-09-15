"""CSRF protection, with Flask-WTF enabled as in production."""
import re
from pathlib import Path

import pytest

from app import create_app
from auth.invites import issue_invite
from auth.passwords import hash_password
from db import db
from db.models import Organisation, Role, User

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"
PASSWORD = "correct horse battery staple"
OWNER_EMAIL = "owner@alpha.test"


@pytest.fixture()
def csrf_app(tmp_path):
    app = create_app(
        {"SQLALCHEMY_DATABASE_URI": "sqlite://", "STORAGE_LOCAL_ROOT": str(tmp_path / "storage"), "WTF_CSRF_ENABLED": True},
        env="testing",
    )
    with app.app_context():
        db.create_all()
        organisation = Organisation(name="Alpha", slug="alpha")
        owner = User(organisation=organisation, email=OWNER_EMAIL, role=Role.OWNER, password_hash=hash_password(PASSWORD))
        db.session.add_all([organisation, owner])
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


def _token_from(client, path: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', client.get(path).get_data(as_text=True))
    assert match, f"no CSRF token on {path}"
    return match.group(1)


def _logged_in(app):
    client = app.test_client()
    token = _token_from(client, "/login")
    response = client.post("/login", data={"email": OWNER_EMAIL, "password": PASSWORD, "csrf_token": token})
    assert response.status_code == 302
    return client


def test_login_with_the_pages_token_succeeds(csrf_app):
    assert _logged_in(csrf_app).get("/").status_code == 200


def test_login_without_a_token_is_rejected(csrf_app):
    response = csrf_app.test_client().post("/login", data={"email": OWNER_EMAIL, "password": PASSWORD})

    assert response.status_code == 400


def test_a_token_from_another_session_is_rejected(csrf_app):
    stolen = _token_from(csrf_app.test_client(), "/login")

    response = csrf_app.test_client().post(
        "/login", data={"email": OWNER_EMAIL, "password": PASSWORD, "csrf_token": stolen}
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "path",
    ["/upload-train", "/upload-predict", "/upload-actual", "/train", "/recheck-comparison", "/change-password", "/logout"],
)
def test_every_state_changing_route_rejects_a_post_without_a_token(csrf_app, path):
    client = _logged_in(csrf_app)

    assert client.post(path).status_code == 400
    assert client.get("/").status_code == 200  # nothing happened, not even a logout


def test_invite_acceptance_rejects_a_post_without_a_token(csrf_app):
    user = User(organisation=db.session.scalars(db.select(Organisation)).one(), email="new@alpha.test", role=Role.STAFF)
    token = issue_invite(user)
    db.session.add(user)
    db.session.commit()

    response = csrf_app.test_client().post(
        f"/invite/{token}", data={"password": "a brand new passphrase", "confirm": "a brand new passphrase"}
    )

    assert response.status_code == 400
    assert db.session.get(User, user.id).password_hash is None


def test_every_post_form_in_the_templates_carries_a_csrf_token():
    offenders = [
        template.name
        for template in sorted(TEMPLATES.glob("*.html"))
        for form in re.findall(r"<form\b.*?</form>", template.read_text(encoding="utf-8"), re.DOTALL | re.IGNORECASE)
        if re.search(r'method\s*=\s*"post"', form, re.IGNORECASE) and "csrf_token()" not in form
    ]

    assert offenders == []
