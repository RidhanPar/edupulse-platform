import hashlib
import re
from datetime import timedelta

import pytest
from sqlalchemy import select

from auth import passwords
from auth.invites import consume_invite, issue_invite, pending_invite
from auth.views import INVALID_INVITE
from db import db
from db.models import AuditEvent, Role, User, utcnow

NEW_PASSWORD = "a brand new passphrase"


@pytest.fixture()
def invited(make_org):
    """A user with a fresh invite and no password. Returns (user, raw token)."""

    def _invited(slug: str = "alpha", role: Role = Role.STAFF):
        _, user = make_org(slug, role)
        token = issue_invite(user)
        db.session.commit()
        return user, token

    return _invited


def _accept(client, token: str, password: str = NEW_PASSWORD, confirm: str | None = None):
    return client.post(f"/invite/{token}", data={"password": password, "confirm": confirm or password})


def _page(response) -> str:
    return re.sub(r'name="csrf_token" value="[^"]*"', 'name="csrf_token"', response.get_data(as_text=True))


def test_only_a_hash_of_the_token_is_stored(invited):
    user, token = invited()

    assert user.invite_token_hash == hashlib.sha256(token.encode()).hexdigest()
    assert token not in {str(value) for value in vars(user).values()}


def test_accepting_an_invite_sets_the_password_and_allows_login(client, invited, login):
    user, token = invited()
    assert user.email in client.get(f"/invite/{token}").get_data(as_text=True)

    response = _accept(client, token)

    assert response.status_code == 302 and response.headers["Location"] == "/login"
    db.session.refresh(user)
    assert passwords.verify_password(NEW_PASSWORD, user.password_hash)
    assert (user.invite_token_hash, user.invite_expires_at) == (None, None)
    [event] = db.session.scalars(select(AuditEvent).where(AuditEvent.action == "invite_accepted")).all()
    assert (event.user_id, event.organisation_id) == (user.id, user.organisation_id)
    assert login(client, user.email, NEW_PASSWORD).status_code == 302


def test_an_invite_link_works_exactly_once(app, invited):
    user, token = invited()
    assert _accept(app.test_client(), token).status_code == 302
    db.session.refresh(user)
    first_hash = user.password_hash

    for response in (app.test_client().get(f"/invite/{token}"), _accept(app.test_client(), token, "attacker's passphrase")):
        assert response.status_code == 404
        assert INVALID_INVITE in response.get_data(as_text=True)
    db.session.refresh(user)
    assert user.password_hash == first_hash


def test_expired_used_and_unknown_invites_are_indistinguishable(app, invited):
    expired_user, expired_token = invited("alpha")
    expired_user.invite_expires_at = utcnow() - timedelta(minutes=1)
    db.session.commit()
    _, used_token = invited("beta")
    _accept(app.test_client(), used_token)

    responses = {
        name: app.test_client().get(f"/invite/{token}")
        for name, token in {"expired": expired_token, "used": used_token, "unknown": "not-a-real-token"}.items()
    }

    assert {response.status_code for response in responses.values()} == {404}
    assert len({_page(response) for response in responses.values()}) == 1
    assert INVALID_INVITE in _page(responses["expired"])


def test_consumption_is_atomic_when_two_submissions_race(invited):
    user, token = invited()
    # Both requests pass the page's validity check before either submits...
    assert pending_invite(token) is not None
    assert pending_invite(token) is not None

    # ...but the token is re-checked inside the UPDATE, so only one can win.
    assert consume_invite(token, NEW_PASSWORD) is not None
    assert consume_invite(token, "a second racing passphrase") is None
    db.session.refresh(user)
    assert passwords.verify_password(NEW_PASSWORD, user.password_hash)


@pytest.mark.parametrize("suspend", ["user", "organisation"])
def test_invites_for_deactivated_users_or_suspended_organisations_are_invalid(client, invited, suspend):
    user, token = invited()
    (user if suspend == "user" else user.organisation).is_active = False
    db.session.commit()

    response = _accept(client, token)

    assert response.status_code == 404
    assert INVALID_INVITE in response.get_data(as_text=True)
    assert db.session.get(User, user.id).password_hash is None


def test_invite_password_must_meet_the_policy(client, invited):
    user, token = invited()

    response = _accept(client, token, "too short")

    assert response.status_code == 200
    assert "Use between 12 and 256 characters." in response.get_data(as_text=True)
    assert pending_invite(token) is not None
