import re
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import select

from auth import PUBLIC_ENDPOINTS, passwords
from auth.roles import require_role
from auth.views import LOGIN_FAILED, safe_next_url
from db import db
from db.models import AuditEvent, Role, UserSession

NEW_PASSWORD = "a brand new passphrase"


def _events(action: str) -> list[AuditEvent]:
    return db.session.scalars(select(AuditEvent).where(AuditEvent.action == action)).all()


def _without_csrf_token(page: str) -> str:
    return re.sub(r'name="csrf_token" value="[^"]*"', 'name="csrf_token"', page)


def test_only_login_invite_healthz_and_static_are_public():
    assert PUBLIC_ENDPOINTS == {"auth.login", "auth.invite", "healthz", "static"}


def test_healthz_is_public(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_login_page_is_public(client):
    assert client.get("/login").status_code == 200


def test_every_protected_route_redirects_anonymous_users_to_login(app, client):
    checked = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint in PUBLIC_ENDPOINTS:
            continue
        method = "GET" if "GET" in rule.methods else "POST"

        response = client.open(rule.rule, method=method)

        location = urlsplit(response.headers.get("Location", ""))
        assert response.status_code == 302, rule.rule
        assert location.path == "/login", rule.rule
        assert parse_qs(location.query)["next"] == [rule.rule], rule.rule
        checked.append(rule.rule)
    assert len(checked) == 13


def test_api_requests_get_401_instead_of_a_login_redirect(app, client):
    app.add_url_rule("/api/ping", endpoint="api_ping", view_func=require_role(Role.VIEWER)(lambda: "pong"))

    assert client.get("/api/ping").status_code == 401


def test_login_logs_the_user_in_and_audits_it(account, client, login):
    organisation, user = account("alpha")

    response = login(client, "  OWNER@Alpha.test ")

    assert response.status_code == 302 and response.headers["Location"] == "/"
    assert client.get("/").status_code == 200
    [event] = _events("login_success")
    assert (event.user_id, event.organisation_id) == (user.id, organisation.id)
    assert event.ip_address == "127.0.0.1"
    assert event.user_agent.startswith("Werkzeug/")
    db.session.refresh(user)
    assert user.last_login_at is not None


def test_login_rotates_the_session_id_so_a_planted_id_is_useless(app, account, login):
    _, user = account("alpha")
    victim = app.test_client()
    with victim.session_transaction() as planted:
        planted["planted_by"] = "attacker"
    planted_sid = victim.get_cookie("session").value

    login(victim, user.email)

    assert victim.get_cookie("session").value != planted_sid
    assert db.session.get(UserSession, f"session:{planted_sid}") is None
    attacker = app.test_client()
    attacker.set_cookie("session", planted_sid)
    assert attacker.get("/").status_code == 302
    assert victim.get("/").status_code == 200


def test_session_cookie_is_secure_httponly_samesite_lax_and_holds_only_an_id(account, client, login):
    _, user = account("alpha")

    response = login(client, user.email)

    [cookie] = [header for header in response.headers.getlist("Set-Cookie") if header.startswith("session=")]
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=Lax" in cookie
    assert str(user.id) not in cookie
    sid = cookie.split(";", 1)[0].split("=", 1)[1]
    stored = db.session.get(UserSession, f"session:{sid}")
    assert stored is not None and stored.user_id == user.id


def test_failed_login_for_a_registered_email_records_the_user_and_organisation(account, client, login):
    organisation, user = account("alpha")

    response = login(client, user.email, "not the password at all")

    assert response.status_code == 200 and LOGIN_FAILED in response.get_data(as_text=True)
    [event] = _events("login_failure")
    assert (event.user_id, event.organisation_id) == (user.id, organisation.id)
    assert event.details == {"reason": "wrong_password"}
    assert client.get("/").status_code == 302


def test_failed_login_for_an_unknown_email_leaves_user_and_organisation_null(client, login):
    login(client, "nobody@nowhere.test", "whatever password")

    [event] = _events("login_failure")
    assert (event.user_id, event.organisation_id) == (None, None)
    assert event.details == {"reason": "unknown_email"}


def test_every_login_failure_looks_identical_to_the_client(app, account, make_org, login, password):
    _, wrong_password = account("alpha")
    _, deactivated = account("beta")
    suspended_organisation, suspended = account("gamma")
    _, never_accepted_invite = make_org("delta")
    deactivated.is_active = False
    suspended_organisation.is_active = False
    db.session.commit()
    attempts = {
        "wrong_password": (wrong_password.email, "not the password at all"),
        "unknown_email": ("nobody@nowhere.test", password),
        "user_deactivated": (deactivated.email, password),
        "organisation_suspended": (suspended.email, password),
        "invite_not_accepted": (never_accepted_invite.email, password),
    }

    pages = {}
    for reason, (email, attempt_password) in attempts.items():
        response = login(app.test_client(), email, attempt_password)
        assert response.status_code == 200, reason
        pages[reason] = _without_csrf_token(response.get_data(as_text=True)).replace(email, "EMAIL")

    assert len(set(pages.values())) == 1
    page = pages["organisation_suspended"]
    assert LOGIN_FAILED in page
    assert not re.search(r"suspend|deactivat|inactive|invite", page, re.IGNORECASE)
    assert sorted(event.details["reason"] for event in _events("login_failure")) == sorted(attempts)


def test_unknown_email_still_costs_one_password_verification(client, login, monkeypatch):
    verified_against = []
    real_verify = passwords.verify_password
    monkeypatch.setattr(
        passwords, "verify_password", lambda pw, pw_hash: verified_against.append(pw_hash) or real_verify(pw, pw_hash)
    )

    login(client, "nobody@nowhere.test", "whatever password")

    assert verified_against == [passwords.dummy_hash()]


def test_suspended_organisation_still_verifies_the_real_password(account, client, login, monkeypatch):
    organisation, user = account("alpha")
    organisation.is_active = False
    db.session.commit()
    verified_against = []
    real_verify = passwords.verify_password
    monkeypatch.setattr(
        passwords, "verify_password", lambda pw, pw_hash: verified_against.append(pw_hash) or real_verify(pw, pw_hash)
    )

    login(client, user.email)

    assert verified_against == [user.password_hash]


@pytest.mark.parametrize(
    "target",
    ["https://evil.example/", "//evil.example/", "/\\evil.example", "/\t/evil.example", "javascript:alert(1)", "", None],
)
def test_unsafe_next_targets_are_rejected(target):
    assert safe_next_url(target) is None


def test_login_follows_only_a_local_next_path(app, account, login):
    _, user = account("alpha")

    local = login(app.test_client(), user.email, next_url="/results?risk=High")
    external = login(app.test_client(), user.email, next_url="https://evil.example/")

    assert local.headers["Location"] == "/results?risk=High"
    assert external.headers["Location"] == "/"


def test_logout_is_post_only_audited_and_ends_the_session(app, account, login):
    _, user = account("alpha")
    client = app.test_client()
    login(client, user.email)
    sid = client.get_cookie("session").value

    assert client.get("/logout").status_code == 405
    response = client.post("/logout")

    assert response.status_code == 302 and response.headers["Location"] == "/login"
    assert len(_events("logout")) == 1
    assert client.get("/").status_code == 302
    replay = app.test_client()
    replay.set_cookie("session", sid)
    assert replay.get("/").status_code == 302


def test_change_password_requires_the_current_password(account, client, login):
    _, user = account("alpha")
    login(client, user.email)
    old_hash = user.password_hash

    response = client.post(
        "/change-password",
        data={"current_password": "not my password", "password": NEW_PASSWORD, "confirm": NEW_PASSWORD},
    )

    assert response.status_code == 200
    assert "Your current password is incorrect." in response.get_data(as_text=True)
    db.session.refresh(user)
    assert user.password_hash == old_hash


@pytest.mark.parametrize(
    ("new_password", "confirm", "message"),
    [
        ("short", "short", "Use between 12 and 256 characters."),
        (NEW_PASSWORD, "a different passphrase", "The passwords do not match."),
    ],
)
def test_change_password_validates_the_new_password(account, client, login, password, new_password, confirm, message):
    _, user = account("alpha")
    login(client, user.email)

    response = client.post(
        "/change-password", data={"current_password": password, "password": new_password, "confirm": confirm}
    )

    assert response.status_code == 200
    assert message in response.get_data(as_text=True)


def test_changing_password_ends_other_sessions_and_rotates_this_one(app, account, login, password):
    _, user = account("alpha")
    this_device, other_device = app.test_client(), app.test_client()
    login(this_device, user.email)
    login(other_device, user.email)
    sid_before = this_device.get_cookie("session").value

    response = this_device.post(
        "/change-password", data={"current_password": password, "password": NEW_PASSWORD, "confirm": NEW_PASSWORD}
    )

    assert response.status_code == 302
    assert this_device.get_cookie("session").value != sid_before
    assert this_device.get("/").status_code == 200
    assert other_device.get("/").status_code == 302
    assert len(_events("password_changed")) == 1
    assert login(app.test_client(), user.email, password).status_code == 200  # old password now fails
    assert login(app.test_client(), user.email, NEW_PASSWORD).status_code == 302


def test_passwords_are_argon2id_and_upgraded_when_parameters_change(account, client, login, password):
    _, user = account("alpha")
    assert user.password_hash.startswith("$argon2id$")
    user.password_hash = passwords._context(2048, 1, 1).hash(password)
    db.session.commit()
    assert passwords.needs_rehash(user.password_hash)

    login(client, user.email)

    db.session.refresh(user)
    assert not passwords.needs_rehash(user.password_hash)
    assert passwords.verify_password(password, user.password_hash)
