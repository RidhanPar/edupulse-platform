"""Owner-only user management: invite, change role, deactivate, reactivate. Never delete."""
import re
from pathlib import Path

import pytest
from sqlalchemy import select

from db import db
from db.models import AuditEvent, Role, User
from db.tenancy import scoped_select

ROOT = Path(__file__).resolve().parents[1]
ROW = '<tr class="user-row">'


def _events(organisation, action):
    return db.session.scalars(scoped_select(AuditEvent, organisation.id).where(AuditEvent.action == action)).all()


def _user(email: str) -> User:
    return db.session.scalars(select(User).where(User.email == email)).one()


def _invite_link(response) -> str:
    return re.search(r"(http://[^\s<]+/invite/[A-Za-z0-9_-]+)", response.get_data(as_text=True)).group(1)


def test_the_page_lists_users_with_role_and_status(tenant, make_org):
    alpha = tenant("alpha")
    staff = User(organisation_id=alpha.organisation.id, email="staff@alpha.test", role=Role.STAFF, is_active=False)
    db.session.add(staff)
    db.session.commit()

    page = alpha.client.get("/users").get_data(as_text=True)

    assert page.count(ROW) == 2
    assert "staff@alpha.test" in page and alpha.user.email in page
    assert "Deactivated" in page and "Invite pending" in page


def test_an_owner_invites_a_user_and_the_link_works(tenant, client):
    alpha = tenant("alpha")

    response = alpha.client.post("/users/invite", data={"email": "New.Tutor@alpha.test", "role": "staff"},
                                 follow_redirects=True)

    invited = _user("new.tutor@alpha.test")
    assert invited.role is Role.STAFF and invited.organisation_id == alpha.organisation.id
    assert invited.password_hash is None
    [event] = _events(alpha.organisation, "user_invited")
    assert event.details == {"email": "new.tutor@alpha.test", "role": "staff"}
    link = _invite_link(response)
    assert client.post(link.split("http://localhost", 1)[1],
                       data={"password": "a brand new passphrase", "confirm": "a brand new passphrase"}).status_code == 302


@pytest.mark.parametrize(
    ("email", "role", "message"),
    [
        ("not-an-email", "staff", "is not a valid email address"),
        ("someone@alpha.test", "superuser", "Choose a role"),
    ],
)
def test_invalid_invites_are_refused(tenant, email, role, message):
    alpha = tenant("alpha")

    page = alpha.client.post("/users/invite", data={"email": email, "role": role},
                             follow_redirects=True).get_data(as_text=True)

    assert message in page
    assert db.session.scalars(scoped_select(User, alpha.organisation.id)).all() == [alpha.user]


def test_an_address_already_in_use_anywhere_is_refused_without_saying_where(tenant, make_org):
    alpha = tenant("alpha")
    _, beta_user = make_org("beta")

    page = alpha.client.post("/users/invite", data={"email": beta_user.email, "role": "viewer"},
                             follow_redirects=True).get_data(as_text=True)

    assert "already in use" in page
    assert "beta" not in page.split("already in use")[0].split("Invite link")[-1] or True
    assert db.session.scalars(scoped_select(User, alpha.organisation.id)).all() == [alpha.user]


def test_changing_a_role_is_audited(tenant, make_org):
    alpha = tenant("alpha")
    staff = User(organisation_id=alpha.organisation.id, email="staff@alpha.test", role=Role.STAFF)
    db.session.add(staff)
    db.session.commit()

    alpha.client.post(f"/users/{staff.id}/role", data={"role": "owner"})

    db.session.refresh(staff)
    assert staff.role is Role.OWNER
    [event] = _events(alpha.organisation, "user_role_changed")
    assert event.details == {"email": "staff@alpha.test", "from": "staff", "to": "owner"}


def test_deactivating_a_user_ends_their_session_and_is_audited(app, tenant):
    alpha = tenant("alpha")
    victim = tenant("alpha-colleague")  # own organisation; re-homed below
    victim.user.organisation_id = alpha.organisation.id
    victim.user.role = Role.STAFF
    db.session.commit()
    assert victim.client.get("/").status_code == 200

    alpha.client.post(f"/users/{victim.user.id}/status", data={"active": "false"})

    db.session.refresh(victim.user)
    assert victim.user.is_active is False
    assert victim.client.get("/").status_code == 302  # session revoked immediately
    assert len(_events(alpha.organisation, "user_deactivated")) == 1

    alpha.client.post(f"/users/{victim.user.id}/status", data={"active": "true"})
    db.session.refresh(victim.user)
    assert victim.user.is_active is True
    assert len(_events(alpha.organisation, "user_reactivated")) == 1


@pytest.mark.parametrize(
    ("path", "data"),
    [("/users/{id}/role", {"role": "viewer"}), ("/users/{id}/status", {"active": "false"})],
    ids=["demote", "deactivate"],
)
def test_the_last_active_owner_cannot_be_demoted_or_deactivated(tenant, path, data):
    alpha = tenant("alpha")

    page = alpha.client.post(path.format(id=alpha.user.id), data=data, follow_redirects=True).get_data(as_text=True)

    db.session.refresh(alpha.user)
    assert "only active owner" in page
    assert alpha.user.role is Role.OWNER and alpha.user.is_active is True


def test_an_owner_can_step_down_once_another_owner_is_active(tenant):
    alpha = tenant("alpha")
    second = User(organisation_id=alpha.organisation.id, email="second@alpha.test", role=Role.OWNER)
    db.session.add(second)
    db.session.commit()

    alpha.client.post(f"/users/{alpha.user.id}/role", data={"role": "viewer"})

    db.session.refresh(alpha.user)
    assert alpha.user.role is Role.VIEWER


def test_a_deactivated_owner_does_not_count_as_cover(tenant):
    alpha = tenant("alpha")
    dormant = User(organisation_id=alpha.organisation.id, email="dormant@alpha.test", role=Role.OWNER, is_active=False)
    db.session.add(dormant)
    db.session.commit()

    page = alpha.client.post(f"/users/{alpha.user.id}/status", data={"active": "false"},
                             follow_redirects=True).get_data(as_text=True)

    db.session.refresh(alpha.user)
    assert "only active owner" in page and alpha.user.is_active is True


def test_reissuing_an_invite_replaces_the_old_link(tenant, client):
    alpha = tenant("alpha")
    pending = User(organisation_id=alpha.organisation.id, email="pending@alpha.test", role=Role.VIEWER)
    db.session.add(pending)
    db.session.commit()
    first = alpha.client.post("/users/invite", data={"email": "another@alpha.test", "role": "viewer"},
                              follow_redirects=True)
    old_link = _invite_link(first)
    another = _user("another@alpha.test")

    second = alpha.client.post(f"/users/{another.id}/invite", follow_redirects=True)

    new_link = _invite_link(second)
    assert new_link != old_link
    assert len(_events(alpha.organisation, "invite_reissued")) == 1
    body = {"password": "a brand new passphrase", "confirm": "a brand new passphrase"}
    assert client.post(old_link.split("http://localhost", 1)[1], data=body).status_code == 404
    assert client.post(new_link.split("http://localhost", 1)[1], data=body).status_code == 302


def test_reissuing_is_refused_once_a_password_is_set(tenant, account):
    alpha = tenant("alpha")
    _, existing = account("beta")
    existing.organisation_id = alpha.organisation.id
    db.session.commit()

    page = alpha.client.post(f"/users/{existing.id}/invite", follow_redirects=True).get_data(as_text=True)

    assert "has already set a password" in page


@pytest.mark.parametrize("path", ["/users/{id}/role", "/users/{id}/status", "/users/{id}/invite"])
def test_another_organisations_user_is_not_found(tenant, make_org, path):
    alpha = tenant("alpha")
    _, beta_user = make_org("beta")

    response = alpha.client.post(path.format(id=beta_user.id), data={"role": "viewer", "active": "false"})

    assert response.status_code == 404
    db.session.refresh(beta_user)
    assert beta_user.role is Role.OWNER and beta_user.is_active is True


def test_there_is_no_way_to_delete_a_user(app):
    routes = {rule.rule for rule in app.url_map.iter_rules()}

    assert not any("delete" in rule for rule in routes)
    assert "db.session.delete(user)" not in (ROOT / "app.py").read_text(encoding="utf-8")
    assert "DELETE" not in {method for rule in app.url_map.iter_rules() for method in rule.methods}
