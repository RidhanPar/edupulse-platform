import re

from sqlalchemy import select

from db import db
from db.models import Organisation, Role, User

INVITE_LINK = re.compile(r"(https?://\S+)/invite/(\S+)")
NEW_PASSWORD = "a brand new passphrase"


def _run(app, *args):
    return app.test_cli_runner().invoke(args=list(args))


def _accept(client, token):
    return client.post(f"/invite/{token}", data={"password": NEW_PASSWORD, "confirm": NEW_PASSWORD})


def test_create_org_creates_the_owner_and_prints_a_working_invite_link(app, client, login):
    result = _run(app, "create-org", "Northfield College", "Owner@Northfield.ac.uk", "--base-url", "https://edupulse.example/")

    assert result.exit_code == 0, result.output
    base, token = INVITE_LINK.search(result.output).groups()
    assert base == "https://edupulse.example"
    organisation = db.session.scalars(select(Organisation).where(Organisation.slug == "northfield-college")).one()
    [owner] = organisation.users
    assert (owner.email, owner.role, owner.password_hash) == ("owner@northfield.ac.uk", Role.OWNER, None)
    assert _accept(client, token).status_code == 302
    assert login(client, "owner@northfield.ac.uk", NEW_PASSWORD).status_code == 302


def test_create_org_refuses_duplicates_and_bad_emails(app):
    assert _run(app, "create-org", "Northfield College", "owner@northfield.ac.uk").exit_code == 0

    same_slug = _run(app, "create-org", "Northfield  College", "other@northfield.ac.uk")
    same_email = _run(app, "create-org", "Southfield College", "OWNER@northfield.ac.uk")
    bad_email = _run(app, "create-org", "Eastfield College", "not-an-email")

    assert same_slug.exit_code != 0 and "already exists" in same_slug.output
    assert same_email.exit_code != 0 and "already exists" in same_email.output
    assert bad_email.exit_code != 0 and "not a valid email" in bad_email.output
    assert len(db.session.scalars(select(Organisation)).all()) == 1


def test_create_user_adds_an_invited_user_with_the_given_role(app, client):
    _run(app, "create-org", "Northfield College", "owner@northfield.ac.uk")

    result = _run(app, "create-user", "northfield-college", "tutor@northfield.ac.uk", "--role", "staff")

    assert result.exit_code == 0, result.output
    user = db.session.scalars(select(User).where(User.email == "tutor@northfield.ac.uk")).one()
    assert user.role is Role.STAFF and user.organisation.slug == "northfield-college"
    assert _accept(client, INVITE_LINK.search(result.output).group(2)).status_code == 302


def test_create_user_requires_an_existing_organisation_and_a_role(app):
    unknown_org = _run(app, "create-user", "nowhere", "tutor@nowhere.ac.uk", "--role", "staff")
    missing_role = _run(app, "create-user", "nowhere", "tutor@nowhere.ac.uk")

    assert unknown_org.exit_code != 0 and "No organisation" in unknown_org.output
    assert missing_role.exit_code != 0 and "--role" in missing_role.output


def test_reissue_invite_replaces_the_old_link_for_pending_users_only(app, client):
    first = _run(app, "create-org", "Northfield College", "owner@northfield.ac.uk")
    old_token = INVITE_LINK.search(first.output).group(2)

    reissued = _run(app, "reissue-invite", "owner@northfield.ac.uk")

    assert reissued.exit_code == 0, reissued.output
    new_token = INVITE_LINK.search(reissued.output).group(2)
    assert _accept(client, old_token).status_code == 404
    assert _accept(client, new_token).status_code == 302
    refused = _run(app, "reissue-invite", "owner@northfield.ac.uk")
    assert refused.exit_code != 0 and "already set a password" in refused.output
