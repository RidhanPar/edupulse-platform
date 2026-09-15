"""Account administration commands. There is no public sign-up."""
from __future__ import annotations

import re

import click
from flask.cli import with_appcontext
from sqlalchemy import select

from auth.invites import issue_invite
from auth.views import normalise_email
from db import db
from db.models import Organisation, Role, User

EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
SLUG_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")

base_url_option = click.option(
    "--base-url",
    envvar="APP_BASE_URL",
    default="http://127.0.0.1:5000",
    show_default=True,
    help="Public URL of the app, used to build the invite link. Also read from APP_BASE_URL.",
)


def _new_email(value: str) -> str:
    email = normalise_email(value)
    if len(email) > 320 or not EMAIL_PATTERN.fullmatch(email):
        raise click.BadParameter(f"{value!r} is not a valid email address.")
    if db.session.scalars(select(User.id).where(User.email == email)).first() is not None:
        raise click.ClickException(f"A user with the email {email} already exists.")
    return email


def _echo_invite(user: User, token: str, base_url: str) -> None:
    click.echo(f"One-time invite link for {user.email} (expires {user.invite_expires_at:%Y-%m-%d %H:%M} UTC):")
    click.echo(f"{base_url.rstrip('/')}/invite/{token}")


@click.command("create-org")
@click.argument("name")
@click.argument("owner_email")
@click.option("--slug", help="URL-safe identifier. Derived from NAME if omitted.")
@base_url_option
@with_appcontext
def create_org_command(name: str, owner_email: str, slug: str | None, base_url: str) -> None:
    """Create an organisation and its owner, and print the owner's invite link."""
    name = name.strip()
    slug = slug or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not name or len(name) > 200:
        raise click.BadParameter("NAME must be between 1 and 200 characters.")
    if len(slug) > 100 or not SLUG_PATTERN.fullmatch(slug):
        raise click.BadParameter(f"{slug!r} is not a valid slug: use lowercase letters, digits and hyphens.")
    if db.session.scalars(select(Organisation.id).where(Organisation.slug == slug)).first() is not None:
        raise click.ClickException(f"An organisation with the slug {slug} already exists.")
    email = _new_email(owner_email)

    organisation = Organisation(name=name, slug=slug)
    owner = User(organisation=organisation, email=email, role=Role.OWNER)
    token = issue_invite(owner)
    db.session.add_all([organisation, owner])
    db.session.commit()

    click.echo(f"Created organisation {name} ({slug}) with owner {email}.")
    _echo_invite(owner, token, base_url)


@click.command("create-user")
@click.argument("organisation_slug")
@click.argument("email")
@click.option("--role", type=click.Choice([role.value for role in Role]), required=True)
@base_url_option
@with_appcontext
def create_user_command(organisation_slug: str, email: str, role: str, base_url: str) -> None:
    """Add a user to an existing organisation, and print their invite link."""
    organisation = db.session.scalars(
        select(Organisation).where(Organisation.slug == organisation_slug)
    ).one_or_none()
    if organisation is None:
        raise click.ClickException(f"No organisation with the slug {organisation_slug}.")
    user = User(organisation=organisation, email=_new_email(email), role=Role(role))
    token = issue_invite(user)
    db.session.add(user)
    db.session.commit()

    click.echo(f"Created {role} {user.email} in {organisation.name}.")
    _echo_invite(user, token, base_url)


@click.command("reissue-invite")
@click.argument("email")
@base_url_option
@with_appcontext
def reissue_invite_command(email: str, base_url: str) -> None:
    """Replace the invite of a user who has not set a password yet. The old link stops working."""
    user = db.session.scalars(select(User).where(User.email == normalise_email(email))).one_or_none()
    if user is None:
        raise click.ClickException(f"No user with the email {email}.")
    if user.password_hash is not None:
        raise click.ClickException(f"{user.email} has already set a password; invites are only for new accounts.")
    token = issue_invite(user)
    db.session.commit()
    _echo_invite(user, token, base_url)


COMMANDS = (create_org_command, create_user_command, reissue_invite_command)
