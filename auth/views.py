"""Login, logout, password change and invite acceptance."""
from __future__ import annotations

from urllib.parse import urlsplit

from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_user, logout_user
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from auth import passwords
from auth.forms import ChangePasswordForm, LoginForm, SetPasswordForm
from auth.invites import consume_invite, pending_invite
from auth.roles import require_role
from db import db
from db.audit import record_audit_event
from db.models import Role, User, utcnow

bp = Blueprint("auth", __name__)

# One message for every failure, so a response never reveals whether an address is
# registered, the password was wrong, the account is deactivated or the organisation
# suspended. The audit event records the real reason.
LOGIN_FAILED = "Invalid email or password."
INVALID_INVITE = "This invite link is invalid or has expired."


def normalise_email(email: str | None) -> str:
    return (email or "").strip().lower()


def safe_next_url(target: str | None) -> str | None:
    """Accept only a local path, so the login page cannot bounce a user to another site."""
    if not target or not target.startswith("/") or target.startswith("//"):
        return None
    if "\\" in target or any(character.isspace() or ord(character) < 32 for character in target):
        return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc:
        return None
    return target


def authenticate(email: str, password: str) -> User | None:
    """The user for these credentials if they may log in; otherwise None, with the attempt audited."""
    user = db.session.scalars(
        select(User).options(joinedload(User.organisation)).where(User.email == normalise_email(email))
    ).one_or_none()
    stored_hash = user.password_hash if user is not None else None
    # Exactly one Argon2 verification on every path, so timing does not reveal registered addresses.
    password_ok = passwords.verify_password(password, stored_hash or passwords.dummy_hash()) and stored_hash is not None

    if user is None:
        reason = "unknown_email"
    elif not password_ok:
        reason = "wrong_password" if stored_hash else "invite_not_accepted"
    elif not user.is_active:
        reason = "user_deactivated"
    elif not user.organisation.is_active:
        reason = "organisation_suspended"
    else:
        return user

    record_audit_event("login_failure", user=user, details={"reason": reason})
    db.session.commit()
    return None


def _begin_session(user: User, password: str) -> None:
    # Rotate the session id at the moment of login, so an id planted before login (session
    # fixation) is discarded and never becomes authenticated.
    current_app.session_interface.regenerate(session)
    session.clear()
    login_user(user)
    user.last_login_at = utcnow()
    if passwords.needs_rehash(user.password_hash):
        user.password_hash = passwords.hash_password(password)
    record_audit_event("login_success", user=user)
    db.session.commit()


@bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    form = LoginForm()
    if form.validate_on_submit():
        user = authenticate(form.email.data, form.password.data)
        if user is not None:
            _begin_session(user, form.password.data)
            return redirect(safe_next_url(request.args.get("next")) or url_for("home"))
        flash(LOGIN_FAILED, "danger")
    return render_template("login.html", form=form)


@bp.route("/logout", methods=["POST"])
@require_role(Role.VIEWER)
def logout():
    record_audit_event("logout", user=current_user)
    db.session.commit()
    logout_user()
    current_app.session_interface.regenerate(session)
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("auth.login"))


@bp.route("/change-password", methods=["GET", "POST"])
@require_role(Role.VIEWER)
def change_password():
    form = ChangePasswordForm()
    if form.validate_on_submit():
        if not passwords.verify_password(form.current_password.data, current_user.password_hash):
            form.current_password.errors.append("Your current password is incorrect.")
        else:
            current_user.password_hash = passwords.hash_password(form.password.data)
            record_audit_event("password_changed", user=current_user)
            db.session.commit()
            # End every other session for this user, and rotate this one's id.
            interface = current_app.session_interface
            interface.regenerate(session)
            interface.revoke_user_sessions(current_user.id)
            flash("Your password has been changed.", "success")
            return redirect(url_for("home"))
    return render_template("change_password.html", form=form)


@bp.route("/invite/<token>", methods=["GET", "POST"])
def invite(token: str):
    user = pending_invite(token)
    if user is None:
        return render_template("error.html", message=INVALID_INVITE), 404
    form = SetPasswordForm()
    if form.validate_on_submit():
        if consume_invite(token, form.password.data) is None:
            return render_template("error.html", message=INVALID_INVITE), 404
        flash("Your password is set. Log in to continue.", "success")
        return redirect(url_for("auth.login"))
    return render_template("accept_invite.html", form=form, email=user.email)
