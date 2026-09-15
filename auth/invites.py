"""One-time invite tokens. Only a SHA-256 of each token is ever stored."""
from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta

from sqlalchemy import select, update

from auth import passwords
from db import db
from db.audit import record_audit_event
from db.models import Organisation, User, utcnow

INVITE_TTL = timedelta(days=7)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_invite(user: User) -> str:
    """Give `user` a new invite, replacing any earlier one. Returns the raw token; the caller commits."""
    token = secrets.token_urlsafe(32)
    user.invite_token_hash = _token_hash(token)
    user.invite_expires_at = utcnow() + INVITE_TTL
    return token


def _redeemable(token: str) -> tuple:
    # Unknown, already used, expired, deactivated and suspended all fail these same
    # conditions, so no caller can tell them apart.
    return (
        User.invite_token_hash == _token_hash(token),
        User.invite_expires_at > utcnow(),
        User.is_active.is_(True),
        User.organisation_id.in_(select(Organisation.id).where(Organisation.is_active.is_(True))),
    )


def pending_invite(token: str) -> User | None:
    return db.session.scalars(select(User).where(*_redeemable(token))).one_or_none()


def consume_invite(token: str, password: str) -> User | None:
    """Set the password and burn the token in one UPDATE.

    The token is re-checked inside that statement, so two concurrent submissions of the
    same link cannot both succeed: whichever runs second matches no row.
    """
    result = db.session.execute(
        update(User)
        .where(*_redeemable(token))
        .values(password_hash=passwords.hash_password(password), invite_token_hash=None, invite_expires_at=None)
        .returning(User.id)
        .execution_options(synchronize_session=False)
    )
    user_id = result.scalar_one_or_none()
    if user_id is None:
        db.session.rollback()
        return None
    user = db.session.get(User, user_id)
    db.session.refresh(user)
    record_audit_event("invite_accepted", user=user)
    db.session.commit()
    return user
