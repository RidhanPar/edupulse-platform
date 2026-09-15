"""Role checks. Roles are ranked: viewer < staff < owner."""
from __future__ import annotations

import functools

from flask import abort, current_app, request
from flask_login import current_user

from db.models import Role

ROLE_RANK = {Role.VIEWER: 0, Role.STAFF: 1, Role.OWNER: 2}
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def user_has_role(user, minimum) -> bool:
    return bool(user.is_authenticated) and ROLE_RANK[user.role] >= ROLE_RANK[Role(minimum)]


def require_role(minimum: Role, *, write: Role | None = None):
    """Require `minimum` to read, and `write` (defaulting to `minimum`) for POST and other writes."""
    write = write or minimum

    def decorator(view):
        @functools.wraps(view)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                return current_app.login_manager.unauthorized()
            needed = minimum if request.method in SAFE_METHODS else write
            if not user_has_role(current_user, needed):
                abort(403)
            return view(*args, **kwargs)

        wrapped.required_roles = (minimum, write)
        return wrapped

    return decorator
