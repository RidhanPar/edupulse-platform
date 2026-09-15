"""Authentication and authorisation: Flask-Login, server-side sessions, CSRF and roles.

Every endpoint requires a logged-in user unless it is listed in PUBLIC_ENDPOINTS, and
every other endpoint must declare a role with auth.roles.require_role or the app
refuses to start. Forgetting either fails closed.
"""
from __future__ import annotations

import uuid

from flask import Flask, abort, g, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user
from flask_wtf.csrf import CSRFError, CSRFProtect

from auth.roles import user_has_role
from auth.sessions import DatabaseSessionInterface
from db import db
from db.models import User

PUBLIC_ENDPOINTS = frozenset({"auth.login", "auth.invite", "healthz", "static"})

login_manager = LoginManager()
csrf = CSRFProtect()


@login_manager.user_loader
def _load_user(user_id: str) -> User | None:
    try:
        user = db.session.get(User, uuid.UUID(user_id))
    except (TypeError, ValueError):
        return None
    # Deactivated users and suspended organisations lose access on their next request.
    if user is None or not user.is_active or not user.organisation.is_active:
        return None
    return user


@login_manager.unauthorized_handler
def _unauthorized():
    if request.path == "/api" or request.path.startswith("/api/"):
        abort(401)
    target = request.full_path if request.query_string else request.path
    return redirect(url_for("auth.login", next=target))


def _require_login():
    # Flask-Login caches the loaded user on g, which lives on the app context. Drop any
    # cached user so identity always comes from this request's session, even if an app
    # context outlives a single request (as it does in the test suite).
    g.pop("_login_user", None)
    if request.endpoint is None or request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if not current_user.is_authenticated:
        return login_manager.unauthorized()
    return None


def _csrf_failed(error: CSRFError):
    message = "This form has expired or did not come from EduPulse. Reload the page and try again."
    return render_template("error.html", message=message), 400


def init_auth(app: Flask) -> None:
    from auth import cli, views

    app.session_interface = DatabaseSessionInterface(app)
    csrf.init_app(app)
    login_manager.init_app(app)
    app.before_request(_require_login)
    app.register_blueprint(views.bp)
    app.register_error_handler(CSRFError, _csrf_failed)
    for command in cli.COMMANDS:
        app.cli.add_command(command)
    app.jinja_env.globals["user_has_role"] = lambda role: user_has_role(current_user, role)


def check_route_roles(app: Flask) -> None:
    """Refuse to start if any non-public endpoint forgot to declare a role."""
    undeclared = sorted(
        endpoint
        for endpoint, view in app.view_functions.items()
        if endpoint not in PUBLIC_ENDPOINTS and not hasattr(view, "required_roles")
    )
    if undeclared:
        raise RuntimeError(f"endpoints without a role requirement: {', '.join(undeclared)}")
