"""Client address resolution behind the proxy, and rate limits on credential endpoints.

Order matters: the limiter keys on request.remote_addr, which is only the real client
once ProxyFix has rewritten it from X-Forwarded-For. Without ProxyFix every request on
Render appears to come from the load balancer, so a handful of failed logins anywhere
on the platform would lock every user out of the login page.
"""
from __future__ import annotations

from flask import Flask, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

# Counted per client address, and only for failed attempts (see _failed_attempt), so a
# school's shared NAT address is not locked out by its users logging in successfully.
LOGIN_RATE_LIMIT = "5 per minute;30 per hour"

# Endpoints that accept a secret and are open to anonymous users.
RATE_LIMITED_ENDPOINTS = ("auth.login", "auth.invite")

RATE_LIMITED_MESSAGE = "Too many attempts from your network. Wait a minute and try again."


def trust_proxy(app: Flask) -> None:
    """Take the client address and scheme from the proxy's forwarding headers.

    Trusts exactly PROXY_HOPS proxies (1 on Render). Only the last address in
    X-Forwarded-For, the one appended by our own proxy, is believed; anything a client
    prepends to that header is ignored.
    """
    hops = app.config["PROXY_HOPS"]
    if hops:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=hops, x_proto=hops)
    app.logger.info("Trusting %d proxy hop(s) for X-Forwarded-For and X-Forwarded-Proto (PROXY_HOPS)", hops)
    _log_first_forwarded_header(app)


def _log_first_forwarded_header(app: Flask) -> None:
    """Log the raw X-Forwarded-For of the first real request this process serves.

    The hop count must come from what production actually sends, not a guess. A browser
    sends no X-Forwarded-For of its own, so on a normal request the number of entries
    equals the number of proxies in front of the app: set PROXY_HOPS to that number.
    Health checks are skipped because the platform's checker may not route like a user.
    """
    logged = False

    @app.before_request
    def _log_once():
        nonlocal logged
        if logged or request.endpoint in ("healthz", "static"):
            return
        logged = True
        raw = request.headers.get("X-Forwarded-For", "")
        entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
        original_peer = request.environ.get("werkzeug.proxy_fix.orig", {}).get("REMOTE_ADDR", request.remote_addr)
        app.logger.info(
            "First request: X-Forwarded-For=%r has %d entr%s, connecting peer=%s, PROXY_HOPS=%d. "
            "If this was an ordinary browser request, PROXY_HOPS should equal the number of entries.",
            raw, len(entries), "y" if len(entries) == 1 else "ies", original_peer, app.config["PROXY_HOPS"],
        )


def _failed_attempt(response) -> bool:
    # A successful login or invite acceptance redirects; failures re-render the form.
    return response.status_code != 302


def init_rate_limits(app: Flask) -> Limiter:
    """Give this app its own limiter and apply the credential limits to its views.

    A limiter per app (not a module global) keeps each app's counters and storage
    separate. The storage is in-process memory (RATELIMIT_STORAGE_URI=memory://),
    which is correct only while Gunicorn runs a single worker: see render.yaml.
    """
    limiter = Limiter(get_remote_address, app=app, default_limits=[])
    for endpoint in RATE_LIMITED_ENDPOINTS:
        app.view_functions[endpoint] = limiter.limit(
            LOGIN_RATE_LIMIT, methods=["POST"], deduct_when=_failed_attempt
        )(app.view_functions[endpoint])

    @app.errorhandler(429)
    def _rate_limited(error):
        return render_template("error.html", message=RATE_LIMITED_MESSAGE), 429

    app.extensions["edupulse_limiter"] = limiter
    return limiter
