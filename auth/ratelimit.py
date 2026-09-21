"""Client address resolution behind the proxy, and rate limits on credential endpoints.

Order matters: the limiter keys on request.remote_addr, which is only the real client
once ProxyFix has rewritten it from X-Forwarded-For. Without ProxyFix every request on
Render appears to come from the load balancer, so a handful of failed logins anywhere
on the platform would lock every user out of the login page.
"""
from __future__ import annotations

from flask import Flask, render_template
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
