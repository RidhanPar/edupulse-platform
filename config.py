"""Application configuration, built from environment variables."""
from __future__ import annotations

import os
import secrets

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# Environments that may run with a generated secret key and a local SQLite file.
# Anything else, including an unset FLASK_ENV, is treated as production.
LOCAL_ENVS = {"development", "testing"}


class ConfigError(RuntimeError):
    """Raised when the environment cannot produce a safe configuration."""


def resolve_env(env: str | None = None) -> str:
    return (env or os.environ.get("FLASK_ENV") or "production").strip().lower()


def database_url(env: str) -> str | None:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url.startswith("postgres://"):
        # Render hands out postgres://, which SQLAlchemy 2 rejects.
        url = "postgresql://" + url[len("postgres://"):]
    if url:
        return url
    if env in LOCAL_ENVS:
        # Relative SQLite paths resolve inside the Flask instance folder.
        return "sqlite:///edupulse.db"
    return None


def build_config(env: str) -> dict:
    missing = []

    # A generated fallback key differs per Gunicorn worker, so a session signed
    # by one worker is rejected by the next and users are logged out at random.
    secret_key = os.environ.get("FLASK_SECRET_KEY", "").strip()
    if not secret_key:
        if env in LOCAL_ENVS:
            secret_key = secrets.token_hex(32)
        else:
            missing.append("FLASK_SECRET_KEY")

    # Falling back to SQLite in production would put tenant data back on
    # Render's ephemeral disk.
    db_url = database_url(env)
    if db_url is None:
        missing.append("DATABASE_URL")

    if missing:
        raise ConfigError(
            f"missing required environment variable(s) {', '.join(missing)} "
            f"(FLASK_ENV={env!r}). Set them, or set FLASK_ENV=development for local work."
        )

    return {
        "ENV_NAME": env,
        "TESTING": env == "testing",
        "SECRET_KEY": secret_key,
        "MAX_CONTENT_LENGTH": MAX_UPLOAD_BYTES,
        "SQLALCHEMY_DATABASE_URI": db_url,
        "SQLALCHEMY_ENGINE_OPTIONS": {"pool_pre_ping": True},
    }
