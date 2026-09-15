"""Application configuration, built from environment variables."""
from __future__ import annotations

import os
import secrets
from datetime import timedelta
from pathlib import Path

MAX_UPLOAD_BYTES = 5 * 1024 * 1024

# Environments that may run with a generated secret key, a local SQLite file and
# local file storage. Anything else, including an unset FLASK_ENV, is production.
LOCAL_ENVS = {"development", "testing"}

STORAGE_BACKENDS = ("local", "s3")
S3_SETTINGS = ("STORAGE_ENDPOINT", "STORAGE_BUCKET", "STORAGE_ACCESS_KEY", "STORAGE_SECRET_KEY")
DEFAULT_LOCAL_STORAGE_ROOT = Path(__file__).resolve().parent / "var" / "storage"

# OWASP's recommended Argon2id parameters (19 MiB, 2 iterations, 1 lane): strong, and
# small enough that concurrent logins do not exhaust a small instance's memory.
ARGON2_PARAMS = {"memory_cost": 19456, "time_cost": 2, "parallelism": 1}
# Deliberately weak so the test suite stays fast. Only used when FLASK_ENV=testing.
ARGON2_TEST_PARAMS = {"memory_cost": 1024, "time_cost": 1, "parallelism": 1}

# Idle timeout: each request pushes the expiry forward by this much.
SESSION_LIFETIME = timedelta(hours=8)


class ConfigError(RuntimeError):
    """Raised when the environment cannot produce a safe configuration."""


def resolve_env(env: str | None = None) -> str:
    return (env or os.environ.get("FLASK_ENV") or "production").strip().lower()


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def database_url(env: str) -> str | None:
    url = _env("DATABASE_URL")
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
    problems = []

    # A generated fallback key differs per Gunicorn worker, so a session signed
    # by one worker is rejected by the next and users are logged out at random.
    secret_key = _env("FLASK_SECRET_KEY")
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

    # The same applies to uploaded files and model artifacts.
    storage_backend = _env("STORAGE_BACKEND").lower() or ("local" if env in LOCAL_ENVS else "")
    if not storage_backend:
        missing.append("STORAGE_BACKEND")
    elif storage_backend not in STORAGE_BACKENDS:
        problems.append(f"STORAGE_BACKEND must be one of {', '.join(STORAGE_BACKENDS)}, not {storage_backend!r}")
    elif storage_backend == "local" and env not in LOCAL_ENVS:
        problems.append("STORAGE_BACKEND=local is only allowed when FLASK_ENV is development or testing")
    elif storage_backend == "s3":
        missing.extend(name for name in S3_SETTINGS if not _env(name))

    if missing:
        problems.insert(0, f"missing required environment variable(s) {', '.join(missing)}")
    if problems:
        raise ConfigError(
            f"{'; '.join(problems)} (FLASK_ENV={env!r}). "
            "Set them, or set FLASK_ENV=development for local work."
        )

    return {
        "ENV_NAME": env,
        "TESTING": env == "testing",
        "SECRET_KEY": secret_key,
        "MAX_CONTENT_LENGTH": MAX_UPLOAD_BYTES,
        "SQLALCHEMY_DATABASE_URI": db_url,
        "SQLALCHEMY_ENGINE_OPTIONS": {"pool_pre_ping": True},
        "STORAGE_BACKEND": storage_backend,
        "STORAGE_LOCAL_ROOT": _env("STORAGE_LOCAL_ROOT") or str(DEFAULT_LOCAL_STORAGE_ROOT),
        **{name: _env(name) for name in S3_SETTINGS},
        # Server-side sessions (auth.sessions): the cookie carries only a random id.
        "SESSION_COOKIE_SECURE": True,
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "PERMANENT_SESSION_LIFETIME": SESSION_LIFETIME,
        # Delete expired session rows on average once every N requests.
        "SESSION_CLEANUP_N_REQUESTS": 200,
        # CSRF tokens last as long as the session, instead of failing on a form left open an hour.
        "WTF_CSRF_TIME_LIMIT": None,
        "ARGON2_PARAMS": ARGON2_TEST_PARAMS if env == "testing" else ARGON2_PARAMS,
        "MODEL_CACHE_SIZE": 4,
    }
