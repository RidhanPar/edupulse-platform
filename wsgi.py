"""WSGI entry point for `gunicorn wsgi:app` and the `flask` CLI."""
import sys

from app import create_app
from config import ConfigError

try:
    app = create_app()
except ConfigError as exc:
    print(f"EduPulse cannot start: {exc}", file=sys.stderr)
    # Gunicorn treats exit code 4 as "App failed to load" and halts, rather
    # than logging a traceback and respawning workers in a loop.
    raise SystemExit(4)
