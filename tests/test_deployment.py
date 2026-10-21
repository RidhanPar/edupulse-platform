"""The Render blueprint matches what the application needs, and operational logging works."""
import logging
import re
from pathlib import Path

import pytest

from app import create_app
from config import PRODUCTION_REQUIRED

ROOT = Path(__file__).resolve().parents[1]
BLUEPRINT = (ROOT / "render.yaml").read_text(encoding="utf-8")


def _env_block(key: str) -> str:
    match = re.search(rf"- key: {key}\n((?:\s{{8,}}.*\n)*)", BLUEPRINT)
    assert match, f"{key} is not declared in render.yaml"
    return match.group(1)


@pytest.mark.parametrize("variable", PRODUCTION_REQUIRED)
def test_every_variable_production_requires_is_declared(variable):
    _env_block(variable)


@pytest.mark.parametrize("secret", ["FLASK_SECRET_KEY", "MODEL_SIGNING_KEY"])
def test_secrets_are_generated_by_render_not_written_here(secret):
    assert "generateValue: true" in _env_block(secret)


def test_storage_credentials_are_set_in_the_dashboard_not_the_repo():
    for key in ("STORAGE_ENDPOINT", "STORAGE_BUCKET", "STORAGE_ACCESS_KEY", "STORAGE_SECRET_KEY"):
        assert "sync: false" in _env_block(key), key


def test_the_database_url_comes_from_the_managed_database():
    assert "fromDatabase:" in _env_block("DATABASE_URL")


def test_the_service_is_paid_migrates_before_deploying_and_is_health_checked():
    assert re.search(r"type: web[\s\S]*?plan: starter", BLUEPRINT)
    assert "preDeployCommand: flask --app wsgi db upgrade" in BLUEPRINT
    assert "healthCheckPath: /healthz" in BLUEPRINT
    assert "FLASK_ENV" in BLUEPRINT and "value: production" in BLUEPRINT


def test_everything_runs_in_the_eu():
    assert BLUEPRINT.count("region: frankfurt") == 2


def test_the_configured_hop_count_is_logged_at_startup(caplog, tmp_path):
    caplog.set_level(logging.INFO)

    create_app({"SQLALCHEMY_DATABASE_URI": "sqlite://", "STORAGE_LOCAL_ROOT": str(tmp_path), "PROXY_HOPS": 2},
               env="testing")

    assert "Trusting 2 proxy hop(s)" in caplog.text


def test_the_first_real_request_logs_its_forwarded_header_once(app, caplog):
    caplog.set_level(logging.INFO)
    client = app.test_client()
    client.environ_base.update(REMOTE_ADDR="10.0.0.1", HTTP_X_FORWARDED_FOR="203.0.113.5")

    client.get("/healthz")  # skipped: the platform's checker may not route like a user
    client.get("/login")
    client.get("/login")

    lines = [record.getMessage() for record in caplog.records if record.getMessage().startswith("First request")]
    assert len(lines) == 1
    assert "X-Forwarded-For='203.0.113.5' has 1 entry" in lines[0]
    assert "connecting peer=10.0.0.1" in lines[0] and "PROXY_HOPS=1" in lines[0]
