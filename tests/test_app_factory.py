import os
import subprocess
import sys
from pathlib import Path

import pytest

from app import create_app
from config import ConfigError, database_url

ROOT = Path(__file__).resolve().parents[1]
CONFIG_VARS = ("FLASK_ENV", "FLASK_SECRET_KEY", "DATABASE_URL")


@pytest.fixture()
def clean_env(monkeypatch):
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _run_python(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in CONFIG_VARS}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize("env", ["production", None, "staging"])
def test_non_local_env_refuses_to_boot_without_secret_key(clean_env, env):
    clean_env.setenv("DATABASE_URL", "sqlite://")

    with pytest.raises(ConfigError, match="FLASK_SECRET_KEY"):
        create_app(env=env)


def test_production_refuses_to_boot_without_database_url(clean_env):
    clean_env.setenv("FLASK_SECRET_KEY", "k" * 64)

    with pytest.raises(ConfigError, match="DATABASE_URL"):
        create_app(env="production")


def test_production_boots_with_required_variables(clean_env):
    clean_env.setenv("FLASK_SECRET_KEY", "k" * 64)
    clean_env.setenv("DATABASE_URL", "sqlite://")

    app = create_app(env="production")

    assert app.secret_key == "k" * 64
    assert app.testing is False


@pytest.mark.parametrize("env", ["development", "testing"])
def test_local_envs_generate_a_secret_key(clean_env, env):
    clean_env.setenv("DATABASE_URL", "sqlite://")

    assert len(create_app(env=env).secret_key) == 64


def test_only_local_envs_fall_back_to_sqlite(clean_env):
    assert database_url("development").startswith("sqlite:///")
    assert database_url("production") is None


def test_render_postgres_scheme_is_normalised(clean_env):
    clean_env.setenv("DATABASE_URL", "postgres://u:p@db.internal:5432/edupulse")

    assert database_url("production") == "postgresql://u:p@db.internal:5432/edupulse"


def test_upload_limit_is_still_5_mb(app):
    assert app.config["MAX_CONTENT_LENGTH"] == 5 * 1024 * 1024


def test_wsgi_fails_loudly_without_secret_key():
    result = _run_python("import wsgi")

    assert result.returncode == 4
    assert "FLASK_SECRET_KEY" in result.stderr
    assert "Traceback" not in result.stderr


def test_create_app_does_not_import_ml_libraries():
    result = _run_python(
        "import sys\n"
        "from app import create_app\n"
        "create_app({'SQLALCHEMY_DATABASE_URI': 'sqlite://'}, env='testing')\n"
        "heavy = ['sklearn', 'joblib', 'utils.preprocessing', 'utils.train_model',"
        " 'utils.predict', 'utils.compare_results']\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


EXPECTED_ROUTES = {
    ("/", "home", frozenset({"GET"})),
    ("/upload-train", "upload_train", frozenset({"GET", "POST"})),
    ("/upload-predict", "upload_predict", frozenset({"GET", "POST"})),
    ("/train", "train", frozenset({"GET", "POST"})),
    ("/results", "results", frozenset({"GET"})),
    ("/download-results", "download_results", frozenset({"GET"})),
    ("/explain", "explain", frozenset({"GET"})),
    ("/about", "about", frozenset({"GET"})),
    ("/upload-actual", "upload_actual", frozenset({"GET", "POST"})),
    ("/compare", "compare", frozenset({"GET"})),
    ("/recheck-comparison", "recheck_comparison", frozenset({"POST"})),
}


def test_route_table_is_unchanged(app):
    routes = {
        (rule.rule, rule.endpoint, frozenset(rule.methods - {"HEAD", "OPTIONS"}))
        for rule in app.url_map.iter_rules()
        if rule.endpoint != "static"
    }

    assert routes == EXPECTED_ROUTES


@pytest.mark.parametrize(
    "path",
    ["/", "/about", "/upload-train", "/upload-predict", "/upload-actual", "/train", "/explain", "/results", "/compare"],
)
def test_existing_pages_still_render(client, path):
    response = client.get(path)

    assert response.status_code == 200, response.data[:500]


def test_download_results_still_exports_filtered_csv(client):
    response = client.get("/download-results?risk=High")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    header, *rows = response.data.decode().splitlines()
    assert header == "student_id,student_name,prediction,fail_probability,confidence,risk_level,recommendation"
    assert rows and all(",High," in row for row in rows)
