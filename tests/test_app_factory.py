import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from app import create_app
from config import ConfigError, database_url
from utils.storage import LocalStorage, S3Storage

ROOT = Path(__file__).resolve().parents[1]
S3_VARS = ("STORAGE_ENDPOINT", "STORAGE_BUCKET", "STORAGE_ACCESS_KEY", "STORAGE_SECRET_KEY")
CONFIG_VARS = ("FLASK_ENV", "FLASK_SECRET_KEY", "DATABASE_URL", "STORAGE_BACKEND", "STORAGE_LOCAL_ROOT", *S3_VARS)


@pytest.fixture()
def clean_env(monkeypatch):
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _set_production_env(monkeypatch, *, skip=()):
    values = {
        "FLASK_SECRET_KEY": "k" * 64,
        "DATABASE_URL": "sqlite://",
        "STORAGE_BACKEND": "s3",
        "STORAGE_ENDPOINT": "https://s3.us-east-1.amazonaws.com",
        "STORAGE_BUCKET": "edupulse-test",
        "STORAGE_ACCESS_KEY": "test-access-key",
        "STORAGE_SECRET_KEY": "test-secret-key",
    }
    for name, value in values.items():
        if name not in skip:
            monkeypatch.setenv(name, value)


def _run_python(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in CONFIG_VARS}
    return subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120
    )


@pytest.mark.parametrize("env", ["production", None, "staging"])
def test_non_local_env_refuses_to_boot_without_secret_key(clean_env, env):
    _set_production_env(clean_env, skip={"FLASK_SECRET_KEY"})

    with pytest.raises(ConfigError, match="FLASK_SECRET_KEY"):
        create_app(env=env)


def test_production_refuses_to_boot_without_database_url(clean_env):
    _set_production_env(clean_env, skip={"DATABASE_URL"})

    with pytest.raises(ConfigError, match="DATABASE_URL"):
        create_app(env="production")


def test_production_refuses_to_boot_without_storage_backend(clean_env):
    _set_production_env(clean_env, skip={"STORAGE_BACKEND"})

    with pytest.raises(ConfigError, match="STORAGE_BACKEND"):
        create_app(env="production")


def test_production_refuses_local_storage(clean_env):
    _set_production_env(clean_env)
    clean_env.setenv("STORAGE_BACKEND", "local")

    with pytest.raises(ConfigError, match="STORAGE_BACKEND=local is only allowed"):
        create_app(env="production")


def test_s3_storage_names_every_missing_setting(clean_env):
    _set_production_env(clean_env, skip=set(S3_VARS))

    with pytest.raises(ConfigError) as excinfo:
        create_app(env="production")

    assert all(name in str(excinfo.value) for name in S3_VARS)


def test_unknown_storage_backend_is_refused(clean_env):
    _set_production_env(clean_env)
    clean_env.setenv("STORAGE_BACKEND", "ftp")

    with pytest.raises(ConfigError, match="STORAGE_BACKEND must be one of"):
        create_app(env="production")


def test_production_boots_with_required_variables(clean_env):
    _set_production_env(clean_env)
    clean_env.setenv("AWS_DEFAULT_REGION", "us-east-1")

    app = create_app(env="production")

    assert app.secret_key == "k" * 64
    assert app.testing is False
    assert isinstance(app.extensions["storage"], S3Storage)


@pytest.mark.parametrize("env", ["development", "testing"])
def test_local_envs_generate_a_secret_key_and_use_local_storage(clean_env, env):
    clean_env.setenv("DATABASE_URL", "sqlite://")

    app = create_app(env=env)

    assert len(app.secret_key) == 64
    assert isinstance(app.extensions["storage"], LocalStorage)


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


def test_create_app_does_not_import_ml_or_cloud_libraries():
    result = _run_python(
        "import sys\n"
        "from app import create_app\n"
        "create_app({'SQLALCHEMY_DATABASE_URI': 'sqlite://'}, env='testing')\n"
        "heavy = ['sklearn', 'joblib', 'boto3', 'utils.preprocessing', 'utils.train_model',"
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


FILESYSTEM_ACCESS = re.compile(
    r"\bpathlib\b|\bPath\(|\bopen\(|\.save\(|\.write_(?:text|bytes)\(|\.read_(?:text|bytes)\("
    r"|\.mkdir\(|\bos\.path\b|data/raw|best_model\.pkl|MODELS_DIR|RAW_DIR"
)


def test_app_and_ml_utils_do_not_touch_the_filesystem():
    sources = [ROOT / "app.py", *sorted((ROOT / "utils").glob("*.py"))]

    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{lineno}: {line.strip()}"
        for path in sources
        if path.name != "storage.py"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if FILESYSTEM_ACCESS.search(line)
    ]

    assert offenders == [], "Only utils/storage.py may touch the filesystem"
