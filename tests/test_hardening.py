"""Proxy handling, login rate limits, and upload hardening."""
import io
from pathlib import Path

import pytest
from sqlalchemy import select

from app import create_app
from auth.passwords import hash_password
from auth.ratelimit import RATE_LIMITED_MESSAGE
from db import db
from db.models import AuditEvent, Dataset, Organisation, Role, User
from db.tenancy import scoped_select

ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "correct horse battery staple"
EMAIL = "owner@alpha.test"
PROXY = "10.0.0.1"  # the load balancer: every request reaches the app from here


@pytest.fixture()
def limited_app(tmp_path):
    """An app with rate limits on, as in production, behind one trusted proxy."""
    app = create_app(
        {
            "SQLALCHEMY_DATABASE_URI": "sqlite://",
            "STORAGE_LOCAL_ROOT": str(tmp_path / "storage"),
            "WTF_CSRF_ENABLED": False,
            "RATELIMIT_ENABLED": True,
        },
        env="testing",
    )
    with app.app_context():
        db.create_all()
        organisation = Organisation(name="Alpha", slug="alpha")
        db.session.add_all([organisation, User(organisation=organisation, email=EMAIL, role=Role.OWNER,
                                               password_hash=hash_password(PASSWORD))])
        db.session.commit()
        yield app
        db.session.remove()
        db.drop_all()


def _via_proxy(app, forwarded_for: str):
    """A client whose requests arrive through the proxy, which appends the real client address."""
    return _client(app, REMOTE_ADDR=PROXY, HTTP_X_FORWARDED_FOR=forwarded_for)


def _client(app, **environ):
    client = app.test_client()
    client.environ_base.update(environ)
    return client


def _fail_login(client):
    return client.post("/login", data={"email": EMAIL, "password": "wrong password here"})


def _login(client):
    return client.post("/login", data={"email": EMAIL, "password": PASSWORD})


def test_the_client_address_comes_from_the_proxy_header(limited_app):
    client = _via_proxy(limited_app, "203.0.113.5")

    _login(client)

    event = db.session.scalars(select(AuditEvent).where(AuditEvent.action == "login_success")).one()
    assert event.ip_address == "203.0.113.5"


def test_only_the_last_hop_is_trusted(limited_app):
    # The client sent "X-Forwarded-For: 198.51.100.66"; the proxy appended the real address.
    client = _via_proxy(limited_app, "198.51.100.66, 203.0.113.5")

    _fail_login(client)

    event = db.session.scalars(select(AuditEvent).where(AuditEvent.action == "login_failure")).one()
    assert event.ip_address == "203.0.113.5"


def test_the_login_limit_locks_out_one_client_not_everyone_behind_the_proxy(limited_app):
    attacker, colleague = _via_proxy(limited_app, "203.0.113.5"), _via_proxy(limited_app, "203.0.113.9")

    assert [_fail_login(attacker).status_code for _ in range(5)] == [200] * 5
    blocked = _fail_login(attacker)

    assert blocked.status_code == 429
    assert RATE_LIMITED_MESSAGE in blocked.get_data(as_text=True)
    assert "Retry-After" in blocked.headers
    assert _login(attacker).status_code == 429  # even the right password, until the window passes
    assert _login(colleague).status_code == 302  # same proxy address, different client: unaffected


def test_rotating_a_spoofed_forwarded_address_does_not_evade_the_limit(limited_app):
    for attempt in range(5):
        assert _fail_login(_via_proxy(limited_app, f"198.51.100.{attempt}, 203.0.113.5")).status_code == 200

    assert _fail_login(_via_proxy(limited_app, "198.51.100.99, 203.0.113.5")).status_code == 429


def test_successful_logins_do_not_count_towards_the_limit(limited_app):
    school_nat = _via_proxy(limited_app, "203.0.113.20")

    for _ in range(8):
        assert _login(school_nat).status_code == 302
        school_nat.post("/logout")

    assert _fail_login(school_nat).status_code == 200


def test_the_login_page_itself_is_not_limited(limited_app):
    client = _via_proxy(limited_app, "203.0.113.5")
    for _ in range(6):
        _fail_login(client)

    assert client.get("/login").status_code == 200


def test_invite_acceptance_is_limited_too(limited_app):
    client = _via_proxy(limited_app, "203.0.113.5")
    for _ in range(5):
        client.post("/invite/not-a-real-token", data={"password": "x" * 12, "confirm": "x" * 12})

    assert client.post("/invite/not-a-real-token", data={"password": "x" * 12, "confirm": "x" * 12}).status_code == 429


def test_without_a_proxy_forwarded_headers_are_ignored(tmp_path):
    app = create_app(
        {"SQLALCHEMY_DATABASE_URI": "sqlite://", "STORAGE_LOCAL_ROOT": str(tmp_path), "PROXY_HOPS": 0,
         "WTF_CSRF_ENABLED": False},
        env="testing",
    )
    with app.app_context():
        db.create_all()
        client = _client(app, REMOTE_ADDR="192.0.2.1", HTTP_X_FORWARDED_FOR="203.0.113.5")

        client.post("/login", data={"email": "nobody@nowhere.test", "password": "whatever"})

        assert db.session.scalars(select(AuditEvent.ip_address)).one() == "192.0.2.1"
        db.session.remove()


MALFORMED_UPLOADS = {
    "empty file": b"",
    "whitespace only": b"   \n\n  \n",
    "binary data": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\xff\xfe\xfd" * 20,
    "not utf-8": "student_id,student_name\nS-1,Ren\xe9e\n".encode("latin-1"),
    "header only": b"student_id,student_name,attendance\n",
}


@pytest.mark.parametrize("path", ["/upload-train", "/upload-predict", "/upload-actual"])
@pytest.mark.parametrize("content", MALFORMED_UPLOADS.values(), ids=MALFORMED_UPLOADS.keys())
def test_a_malformed_csv_gets_a_clean_error_and_stores_nothing(tenant, upload, tmp_path, path, content):
    alpha = tenant("alpha")

    response = upload(alpha.client, path, content, filename="students.csv")
    page = alpha.client.get(path).get_data(as_text=True)

    assert response.status_code == 302
    assert "Upload failed: the file" in page
    assert "Traceback" not in page and "Error tokenizing" not in page and "codec" not in page
    assert db.session.scalars(scoped_select(Dataset, alpha.organisation.id, include_deleted=True)).all() == []
    assert db.session.scalars(scoped_select(AuditEvent, alpha.organisation.id)
                              .where(AuditEvent.action == "dataset_uploaded")).all() == []
    stored = [p for p in (tmp_path / "storage").rglob("*") if p.is_file()] if (tmp_path / "storage").exists() else []
    assert stored == []


def test_an_upload_over_5_mb_is_refused_cleanly(tenant, upload, tmp_path):
    alpha = tenant("alpha")

    response = upload(alpha.client, "/upload-train", b"a,b\n" + b"1,2\n" * (1_400_000), filename="big.csv")

    assert response.status_code == 413
    assert "larger than the 5 MB upload limit" in response.get_data(as_text=True)
    assert db.session.scalars(scoped_select(Dataset, alpha.organisation.id, include_deleted=True)).all() == []


def test_render_runs_exactly_one_worker_while_limits_are_in_memory():
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")

    assert "--workers 1" in render
    assert "Redis" in render


def test_matplotlib_is_no_longer_a_dependency():
    assert "matplotlib" not in (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
