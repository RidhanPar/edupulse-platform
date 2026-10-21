"""Stored models are signed, and nothing unsigned or altered is ever unpickled."""
import io
import uuid

import joblib
import pytest
from sqlalchemy import select

from app import create_app
from config import ConfigError
from db import db
from db.models import AuditEvent
from utils import predict
from utils.signing import MAGIC, ModelSignatureError, sign, verify
from utils.model_cache import ModelCache
from utils.storage import storage_for

KEY, OLD_KEY = "current-signing-key", "previous-signing-key"


def test_a_signed_payload_round_trips():
    assert verify(sign(b"model bytes", KEY), [KEY]) == b"model bytes"


@pytest.mark.parametrize(
    ("blob", "message"),
    [
        (b"raw pickle bytes with no signature", "not signed"),
        (MAGIC + b"too short", "not signed"),
        (sign(b"model bytes", "someone else's key"), "does not match"),
        (sign(b"model bytes", KEY)[:-1] + b"X", "does not match"),
    ],
    ids=["unsigned", "truncated", "wrong key", "tampered payload"],
)
def test_anything_unsigned_or_altered_is_refused(blob, message):
    with pytest.raises(ModelSignatureError, match=message):
        verify(blob, [KEY])


def test_a_previous_key_still_verifies_during_rotation():
    assert verify(sign(b"model bytes", OLD_KEY), [KEY, OLD_KEY]) == b"model bytes"
    with pytest.raises(ModelSignatureError):
        verify(sign(b"model bytes", OLD_KEY), [KEY])


def test_signing_requires_a_key():
    with pytest.raises(ValueError):
        sign(b"model bytes", "")


def test_an_unsigned_model_is_never_unpickled(monkeypatch):
    buffer = io.BytesIO()
    joblib.dump({"harmless": "object"}, buffer)
    monkeypatch.setattr(predict.joblib, "load", lambda *_: pytest.fail("unpickled an unsigned model"))

    with pytest.raises(ModelSignatureError):
        predict.load_model(buffer.getvalue(), [KEY])


def test_the_app_refuses_a_replaced_model_file_and_audits_it(ready_tenant):
    alpha = ready_tenant("alpha")
    buffer = io.BytesIO()
    joblib.dump({"planted": "by someone with bucket access"}, buffer)
    storage_for(alpha.organisation.id).put(alpha.artifact.storage_key, buffer.getvalue())
    # A fresh worker: nothing verified earlier is still held in memory.
    alpha.client.application.extensions["model_cache"] = ModelCache(4)

    response = alpha.client.post("/predict", follow_redirects=True)

    assert "model file is not signed" in response.get_data(as_text=True)
    [event] = db.session.scalars(select(AuditEvent).where(AuditEvent.action == "model_signature_rejected")).all()
    assert (event.entity_id, event.organisation_id) == (str(alpha.artifact.id), alpha.organisation.id)


def test_production_refuses_to_start_without_a_model_signing_key(monkeypatch, tmp_path):
    for name, value in {
        "FLASK_SECRET_KEY": "k" * 64, "DATABASE_URL": "sqlite://", "STORAGE_BACKEND": "s3",
        "STORAGE_ENDPOINT": "https://s3.eu-central-1.amazonaws.com", "STORAGE_BUCKET": "b",
        "STORAGE_ACCESS_KEY": "a", "STORAGE_SECRET_KEY": "s",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("MODEL_SIGNING_KEY", raising=False)

    with pytest.raises(ConfigError, match="MODEL_SIGNING_KEY"):
        create_app(env="production")


def test_the_signing_key_is_independent_of_the_session_secret(monkeypatch):
    monkeypatch.setenv("MODEL_SIGNING_KEY", KEY)
    monkeypatch.setenv("MODEL_SIGNING_KEY_PREVIOUS", OLD_KEY)
    monkeypatch.setenv("FLASK_SECRET_KEY", "a session secret that will be rotated")

    app = create_app({"SQLALCHEMY_DATABASE_URI": "sqlite://"}, env="testing")

    assert app.config["MODEL_SIGNING_KEYS"] == [KEY, OLD_KEY]
    assert app.secret_key not in app.config["MODEL_SIGNING_KEYS"]
