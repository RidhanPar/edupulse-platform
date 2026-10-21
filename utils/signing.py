"""HMAC signatures on serialised models, checked before anything is unpickled.

Unpickling runs code, so a model file is only as trustworthy as whoever could write
to the bucket. Every model is written as MAGIC + HMAC-SHA256(key, MAGIC + payload) +
payload, and verify() refuses anything else: unsigned files, tampered files, and files
signed with an unknown key all raise before the payload is touched.

The key is MODEL_SIGNING_KEY, deliberately separate from FLASK_SECRET_KEY so that
rotating the session secret never makes stored models unloadable. To rotate the
signing key itself, move the old value to MODEL_SIGNING_KEY_PREVIOUS: models signed
with either key verify, and new ones are signed with the current key.
"""
from __future__ import annotations

import hashlib
import hmac
from collections.abc import Sequence

MAGIC = b"EDUPULSE-MODEL-v1\n"
_MAC_LENGTH = hashlib.sha256().digest_size


class ModelSignatureError(ValueError):
    """A stored model is unsigned or its signature does not verify. It is never loaded."""


def _mac(key: str, payload: bytes) -> bytes:
    return hmac.new(key.encode("utf-8"), MAGIC + payload, hashlib.sha256).digest()


def sign(payload: bytes, key: str) -> bytes:
    if not key:
        raise ValueError("a signing key is required")
    return MAGIC + _mac(key, payload) + payload


def verify(blob: bytes, keys: Sequence[str]) -> bytes:
    """The signed payload, or ModelSignatureError. Fails closed on anything unexpected."""
    if not blob.startswith(MAGIC) or len(blob) <= len(MAGIC) + _MAC_LENGTH:
        raise ModelSignatureError("model file is not signed")
    mac = blob[len(MAGIC):len(MAGIC) + _MAC_LENGTH]
    payload = blob[len(MAGIC) + _MAC_LENGTH:]
    for key in keys:
        if key and hmac.compare_digest(mac, _mac(key, payload)):
            return payload
    raise ModelSignatureError("model signature does not match any configured signing key")
