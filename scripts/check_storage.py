"""Smoke-test the configured object storage before real data goes into it.

    python scripts/check_storage.py

Run it from the Render shell after setting the STORAGE_* variables. It writes one small
object under a reserved organisation id that no real organisation can have, reads it
back, confirms the provider encrypted it at rest, and removes it again. Nothing else in
the bucket is read or touched.
"""
from __future__ import annotations

import sys
from pathlib import Path
import uuid

# Reserved for this check. A real organisation's random uuid4() matching it is a
# 1 in 2**122 chance, so in practice no tenant's data shares this prefix.
SMOKE_TEST_ORGANISATION = uuid.UUID("00000000-0000-4000-8000-000000000000")

# Run as `python scripts/<name>.py` from the repository root: make the app importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    from wsgi import app
    from utils.storage import S3Storage, TenantStorage, dataset_key

    backend = app.extensions["storage"]
    if not isinstance(backend, S3Storage):
        print(f"STORAGE_BACKEND is not s3 ({type(backend).__name__}); nothing to check.")
        return 1

    storage = TenantStorage(backend, SMOKE_TEST_ORGANISATION)
    key = dataset_key(SMOKE_TEST_ORGANISATION, uuid.uuid4())
    payload = b"student_id,target\nSMOKE-TEST,0\n"
    print(f"Bucket {backend.bucket}, test object {key}")
    try:
        storage.put(key, payload)
        print("[OK] write")
        assert storage.get(key) == payload, "read back different bytes"
        print("[OK] read back the same bytes")
        encryption = backend._client.head_object(Bucket=backend.bucket, Key=key).get("ServerSideEncryption")
        assert encryption, "the provider did not report encryption at rest"
        print(f"[OK] encrypted at rest ({encryption})")
    finally:
        backend._client.delete_object(Bucket=backend.bucket, Key=key)
        print("[OK] test object removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
