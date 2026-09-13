"""Object storage for tenant data, namespaced by organisation.

Keys are built only from database ids (dataset_key, model_key), never from
anything a user supplies such as an uploaded filename. Application code reaches
storage only through TenantStorage, which refuses any key outside the requesting
organisation's namespace before a backend is touched, so a bug elsewhere cannot
read or write another tenant's objects through this layer.
"""
from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path

from flask import current_app

from config import ConfigError
from db import db
from db.models import Dataset, utcnow
from db.tenancy import scoped_select

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
KEY_PATTERN = re.compile(rf"org/(?P<organisation_id>{_UUID})/(?P<kind>datasets/{_UUID}\.csv|models/{_UUID}\.pkl)")


class StorageError(Exception):
    """Base class for storage failures."""


class InvalidStorageKey(StorageError):
    """The key is not one this application could have generated."""


class CrossTenantAccess(StorageError):
    """The key belongs to a different organisation from the one asking for it."""


class ObjectNotFound(StorageError):
    """No object (or dataset record) exists for the key."""


def dataset_key(organisation_id, dataset_id) -> str:
    return f"org/{uuid.UUID(str(organisation_id))}/datasets/{uuid.UUID(str(dataset_id))}.csv"


def model_key(organisation_id, artifact_id) -> str:
    return f"org/{uuid.UUID(str(organisation_id))}/models/{uuid.UUID(str(artifact_id))}.pkl"


class LocalStorage:
    """Filesystem backend for development and tests."""

    def __init__(self, root) -> None:
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root):
            raise InvalidStorageKey(f"key escapes the storage root: {key!r}")
        return path

    def write(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file and rename, so a reader never sees a partial object.
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".partial-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def read(self, key: str) -> bytes:
        try:
            return self._path(key).read_bytes()
        except FileNotFoundError:
            raise ObjectNotFound(key) from None


class S3Storage:
    """S3-compatible backend (AWS S3, Cloudflare R2, MinIO and similar)."""

    def __init__(self, endpoint_url: str, bucket: str, access_key: str, secret_key: str) -> None:
        import boto3  # only production needs it, so keep it off every local cold start

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self.bucket = bucket

    def write(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    def read(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
                raise ObjectNotFound(key) from None
            raise
        return response["Body"].read()


class TenantStorage:
    """put / get / delete, confined to one organisation's namespace."""

    def __init__(self, backend, organisation_id) -> None:
        self._backend = backend
        self.organisation_id = uuid.UUID(str(organisation_id))

    def _authorise(self, key: str) -> re.Match:
        match = KEY_PATTERN.fullmatch(key) if isinstance(key, str) else None
        if match is None:
            raise InvalidStorageKey(f"not an application storage key: {key!r}")
        if match["organisation_id"] != str(self.organisation_id):
            raise CrossTenantAccess(f"key {key!r} is outside organisation {self.organisation_id}")
        return match

    def put(self, key: str, data: bytes) -> None:
        self._authorise(key)
        self._backend.write(key, data)

    def get(self, key: str) -> bytes:
        self._authorise(key)
        return self._backend.read(key)

    def delete(self, key: str) -> None:
        """Soft delete: mark the dataset stored under `key` as deleted and keep the object.

        Objects are removed only by the retention purge job, which is not built yet.
        Model artifacts are deactivated, never deleted. The caller commits.
        """
        if self._authorise(key)["kind"].startswith("models/"):
            raise StorageError("model artifacts are deactivated, not deleted")
        dataset = db.session.scalars(
            scoped_select(Dataset, self.organisation_id, include_deleted=True).where(Dataset.storage_key == key)
        ).one_or_none()
        if dataset is None:
            raise ObjectNotFound(key)
        if dataset.deleted_at is None:
            dataset.deleted_at = utcnow()
            db.session.flush()


def build_backend(config) -> LocalStorage | S3Storage:
    backend = config["STORAGE_BACKEND"]
    if backend == "local":
        return LocalStorage(config["STORAGE_LOCAL_ROOT"])
    if backend == "s3":
        return S3Storage(
            endpoint_url=config["STORAGE_ENDPOINT"],
            bucket=config["STORAGE_BUCKET"],
            access_key=config["STORAGE_ACCESS_KEY"],
            secret_key=config["STORAGE_SECRET_KEY"],
        )
    raise ConfigError(f"unknown STORAGE_BACKEND {backend!r}")


def storage_for(organisation_id) -> TenantStorage:
    """Storage confined to `organisation_id`: pass the requesting organisation, never the object's."""
    return TenantStorage(current_app.extensions["storage"], organisation_id)
