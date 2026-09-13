import uuid

import boto3
import pytest
from moto import mock_aws

from config import ConfigError
from db import db
from db.models import Dataset, DatasetKind
from db.tenancy import scoped_select
from utils.storage import (
    CrossTenantAccess,
    InvalidStorageKey,
    LocalStorage,
    ObjectNotFound,
    S3Storage,
    StorageError,
    TenantStorage,
    build_backend,
    dataset_key,
    model_key,
    storage_for,
)

# Contains hex letters, so an upper-cased copy is a genuinely different key.
ORG = uuid.UUID("abcdef01-2345-4678-89ab-cdef01234567")
OTHER_ORG = uuid.UUID("22222222-2222-4222-8222-222222222222")
ITEM = uuid.UUID("33333333-3333-4333-8333-333333333333")


def test_keys_are_built_from_database_ids_only():
    assert dataset_key(ORG, ITEM) == f"org/{ORG}/datasets/{ITEM}.csv"
    assert model_key(str(ORG), str(ITEM)) == f"org/{ORG}/models/{ITEM}.pkl"
    with pytest.raises(ValueError):
        dataset_key(ORG, "../../etc/passwd")


def test_local_backend_round_trip_leaves_no_partial_files(tmp_path):
    backend = LocalStorage(tmp_path)
    key = dataset_key(ORG, ITEM)

    backend.write(key, b"a,b\n1,2\n")

    assert backend.read(key) == b"a,b\n1,2\n"
    folder = tmp_path / "org" / str(ORG) / "datasets"
    assert [path.name for path in folder.iterdir()] == [f"{ITEM}.csv"]


def test_local_backend_reports_missing_objects(tmp_path):
    with pytest.raises(ObjectNotFound):
        LocalStorage(tmp_path).read(dataset_key(ORG, ITEM))


@pytest.mark.parametrize("key", ["../outside.csv", "org/../../outside.csv"])
def test_local_backend_refuses_paths_outside_its_root(tmp_path, key):
    backend = LocalStorage(tmp_path / "root")

    with pytest.raises(InvalidStorageKey):
        backend.write(key, b"x")

    assert not (tmp_path / "outside.csv").exists()


def test_tenant_storage_round_trip(tmp_path):
    storage = TenantStorage(LocalStorage(tmp_path), ORG)
    key = dataset_key(ORG, ITEM)

    storage.put(key, b"data")

    assert storage.get(key) == b"data"


def test_get_refuses_another_organisations_key(tmp_path):
    backend = LocalStorage(tmp_path)
    other_key = dataset_key(OTHER_ORG, ITEM)
    TenantStorage(backend, OTHER_ORG).put(other_key, b"beta's students")

    with pytest.raises(CrossTenantAccess):
        TenantStorage(backend, ORG).get(other_key)


def test_put_refuses_another_organisations_key(tmp_path):
    backend = LocalStorage(tmp_path)

    with pytest.raises(CrossTenantAccess):
        TenantStorage(backend, ORG).put(dataset_key(OTHER_ORG, ITEM), b"planted")

    with pytest.raises(ObjectNotFound):
        backend.read(dataset_key(OTHER_ORG, ITEM))


MALFORMED_KEYS = [
    None,
    "",
    "../../etc/passwd",
    f"org/{ORG}/datasets/../../{OTHER_ORG}/datasets/{ITEM}.csv",
    f"/org/{ORG}/datasets/{ITEM}.csv",
    f"org/{ORG}\\datasets\\{ITEM}.csv",
    f"org/{ORG}/datasets/{ITEM}.exe",
    f"org/{ORG}/datasets/passwd.csv",
    f"org/{str(ORG).upper()}/datasets/{ITEM}.csv",
    f"org/{ORG}/datasets/{ITEM}.csv\n",
    f"org/{ORG}/models/{ITEM}.csv",
]


@pytest.mark.parametrize("key", MALFORMED_KEYS)
def test_malformed_keys_are_refused_before_reaching_the_backend(tmp_path, key):
    storage = TenantStorage(LocalStorage(tmp_path), ORG)

    with pytest.raises(InvalidStorageKey):
        storage.get(key)
    with pytest.raises(InvalidStorageKey):
        storage.put(key, b"x")

    assert list(tmp_path.iterdir()) == []


def _stored_dataset(organisation, user, data: bytes = b"student_id,target\nS-1,0\n") -> Dataset:
    dataset_id = uuid.uuid4()
    dataset = Dataset(
        id=dataset_id,
        organisation_id=organisation.id,
        kind=DatasetKind.TRAINING,
        original_filename="training.csv",
        storage_key=dataset_key(organisation.id, dataset_id),
        row_count=1,
        column_names=["student_id", "target"],
        uploaded_by=user.id,
    )
    storage_for(organisation.id).put(dataset.storage_key, data)
    db.session.add(dataset)
    db.session.commit()
    return dataset


def test_delete_is_a_soft_delete_that_keeps_the_object(make_org):
    organisation, user = make_org("alpha")
    dataset = _stored_dataset(organisation, user)
    storage = storage_for(organisation.id)

    storage.delete(dataset.storage_key)
    db.session.commit()

    assert dataset.deleted_at is not None
    assert storage.get(dataset.storage_key) == b"student_id,target\nS-1,0\n"
    assert db.session.scalars(scoped_select(Dataset, organisation.id)).all() == []
    assert db.session.scalars(scoped_select(Dataset, organisation.id, include_deleted=True)).all() == [dataset]


def test_delete_refuses_another_organisations_key(make_org):
    org_a, _ = make_org("alpha")
    org_b, user_b = make_org("beta")
    dataset_b = _stored_dataset(org_b, user_b)

    with pytest.raises(CrossTenantAccess):
        storage_for(org_a.id).delete(dataset_b.storage_key)

    db.session.expire_all()
    assert db.session.get(Dataset, dataset_b.id).deleted_at is None


def test_delete_of_an_unknown_key_raises_not_found(make_org):
    organisation, _ = make_org("alpha")

    with pytest.raises(ObjectNotFound):
        storage_for(organisation.id).delete(dataset_key(organisation.id, uuid.uuid4()))


def test_model_artifacts_cannot_be_deleted(make_org):
    organisation, _ = make_org("alpha")

    with pytest.raises(StorageError, match="deactivated, not deleted"):
        storage_for(organisation.id).delete(model_key(organisation.id, uuid.uuid4()))


@pytest.fixture()
def s3_backend(monkeypatch, tmp_path):
    # Isolate from any real AWS profile or region on the developer's machine.
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="edupulse-test")
        yield S3Storage(
            endpoint_url="https://s3.us-east-1.amazonaws.com",
            bucket="edupulse-test",
            access_key="testing",
            secret_key="testing",
        )


def test_s3_backend_round_trip(s3_backend):
    key = dataset_key(ORG, ITEM)

    s3_backend.write(key, b"student_id,target\n")

    assert s3_backend.read(key) == b"student_id,target\n"


def test_s3_backend_reports_missing_objects(s3_backend):
    with pytest.raises(ObjectNotFound):
        s3_backend.read(dataset_key(ORG, ITEM))


def test_s3_tenant_storage_refuses_cross_tenant_reads(s3_backend):
    other_key = dataset_key(OTHER_ORG, ITEM)
    TenantStorage(s3_backend, OTHER_ORG).put(other_key, b"beta's students")

    with pytest.raises(CrossTenantAccess):
        TenantStorage(s3_backend, ORG).get(other_key)


def test_build_backend_selects_the_configured_implementation(tmp_path, s3_backend):
    local = build_backend({"STORAGE_BACKEND": "local", "STORAGE_LOCAL_ROOT": str(tmp_path)})
    s3 = build_backend({
        "STORAGE_BACKEND": "s3",
        "STORAGE_ENDPOINT": "https://s3.us-east-1.amazonaws.com",
        "STORAGE_BUCKET": "edupulse-test",
        "STORAGE_ACCESS_KEY": "testing",
        "STORAGE_SECRET_KEY": "testing",
    })

    assert isinstance(local, LocalStorage) and local.root == tmp_path.resolve()
    assert isinstance(s3, S3Storage) and s3.bucket == "edupulse-test"
    with pytest.raises(ConfigError):
        build_backend({"STORAGE_BACKEND": "ftp"})
