"""The three CSV slots and the model artifact resolve per organisation through storage."""
import io
from datetime import timedelta

import pandas as pd
import pytest

from db import db
from db.models import Dataset, DatasetKind, ModelArtifact
from db.tenancy import scoped_select
from utils.storage import KEY_PATTERN, dataset_key, model_key, storage_for

ALL_PAGES = [
    "/", "/about", "/upload-train", "/upload-predict", "/upload-actual",
    "/train", "/explain", "/results", "/compare", "/download-results",
]


def _with_marker(data: bytes, column: int, marker: str) -> bytes:
    """Replace one value in the first data row, so a page can be checked for whose data it shows."""
    header, first, rest = data.decode().split("\n", 2)
    fields = first.split(",")
    fields[column] = marker
    return "\n".join([header, ",".join(fields), rest]).encode()


def _datasets(organisation, include_deleted=False) -> list[Dataset]:
    return db.session.scalars(scoped_select(Dataset, organisation.id, include_deleted=include_deleted)).all()


@pytest.mark.parametrize("path", ALL_PAGES)
def test_requests_without_a_user_are_refused(client, path):
    assert client.get(path).status_code == 401


def test_requests_from_a_deactivated_user_are_refused(tenant):
    alpha = tenant("alpha")
    alpha.user.is_active = False
    db.session.commit()

    assert alpha.client.get("/").status_code == 401


def test_requests_from_a_suspended_organisation_are_refused(tenant):
    alpha = tenant("alpha")
    alpha.organisation.is_active = False
    db.session.commit()

    assert alpha.client.get("/").status_code == 401


def test_upload_stores_the_file_under_the_organisations_namespace(tenant, upload, csv_fixtures):
    alpha = tenant("alpha")

    response = upload(alpha.client, "/upload-train", csv_fixtures["training"], filename="Spring term.csv")

    assert response.status_code == 302
    [dataset] = _datasets(alpha.organisation)
    assert dataset.kind is DatasetKind.TRAINING
    assert dataset.storage_key == dataset_key(alpha.organisation.id, dataset.id)
    assert dataset.original_filename == "Spring term.csv"
    assert dataset.uploaded_by == alpha.user.id
    expected = pd.read_csv(io.BytesIO(csv_fixtures["training"]))
    assert dataset.row_count == len(expected)
    assert dataset.column_names == list(expected.columns)
    assert storage_for(alpha.organisation.id).get(dataset.storage_key) == csv_fixtures["training"]


def test_path_traversal_in_the_filename_cannot_influence_storage(tenant, upload, csv_fixtures, tmp_path):
    alpha = tenant("alpha")

    upload(alpha.client, "/upload-train", csv_fixtures["training"], filename="../../etc/passwd.csv")

    [dataset] = _datasets(alpha.organisation)
    assert dataset.storage_key == f"org/{alpha.organisation.id}/datasets/{dataset.id}.csv"
    assert KEY_PATTERN.fullmatch(dataset.storage_key)
    assert dataset.original_filename == "../../etc/passwd.csv"
    written = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())
    assert written == [f"storage/org/{alpha.organisation.id}/datasets/{dataset.id}.csv"]
    assert not (tmp_path.parent / "etc").exists()


UPLOAD_SLOTS = [
    ("/upload-train", "training", 1),
    ("/upload-predict", "prediction", 1),
    ("/upload-actual", "actual", 0),
]


@pytest.mark.parametrize(("path", "name", "column"), UPLOAD_SLOTS, ids=[slot[1] for slot in UPLOAD_SLOTS])
def test_each_csv_slot_resolves_per_organisation(tenant, upload, csv_fixtures, path, name, column):
    alpha, beta = tenant("alpha"), tenant("beta")

    upload(alpha.client, path, _with_marker(csv_fixtures[name], column, "ALPHA-ONLY-STUDENT"))
    assert "ALPHA-ONLY-STUDENT" in alpha.client.get(path).get_data(as_text=True)
    assert "ALPHA-ONLY-STUDENT" not in beta.client.get(path).get_data(as_text=True)

    upload(beta.client, path, _with_marker(csv_fixtures[name], column, "BETA-ONLY-STUDENT"))
    alpha_page = alpha.client.get(path).get_data(as_text=True)
    beta_page = beta.client.get(path).get_data(as_text=True)

    assert "ALPHA-ONLY-STUDENT" in alpha_page and "BETA-ONLY-STUDENT" not in alpha_page
    assert "BETA-ONLY-STUDENT" in beta_page and "ALPHA-ONLY-STUDENT" not in beta_page


def test_training_stores_the_model_under_the_organisations_namespace(tenant, upload, csv_fixtures):
    alpha, beta = tenant("alpha"), tenant("beta")
    upload(alpha.client, "/upload-train", csv_fixtures["training"])

    assert alpha.client.post("/train").status_code == 200

    [artifact] = db.session.scalars(scoped_select(ModelArtifact, alpha.organisation.id)).all()
    assert artifact.is_active
    assert artifact.storage_key == model_key(alpha.organisation.id, artifact.id)
    assert artifact.trained_by == alpha.user.id
    assert len(storage_for(alpha.organisation.id).get(artifact.storage_key)) > 0
    assert db.session.scalars(scoped_select(ModelArtifact, beta.organisation.id)).all() == []

    response = beta.client.post("/train")  # beta has no training data of its own
    assert response.status_code == 302 and response.headers["Location"].endswith("/upload-train")


def test_model_artifact_resolves_per_organisation(ready_tenant, tenant, upload, csv_fixtures):
    alpha = ready_tenant("alpha")
    beta = tenant("beta")
    upload(beta.client, "/upload-predict", csv_fixtures["prediction"])

    assert alpha.client.get("/results").status_code == 200
    assert alpha.client.get("/download-results").status_code == 200
    assert alpha.client.get("/explain").status_code == 200

    for path in ("/results", "/download-results", "/explain"):
        response = beta.client.get(path)
        assert response.status_code == 302 and response.headers["Location"].endswith("/train"), path
    assert beta.client.get("/compare").status_code == 200  # renders its own "train first" state


def test_retraining_keeps_one_active_model_and_the_previous_artifact(ready_tenant):
    alpha = ready_tenant("alpha")
    first = alpha.artifact

    assert alpha.client.post("/train").status_code == 200

    artifacts = db.session.scalars(scoped_select(ModelArtifact, alpha.organisation.id)).all()
    assert len(artifacts) == 2
    [active] = [artifact for artifact in artifacts if artifact.is_active]
    assert active.id != first.id
    assert len(storage_for(alpha.organisation.id).get(first.storage_key)) > 0


def test_soft_deleted_datasets_are_never_used(tenant, upload, csv_fixtures):
    alpha = tenant("alpha")
    upload(alpha.client, "/upload-train", _with_marker(csv_fixtures["training"], 1, "OLDER-UPLOAD"))
    upload(alpha.client, "/upload-train", _with_marker(csv_fixtures["training"], 1, "NEWER-UPLOAD"))
    older, newer = sorted(_datasets(alpha.organisation), key=lambda d: d.uploaded_at)
    older.uploaded_at = newer.uploaded_at - timedelta(minutes=1)  # independent of clock resolution
    db.session.commit()
    assert "NEWER-UPLOAD" in alpha.client.get("/upload-train").get_data(as_text=True)

    storage_for(alpha.organisation.id).delete(newer.storage_key)
    db.session.commit()
    page = alpha.client.get("/upload-train").get_data(as_text=True)
    assert "OLDER-UPLOAD" in page and "NEWER-UPLOAD" not in page

    storage_for(alpha.organisation.id).delete(older.storage_key)
    db.session.commit()
    response = alpha.client.post("/train")
    assert response.status_code == 302 and response.headers["Location"].endswith("/upload-train")
    assert len(_datasets(alpha.organisation, include_deleted=True)) == 2


@pytest.mark.parametrize(
    "path",
    ["/", "/about", "/upload-train", "/upload-predict", "/upload-actual", "/train", "/explain", "/results", "/compare"],
)
def test_existing_pages_render_for_an_organisation_with_data(ready_tenant, path):
    response = ready_tenant("alpha").client.get(path)

    assert response.status_code == 200, response.data[:500]


def test_download_results_exports_the_filtered_csv(ready_tenant):
    response = ready_tenant("alpha").client.get("/download-results?risk=High")

    assert response.status_code == 200
    assert response.mimetype == "text/csv"
    header, *rows = response.data.decode().splitlines()
    assert header == "student_id,student_name,prediction,fail_probability,confidence,risk_level,recommendation"
    assert rows and all(",High," in row for row in rows)
