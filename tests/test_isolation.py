"""Organisation B can never read, train on, or export organisation A's data, route by route.

The query layer is tested in test_tenancy.py. This file checks the guarantee where it
matters, at every page and action a logged-in user of another organisation can reach,
using data that is recognisable on sight: A's students are MATnnn, B's are BTAnnn.
"""
import csv
import io
import re

import pytest

from db import db
from db.models import AuditEvent, Dataset, ModelArtifact
from db.tenancy import scoped_select

ALPHA_STUDENT = re.compile(r"\bMAT\d{3}\b")
BETA_STUDENT = re.compile(r"\bBTA\d{3}\b")
ALPHA_MODEL_NAME = "ALPHA-ONLY-MODEL"

READ_ROUTES = [
    "/", "/about", "/upload-train", "/upload-predict", "/upload-actual", "/train",
    "/results", "/results?name=MAT", "/download-results", "/explain", "/compare", "/audit", "/users",
]


def _as_beta(data: bytes) -> bytes:
    return data.replace(b"MAT", b"BTA").replace(b"Student_", b"BetaStudent_")


@pytest.fixture()
def tenants(ready_tenant, tenant, upload, csv_fixtures):
    alpha = ready_tenant("alpha")
    alpha.artifact.algorithm_name = ALPHA_MODEL_NAME
    db.session.commit()

    beta = tenant("beta")
    for path, name in (("/upload-train", "training"), ("/upload-predict", "prediction"), ("/upload-actual", "actual")):
        assert upload(beta.client, path, _as_beta(csv_fixtures[name])).status_code == 302
    assert beta.client.post("/train").status_code == 200
    return alpha, beta


@pytest.mark.parametrize("path", READ_ROUTES)
def test_no_page_shows_another_organisations_data(tenants, path):
    alpha, beta = tenants

    response = beta.client.get(path, follow_redirects=True)

    body = response.get_data(as_text=True)
    assert response.status_code == 200, path
    # Served to B, logged in. Without this a silent redirect to the login page would pass.
    assert beta.user.email in body or path == "/download-results", f"{path} was not served to organisation B"
    assert not ALPHA_STUDENT.search(body), f"{path} shows organisation A's students"
    assert alpha.user.email not in body, f"{path} shows organisation A's users"
    assert ALPHA_MODEL_NAME not in body, f"{path} shows organisation A's model"


@pytest.mark.parametrize("path", ["/results", "/download-results", "/compare"])
def test_each_organisation_sees_its_own_students(tenants, path):
    alpha, beta = tenants

    alpha_body = alpha.client.get(path).get_data(as_text=True)
    beta_body = beta.client.get(path).get_data(as_text=True)

    assert ALPHA_STUDENT.search(alpha_body) and not BETA_STUDENT.search(alpha_body)
    assert BETA_STUDENT.search(beta_body) and not ALPHA_STUDENT.search(beta_body)


def test_an_export_contains_only_the_exporting_organisations_students(tenants):
    _, beta = tenants

    response = beta.client.get("/download-results")

    assert response.mimetype == "text/csv"  # a real export, not a redirect to the login page
    rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))

    assert rows and all(BETA_STUDENT.fullmatch(row["student_id"]) for row in rows)


def test_training_uses_only_the_trainers_own_dataset(tenants):
    alpha, beta = tenants
    alpha_datasets = {d.id for d in db.session.scalars(scoped_select(Dataset, alpha.organisation.id))}

    assert beta.client.post("/train").status_code == 200

    beta_models = db.session.scalars(scoped_select(ModelArtifact, beta.organisation.id)).all()
    alpha_models = db.session.scalars(scoped_select(ModelArtifact, alpha.organisation.id)).all()
    assert len(beta_models) == 2 and len(alpha_models) == 1  # beta trained twice, alpha untouched
    trained = db.session.scalars(
        scoped_select(AuditEvent, beta.organisation.id).where(AuditEvent.action == "model_trained")
    ).all()
    assert trained and all(event.details["training_dataset_id"] not in {str(i) for i in alpha_datasets}
                           for event in trained)


def test_addressing_another_organisations_records_by_id_finds_nothing(tenants):
    alpha, beta = tenants
    alpha_dataset = db.session.scalars(scoped_select(Dataset, alpha.organisation.id)).first()

    assert beta.client.get(f"/map-columns/{alpha_dataset.id}").status_code == 404
    assert beta.client.post(f"/map-columns/{alpha_dataset.id}", data={"student_id": "x"}).status_code == 404
    for action in ("role", "status", "invite"):
        assert beta.client.post(f"/users/{alpha.user.id}/{action}", data={"role": "viewer"}).status_code == 404


def test_the_audit_log_cannot_be_filtered_onto_another_organisations_user(tenants):
    alpha, beta = tenants

    body = beta.client.get(f"/audit?user={alpha.user.id}").get_data(as_text=True)

    assert alpha.user.email not in body
