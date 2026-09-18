"""Processing of student data writes audit events (GDPR Article 30 records)."""
import csv
import io
import re
from pathlib import Path

from db import db
from db.audit import AUDIT_ACTIONS
from db.models import AuditEvent, Dataset, DatasetKind, ModelArtifact
from db.tenancy import scoped_select

ROOT = Path(__file__).resolve().parents[1]


def _events(organisation, action: str) -> list[AuditEvent]:
    return db.session.scalars(scoped_select(AuditEvent, organisation.id).where(AuditEvent.action == action)).all()


def test_dataset_upload_is_audited(tenant, upload, csv_fixtures):
    alpha = tenant("alpha")

    upload(alpha.client, "/upload-predict", csv_fixtures["prediction"])

    [dataset] = db.session.scalars(scoped_select(Dataset, alpha.organisation.id)).all()
    [event] = _events(alpha.organisation, "dataset_uploaded")
    assert (event.user_id, event.entity_type, event.entity_id) == (alpha.user.id, "dataset", str(dataset.id))
    assert event.details == {"kind": "prediction", "row_count": dataset.row_count}
    assert event.ip_address == "127.0.0.1"


def test_a_rejected_upload_writes_no_audit_event(tenant, upload):
    alpha = tenant("alpha")

    upload(alpha.client, "/upload-train", b"irrelevant", filename="notes.txt")

    assert _events(alpha.organisation, "dataset_uploaded") == []


def test_model_training_is_audited(tenant, upload, csv_fixtures):
    alpha = tenant("alpha")
    upload(alpha.client, "/upload-train", csv_fixtures["training"])
    [dataset] = db.session.scalars(scoped_select(Dataset, alpha.organisation.id)).all()

    assert alpha.client.post("/train").status_code == 200

    [artifact] = db.session.scalars(scoped_select(ModelArtifact, alpha.organisation.id)).all()
    [event] = _events(alpha.organisation, "model_trained")
    assert (event.user_id, event.entity_type, event.entity_id) == (alpha.user.id, "model_artifact", str(artifact.id))
    assert event.details == {"algorithm": artifact.algorithm_name, "training_dataset_id": str(dataset.id)}


def test_viewing_results_records_a_prediction_run_with_its_filters(ready_tenant):
    alpha = ready_tenant("alpha")

    page = alpha.client.get("/results?risk=High&name=MAT").get_data(as_text=True)

    [prediction_dataset] = db.session.scalars(
        scoped_select(Dataset, alpha.organisation.id).where(Dataset.kind == DatasetKind.PREDICTION)
    ).all()
    [event] = _events(alpha.organisation, "prediction_run")
    total = int(re.search(r"Total students</span><strong>(\d+)</strong>", page).group(1))
    assert (event.user_id, event.entity_type, event.entity_id) == (alpha.user.id, "model_artifact", str(alpha.artifact.id))
    assert event.details == {
        "page": "results",
        "prediction_dataset_id": str(prediction_dataset.id),
        "row_count": total,
        "filters": {"name": "mat", "risk": "High"},
    }


def test_viewing_the_comparison_records_a_prediction_run(ready_tenant):
    alpha = ready_tenant("alpha")

    page = alpha.client.get("/compare").get_data(as_text=True)

    [event] = _events(alpha.organisation, "prediction_run")
    total = int(re.search(r"Total compared</span><strong>(\d+)</strong>", page).group(1))
    assert event.details["page"] == "compare"
    assert event.details["row_count"] == total


def test_an_export_records_the_user_row_count_and_filters(ready_tenant):
    alpha = ready_tenant("alpha")

    response = alpha.client.get("/download-results?risk=High")

    exported_rows = list(csv.reader(io.StringIO(response.get_data(as_text=True))))[1:]
    [event] = _events(alpha.organisation, "results_exported")
    assert event.user_id == alpha.user.id
    assert event.details["row_count"] == len(exported_rows) > 0
    assert event.details["filters"] == {"risk": "High"}
    assert _events(alpha.organisation, "prediction_run") == []  # recorded once, as an export


def test_no_prediction_event_when_there_is_no_model(tenant):
    alpha = tenant("alpha")

    assert alpha.client.get("/results").status_code == 302

    assert _events(alpha.organisation, "prediction_run") == []


def test_every_recorded_action_is_filterable_on_the_audit_page():
    sources = [ROOT / "app.py", *sorted((ROOT / "auth").glob("*.py"))]
    recorded = {
        action
        for path in sources
        for action in re.findall(
            r'(?:record_audit_event|_record_prediction_use)\(\s*"([a-z_]+)"', path.read_text(encoding="utf-8")
        )
    }

    assert {"login_success", "login_failure", "logout", "dataset_uploaded", "model_trained", "prediction_run", "results_exported"} <= recorded
    assert recorded <= set(AUDIT_ACTIONS)
