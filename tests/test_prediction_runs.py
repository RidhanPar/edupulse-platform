"""Stored prediction runs: results are read, not recomputed, and risk is tracked over time."""
import io

import pandas as pd
import pytest

from db import db
from db.models import AuditEvent, PredictionRun, Role, StudentPrediction
from db.tenancy import scoped_select

# Student MAT384 is the first row of tests/fixtures/prediction.csv.
WATCHED = "MAT384"
STRONG = {"attendance": 100, "assignment_score": 20, "quiz_score": 20, "study_time": 10,
          "lms_activity": 100, "previous_grade": 20, "missed_submissions": 0}
STRUGGLING = {"attendance": 15, "assignment_score": 1, "quiz_score": 1, "study_time": 0,
              "lms_activity": 2, "previous_grade": 1, "missed_submissions": 6}


def _with_student(data: bytes, student_id: str, values: dict) -> bytes:
    df = pd.read_csv(io.BytesIO(data))
    for column, value in values.items():
        df.loc[df["student_id"] == student_id, column] = value
    return df.to_csv(index=False).encode()


def _runs(organisation) -> list[PredictionRun]:
    return db.session.scalars(
        scoped_select(PredictionRun, organisation.id).order_by(PredictionRun.created_at)
    ).all()


def _row(run, organisation, student_id=WATCHED) -> StudentPrediction:
    return db.session.scalars(
        scoped_select(StudentPrediction, organisation.id)
        .where(StudentPrediction.run_id == run.id, StudentPrediction.student_id == student_id)
    ).one()


@pytest.fixture()
def trained_tenant(tenant, upload, csv_fixtures):
    """A tenant with a trained model and no prediction data yet."""
    alpha = tenant("alpha")
    upload(alpha.client, "/upload-train", csv_fixtures["training"])
    assert alpha.client.post("/train").status_code == 200
    return alpha


def test_uploading_prediction_data_runs_and_stores_predictions(trained_tenant, upload, csv_fixtures):
    alpha = trained_tenant

    upload(alpha.client, "/upload-predict", csv_fixtures["prediction"])

    [run] = _runs(alpha.organisation)
    expected = len(pd.read_csv(io.BytesIO(csv_fixtures["prediction"])))
    stored = db.session.scalars(scoped_select(StudentPrediction, alpha.organisation.id)).all()
    assert run.row_count == len(stored) == expected
    assert run.previous_run_id is None
    assert set(_row(run, alpha.organisation).features) == set(run.model_artifact.features)
    assert "Predictions updated for" in alpha.client.get("/upload-predict").get_data(as_text=True)


def test_viewing_results_never_loads_the_model_or_rereads_the_prediction_file(ready_tenant, monkeypatch):
    alpha = ready_tenant("alpha")
    [run] = _runs(alpha.organisation)
    forbidden = {alpha.artifact.storage_key, run.prediction_dataset.storage_key}
    backend = alpha.client.application.extensions["storage"]
    original_read = backend.read
    reads = []
    monkeypatch.setattr(backend, "read", lambda key: reads.append(key) or original_read(key))

    for path in ("/results", "/results?risk=High", "/download-results", "/compare", f"/students/{WATCHED}"):
        assert alpha.client.get(path).status_code == 200, path

    assert not forbidden & set(reads)
    # Only the comparison reads a file: the actual outcomes, which are not part of a run.
    assert set(reads) <= {key for key in reads if key.endswith(".csv")} and len(set(reads)) <= 1


def test_risk_change_is_tracked_from_one_run_to_the_next(trained_tenant, upload, csv_fixtures):
    alpha = trained_tenant
    upload(alpha.client, "/upload-predict", _with_student(csv_fixtures["prediction"], WATCHED, STRONG))
    upload(alpha.client, "/upload-predict", _with_student(csv_fixtures["prediction"], WATCHED, STRUGGLING))

    first, second = _runs(alpha.organisation)
    before, after = _row(first, alpha.organisation), _row(second, alpha.organisation)
    assert second.previous_run_id == first.id
    assert (before.risk_level, after.risk_level) == ("Low", "High")
    assert after.previous_risk_level == "Low"

    page = alpha.client.get("/results?change=up").get_data(as_text=True)
    assert "Risk rose since last run" in page
    assert WATCHED in page and "Rose from Low" in page


def test_the_first_run_claims_no_changes(trained_tenant, upload, csv_fixtures):
    alpha = trained_tenant
    upload(alpha.client, "/upload-predict", csv_fixtures["prediction"])

    page = alpha.client.get("/results").get_data(as_text=True)

    assert "This is the first run" in page
    assert "Since Last Run" not in page and "Risk rose since last run" not in page


def test_retraining_reruns_predictions_with_the_new_model(ready_tenant):
    alpha = ready_tenant("alpha")

    assert alpha.client.post("/train").status_code == 200

    first, second = _runs(alpha.organisation)
    assert second.model_artifact_id != first.model_artifact_id
    assert "has changed since these predictions were made" not in alpha.client.get("/results").get_data(as_text=True)


def test_results_say_when_they_are_out_of_date(ready_tenant, upload, csv_fixtures):
    alpha = ready_tenant("alpha")
    # A new prediction file the model cannot use: it is stored, but no new run is made.
    broken = pd.read_csv(io.BytesIO(csv_fixtures["prediction"])).drop(columns=["attendance"])
    upload(alpha.client, "/upload-predict", broken.to_csv(index=False).encode())

    page = alpha.client.get("/results").get_data(as_text=True)

    assert "has changed since these predictions were made" in page
    assert len(_runs(alpha.organisation)) == 1


def test_the_student_page_shows_history_and_is_audited(ready_tenant):
    alpha = ready_tenant("alpha")
    alpha.client.post("/predict")

    page = alpha.client.get(f"/students/{WATCHED}").get_data(as_text=True)

    assert "Student_384" in page
    assert page.count('<tr class="history-row">') == 2
    assert "Indicators Used" in page and "attendance" in page
    [event] = db.session.scalars(
        scoped_select(AuditEvent, alpha.organisation.id).where(AuditEvent.action == "student_viewed")
    ).all()
    assert (event.user_id, event.entity_type, event.entity_id) == (alpha.user.id, "student", WATCHED)


def test_an_unknown_student_is_not_found(ready_tenant):
    assert ready_tenant("alpha").client.get("/students/NOBODY").status_code == 404


def test_the_same_student_id_in_two_organisations_never_mixes(ready_tenant, tenant, upload, csv_fixtures):
    alpha = ready_tenant("alpha")
    beta = tenant("beta")
    renamed = csv_fixtures["prediction"].replace(b"Student_384", b"Beta Person")
    upload(beta.client, "/upload-train", csv_fixtures["training"])
    beta.client.post("/train")
    upload(beta.client, "/upload-predict", renamed)

    alpha_page = alpha.client.get(f"/students/{WATCHED}").get_data(as_text=True)
    beta_page = beta.client.get(f"/students/{WATCHED}").get_data(as_text=True)

    assert "Student_384" in alpha_page and "Beta Person" not in alpha_page
    assert "Beta Person" in beta_page and "Student_384" not in beta_page
    assert alpha_page.count('<tr class="history-row">') == 1


def test_viewers_read_results_but_only_staff_run_predictions(ready_tenant):
    viewer = ready_tenant("alpha", Role.VIEWER)

    page = viewer.client.get("/results").get_data(as_text=True)

    assert "Total students" in page and "Run Predictions" not in page
    assert viewer.client.post("/predict").status_code == 403
    assert len(_runs(viewer.organisation)) == 1
