"""A file that supplies only some features trains a model on those, and says so."""
import io

import pandas as pd
import pytest

from db import db
from db.models import ModelArtifact
from db.tenancy import scoped_select
from utils.preprocessing import FEATURE_COLUMNS, available_features, missing_features, validate_columns

SUPPLIED = ["attendance", "quiz_score", "previous_grade"]
ABSENT = [feature for feature in FEATURE_COLUMNS if feature not in SUPPLIED]


def _drop_absent_features(data: bytes) -> bytes:
    df = pd.read_csv(io.BytesIO(data))
    return df.drop(columns=ABSENT).to_csv(index=False).encode()


@pytest.fixture()
def partial_csvs(csv_fixtures) -> dict[str, bytes]:
    return {name: _drop_absent_features(data) for name, data in csv_fixtures.items() if name != "actual"}


def test_available_and_missing_features_are_reported(partial_csvs):
    df = pd.read_csv(io.BytesIO(partial_csvs["training"]))

    assert available_features(df) == SUPPLIED
    assert missing_features(df) == ABSENT
    assert validate_columns(df) == (True, [])


def test_a_file_with_no_known_features_is_rejected():
    df = pd.DataFrame({"student_id": ["S-1"], "student_name": ["Ada"], "target": [1]})

    valid, missing = validate_columns(df)

    assert valid is False
    assert any("at least one feature column" in item for item in missing)


def test_training_records_only_the_features_supplied(tenant, upload, partial_csvs):
    alpha = tenant("alpha")
    upload(alpha.client, "/upload-train", partial_csvs["training"])

    assert alpha.client.post("/train").status_code == 200

    [artifact] = db.session.scalars(scoped_select(ModelArtifact, alpha.organisation.id)).all()
    assert artifact.features == SUPPLIED
    # Nothing was invented for the absent columns, so they carry no importance at all.
    assert sorted(artifact.feature_importances) == sorted(SUPPLIED)


def test_the_explain_page_names_the_features_used_and_those_not_supplied(tenant, upload, partial_csvs):
    alpha = tenant("alpha")
    upload(alpha.client, "/upload-train", partial_csvs["training"])
    alpha.client.post("/train")

    page = alpha.client.get("/explain").get_data(as_text=True)

    assert "Features This Model Used" in page
    assert all(feature in page for feature in SUPPLIED)
    assert "Not supplied by your training file" in page
    assert all(feature in page for feature in ABSENT)


def test_predictions_run_on_the_same_reduced_feature_set(tenant, upload, partial_csvs):
    alpha = tenant("alpha")
    upload(alpha.client, "/upload-train", partial_csvs["training"])
    alpha.client.post("/train")
    upload(alpha.client, "/upload-predict", partial_csvs["prediction"])

    results = alpha.client.get("/results")
    export = alpha.client.get("/download-results")

    assert results.status_code == 200
    assert export.status_code == 200
    assert export.get_data(as_text=True).splitlines()[0].startswith("student_id,student_name,prediction")


def test_a_prediction_file_missing_a_trained_feature_is_refused(tenant, upload, csv_fixtures, partial_csvs):
    alpha = tenant("alpha")
    upload(alpha.client, "/upload-train", csv_fixtures["training"])  # trained on all seven
    alpha.client.post("/train")
    upload(alpha.client, "/upload-predict", partial_csvs["prediction"])  # supplies only three

    response = alpha.client.get("/results", follow_redirects=True)

    page = response.get_data(as_text=True)
    assert "missing columns this model was trained on" in page
    assert all(feature in page for feature in ABSENT)
