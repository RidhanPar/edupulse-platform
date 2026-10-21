"""Templates and the data dictionary: what an institution needs to prepare its first file."""
import csv
import io

import pytest

from db import db
from db.models import Dataset, Role
from db.tenancy import scoped_select
from utils.data_dictionary import DATA_DICTIONARY, EXAMPLE_ROWS
from utils.preprocessing import DISPLAY_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN

EXPECTED_HEADERS = {
    "training": DISPLAY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN],
    "prediction": DISPLAY_COLUMNS + FEATURE_COLUMNS,
    "actual": ["student_id", TARGET_COLUMN],
}


def test_every_column_the_app_reads_is_explained():
    assert set(DATA_DICTIONARY) == set(DISPLAY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])
    assert all(set(row) == set(DATA_DICTIONARY) for row in EXAMPLE_ROWS)


@pytest.mark.parametrize("kind", EXPECTED_HEADERS)
def test_each_template_has_exactly_the_expected_columns(tenant, kind):
    alpha = tenant("alpha")

    response = alpha.client.get(f"/templates/{kind}.csv")

    assert response.status_code == 200 and response.mimetype == "text/csv"
    assert f"edupulse-{kind}-template.csv" in response.headers["Content-Disposition"]
    rows = list(csv.reader(io.StringIO(response.get_data(as_text=True))))
    assert rows[0] == EXPECTED_HEADERS[kind]
    assert len(rows) == 1 + len(EXAMPLE_ROWS)


@pytest.mark.parametrize(("kind", "path"), [("prediction", "/upload-predict"), ("actual", "/upload-actual")])
def test_a_template_uploads_without_needing_a_column_mapping(tenant, upload, kind, path):
    alpha = tenant("alpha")
    template = alpha.client.get(f"/templates/{kind}.csv").get_data()

    response = upload(alpha.client, path, template, filename=f"{kind}.csv")

    assert response.status_code == 302 and "map-columns" not in response.headers["Location"]
    assert len(db.session.scalars(scoped_select(Dataset, alpha.organisation.id)).all()) == 1


def test_an_unknown_template_is_not_found(tenant):
    assert tenant("alpha").client.get("/templates/passwords.csv").status_code == 404


def test_viewers_cannot_download_templates(tenant):
    assert tenant("alpha", Role.VIEWER).client.get("/templates/training.csv").status_code == 403


@pytest.mark.parametrize("path", ["/upload-train", "/upload-predict", "/upload-actual"])
def test_upload_pages_explain_each_column_and_offer_a_template(tenant, path):
    page = tenant("alpha").client.get(path).get_data(as_text=True)

    assert "Download a template CSV" in page
    assert "What each column means" in page
    assert DATA_DICTIONARY["student_id"].split(".")[0] in page
    # Files with a name column say a real name is never needed. Actual results have no names.
    assert ("It can be a pseudonym" in page) is (path != "/upload-actual")
