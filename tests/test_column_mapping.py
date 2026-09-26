"""Files that use an institution's own column names are mapped, not rejected."""
import pandas as pd
import pytest

from db import db
from db.models import AuditEvent, ColumnMapping, Dataset, ModelArtifact
from db.tenancy import scoped_select
from utils.column_mapping import apply_mapping, guess_mapping, mapping_problems

EXPECTED = ["student_id", "student_name", "attendance", "quiz_score", "target"]

NAMES = ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Katherine Johnson"]


def _their_export(rows: int = 40) -> bytes:
    """A plausible export: their own column names, their own order, an extra column."""
    lines = ["Student No,Full Name,Attendance %,Quiz Avg,Tutor Group,Final Result"]
    for index in range(rows):
        passing = index % 2 == 0
        lines.append(
            f"MAT{100 + index},{NAMES[index % len(NAMES)]},{90 if passing else 50},"
            f"{80 if passing else 40},4{'B' if passing else 'C'},{0 if passing else 1}"
        )
    return ("\n".join(lines) + "\n").encode()


THEIR_CSV = _their_export()


def test_guessing_matches_exact_names_synonyms_and_near_spellings():
    columns = ["Student No", "Full Name", "Attendance %", "Quiz Avg", "Tutor Group", "Final Result"]

    guesses = guess_mapping(EXPECTED, columns)

    assert guesses["student_id"] == "Student No"
    assert guesses["student_name"] == "Full Name"
    assert guesses["attendance"] == "Attendance %"
    assert guesses["quiz_score"] == "Quiz Avg"
    assert guesses["target"] == "Final Result"


def test_each_column_is_only_guessed_once_and_unknowns_are_left_blank():
    guesses = guess_mapping(EXPECTED, ["student_id", "mystery"])

    assert guesses["student_id"] == "student_id"
    assert [field for field, source in guesses.items() if source] == ["student_id"]


def test_applying_a_mapping_never_overwrites_correctly_named_columns():
    df = pd.DataFrame({"student_id": ["S-1"], "Attendance %": [90], "attendance": [50]})

    mapped = apply_mapping(df, {"student_id": "student_id", "attendance": "Attendance %"})

    assert list(mapped["attendance"]) == [50]  # the file's own attendance column wins
    assert "Attendance %" in mapped.columns


@pytest.mark.parametrize(
    ("mapping", "problem"),
    [
        ({"student_id": "Nope"}, "not in the uploaded file"),
        ({"student_id": "A", "student_name": "A"}, "only be used once"),
        ({"student_name": "B"}, "These are required"),
    ],
)
def test_a_mapping_that_cannot_work_is_explained(mapping, problem):
    problems = mapping_problems(mapping, ["A", "B"], ["student_id"])

    assert any(problem in message for message in problems)


def _upload_their_file(tenant_client, upload, path="/upload-train", data=THEIR_CSV):
    return upload(tenant_client, path, data, filename="term-export.csv")


def test_an_unrecognised_file_goes_to_the_mapping_screen_with_guesses(tenant, upload):
    alpha = tenant("alpha")

    response = _upload_their_file(alpha.client, upload)

    assert response.status_code == 302 and "/map-columns/" in response.headers["Location"]
    page = alpha.client.get(response.headers["Location"]).get_data(as_text=True)
    assert "We could not find these columns" in page
    assert '<option value="Student No" selected>' in page
    assert '<option value="Final Result" selected>' in page
    # The file is kept as uploaded while its columns are being matched.
    [dataset] = db.session.scalars(scoped_select(Dataset, alpha.organisation.id)).all()
    assert dataset.column_names[0] == "Student No"


def test_saving_a_mapping_makes_the_uploaded_file_usable(tenant, upload):
    alpha = tenant("alpha")
    location = _upload_their_file(alpha.client, upload).headers["Location"]

    saved = alpha.client.post(location, data={
        "student_id": "Student No", "student_name": "Full Name", "attendance": "Attendance %",
        "quiz_score": "Quiz Avg", "target": "Final Result",
    }, follow_redirects=True)

    page = saved.get_data(as_text=True)
    assert "Column mapping saved" in page
    assert "Ada Lovelace" in page  # the preview now shows the mapped file
    [event] = db.session.scalars(
        scoped_select(AuditEvent, alpha.organisation.id).where(AuditEvent.action == "column_mapping_saved")
    ).all()
    assert event.details["kind"] == "training"
    assert event.details["mapping"]["student_id"] == "Student No"


def test_a_saved_mapping_is_reused_by_the_next_upload(tenant, upload):
    alpha = tenant("alpha")
    location = _upload_their_file(alpha.client, upload).headers["Location"]
    alpha.client.post(location, data={
        "student_id": "Student No", "student_name": "Full Name", "attendance": "Attendance %",
        "quiz_score": "Quiz Avg", "target": "Final Result",
    })

    second = _upload_their_file(alpha.client, upload, data=THEIR_CSV.replace(b"Ada Lovelace", b"Ida Rhodes"))

    assert second.status_code == 302 and "/map-columns/" not in second.headers["Location"]
    page = alpha.client.get("/upload-train").get_data(as_text=True)
    assert "uploaded successfully" in page and "Ida Rhodes" in page


def test_a_mapping_belongs_to_one_organisation(tenant, upload):
    alpha, beta = tenant("alpha"), tenant("beta")
    location = _upload_their_file(alpha.client, upload).headers["Location"]
    alpha.client.post(location, data={
        "student_id": "Student No", "student_name": "Full Name", "attendance": "Attendance %",
        "quiz_score": "Quiz Avg", "target": "Final Result",
    })

    beta_response = _upload_their_file(beta.client, upload)

    assert "/map-columns/" in beta_response.headers["Location"]  # beta must confirm its own mapping
    assert db.session.scalars(scoped_select(ColumnMapping, beta.organisation.id)).all() == []


def test_a_mapping_missing_a_required_column_is_refused(tenant, upload):
    alpha = tenant("alpha")
    location = _upload_their_file(alpha.client, upload).headers["Location"]

    page = alpha.client.post(location, data={"student_id": "Student No", "attendance": "Attendance %"},
                             follow_redirects=True).get_data(as_text=True)

    assert "These are required: student_name, target." in page
    assert db.session.scalars(scoped_select(ColumnMapping, alpha.organisation.id)).all() == []


def test_a_mapping_with_no_feature_columns_is_refused(tenant, upload):
    alpha = tenant("alpha")
    location = _upload_their_file(alpha.client, upload).headers["Location"]

    page = alpha.client.post(location, data={
        "student_id": "Student No", "student_name": "Full Name", "target": "Final Result",
    }, follow_redirects=True).get_data(as_text=True)

    assert "Map at least one feature column" in page


def test_mapped_files_train_and_predict(tenant, upload):
    alpha = tenant("alpha")
    mapping = {"student_id": "Student No", "student_name": "Full Name", "attendance": "Attendance %",
               "quiz_score": "Quiz Avg", "target": "Final Result"}
    alpha.client.post(_upload_their_file(alpha.client, upload).headers["Location"], data=mapping)

    assert alpha.client.post("/train").status_code == 200

    [artifact] = db.session.scalars(scoped_select(ModelArtifact, alpha.organisation.id)).all()
    assert artifact.features == ["attendance", "quiz_score"]


def test_the_mapping_screen_belongs_to_the_dataset_s_organisation(tenant, upload, make_org):
    alpha, beta = tenant("alpha"), tenant("beta")
    location = _upload_their_file(alpha.client, upload).headers["Location"]

    assert beta.client.get(location).status_code == 404
