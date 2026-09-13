import re
import uuid
from pathlib import Path

import pytest
from werkzeug.exceptions import NotFound

from db import db
from db.models import AuditEvent, Dataset, DatasetKind, ModelArtifact, Organisation
from db.tenancy import TenancyError, scoped_get, scoped_get_or_404, scoped_select

ROOT = Path(__file__).resolve().parents[1]
TENANT_MODELS = [Dataset, ModelArtifact, AuditEvent]


def _seed(organisation, user) -> dict:
    rows = {
        Dataset: Dataset(
            organisation_id=organisation.id,
            kind=DatasetKind.TRAINING,
            original_filename="training.csv",
            storage_key=f"org/{organisation.id}/datasets/{uuid.uuid4()}.csv",
            row_count=10,
            column_names=["student_id", "target"],
            uploaded_by=user.id,
        ),
        ModelArtifact: ModelArtifact(
            organisation_id=organisation.id,
            algorithm_name="Random Forest",
            metrics={},
            feature_importances={},
            model_comparison=[],
            storage_key=f"org/{organisation.id}/models/{uuid.uuid4()}.pkl",
            trained_by=user.id,
            is_active=True,
        ),
        AuditEvent: AuditEvent(organisation_id=organisation.id, user_id=user.id, action="login_success"),
    }
    db.session.add_all(rows.values())
    db.session.commit()
    return rows


@pytest.fixture()
def tenants(make_org):
    org_a, user_a = make_org("alpha")
    org_b, user_b = make_org("beta")
    return (org_a, _seed(org_a, user_a)), (org_b, _seed(org_b, user_b))


@pytest.mark.parametrize("model", TENANT_MODELS)
def test_scoped_select_returns_only_own_rows(tenants, model):
    (org_a, rows_a), (org_b, rows_b) = tenants

    assert db.session.scalars(scoped_select(model, org_a.id)).all() == [rows_a[model]]
    assert db.session.scalars(scoped_select(model, org_b.id)).all() == [rows_b[model]]


@pytest.mark.parametrize("model", TENANT_MODELS)
def test_scoped_get_hides_other_organisations_rows(tenants, model):
    (org_a, rows_a), (_, rows_b) = tenants

    assert scoped_get(model, org_a.id, rows_a[model].id) is rows_a[model]
    assert scoped_get(model, org_a.id, rows_b[model].id) is None
    with pytest.raises(NotFound):
        scoped_get_or_404(model, org_a.id, rows_b[model].id)


def test_scoped_get_accepts_string_ids_and_ignores_malformed_ones(tenants):
    (org_a, rows_a), _ = tenants
    dataset = rows_a[Dataset]

    assert scoped_get(Dataset, str(org_a.id), str(dataset.id)) is dataset
    assert scoped_get(Dataset, org_a.id, "not-a-uuid") is None


def test_audit_events_without_organisation_are_never_returned(tenants):
    (org_a, _), _ = tenants
    db.session.add(AuditEvent(organisation_id=None, action="login_failure"))
    db.session.commit()

    actions = [event.action for event in db.session.scalars(scoped_select(AuditEvent, org_a.id))]

    assert actions == ["login_success"]


@pytest.mark.parametrize("organisation_id", [None, "", "not-a-uuid"])
def test_query_without_valid_organisation_is_refused(app, organisation_id):
    with pytest.raises(TenancyError):
        scoped_select(Dataset, organisation_id)


def test_non_tenant_models_are_refused(app):
    with pytest.raises(TypeError):
        scoped_select(Organisation, uuid.uuid4())


HAND_WRITTEN_FILTER = re.compile(r"organisation_id\s*==|filter_by\([^)]*organisation_id")
SKIPPED_DIRS = {".git", ".venv", "venv", "tests", "migrations", "instance"}


def test_no_hand_written_organisation_filters():
    sources = sorted(ROOT.glob("*.py"))
    for directory in sorted(ROOT.iterdir()):
        if directory.is_dir() and directory.name not in SKIPPED_DIRS:
            sources.extend(sorted(directory.rglob("*.py")))

    offenders = [
        f"{path.relative_to(ROOT)}:{lineno}"
        for path in sources
        if path != ROOT / "db" / "tenancy.py"
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if HAND_WRITTEN_FILTER.search(line)
    ]

    assert offenders == [], "Use db.tenancy helpers instead of filtering on organisation_id directly"
