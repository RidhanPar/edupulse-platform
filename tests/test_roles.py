import pytest

from auth import PUBLIC_ENDPOINTS, check_route_roles
from auth.roles import ROLE_RANK
from db import db
from db.models import ModelArtifact, Role
from db.tenancy import scoped_select

# (method, path, minimum role) for every non-public route.
ROLE_MAP = [
    ("GET", "/", Role.VIEWER),
    ("GET", "/about", Role.VIEWER),
    ("GET", "/results", Role.VIEWER),
    ("GET", "/explain", Role.VIEWER),
    ("GET", "/compare", Role.VIEWER),
    ("POST", "/recheck-comparison", Role.VIEWER),
    ("GET", "/train", Role.VIEWER),
    ("GET", "/change-password", Role.VIEWER),
    ("GET", "/upload-train", Role.STAFF),
    ("POST", "/upload-train", Role.STAFF),
    ("GET", "/upload-predict", Role.STAFF),
    ("POST", "/upload-predict", Role.STAFF),
    ("GET", "/upload-actual", Role.STAFF),
    ("POST", "/upload-actual", Role.STAFF),
    ("GET", "/download-results", Role.STAFF),
    ("POST", "/train", Role.OWNER),
]


@pytest.mark.parametrize("role", list(Role), ids=lambda role: role.value)
@pytest.mark.parametrize(("method", "path", "minimum"), ROLE_MAP, ids=[f"{m} {p}" for m, p, _ in ROLE_MAP])
def test_role_map(ready_tenant, role, method, path, minimum):
    response = ready_tenant("alpha", role).client.open(path, method=method)

    if ROLE_RANK[role] >= ROLE_RANK[minimum]:
        assert response.status_code in (200, 302), response.status_code
        assert not response.headers.get("Location", "").startswith("/login")
    else:
        assert response.status_code == 403


def test_viewer_cannot_post_to_train(ready_tenant):
    viewer = ready_tenant("alpha", Role.VIEWER)
    before = db.session.scalars(scoped_select(ModelArtifact, viewer.organisation.id)).all()

    assert viewer.client.post("/train").status_code == 403

    assert db.session.scalars(scoped_select(ModelArtifact, viewer.organisation.id)).all() == before


def test_every_non_public_endpoint_declares_a_role(app):
    undeclared = [
        endpoint
        for endpoint, view in app.view_functions.items()
        if endpoint not in PUBLIC_ENDPOINTS and not hasattr(view, "required_roles")
    ]

    assert undeclared == []


def test_an_endpoint_without_a_role_stops_the_app_from_starting(app):
    app.add_url_rule("/forgotten", endpoint="forgotten", view_func=lambda: "no role declared")

    with pytest.raises(RuntimeError, match="forgotten"):
        check_route_roles(app)


@pytest.mark.parametrize(
    ("role", "can_upload_and_export", "can_train"),
    [(Role.VIEWER, False, False), (Role.STAFF, True, False), (Role.OWNER, True, True)],
    ids=lambda value: value.value if isinstance(value, Role) else None,
)
def test_pages_only_offer_actions_the_role_can_perform(ready_tenant, role, can_upload_and_export, can_train):
    client = ready_tenant("alpha", role).client

    assert ("Download Results CSV" in client.get("/results").get_data(as_text=True)) is can_upload_and_export
    assert ("Upload Train Data" in client.get("/").get_data(as_text=True)) is can_upload_and_export
    assert ("Train Models" in client.get("/train").get_data(as_text=True)) is can_train
