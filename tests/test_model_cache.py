import pytest

from db import db
from db.models import ModelArtifact
from db.tenancy import scoped_select
from utils.model_cache import ModelCache


def test_the_loader_runs_once_per_key():
    cache = ModelCache(capacity=2)
    calls = []

    def loader():
        calls.append(1)
        return object()

    first = cache.get_or_load("a", loader)

    assert cache.get_or_load("a", loader) is first
    assert len(calls) == 1


def test_the_least_recently_used_entry_is_evicted():
    cache = ModelCache(capacity=2)
    cache.get_or_load("a", object)
    cache.get_or_load("b", object)
    cache.get_or_load("a", object)  # "a" is now the most recently used

    cache.get_or_load("c", object)

    assert "a" in cache and "c" in cache and "b" not in cache
    assert len(cache) == 2


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        ModelCache(capacity=0)


@pytest.fixture()
def model_reads(app, monkeypatch):
    """Storage keys of every model file read from the storage backend."""
    backend = app.extensions["storage"]
    original_read = backend.read
    reads = []

    def counting_read(key):
        if key.endswith(".pkl"):
            reads.append(key)
        return original_read(key)

    monkeypatch.setattr(backend, "read", counting_read)
    return reads


def test_a_model_is_deserialised_once_rather_than_on_every_request(app, ready_tenant, model_reads):
    alpha = ready_tenant("alpha")

    for path in ("/results", "/results?risk=High", "/download-results", "/compare", "/results"):
        assert alpha.client.get(path).status_code == 200, path

    assert model_reads == [alpha.artifact.storage_key]
    assert (alpha.organisation.id, alpha.artifact.id) in app.extensions["model_cache"]


def test_retraining_is_picked_up_through_the_new_artifact_id(ready_tenant, model_reads):
    alpha = ready_tenant("alpha")
    alpha.client.get("/results")

    assert alpha.client.post("/train").status_code == 200
    alpha.client.get("/results")
    alpha.client.get("/results")

    new = db.session.scalars(
        scoped_select(ModelArtifact, alpha.organisation.id).where(ModelArtifact.is_active.is_(True))
    ).one()
    assert model_reads == [alpha.artifact.storage_key, new.storage_key]


def test_each_organisation_loads_its_own_model(ready_tenant, model_reads):
    alpha, beta = ready_tenant("alpha"), ready_tenant("beta")

    alpha.client.get("/results")
    beta.client.get("/results")
    alpha.client.get("/results")

    assert model_reads == [alpha.artifact.storage_key, beta.artifact.storage_key]


def test_the_cache_holds_models_not_predictions(app, ready_tenant):
    alpha = ready_tenant("alpha")
    alpha.client.get("/results")

    cached = app.extensions["model_cache"].get_or_load(
        (alpha.organisation.id, alpha.artifact.id), lambda: pytest.fail("the model should already be cached")
    )

    assert hasattr(cached, "predict_proba")
