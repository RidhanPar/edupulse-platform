"""The single place that scopes tenant-owned queries to an organisation.

Dataset, ModelArtifact and AuditEvent rows must be selected through these
helpers only. tests/test_tenancy.py fails the build if an organisation filter
is written by hand anywhere else.
"""
from __future__ import annotations

import uuid

from flask import abort
from sqlalchemy import Select, select

from db import db
from db.models import AuditEvent, Dataset, ModelArtifact

TENANT_MODELS = (Dataset, ModelArtifact, AuditEvent)


class TenancyError(RuntimeError):
    """Raised when a tenant-scoped query is attempted without a valid tenant."""


def _as_uuid(value) -> uuid.UUID | None:
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


def scoped_select(model, organisation_id) -> Select:
    """SELECT for `model` restricted to one organisation. Chain further clauses onto it."""
    if model not in TENANT_MODELS:
        raise TypeError(f"{model.__name__} is not a tenant-scoped model")
    org_id = _as_uuid(organisation_id)
    if org_id is None:
        raise TenancyError(f"a valid organisation_id is required to query {model.__name__}")
    return select(model).where(model.organisation_id == org_id)


def scoped_get(model, organisation_id, entity_id):
    """One row by id, or None if it is missing or belongs to another organisation."""
    stmt = scoped_select(model, organisation_id)
    row_id = _as_uuid(entity_id)
    if row_id is None:
        return None
    return db.session.scalars(stmt.where(model.id == row_id)).one_or_none()


def scoped_get_or_404(model, organisation_id, entity_id):
    """Like scoped_get, but 404s: another tenant's row is indistinguishable from a missing one."""
    row = scoped_get(model, organisation_id, entity_id)
    if row is None:
        abort(404)
    return row
