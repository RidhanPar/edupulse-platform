"""ORM models. Every tenant-owned table carries organisation_id.

JSON columns use the generic JSON type, which is json rather than jsonb on
PostgreSQL. That is fine while they are stored and read back whole; if metrics
ever need to be queried or indexed, migrate those columns to JSONB.
"""
from __future__ import annotations

import calendar
import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Enum, ForeignKey, Index, Integer, String, Uuid, event, text, true
from sqlalchemy.orm import Mapped, ORMExecuteState, Session, mapped_column, relationship

from db import db

# Default storage limit for uploaded datasets, per GDPR Article 5(1)(e). To become
# a per-organisation setting; the purge job that acts on it is not built yet.
DATASET_RETENTION_MONTHS = 24


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def add_months(value: datetime, months: int) -> datetime:
    """Calendar-month arithmetic, clamped to the last day of a shorter month."""
    month_index = value.month - 1 + months
    year, month = value.year + month_index // 12, month_index % 12 + 1
    return value.replace(year=year, month=month, day=min(value.day, calendar.monthrange(year, month)[1]))


def _enum(enum_cls: type[enum.Enum], name: str) -> Enum:
    # Store the lowercase value ("owner"), not the member name ("OWNER").
    return Enum(
        enum_cls,
        name=name,
        values_callable=lambda members: [m.value for m in members],
        validate_strings=True,
        create_constraint=True,
    )


def _user_fk() -> ForeignKey:
    # Users are deactivated via is_active, never deleted, so nothing that
    # references a user may silently lose that reference.
    return ForeignKey("users.id", ondelete="RESTRICT")


class Role(str, enum.Enum):
    OWNER = "owner"
    STAFF = "staff"
    VIEWER = "viewer"


class DatasetKind(str, enum.Enum):
    TRAINING = "training"
    PREDICTION = "prediction"
    ACTUAL = "actual"


class PlanTier(str, enum.Enum):
    PILOT = "pilot"  # unpaid or heavily discounted reference customers
    SMALL = "small"  # under 1,000 students
    MEDIUM = "medium"  # 1,000 to 3,000 students
    LARGE = "large"  # above 3,000 students


class Organisation(db.Model):
    __tablename__ = "organisations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    plan_tier: Mapped[PlanTier] = mapped_column(_enum(PlanTier, "plan_tier"), default=PlanTier.PILOT)
    # A suspended organisation keeps its data, but its users cannot log in.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())

    users: Mapped[list[User]] = relationship(back_populates="organisation")


class User(db.Model):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organisation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organisations.id"), index=True)
    # Deliberately unique across all organisations, not per (organisation_id, email):
    # login needs only an email and password. The accepted cost is that one person
    # cannot hold accounts at two institutions under the same address. Do not change
    # this without redesigning login.
    email: Mapped[str] = mapped_column(String(320), unique=True)
    # Null until an invited user sets their first password.
    password_hash: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(_enum(Role, "user_role"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # SHA-256 of the one-time invite token. The raw token exists only in the link.
    invite_token_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    invite_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    organisation: Mapped[Organisation] = relationship(back_populates="users")


class Dataset(db.Model):
    __tablename__ = "datasets"
    __table_args__ = (
        # Serves "latest dataset of this kind for this organisation".
        Index("ix_datasets_org_kind_uploaded", "organisation_id", "kind", "uploaded_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organisation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organisations.id"))
    kind: Mapped[DatasetKind] = mapped_column(_enum(DatasetKind, "dataset_kind"))
    original_filename: Mapped[str] = mapped_column(String(255))
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    row_count: Mapped[int] = mapped_column(Integer)
    column_names: Mapped[list[str]] = mapped_column(JSON)
    uploaded_by: Mapped[uuid.UUID] = mapped_column(_user_fk())
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # Set on insert to uploaded_at + DATASET_RETENTION_MONTHS unless given explicitly.
    retention_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # Soft-delete marker for the future retention purge.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)


@event.listens_for(Dataset, "before_insert")
def _set_default_retention(mapper, connection, target: Dataset) -> None:
    if target.uploaded_at is None:
        target.uploaded_at = utcnow()
    if target.retention_expires_at is None:
        target.retention_expires_at = add_months(target.uploaded_at, DATASET_RETENTION_MONTHS)


class ModelArtifact(db.Model):
    __tablename__ = "model_artifacts"
    __table_args__ = (
        # At most one active model per organisation, enforced by the database.
        Index(
            "uq_model_artifacts_one_active_per_org",
            "organisation_id",
            unique=True,
            postgresql_where=text("is_active"),
            sqlite_where=text("is_active"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organisation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organisations.id"), index=True)
    algorithm_name: Mapped[str] = mapped_column(String(100))
    metrics: Mapped[dict] = mapped_column(JSON)
    feature_importances: Mapped[dict] = mapped_column(JSON)
    model_comparison: Mapped[list[dict]] = mapped_column(JSON)
    storage_key: Mapped[str] = mapped_column(String(512), unique=True)
    trained_by: Mapped[uuid.UUID] = mapped_column(_user_fk())
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditEvent(db.Model):
    """Append-only: rows are inserted, never updated or deleted."""

    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_org_created", "organisation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Null only when no tenant can be resolved, e.g. a failed login for an email
    # that matches no account. Tenant-scoped queries never return such rows.
    organisation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organisations.id"))
    user_id: Mapped[uuid.UUID | None] = mapped_column(_user_fk())
    action: Mapped[str] = mapped_column(String(64))
    entity_type: Mapped[str | None] = mapped_column(String(64))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(512))
    # Structured context, e.g. the row count and applied filters of an export.
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AppendOnlyViolation(RuntimeError):
    """Raised when code tries to modify or remove an audit record."""


@event.listens_for(AuditEvent, "before_update")
def _refuse_audit_update(mapper, connection, target: AuditEvent) -> None:
    raise AppendOnlyViolation("audit_events is append-only: rows cannot be updated")


@event.listens_for(AuditEvent, "before_delete")
def _refuse_audit_delete(mapper, connection, target: AuditEvent) -> None:
    raise AppendOnlyViolation("audit_events is append-only: rows cannot be deleted")


@event.listens_for(Session, "do_orm_execute")
def _refuse_bulk_audit_changes(state: ORMExecuteState) -> None:
    # Bulk update()/delete() statements bypass the per-row mapper events above.
    if (state.is_update or state.is_delete) and any(m.class_ is AuditEvent for m in state.all_mappers):
        raise AppendOnlyViolation("audit_events is append-only: bulk UPDATE and DELETE are refused")
