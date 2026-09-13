"""ORM models. Every tenant-owned table carries organisation_id."""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Enum, ForeignKey, Index, Integer, String, Uuid, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _enum(enum_cls: type[enum.Enum], name: str) -> Enum:
    # Store the lowercase value ("owner"), not the member name ("OWNER").
    return Enum(
        enum_cls,
        name=name,
        values_callable=lambda members: [m.value for m in members],
        validate_strings=True,
        create_constraint=True,
    )


class Role(str, enum.Enum):
    OWNER = "owner"
    STAFF = "staff"
    VIEWER = "viewer"


class DatasetKind(str, enum.Enum):
    TRAINING = "training"
    PREDICTION = "prediction"
    ACTUAL = "actual"


class Organisation(db.Model):
    __tablename__ = "organisations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(100), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    plan_tier: Mapped[str] = mapped_column(String(32), default="standard")

    users: Mapped[list[User]] = relationship(back_populates="organisation")


class User(db.Model):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    organisation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("organisations.id"), index=True)
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
    uploaded_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


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
    trained_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditEvent(db.Model):
    __tablename__ = "audit_events"
    __table_args__ = (
        Index("ix_audit_events_org_created", "organisation_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # Null only when no tenant can be resolved, e.g. a failed login for an email
    # that matches no account. Tenant-scoped queries never return such rows.
    organisation_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("organisations.id"))
    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(64))
    entity_type: Mapped[str | None] = mapped_column(String(64))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    ip_address: Mapped[str | None] = mapped_column(String(45))
    user_agent: Mapped[str | None] = mapped_column(String(512))
    # Structured context, e.g. the row count and applied filters of an export.
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
