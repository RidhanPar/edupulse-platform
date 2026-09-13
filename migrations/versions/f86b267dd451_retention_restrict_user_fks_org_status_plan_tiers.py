"""Dataset retention, RESTRICT on user references, organisation status and plan tiers

Revision ID: f86b267dd451
Revises: cb67422390af
Create Date: 2026-09-13 18:45:30.556091

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f86b267dd451'
down_revision = 'cb67422390af'
branch_labels = None
depends_on = None

PLAN_TIERS = ('pilot', 'small', 'medium', 'large')
plan_tier = sa.Enum(*PLAN_TIERS, name='plan_tier', create_constraint=True)

# Every foreign key that references users.id: (table, column, constraint name).
USER_FOREIGN_KEYS = (
    ('audit_events', 'user_id', 'fk_audit_events_user_id_users'),
    ('datasets', 'uploaded_by', 'fk_datasets_uploaded_by_users'),
    ('model_artifacts', 'trained_by', 'fk_model_artifacts_trained_by_users'),
)


def _set_user_fk_ondelete(ondelete):
    for table, column, name in USER_FOREIGN_KEYS:
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_constraint(name, type_='foreignkey')
            batch_op.create_foreign_key(name, 'users', [column], ['id'], ondelete=ondelete)


def upgrade():
    bind = op.get_bind()

    # Users are deactivated via is_active, never deleted.
    _set_user_fk_ondelete('RESTRICT')

    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.add_column(sa.Column('retention_expires_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True))
        batch_op.create_index(batch_op.f('ix_datasets_deleted_at'), ['deleted_at'], unique=False)

    # Existing datasets get the same 24-month retention that new uploads receive.
    if bind.dialect.name == 'postgresql':
        op.execute("UPDATE datasets SET retention_expires_at = uploaded_at + interval '24 months'")
    else:
        op.execute("UPDATE datasets SET retention_expires_at = datetime(uploaded_at, '+24 months')")

    # Organisations created before tiers existed (default 'standard') are early customers: pilots.
    op.execute(
        "UPDATE organisations SET plan_tier = 'pilot' "
        "WHERE plan_tier NOT IN ('pilot', 'small', 'medium', 'large')"
    )
    plan_tier.create(bind, checkfirst=True)  # CREATE TYPE on PostgreSQL; no-op on SQLite
    with op.batch_alter_table('organisations', schema=None) as batch_op:
        batch_op.add_column(sa.Column('is_active', sa.Boolean(), server_default=sa.true(), nullable=False))
        batch_op.alter_column(
            'plan_tier',
            existing_type=sa.String(length=32),
            type_=plan_tier,
            existing_nullable=False,
            postgresql_using='plan_tier::plan_tier',
        )


def downgrade():
    bind = op.get_bind()

    with op.batch_alter_table('organisations', schema=None) as batch_op:
        batch_op.alter_column(
            'plan_tier',
            existing_type=plan_tier,
            type_=sa.String(length=32),
            existing_nullable=False,
            postgresql_using='plan_tier::text',
        )
        batch_op.drop_column('is_active')
    plan_tier.drop(bind, checkfirst=True)

    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_datasets_deleted_at'))
        batch_op.drop_column('deleted_at')
        batch_op.drop_column('retention_expires_at')

    _set_user_fk_ondelete(None)
