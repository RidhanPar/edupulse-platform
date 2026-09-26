"""Record a model's features, and store per-organisation column mappings

Revision ID: 5596dac8a7b4
Revises: c3f1a9d2e7b4
Create Date: 2026-09-26 22:23:34.388444

Models trained before this revision required every feature column, so their features
are the full standard list; existing rows are backfilled with it before the column
becomes NOT NULL.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '5596dac8a7b4'
down_revision = 'c3f1a9d2e7b4'
branch_labels = None
depends_on = None

# utils.preprocessing.FEATURE_COLUMNS as of this revision. Copied, not imported: a
# migration must keep working when the application's list changes.
LEGACY_FEATURES = (
    '["attendance", "assignment_score", "quiz_score", "study_time", '
    '"lms_activity", "previous_grade", "missed_submissions"]'
)


def _dataset_kind(bind):
    """The dataset_kind enum, reusing the type the datasets table already created.

    PostgreSQL enum types are database-wide: creating this table must not try to
    CREATE TYPE a second time. On SQLite the enum is a CHECK constraint per table.
    """
    if bind.dialect.name == 'postgresql':
        return postgresql.ENUM('training', 'prediction', 'actual', name='dataset_kind', create_type=False)
    return sa.Enum('training', 'prediction', 'actual', name='dataset_kind', create_constraint=True)


def upgrade():
    op.create_table(
        'column_mappings',
        sa.Column('id', sa.Uuid(), nullable=False),
        sa.Column('organisation_id', sa.Uuid(), nullable=False),
        sa.Column('kind', _dataset_kind(op.get_bind()), nullable=False),
        sa.Column('mapping', sa.JSON(), nullable=False),
        sa.Column('updated_by', sa.Uuid(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(['organisation_id'], ['organisations.id'], name=op.f('fk_column_mappings_organisation_id_organisations')),
        sa.ForeignKeyConstraint(['updated_by'], ['users.id'], name=op.f('fk_column_mappings_updated_by_users'), ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_column_mappings')),
        sa.UniqueConstraint('organisation_id', 'kind', name='uq_column_mappings_organisation_id_kind'),
    )
    with op.batch_alter_table('column_mappings', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_column_mappings_organisation_id'), ['organisation_id'], unique=False)

    # Add nullable, backfill, then enforce NOT NULL, so a table with existing models migrates.
    with op.batch_alter_table('model_artifacts', schema=None) as batch_op:
        batch_op.add_column(sa.Column('features', sa.JSON(), nullable=True))
    op.execute(f"UPDATE model_artifacts SET features = '{LEGACY_FEATURES}' WHERE features IS NULL")
    with op.batch_alter_table('model_artifacts', schema=None) as batch_op:
        batch_op.alter_column('features', existing_type=sa.JSON(), nullable=False)


def downgrade():
    with op.batch_alter_table('model_artifacts', schema=None) as batch_op:
        batch_op.drop_column('features')

    with op.batch_alter_table('column_mappings', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_column_mappings_organisation_id'))

    op.drop_table('column_mappings')
