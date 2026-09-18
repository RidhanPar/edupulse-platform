"""Index the audit log for keyset pagination by user, action and date

Revision ID: c3f1a9d2e7b4
Revises: 6d7617fc1b41
Create Date: 2026-09-18 10:00:00.000000

Measured on PostgreSQL 16 with 1M rows: the single (organisation_id, created_at) index
left a filter on a rare user, or on an action with no matches, as a full parallel scan
of the table, and every page needed an extra sort because id was not in the index.

On a large production table, create these with CREATE INDEX CONCURRENTLY outside a
transaction instead; at current sizes the brief lock is harmless.
"""
from alembic import op


# revision identifiers, used by Alembic.
revision = 'c3f1a9d2e7b4'
down_revision = '6d7617fc1b41'
branch_labels = None
depends_on = None

NEW_INDEXES = (
    ('ix_audit_events_org_created_id', ['organisation_id', 'created_at', 'id']),
    ('ix_audit_events_org_user_created_id', ['organisation_id', 'user_id', 'created_at', 'id']),
    ('ix_audit_events_org_action_created_id', ['organisation_id', 'action', 'created_at', 'id']),
)


def upgrade():
    with op.batch_alter_table('audit_events', schema=None) as batch_op:
        for name, columns in NEW_INDEXES:
            batch_op.create_index(name, columns, unique=False)
        batch_op.drop_index('ix_audit_events_org_created')


def downgrade():
    with op.batch_alter_table('audit_events', schema=None) as batch_op:
        batch_op.create_index('ix_audit_events_org_created', ['organisation_id', 'created_at'], unique=False)
        for name, _ in reversed(NEW_INDEXES):
            batch_op.drop_index(name)
