"""Record audit personal-data retention in column comments

Revision ID: a4d2e6c81b37
Revises: f86b267dd451
Create Date: 2026-09-13 19:40:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a4d2e6c81b37'
down_revision = 'f86b267dd451'
branch_labels = None
depends_on = None

PERSONAL_DATA_COMMENT = 'Personal data: nulled 90 days after created_at. The audit row itself is kept 24 months.'
PERSONAL_COLUMNS = (('ip_address', sa.String(length=45)), ('user_agent', sa.String(length=512)))


def upgrade():
    with op.batch_alter_table('audit_events', schema=None) as batch_op:
        for name, type_ in PERSONAL_COLUMNS:
            batch_op.alter_column(name, existing_type=type_, existing_nullable=True, comment=PERSONAL_DATA_COMMENT)


def downgrade():
    with op.batch_alter_table('audit_events', schema=None) as batch_op:
        for name, type_ in PERSONAL_COLUMNS:
            batch_op.alter_column(
                name,
                existing_type=type_,
                existing_nullable=True,
                comment=None,
                existing_comment=PERSONAL_DATA_COMMENT,
            )
