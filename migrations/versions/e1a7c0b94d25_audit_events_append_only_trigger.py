"""Enforce the append-only audit log inside PostgreSQL

Revision ID: e1a7c0b94d25
Revises: 5596dac8a7b4
Create Date: 2026-10-21 10:00:00.000000

The application already refuses to update or delete audit rows (db/audit.py), but only
for statements it sends itself. This trigger applies the same rule to every connection:
psql sessions, reporting tools, other services. The one change it allows is the
personal-data redaction: an UPDATE that sets ip_address and user_agent to NULL and
leaves every other column untouched. TRUNCATE is refused too.

A database superuser can still disable triggers. Separating the application's database
role from the table owner is the next step; see docs/DEPLOYMENT.md.

PostgreSQL only. SQLite, used in development and tests, has no equivalent and relies on
the application-level guard.
"""
from alembic import op

# revision identifiers, used by Alembic.
revision = 'e1a7c0b94d25'
down_revision = '5596dac8a7b4'
branch_labels = None
depends_on = None

# json has no equality operator in PostgreSQL, so details is compared as text.
CREATE_FUNCTION = """
CREATE OR REPLACE FUNCTION audit_events_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'UPDATE'
       AND NEW.ip_address IS NULL
       AND NEW.user_agent IS NULL
       AND (NEW.id, NEW.organisation_id, NEW.user_id, NEW.action, NEW.entity_type,
            NEW.entity_id, NEW.details::text, NEW.created_at)
           IS NOT DISTINCT FROM
           (OLD.id, OLD.organisation_id, OLD.user_id, OLD.action, OLD.entity_type,
            OLD.entity_id, OLD.details::text, OLD.created_at)
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'audit_events is append-only: % refused', TG_OP
        USING ERRCODE = 'insufficient_privilege';
END
$$
"""


def upgrade():
    if op.get_bind().dialect.name != 'postgresql':
        return
    op.execute(CREATE_FUNCTION)
    op.execute(
        "CREATE TRIGGER audit_events_append_only BEFORE UPDATE OR DELETE ON audit_events "
        "FOR EACH ROW EXECUTE FUNCTION audit_events_append_only()"
    )
    op.execute(
        "CREATE TRIGGER audit_events_no_truncate BEFORE TRUNCATE ON audit_events "
        "FOR EACH STATEMENT EXECUTE FUNCTION audit_events_append_only()"
    )


def downgrade():
    if op.get_bind().dialect.name != 'postgresql':
        return
    op.execute("DROP TRIGGER IF EXISTS audit_events_no_truncate ON audit_events")
    op.execute("DROP TRIGGER IF EXISTS audit_events_append_only ON audit_events")
    op.execute("DROP FUNCTION IF EXISTS audit_events_append_only()")
