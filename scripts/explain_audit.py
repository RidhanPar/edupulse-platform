"""Measure the /audit page query against a real database. Read-only.

    python scripts/explain_audit.py <organisation-slug>

Run it from the Render shell, where production's environment variables are set. It runs
EXPLAIN (ANALYZE, BUFFERS) for the same query shapes the audit page sends, using that
organisation's real users and actions, and prints the plan, the execution time, and how
many pages came from cache ("hit") versus disk ("read").

Only quote a figure from a run whose "read" count is above zero: that run was cold, so
the time includes going to disk. A run with read = 0 was served from memory and will
look faster than a first visit to the page really is.
"""
from __future__ import annotations

import sys
from pathlib import Path
import uuid
from datetime import timedelta

from sqlalchemy import func, select, text

# Run as `python scripts/<name>.py` from the repository root: make the app importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _scenarios(organisation, AuditEvent, AuditFilters, db, utcnow):
    from db.tenancy import scoped_select

    base = scoped_select(AuditEvent, organisation.id).subquery()
    by_user = db.session.execute(
        select(base.c.user_id, func.count()).where(base.c.user_id.is_not(None))
        .group_by(base.c.user_id).order_by(func.count().desc())
    ).all()
    by_action = db.session.execute(
        select(base.c.action, func.count()).group_by(base.c.action).order_by(func.count())
    ).all()
    total = db.session.execute(select(func.count()).select_from(base)).scalar()
    middle = db.session.execute(
        select(base.c.created_at, base.c.id).order_by(base.c.created_at.desc(), base.c.id.desc())
        .offset(total // 2).limit(1)
    ).first()

    scenarios = {"first page, no filters": (AuditFilters(), None)}
    if middle:
        scenarios[f"page {total // 2:,} rows back (keyset cursor)"] = (AuditFilters(), tuple(middle))
    if by_user:
        scenarios[f"busiest user ({by_user[0][1]:,} events)"] = (AuditFilters(user_id=by_user[0][0]), None)
        scenarios[f"quietest user ({by_user[-1][1]:,} events)"] = (AuditFilters(user_id=by_user[-1][0]), None)
    if by_action:
        scenarios[f"rarest action, {by_action[0][0]} ({by_action[0][1]:,} events)"] = (
            AuditFilters(action=by_action[0][0]), None)
    week_ago = utcnow() - timedelta(days=7)
    scenarios["last 7 days"] = (AuditFilters(start=week_ago), None)
    return total, scenarios


def main(slug: str) -> int:
    from wsgi import app
    from db import db
    from db.audit import AUDIT_PAGE_SIZE, AuditFilters, audit_query
    from db.models import AuditEvent, Organisation, utcnow

    with app.app_context():
        organisation = db.session.scalars(select(Organisation).where(Organisation.slug == slug)).one_or_none()
        if organisation is None:
            print(f"No organisation with slug {slug!r}.")
            return 1
        total, scenarios = _scenarios(organisation, AuditEvent, AuditFilters, db, utcnow)
        print(f"{organisation.name}: {total:,} audit events. Page size {AUDIT_PAGE_SIZE}.\n")

        with db.engine.connect() as connection:
            for label, (filters, before) in scenarios.items():
                statement = audit_query(organisation.id, filters, before=before, limit=AUDIT_PAGE_SIZE + 1)
                compiled = statement.compile(dialect=connection.dialect)
                params = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in compiled.params.items()}
                plan = [row[0] for row in connection.exec_driver_sql(
                    "EXPLAIN (ANALYZE, BUFFERS) " + str(compiled), params)]
                timing = next(line for line in plan if line.startswith("Execution Time"))
                buffers = next((line.strip() for line in plan if "Buffers:" in line), "Buffers: none reported")
                print(f"=== {label}: {timing} | {buffers}")
                for line in plan:
                    if not line.startswith(("Planning", "Execution")):
                        print("    " + line)
                print()
        print("Quote only runs where Buffers shows read=... above zero (a cold read from disk).")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
