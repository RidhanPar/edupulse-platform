"""The owner-only /audit page: bounded keyset pages, filters, tenancy and escaping."""
import html
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from db import db
from db.audit import AUDIT_PAGE_SIZE, AuditFilters, audit_page, decode_cursor, encode_cursor
from db.models import AuditEvent, Role, User

ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
ROW = '<tr class="audit-row">'


def _add(organisation, user, count=1, *, action="prediction_run", start=T0, step=timedelta(minutes=1), **fields):
    events = [
        AuditEvent(
            organisation_id=organisation.id,
            user_id=user.id if user is not None else None,
            action=action,
            created_at=start + index * step,
            **fields,
        )
        for index in range(count)
    ]
    db.session.add_all(events)
    db.session.commit()
    return events


def _ids(events) -> list:
    return [event.id for event in events]


def _walk_older(organisation_id, filters=AuditFilters(), page_size=AUDIT_PAGE_SIZE):
    pages = [audit_page(organisation_id, filters, page_size=page_size)]
    while pages[-1].older_cursor is not None:
        pages.append(audit_page(organisation_id, filters, before=decode_cursor(pages[-1].older_cursor), page_size=page_size))
    return pages


def _older_link(page: str) -> str:
    return html.unescape(re.search(r'href="([^"]+)">Older', page).group(1))


def test_pages_are_bounded_newest_first_and_cover_every_event_once(make_org):
    organisation, user = make_org("alpha")
    events = _add(organisation, user, 120)

    pages = _walk_older(organisation.id)

    assert [len(page.events) for page in pages] == [50, 50, 20]
    assert [event.id for page in pages for event in page.events] == list(reversed(_ids(events)))
    assert pages[0].newer_cursor is None and pages[1].newer_cursor is not None


def test_events_sharing_a_timestamp_are_never_skipped_or_repeated(make_org):
    organisation, user = make_org("alpha")
    events = _add(organisation, user, 10, step=timedelta(0))

    seen = [event.id for page in _walk_older(organisation.id, page_size=3) for event in page.events]

    assert len(seen) == 10 and set(seen) == set(_ids(events))


def test_paging_back_towards_newer_events_returns_the_same_pages(make_org):
    organisation, user = make_org("alpha")
    _add(organisation, user, 120)
    pages = _walk_older(organisation.id)

    back = audit_page(organisation.id, AuditFilters(), after=decode_cursor(pages[2].newer_cursor))
    first = audit_page(organisation.id, AuditFilters(), after=decode_cursor(back.newer_cursor))

    assert _ids(back.events) == _ids(pages[1].events)
    assert _ids(first.events) == _ids(pages[0].events)
    assert first.newer_cursor is None


def test_page_size_is_capped_whatever_the_caller_asks_for(make_org):
    organisation, user = make_org("alpha")
    _add(organisation, user, 80)

    assert len(audit_page(organisation.id, AuditFilters(), page_size=10_000).events) == AUDIT_PAGE_SIZE


def test_filters_by_user_action_and_date_range(make_org):
    organisation, owner = make_org("alpha")
    staff = User(organisation=organisation, email="staff@alpha.test", role=Role.STAFF)
    db.session.add(staff)
    db.session.commit()
    day = timedelta(days=1)
    [first] = _add(organisation, owner, action="prediction_run", start=T0)
    [second] = _add(organisation, staff, action="login_success", start=T0 + day)
    [third] = _add(organisation, staff, action="prediction_run", start=T0 + 2 * day)

    def matching(**filters):
        return set(_ids(audit_page(organisation.id, AuditFilters(**filters)).events))

    assert matching(user_id=staff.id) == {second.id, third.id}
    assert matching(action="prediction_run") == {first.id, third.id}
    assert matching(start=T0 + day, end=T0 + 2 * day) == {second.id}
    assert matching(user_id=staff.id, action="prediction_run", start=T0 + day) == {third.id}


def test_the_audit_page_shows_one_bounded_page_with_user_emails(tenant):
    alpha = tenant("alpha")
    _add(alpha.organisation, alpha.user, 60)

    response = alpha.client.get("/audit")

    page = response.get_data(as_text=True)
    assert response.status_code == 200
    assert page.count(ROW) == AUDIT_PAGE_SIZE
    assert alpha.user.email in page
    assert alpha.client.get(_older_link(page)).get_data(as_text=True).count(ROW) == 10


def test_the_page_size_cannot_be_raised_from_the_query_string(tenant):
    alpha = tenant("alpha")
    _add(alpha.organisation, alpha.user, 60)

    for query in ("page_size=500", "per_page=500", "limit=500"):
        assert alpha.client.get(f"/audit?{query}").get_data(as_text=True).count(ROW) == AUDIT_PAGE_SIZE


def test_filters_come_from_the_query_string_and_survive_paging(tenant):
    alpha = tenant("alpha")
    day_two = T0 + timedelta(days=1)
    _add(alpha.organisation, alpha.user, 60, action="login_success", start=day_two)
    _add(alpha.organisation, alpha.user, 5, action="prediction_run", start=day_two)
    _add(alpha.organisation, alpha.user, 5, action="login_success", start=T0 + timedelta(days=3))

    page = alpha.client.get(
        f"/audit?user={alpha.user.id}&action=login_success&from=2026-09-02&to=2026-09-02"
    ).get_data(as_text=True)

    assert page.count(ROW) == AUDIT_PAGE_SIZE
    older = _older_link(page)
    assert "action=login_success" in older and "from=2026-09-02" in older and "to=2026-09-02" in older
    assert alpha.client.get(older).get_data(as_text=True).count(ROW) == 10


def test_invalid_filter_values_are_ignored_not_trusted(tenant):
    alpha = tenant("alpha")
    _add(alpha.organisation, alpha.user, 3)

    response = alpha.client.get(
        "/audit?user=not-a-uuid&action=drop_table&from=31/12/2026&to=yesterday&before=garbage&after=1.2.3"
    )

    page = response.get_data(as_text=True)
    assert response.status_code == 200
    assert page.count(ROW) == 3
    assert "Dates must be in YYYY-MM-DD format." in page


def test_another_organisations_events_never_appear(tenant, make_org):
    alpha = tenant("alpha")
    beta_organisation, beta_user = make_org("beta")
    _add(alpha.organisation, alpha.user, 2, user_agent="alpha-agent")
    beta_events = _add(beta_organisation, beta_user, 60, user_agent="beta-agent")

    for path in ("/audit", f"/audit?user={beta_user.id}", f"/audit?before={encode_cursor(beta_events[-1])}"):
        page = alpha.client.get(path).get_data(as_text=True)
        assert "beta-agent" not in page and beta_user.email not in page, path
        assert page.count(ROW) == 2, path


def test_failed_logins_with_no_organisation_are_not_shown(tenant):
    alpha = tenant("alpha")
    db.session.add(AuditEvent(organisation_id=None, action="login_failure", user_agent="unscoped-agent", created_at=T0))
    db.session.commit()

    page = alpha.client.get("/audit").get_data(as_text=True)

    assert "unscoped-agent" not in page and page.count(ROW) == 0


def test_attacker_controlled_values_are_escaped(tenant):
    alpha = tenant("alpha")
    _add(
        alpha.organisation,
        alpha.user,
        user_agent='<script>alert("ua")</script>',
        entity_type="dataset",
        entity_id='"><svg onload=alert(1)>',
        details={"filters": {"name": "<img src=x onerror=alert(1)>"}},
    )

    page = alpha.client.get("/audit").get_data(as_text=True)

    assert "<script" not in page
    assert "&lt;script&gt;alert(&#34;ua&#34;)&lt;/script&gt;" in page
    assert "<img src=x" not in page and "<svg onload" not in page


def test_autoescaping_is_on_and_never_switched_off(app):
    assert app.jinja_env.autoescape("audit.html") is True
    templates = sorted((ROOT / "templates").glob("*.html"))
    sources = [ROOT / "app.py", *sorted((ROOT / "auth").glob("*.py")), *sorted((ROOT / "db").glob("*.py"))]
    offenders = [
        path.name
        for path in templates + sources
        if re.search(r"\|\s*safe\b|autoescape\s+false|Markup\(", path.read_text(encoding="utf-8"))
    ]

    assert offenders == []
