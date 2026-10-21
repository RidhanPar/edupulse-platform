# Issues faced and how they were solved

A record of the real problems hit while turning EduPulse from a single user prototype into a
multi tenant application: what was seen, how it was diagnosed, the root cause, the fix, and the
evidence. Wrong first guesses are kept in, because how they were caught is often the most useful
part of the story.

## The five best interview stories

| # | Story | Why it lands |
|---|---|---|
| 1 | [PostgreSQL caught a migration bug that 352 SQLite tests missed](#postgresql-caught-a-migration-bug-that-352-sqlite-tests-missed) | Green test suite, broken deploy. Led to PostgreSQL tests in CI |
| 2 | [The audit page did a full table scan for rare filters](#the-audit-page-did-a-full-table-scan-for-rare-filters) | Measured on 1,000,020 rows: 46.8 ms to 0.061 ms, and why that number is not yet quotable |
| 3 | [One organisation's tests saw another organisation's data](#one-organisations-tests-saw-another-organisations-data) | A tenancy leak that existed only in tests, found and fixed in the app anyway |
| 4 | [A login rate limit that would have locked everyone out](#a-login-rate-limit-that-would-have-locked-everyone-out) | My first framing was too weak. Behind a proxy, every user shares one address |
| 5 | [The session library committed my database transactions](#the-session-library-committed-my-database-transactions) | Read the library source before trusting it, and replaced one backend |

---

## Getting started

### I started work on the wrong copy of the repository
- **Seen:** the clone I first read had the whole virtual environment committed (14,082 tracked files, 37 of them project files), an empty `.gitignore` and no tests, while the brief referred to `tests/test_core.py`.
- **Diagnosed:** the owner pointed me at the authoritative repository. I cloned it fresh and checked that the old copy held nothing unique: its HEAD was an ancestor of the authoritative `main`, it had no other branches or stashes, and the one dangling git tree it contained had all 26 files present in the authoritative history's first commit.
- **Fix:** worked only in the fresh clone. The old copy was abandoned with nothing lost.
- **Say:** "Before deleting anything I proved, with git ancestry and object lookups, that it held nothing we would miss."

## Database layer

### SQLite migrations failed on a populated database
- **Seen:** the second migration passed on an empty database but failed on one with rows: `FOREIGN KEY constraint failed`.
- **Root cause:** SQLite cannot alter a table in place, so Alembic's batch mode copies, drops and renames it. With foreign keys enforced, dropping a table other rows point at fails.
- **Fix:** `migrations/env.py` pauses foreign key enforcement during migrations on SQLite, then runs `PRAGMA foreign_key_check` and fails on any violation. PostgreSQL is unaffected.
- **Evidence:** with the pause disabled, the populated database test fails with the same error. With it, the test passes.

### Append only enforcement missed whole classes of statement
- **Seen:** the first version refused updates and deletes of audit rows through ORM listeners. A bulk `update(AuditEvent)` and raw `text()` SQL went straight past them.
- **Fix:** one engine level check on every statement SQLAlchemy sends, plus a single named function that is allowed to null the two personal fields after 90 days. Tests try 15 ways to change an audit row. Later a PostgreSQL trigger applies the same rule to connections that do not go through the app, tested on a raw driver connection that bypasses the app's own check.
- **Say:** "An ORM hook protects the ORM. An audit log has to be protected where the SQL is executed."

### PostgreSQL caught a migration bug that 352 SQLite tests missed
- **Seen:** all 352 tests passed. Running the same migrations on PostgreSQL 16 failed: `DuplicateObject: type "dataset_kind" already exists`.
- **Root cause:** the new `column_mappings` table reused the `dataset_kind` enum. On SQLite an enum is just a check constraint on each table, so nothing clashed. On PostgreSQL an enum is a database wide type, and creating the table tried to create the type a second time.
- **Fix:** the migration uses `postgresql.ENUM(..., create_type=False)` on PostgreSQL. More importantly, the PostgreSQL checks moved from a throwaway script into `tests/test_postgres.py`, and CI runs them against a `postgres:16` service on every push.
- **Evidence:** the first deploy to a real database would have failed. The original script had also been lost once, when a temporary folder was cleared.
- **Say:** "SQLite in tests is fast but it is a different database. The bug I care about most was invisible to it."

## Storage and tenancy

### Tenancy tests could have passed by accident
- **Seen:** the route by route isolation tests all passed first time, which made me suspicious.
- **Diagnosed:** a page that silently redirected to the login screen would also return 200 and show no other organisation's data, so the checks could pass without testing anything.
- **Fix:** every page check also asserts the page was served to the logged in user. To prove the assertion bites, I logged the user out inside the test: 12 of 13 page checks then failed, and the 13th, the CSV export, has its own check that a real CSV came back.

## Authentication

### One organisation's tests saw another organisation's data
- **Seen:** after adding Flask Login, 10 tests failed. In each, a second organisation's client was served the first organisation's data.
- **Root cause:** Flask Login caches the loaded user on `g`, which lives on the application context. The test suite keeps one application context open across many requests, so the first user stayed cached. Production creates a fresh context per request, so it was not affected there.
- **Fix:** made in the app, not the test fixture. The login check clears the cached user at the start of every request, so identity always comes from that request's session.
- **Say:** "It only happened in tests, but the fix belonged in the app, so nothing depends on how the server happens to create contexts."

### The session library committed my database transactions
- **Seen:** reading Flask Session 0.8.0's SQLAlchemy backend before using it.
- **Root cause:** it calls `commit()` on the application's own database session every time it saves a session, so a view's half finished changes would be committed as a side effect. It also creates its table at start up outside Alembic, and its `regenerate()` skips empty sessions, which leaves a session fixation gap.
- **Fix:** kept Flask Session's interface and wrote a small backend on a migration managed `user_sessions` table. Every session read and write uses its own short connection. Session ids are rotated at login even when the session is empty.

### Moving off passlib without stranding existing passwords
- **Seen:** passlib is unmaintained and warned under argon2-cffi 23.
- **Fix:** before uninstalling it, generated real passlib hashes with both the production and test parameters and saved them as a test fixture. After switching to argon2-cffi directly, a test proves those hashes still verify and are upgraded to the current parameters on the next login.

## Hardening

### A login rate limit that would have locked everyone out
- **Seen:** I first described the missing proxy handling as "the audit log will record the load balancer's address". The owner pointed out the real consequence.
- **Root cause:** behind Render's proxy every request arrives from the same address. A limit of 5 failed logins per address would have locked every user on the platform out after a few failures anywhere.
- **Fix:** ProxyFix trusting exactly one hop, applied before anything else in `create_app`. Tests show one client being blocked while a colleague behind the same proxy is not, and that rotating a fake `X-Forwarded-For` value does not get around the limit. Only failed attempts count, so a school sharing one address is not locked out by normal use.
- **Not guessed:** the hop count is logged at start up, and the first real request logs its raw `X-Forwarded-For`, so the production value is set from evidence.
- **Say:** "My first description was technically true and badly underestimated the impact."

### An oversized upload was reported as a generic failure
- **Seen:** a test sending a file over 5 MB expected a 413 and got a 302 with "Upload failed".
- **Root cause:** the upload routes wrapped everything in `except Exception`, which also caught Flask's `RequestEntityTooLarge`.
- **Fix:** that exception is re raised to a dedicated handler with a clear message.

## Performance

### The audit page did a full table scan for rare filters
- **Seen:** with 1,000,020 seeded audit rows on PostgreSQL 16, filtering by a user with only 20 events took 32.4 ms and filtering by an action with no matches took 46.8 ms. Both plans were parallel scans of the whole table.
- **Wrong first assumption:** that one `(organisation_id, created_at)` index would serve every filter. It does for the date range, but a filter on a rare user or action has to walk the index until it finds enough matching rows, and the planner gave up and scanned the table.
- **Fix:** three indexes, one per filter, each ending in `(created_at, id)`, the keyset order the page uses. Every scenario became a single index scan that stops after one page: 0.066 ms for the rare user and 0.061 ms for the action with no matches.
- **Caveat:** those times are from a laptop with a warm cache and must not be quoted as production figures. `scripts/explain_audit.py` measures the real database and reports whether each run read from disk.
- **Say:** "I can show you the plan before and after, and tell you exactly why the after number is not yet a claim."
