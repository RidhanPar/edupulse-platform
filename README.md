# EduPulse Platform

Flask-based student early-warning prototype for training academic-risk classifiers, reviewing predictions, and comparing predictions with later outcomes.

Live demo: https://edupulse-platform.onrender.com/

## What It Implements

- CSV validation and upload flows for training, prediction, and actual-result datasets
- Column mapping for an institution's own export format, saved for later uploads, plus CSV templates and a data dictionary
- Reproducible comparison of Logistic Regression, Decision Tree, Random Forest, and SVM classifiers, trained on whichever indicators a file supplies
- Persisted best-model artifact and evaluation metrics
- Stored prediction runs: student-level risk bands, deterministic support recommendations, and which students' risk rose since the last run
- A page per student showing their indicators and risk over time
- Prediction-versus-actual comparison
- Organisations, invite-only accounts, roles, user management and an audit log
- Render deployment blueprint, tests (including PostgreSQL), and GitHub Actions CI

## Evidence Boundary

The included data is synthetic demonstration data. Model metrics show the behavior of this controlled prototype and should not be interpreted as evidence of effectiveness in a real educational setting.

Model metrics shown in the application are still produced from synthetic data.

Recommendations are deterministic rules based on input indicators and model risk. The application does not autonomously contact students, change academic records, or replace qualified human review.

## Architecture

```mermaid
flowchart LR
    A[Training CSV] --> M[Column mapping]
    E[Prediction CSV] --> M
    M --> B[Validate and preprocess]
    B --> C[Compare classifiers]
    C --> D[Signed model and metrics]
    D --> F[Prediction run stored per student]
    M --> F
    F --> R[Results, changes since last run, student pages]
    G[Actual outcomes CSV] --> H[Prediction comparison]
    F --> H
```

## Security Controls

- Flask secret is supplied through `FLASK_SECRET_KEY` in deployment
- Uploads are restricted to CSV filenames and a 5 MB request limit
- The repository excludes local virtual environments and secrets
- Every page requires login except the login page, invite links and `/healthz`; there is no public sign-up
- Passwords are hashed with Argon2id; sessions are stored server-side and the session id is rotated at login
- Roles: viewers read results, explanations and comparisons; staff also upload datasets and export results; owners also train models
- CSRF protection on every form; session cookies are Secure, HttpOnly and SameSite=Lax
- Datasets and models are stored per organisation, and every query and storage read is scoped to the requesting organisation
- Stored models are signed with `MODEL_SIGNING_KEY` and verified before loading; an unsigned or altered model is refused
- On PostgreSQL a trigger refuses any change to audit rows except the 90 day removal of IP address and browser details

EduPulse now requires authentication and isolates data per organisation. It is not yet approved for production student data: encryption at rest, data retention policy, breach procedure and a signed Data Processing Agreement are not implemented. Use synthetic or fully anonymised data only.

## Security Behaviour

These are deliberate, not bugs:

- Changing your password logs you out of every other session.
- A session ends after 8 hours without activity, and 12 hours after login even if you stay active.
- Every failed login shows the same message, whatever the reason.
- After 5 failed logins in a minute (or 30 in an hour) from one address, that address must wait before trying again. Successful logins do not count, so a school sharing one network address is not locked out by normal use.
- Uploads that cannot be read as CSV, or that exceed 5 MB, are rejected and nothing is stored.
- Invite links work once and expire after 7 days.
- Owners can filter and page through the audit log, but cannot export or edit it.

## Run Locally

```bash
python -m venv .venv
python -m pip install -r requirements.txt
export FLASK_ENV=development
flask --app wsgi db upgrade
flask --app wsgi create-org "Demo College" owner@example.com
python app.py
```

`create-org` prints a one-time invite link. Open it to set the owner's password, then log in. Add more users with `flask --app wsgi create-user demo-college tutor@example.com --role staff`.

`FLASK_ENV=development` uses a local SQLite database, a generated secret key and file storage under `var/storage/`. Any other value, including leaving it unset, is treated as production: the app refuses to start unless `FLASK_SECRET_KEY`, `MODEL_SIGNING_KEY`, `DATABASE_URL`, `STORAGE_BACKEND=s3`, `STORAGE_ENDPOINT`, `STORAGE_BUCKET`, `STORAGE_ACCESS_KEY` and `STORAGE_SECRET_KEY` are set.

## Deploy

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): Render (web service and PostgreSQL) and S3, all in Frankfurt, with the steps that need an account or a decision marked.

The app trusts one proxy's `X-Forwarded-For` header, as on Render. When running with nothing in front of it, set `PROXY_HOPS=0` so clients cannot spoof their address.

## Documents

- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md): production runbook
- [docs/PRODUCT_STRATEGY.md](docs/PRODUCT_STRATEGY.md): who it is for, what blocks the first paying customer, and the roadmap
- [docs/ISSUES_AND_FIXES.md](docs/ISSUES_AND_FIXES.md): problems hit while building it, how they were diagnosed and fixed

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
# PostgreSQL-only tests (migrations, enum types, the audit trigger) need a disposable database:
TEST_POSTGRES_URL=postgresql://user:pass@localhost:5432/edupulse_test python -m pytest -q tests/test_postgres.py
python -m compileall app.py config.py wsgi.py auth db migrations scripts utils tests
```
