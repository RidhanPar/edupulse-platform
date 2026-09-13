# EduPulse Platform

Flask-based student early-warning prototype for training academic-risk classifiers, reviewing predictions, and comparing predictions with later outcomes.

Live demo: https://edupulse-platform.onrender.com/

## What It Implements

- CSV validation and upload flows for training, prediction, and actual-result datasets
- Reproducible comparison of Logistic Regression, Decision Tree, Random Forest, and SVM classifiers
- Persisted best-model artifact and evaluation metrics
- Student-level risk bands and deterministic support recommendations
- Prediction-versus-actual comparison
- Render deployment blueprint, tests, and GitHub Actions CI

## Evidence Boundary

The included data is synthetic demonstration data. Model metrics show the behavior of this controlled prototype and should not be interpreted as evidence of effectiveness in a real educational setting.

Recommendations are deterministic rules based on input indicators and model risk. The application does not autonomously contact students, change academic records, or replace qualified human review.

## Architecture

```mermaid
flowchart LR
    A[Synthetic training CSV] --> B[Validate and preprocess]
    B --> C[Compare classifiers]
    C --> D[Persist selected model and metrics]
    E[Prediction CSV] --> F[Risk scores and support recommendations]
    D --> F
    G[Actual outcomes CSV] --> H[Prediction comparison]
    F --> H
```

## Security Controls

- Flask secret is supplied through `FLASK_SECRET_KEY` in deployment
- Uploads are restricted to CSV filenames and a 5 MB request limit
- The repository excludes local virtual environments and secrets

This is a public demonstration application without authentication. Do not upload confidential or personal student data.

## Run Locally

```bash
python -m venv .venv
python -m pip install -r requirements.txt
export FLASK_ENV=development
flask --app wsgi db upgrade
python app.py
```

`FLASK_ENV=development` uses a local SQLite database, a generated secret key and file storage under `var/storage/`. Any other value, including leaving it unset, is treated as production: the app refuses to start unless `FLASK_SECRET_KEY`, `DATABASE_URL`, `STORAGE_BACKEND=s3`, `STORAGE_ENDPOINT`, `STORAGE_BUCKET`, `STORAGE_ACCESS_KEY` and `STORAGE_SECRET_KEY` are set.

## Verify

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall app.py config.py wsgi.py db migrations utils tests
```
