from __future__ import annotations

import io
import json
import os
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, Response, current_app
from flask_login import current_user
from sqlalchemy import text
from werkzeug.exceptions import RequestEntityTooLarge

from auth import check_route_roles, init_auth
from auth.ratelimit import trust_proxy
from auth.roles import require_role
from config import build_config, resolve_env
from db import db, migrate
from db.audit import AUDIT_ACTIONS, AUDIT_PAGE_SIZE, AuditFilters, audit_page, decode_cursor, record_audit_event
from db.models import Dataset, DatasetKind, ModelArtifact, Role, User
from db.tenancy import scoped_get, scoped_select
from utils.evaluation import load_importances, load_metrics, load_model_comparison
from utils.model_cache import ModelCache
from utils.storage import build_backend, dataset_key, model_key, storage_for

# utils.preprocessing, utils.train_model, utils.predict and utils.compare_results
# all import scikit-learn (and joblib), which adds seconds to every cold start and
# test run. They are imported inside the handlers that use them, so create_app()
# never loads them. tests/test_app_factory.py enforces this.


def read_csv_flexible(data: bytes) -> pd.DataFrame:
    """
    Reads both comma-separated and semicolon-separated CSV files.
    """
    try:
        df = pd.read_csv(io.BytesIO(data))
        if len(df.columns) == 1:
            df = pd.read_csv(io.BytesIO(data), sep=";")
        return df
    except (pd.errors.ParserError, UnicodeDecodeError):
        df = pd.read_csv(io.BytesIO(data), sep=";")
        return df


def current_organisation_id():
    return current_user.organisation_id


def _latest_dataset(kind: DatasetKind) -> Dataset | None:
    return db.session.scalars(
        scoped_select(Dataset, current_organisation_id())
        .where(Dataset.kind == kind)
        .order_by(Dataset.uploaded_at.desc())
        .limit(1)
    ).first()


def _read_dataset(dataset: Dataset) -> pd.DataFrame:
    # Authorised against the requesting organisation, not the dataset's own.
    return read_csv_flexible(storage_for(current_organisation_id()).get(dataset.storage_key))


class UploadRejected(ValueError):
    """An upload that cannot be read as CSV. Nothing has been stored when this is raised."""


UNREADABLE_CSV = (
    "the file could not be read as CSV. Save it as UTF-8 CSV (comma or semicolon separated) and try again."
)


def _parse_upload(data: bytes) -> pd.DataFrame:
    try:
        df = read_csv_flexible(data)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError):
        raise UploadRejected(UNREADABLE_CSV) from None
    if df.empty or len(df.columns) == 0:
        raise UploadRejected("the file has no data rows.")
    return df


def _store_upload(file, kind: DatasetKind) -> Dataset:
    data = file.read()
    # Parsed in full before anything is written: a file that fails here leaves no
    # object in storage, no Dataset row and no audit event.
    df = _parse_upload(data)
    organisation_id = current_organisation_id()
    dataset_id = uuid.uuid4()
    dataset = Dataset(
        id=dataset_id,
        organisation_id=organisation_id,
        kind=kind,
        # Kept for display only. The storage key is built from ids, never from the filename.
        original_filename=file.filename[:255],
        storage_key=dataset_key(organisation_id, dataset_id),
        row_count=int(len(df)),
        column_names=[str(column) for column in df.columns],
        uploaded_by=current_user.id,
    )
    storage_for(organisation_id).put(dataset.storage_key, data)
    db.session.add(dataset)
    record_audit_event(
        "dataset_uploaded",
        user=current_user,
        entity_type="dataset",
        entity_id=dataset.id,
        details={"kind": kind.value, "row_count": dataset.row_count},
    )
    db.session.commit()
    return dataset


def _active_model() -> ModelArtifact | None:
    return db.session.scalars(
        scoped_select(ModelArtifact, current_organisation_id()).where(ModelArtifact.is_active.is_(True))
    ).one_or_none()


def _load_model(artifact: ModelArtifact):
    from utils.predict import load_model

    # Deserialised once per worker and reused. The artifact came from a tenant-scoped
    # query, and the key includes its organisation, so a hit is never another tenant's model.
    organisation_id = current_organisation_id()
    return current_app.extensions["model_cache"].get_or_load(
        (organisation_id, artifact.id),
        lambda: load_model(storage_for(organisation_id).get(artifact.storage_key)),
    )


def _save_model(artifacts, training_dataset: Dataset) -> ModelArtifact:
    from utils.train_model import serialize_model

    organisation_id = current_organisation_id()
    artifact_id = uuid.uuid4()
    artifact = ModelArtifact(
        id=artifact_id,
        organisation_id=organisation_id,
        algorithm_name=artifacts.model_name,
        metrics=artifacts.metrics,
        feature_importances=artifacts.feature_importances,
        model_comparison=artifacts.model_comparison,
        storage_key=model_key(organisation_id, artifact_id),
        trained_by=current_user.id,
        is_active=True,
    )
    storage_for(organisation_id).put(artifact.storage_key, serialize_model(artifacts.model))
    previous = _active_model()
    if previous is not None:
        # Deactivate first: the database allows one active model per organisation.
        previous.is_active = False
        db.session.flush()
    db.session.add(artifact)
    record_audit_event(
        "model_trained",
        user=current_user,
        entity_type="model_artifact",
        entity_id=artifact.id,
        details={"algorithm": artifact.algorithm_name, "training_dataset_id": str(training_dataset.id)},
    )
    db.session.commit()
    return artifact


def _active_filters(name_query: str, risk_filter: str, prediction_filter: str) -> dict:
    filters = {"name": name_query, "risk": risk_filter, "prediction": prediction_filter}
    return {key: value for key, value in filters.items() if value}


def _record_prediction_use(action: str, artifact: ModelArtifact, *, row_count: int, **details) -> None:
    """Audit a use of the model's predictions over the organisation's student records.

    Viewing /results or /compare and exporting a CSV all process identifiable student
    data, so each one is recorded with who, which model, which dataset and how many rows.
    """
    prediction_dataset = _latest_dataset(DatasetKind.PREDICTION)
    record_audit_event(
        action,
        user=current_user,
        entity_type="model_artifact",
        entity_id=artifact.id,
        details={
            **details,
            "prediction_dataset_id": str(prediction_dataset.id) if prediction_dataset else None,
            "row_count": int(row_count),
        },
    )
    db.session.commit()


def get_training_df(require_target: bool = True):
    from utils.preprocessing import validate_columns

    dataset = _latest_dataset(DatasetKind.TRAINING)
    if dataset is None:
        return None, "No training dataset uploaded yet."
    try:
        df = _read_dataset(dataset)
        ok, missing = validate_columns(df, require_target=require_target)
        if not ok:
            return None, f"Missing required columns in training dataset: {', '.join(missing)}"
        return df, None
    except Exception as e:
        return None, f"Error reading training dataset: {str(e)}"


def get_prediction_df():
    from utils.preprocessing import validate_columns

    dataset = _latest_dataset(DatasetKind.PREDICTION)
    if dataset is None:
        return None, "No prediction dataset uploaded yet."
    try:
        df = _read_dataset(dataset)
        ok, missing = validate_columns(df, require_target=False)
        if not ok:
            return None, f"Missing required columns in prediction dataset: {', '.join(missing)}"
        return df, None
    except Exception as e:
        return None, f"Error reading prediction dataset: {str(e)}"

def get_actual_results_df():
    dataset = _latest_dataset(DatasetKind.ACTUAL)
    if dataset is None:
        return None, "No actual results dataset uploaded yet."
    try:
        df = _read_dataset(dataset)
        required_cols = ["student_id", "target"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"Missing required columns in actual results dataset: {', '.join(missing)}"
        return df, None
    except Exception as e:
        return None, f"Error reading actual results dataset: {str(e)}"


def _preview(kind: DatasetKind, label: str):
    dataset = _latest_dataset(kind)
    if dataset is None:
        return None, None
    try:
        df = _read_dataset(dataset)
        return df.head(10).to_dict(orient="records"), list(df.columns)
    except Exception as e:
        flash(f"Could not preview {label} dataset: {str(e)}", "danger")
        return None, None


def home():
    artifact = _active_model()
    metrics = load_metrics(artifact)
    comparison = load_model_comparison(artifact)
    return render_template("home.html", metrics=metrics, comparison=comparison)


def upload_train():
    from utils.preprocessing import DISPLAY_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN

    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or not file.filename:
                flash("Please select a training CSV file.", "danger")
                return redirect(url_for("upload_train"))

            if not file.filename.lower().endswith(".csv"):
                flash("Only CSV files are allowed for training upload.", "danger")
                return redirect(url_for("upload_train"))

            _store_upload(file, DatasetKind.TRAINING)
            flash("Training dataset uploaded successfully.", "success")
            return redirect(url_for("upload_train"))

        except RequestEntityTooLarge:
            raise  # handled by upload_too_large
        except Exception as e:
            db.session.rollback()
            flash(f"Upload failed: {str(e)}", "danger")
            return redirect(url_for("upload_train"))

    preview, columns = _preview(DatasetKind.TRAINING, "training")

    required = DISPLAY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    return render_template(
        "upload_train.html",
        preview=preview,
        columns=columns,
        required=required
    )


def upload_predict():
    from utils.preprocessing import DISPLAY_COLUMNS, FEATURE_COLUMNS

    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or not file.filename:
                flash("Please select a prediction CSV file.", "danger")
                return redirect(url_for("upload_predict"))

            if not file.filename.lower().endswith(".csv"):
                flash("Only CSV files are allowed for prediction upload.", "danger")
                return redirect(url_for("upload_predict"))

            _store_upload(file, DatasetKind.PREDICTION)
            flash("Prediction dataset uploaded successfully.", "success")
            return redirect(url_for("upload_predict"))

        except RequestEntityTooLarge:
            raise  # handled by upload_too_large
        except Exception as e:
            db.session.rollback()
            flash(f"Upload failed: {str(e)}", "danger")
            return redirect(url_for("upload_predict"))

    preview, columns = _preview(DatasetKind.PREDICTION, "prediction")

    required = DISPLAY_COLUMNS + FEATURE_COLUMNS
    return render_template(
        "upload_predict.html",
        preview=preview,
        columns=columns,
        required=required
    )


def train():
    artifact = _active_model()
    metrics = load_metrics(artifact)
    importances = load_importances(artifact)
    comparison = load_model_comparison(artifact)

    if request.method == "POST":
        from utils.train_model import train_and_select_best

        df, err = get_training_df(require_target=True)
        if err:
            flash(err, "danger")
            return redirect(url_for("upload_train"))

        try:
            artifacts = train_and_select_best(df)
            _save_model(artifacts, _latest_dataset(DatasetKind.TRAINING))
            metrics = {"best_model": artifacts.model_name, **artifacts.metrics}
            importances = artifacts.feature_importances
            comparison = artifacts.model_comparison
            flash(f"Training complete. Best model: {artifacts.model_name}", "success")
        except Exception as e:
            db.session.rollback()
            flash(f"Training failed: {str(e)}", "danger")
            return redirect(url_for("train"))

    return render_template(
        "train.html",
        metrics=metrics,
        importances=importances,
        comparison=comparison
    )


def results():
    from utils.predict import predict_dataframe

    # TODO: predictions are recomputed on every request to /results, /download-results
    # and /compare. The model itself is now cached per worker (see _load_model), but the
    # predictions should be materialised into a table when the prediction dataset or
    # active model changes; recomputing per request becomes a real cost once LLM
    # reasoning is added.
    artifact = _active_model()
    if artifact is None:
        flash("Train the model first.", "warning")
        return redirect(url_for("train"))

    df, err = get_prediction_df()
    if err:
        flash(err, "danger")
        return redirect(url_for("upload_predict"))

    try:
        res = predict_dataframe(df, _load_model(artifact))
    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "danger")
        return redirect(url_for("upload_predict"))

    name_query = request.args.get("name", "").strip().lower()
    risk_filter = request.args.get("risk", "").strip()
    prediction_filter = request.args.get("prediction", "").strip()

    filtered = res.copy()

    if name_query:
        filtered = filtered[
            filtered["student_name"].astype(str).str.lower().str.contains(name_query)
            | filtered["student_id"].astype(str).str.lower().str.contains(name_query)
        ]

    if risk_filter:
        filtered = filtered[filtered["risk_level"] == risk_filter]

    if prediction_filter:
        filtered = filtered[filtered["prediction"] == prediction_filter]

    summary = {
        "total": int(len(filtered)),
        "pass_count": int((filtered["prediction"] == "Pass").sum()),
        "fail_count": int((filtered["prediction"] == "Fail").sum()),
        "high_risk": int((filtered["risk_level"] == "High").sum()),
        "medium_risk": int((filtered["risk_level"] == "Medium").sum()),
        "low_risk": int((filtered["risk_level"] == "Low").sum()),
    }

    records = filtered.head(200).to_dict(orient="records")
    _record_prediction_use(
        "prediction_run",
        artifact,
        row_count=summary["total"],
        page="results",
        filters=_active_filters(name_query, risk_filter, prediction_filter),
    )

    return render_template(
        "results.html",
        records=records,
        summary=summary,
        name_query=name_query,
        risk_filter=risk_filter,
        prediction_filter=prediction_filter
    )

def download_results():
    from utils.predict import predict_dataframe

    artifact = _active_model()
    if artifact is None:
        flash("Train the model first.", "warning")
        return redirect(url_for("train"))

    df, err = get_prediction_df()
    if err:
        flash(err, "danger")
        return redirect(url_for("upload_predict"))

    try:
        res = predict_dataframe(df, _load_model(artifact))
    except Exception as e:
        flash(f"Prediction failed: {str(e)}", "danger")
        return redirect(url_for("upload_predict"))

    name_query = request.args.get("name", "").strip().lower()
    risk_filter = request.args.get("risk", "").strip()
    prediction_filter = request.args.get("prediction", "").strip()

    filtered = res.copy()

    if name_query:
        filtered = filtered[
            filtered["student_name"].astype(str).str.lower().str.contains(name_query)
            | filtered["student_id"].astype(str).str.lower().str.contains(name_query)
        ]

    if risk_filter:
        filtered = filtered[filtered["risk_level"] == risk_filter]

    if prediction_filter:
        filtered = filtered[filtered["prediction"] == prediction_filter]

    export_cols = [
        "student_id",
        "student_name",
        "prediction",
        "fail_probability",
        "confidence",
        "risk_level",
        "recommendation"
    ]

    available_cols = [col for col in export_cols if col in filtered.columns]
    csv_data = filtered[available_cols].to_csv(index=False)
    # An export is an uncontrolled copy of identifiable student data: record who took it,
    # how many rows and which filters produced it.
    _record_prediction_use(
        "results_exported",
        artifact,
        row_count=len(filtered),
        filters=_active_filters(name_query, risk_filter, prediction_filter),
    )

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=prediction_results.csv"}
    )

def explain():
    artifact = _active_model()
    importances = load_importances(artifact)
    metrics = load_metrics(artifact)
    if not importances:
        flash("Train the model first to generate explanations.", "warning")
        return redirect(url_for("train"))
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return render_template("explain.html", ranked=ranked, metrics=metrics)


def about():
    return render_template("about.html")

def upload_actual():
    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or not file.filename:
                flash("Please select an actual results CSV file.", "danger")
                return redirect(url_for("upload_actual"))

            if not file.filename.lower().endswith(".csv"):
                flash("Only CSV files are allowed for actual results upload.", "danger")
                return redirect(url_for("upload_actual"))

            _store_upload(file, DatasetKind.ACTUAL)
            flash("Actual results dataset uploaded successfully.", "success")
            return redirect(url_for("upload_actual"))

        except RequestEntityTooLarge:
            raise  # handled by upload_too_large
        except Exception as e:
            db.session.rollback()
            flash(f"Upload failed: {str(e)}", "danger")
            return redirect(url_for("upload_actual"))

    preview, columns = _preview(DatasetKind.ACTUAL, "actual results")

    required = ["student_id", "target"]
    return render_template(
        "upload_actual.html",
        preview=preview,
        columns=columns,
        required=required
    )

def compare():
    from utils.compare_results import compare_predictions_with_actual
    from utils.predict import predict_dataframe

    artifact = _active_model()
    if artifact is None:
        flash("Train the model first.", "warning")
        return render_template("compare.html", records=None, metrics=None)

    pred_df, pred_err = get_prediction_df()
    if pred_err:
        flash(pred_err, "danger")
        return render_template("compare.html", records=None, metrics=None)

    actual_df, actual_err = get_actual_results_df()
    if actual_err:
        flash(actual_err, "danger")
        return render_template("compare.html", records=None, metrics=None)

    try:
        predicted_results = predict_dataframe(pred_df, _load_model(artifact))
        comparison_df, comparison_metrics = compare_predictions_with_actual(predicted_results, actual_df)
        records = comparison_df.head(200).to_dict(orient="records")
        _record_prediction_use(
            "prediction_run", artifact, row_count=comparison_metrics["total_compared"], page="compare"
        )
        return render_template(
            "compare.html",
            records=records,
            metrics=comparison_metrics
        )
    except Exception as e:
        flash(f"Comparison failed: {str(e)}", "danger")
        return render_template("compare.html", records=None, metrics=None)

def recheck_comparison():
    flash("Comparison metrics refreshed using the current prediction and actual results files.", "success")
    return redirect(url_for("compare"))

def _parse_day(value: str) -> datetime | None:
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _audit_row(event, emails: dict) -> dict:
    details = [
        (key, json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
        for key, value in sorted((event.details or {}).items())
    ]
    entity = " ".join(part for part in (event.entity_type, event.entity_id) if part)
    return {
        "time": _as_utc(event.created_at).strftime("%Y-%m-%d %H:%M:%S"),
        "user": emails.get(event.user_id, "—") if event.user_id else "—",
        "action": event.action,
        "entity": entity,
        "details": details,
        "ip_address": event.ip_address or "",
        "user_agent": event.user_agent or "",
    }


def audit_log():
    """Owner-only, read-only view of the organisation's audit log, one bounded page at a time.

    Filter values that do not parse, or that name another organisation's user, are ignored
    rather than trusted. There is deliberately no export: that would be a bulk extraction
    of personal data in its own right.
    """
    organisation_id = current_organisation_id()
    users = db.session.scalars(scoped_select(User, organisation_id).order_by(User.email)).all()
    emails = {user.id: user.email for user in users}

    form = {key: request.args.get(key, "").strip() for key in ("user", "action", "from", "to")}
    filter_user = scoped_get(User, organisation_id, form["user"]) if form["user"] else None
    action = form["action"] if form["action"] in AUDIT_ACTIONS else None
    start = end = None
    try:
        start = _parse_day(form["from"]) if form["from"] else None
        end = _parse_day(form["to"]) + timedelta(days=1) if form["to"] else None  # "to" is inclusive
    except ValueError:
        start = end = None
        form["from"] = form["to"] = ""
        flash("Dates must be in YYYY-MM-DD format.", "warning")
    if filter_user is None:
        form["user"] = ""
    if action is None:
        form["action"] = ""

    filters = AuditFilters(user_id=filter_user.id if filter_user else None, action=action, start=start, end=end)
    after = decode_cursor(request.args.get("after"))
    before = None if after else decode_cursor(request.args.get("before"))
    page = audit_page(organisation_id, filters, before=before, after=after)

    return render_template(
        "audit.html",
        rows=[_audit_row(event, emails) for event in page.events],
        users=users,
        actions=AUDIT_ACTIONS,
        form=form,
        query={key: value for key, value in form.items() if value},
        newer_cursor=page.newer_cursor,
        older_cursor=page.older_cursor,
        page_size=AUDIT_PAGE_SIZE,
    )

def healthz():
    try:
        db.session.execute(text("SELECT 1"))
    except Exception:
        return {"status": "unavailable"}, 503
    return {"status": "ok"}

def upload_too_large(error):
    return render_template("error.html", message="That file is larger than the 5 MB upload limit."), 413

def forbidden(error):
    return render_template("error.html", message="You do not have permission to do that."), 403

def internal_error(error):
    return render_template("error.html", message="An internal server error occurred. Please check your uploaded dataset and try again."), 500


def _register_routes(app: Flask) -> None:
    # Endpoint names default to the view function names (require_role preserves them),
    # so every url_for() in the templates resolves exactly as it did with @app.route.
    viewer, staff, owner = Role.VIEWER, Role.STAFF, Role.OWNER
    app.add_url_rule("/", view_func=require_role(viewer)(home))
    app.add_url_rule("/upload-train", view_func=require_role(staff)(upload_train), methods=["GET", "POST"])
    app.add_url_rule("/upload-predict", view_func=require_role(staff)(upload_predict), methods=["GET", "POST"])
    app.add_url_rule("/train", view_func=require_role(viewer, write=owner)(train), methods=["GET", "POST"])
    app.add_url_rule("/results", view_func=require_role(viewer)(results))
    app.add_url_rule("/download-results", view_func=require_role(staff)(download_results))
    app.add_url_rule("/explain", view_func=require_role(viewer)(explain))
    app.add_url_rule("/about", view_func=require_role(viewer)(about))
    app.add_url_rule("/audit", view_func=require_role(owner)(audit_log))
    app.add_url_rule("/upload-actual", view_func=require_role(staff)(upload_actual), methods=["GET", "POST"])
    app.add_url_rule("/compare", view_func=require_role(viewer)(compare))
    app.add_url_rule("/recheck-comparison", view_func=require_role(viewer)(recheck_comparison), methods=["POST"])
    app.add_url_rule("/healthz", view_func=healthz)
    app.register_error_handler(403, forbidden)
    app.register_error_handler(413, upload_too_large)
    app.register_error_handler(500, internal_error)


def create_app(config: dict | None = None, env: str | None = None) -> Flask:
    """Build the application. Raises config.ConfigError if the environment is unsafe."""
    app = Flask(__name__)
    app.config.from_mapping(build_config(resolve_env(env)))
    if config:
        app.config.update(config)

    # First, so every later component (sessions, rate limits, audit IPs) sees the real
    # client address rather than the load balancer's.
    trust_proxy(app)
    db.init_app(app)
    migrate.init_app(app, db)
    app.extensions["storage"] = build_backend(app.config)
    app.extensions["model_cache"] = ModelCache(app.config["MODEL_CACHE_SIZE"])
    init_auth(app)
    _register_routes(app)
    check_route_roles(app)
    return app


if __name__ == "__main__":
    from wsgi import app as application  # exits with a clear message if misconfigured

    application.run(debug=os.environ.get("FLASK_DEBUG", "").lower() == "true")
