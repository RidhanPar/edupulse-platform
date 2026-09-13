from __future__ import annotations

import io
import os
import uuid

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, Response, abort, g, session

from config import build_config, resolve_env
from db import db, migrate
from db.models import Dataset, DatasetKind, ModelArtifact, User
from db.tenancy import scoped_select
from utils.evaluation import load_importances, load_metrics, load_model_comparison
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


def _session_user() -> User | None:
    try:
        user_id = uuid.UUID(str(session.get("_user_id")))
    except ValueError:
        return None
    user = db.session.get(User, user_id)
    if user is None or not user.is_active or not user.organisation.is_active:
        return None
    return user


def _load_request_user():
    # Stand-in until section 3 adds Flask-Login, reading the same "_user_id" session
    # key. Without an active user in an active organisation there is no tenant, so
    # the request is refused rather than served from some default organisation.
    if request.endpoint in (None, "static"):
        return
    g.user = _session_user()
    if g.user is None:
        abort(401)


def current_organisation_id():
    return g.user.organisation_id


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


def _store_upload(file, kind: DatasetKind) -> Dataset:
    data = file.read()
    df = read_csv_flexible(data)  # parsed before anything is stored
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
        uploaded_by=g.user.id,
    )
    storage_for(organisation_id).put(dataset.storage_key, data)
    db.session.add(dataset)
    db.session.commit()
    return dataset


def _active_model() -> ModelArtifact | None:
    return db.session.scalars(
        scoped_select(ModelArtifact, current_organisation_id()).where(ModelArtifact.is_active.is_(True))
    ).one_or_none()


def _load_model(artifact: ModelArtifact):
    from utils.predict import load_model

    return load_model(storage_for(current_organisation_id()).get(artifact.storage_key))


def _save_model(artifacts) -> ModelArtifact:
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
        trained_by=g.user.id,
        is_active=True,
    )
    storage_for(organisation_id).put(artifact.storage_key, serialize_model(artifacts.model))
    previous = _active_model()
    if previous is not None:
        # Deactivate first: the database allows one active model per organisation.
        previous.is_active = False
        db.session.flush()
    db.session.add(artifact)
    db.session.commit()
    return artifact


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
            _save_model(artifacts)
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
    # and /compare, including fetching and unpickling the model from object storage.
    # Materialise them into a table when the prediction dataset or active model changes;
    # recomputing per request becomes a real cost once LLM reasoning is added.
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

def internal_error(error):
    return render_template("error.html", message="An internal server error occurred. Please check your uploaded dataset and try again."), 500


def _register_routes(app: Flask) -> None:
    # Endpoint names default to the view function names, so every url_for()
    # in the templates resolves exactly as it did with @app.route.
    app.add_url_rule("/", view_func=home)
    app.add_url_rule("/upload-train", view_func=upload_train, methods=["GET", "POST"])
    app.add_url_rule("/upload-predict", view_func=upload_predict, methods=["GET", "POST"])
    app.add_url_rule("/train", view_func=train, methods=["GET", "POST"])
    app.add_url_rule("/results", view_func=results)
    app.add_url_rule("/download-results", view_func=download_results)
    app.add_url_rule("/explain", view_func=explain)
    app.add_url_rule("/about", view_func=about)
    app.add_url_rule("/upload-actual", view_func=upload_actual, methods=["GET", "POST"])
    app.add_url_rule("/compare", view_func=compare)
    app.add_url_rule("/recheck-comparison", view_func=recheck_comparison, methods=["POST"])
    app.register_error_handler(500, internal_error)


def create_app(config: dict | None = None, env: str | None = None) -> Flask:
    """Build the application. Raises config.ConfigError if the environment is unsafe."""
    app = Flask(__name__)
    app.config.from_mapping(build_config(resolve_env(env)))
    if config:
        app.config.update(config)

    db.init_app(app)
    migrate.init_app(app, db)
    app.extensions["storage"] = build_backend(app.config)
    app.before_request(_load_request_user)
    _register_routes(app)
    return app


if __name__ == "__main__":
    from wsgi import app as application  # exits with a clear message if misconfigured

    application.run(debug=os.environ.get("FLASK_DEBUG", "").lower() == "true")
