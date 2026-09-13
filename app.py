from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, Response

from config import build_config, resolve_env
from db import db, migrate
from utils.evaluation import load_importances, load_metrics, load_model_comparison

# utils.preprocessing, utils.train_model, utils.predict and utils.compare_results
# all import scikit-learn (and joblib), which adds seconds to every cold start and
# test run. They are imported inside the handlers that use them, so create_app()
# never loads them. tests/test_app_factory.py enforces this.

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"

TRAIN_UPLOAD_PATH = RAW_DIR / "training_dataset.csv"
PREDICT_UPLOAD_PATH = RAW_DIR / "prediction_dataset.csv"
ACTUAL_RESULTS_PATH = RAW_DIR / "actual_results.csv"


def read_csv_flexible(path: Path) -> pd.DataFrame:
    """
    Reads both comma-separated and semicolon-separated CSV files.
    """
    try:
        df = pd.read_csv(path)
        if len(df.columns) == 1:
            df = pd.read_csv(path, sep=";")
        return df
    except (pd.errors.ParserError, UnicodeDecodeError):
        df = pd.read_csv(path, sep=";")
        return df


def get_training_df(require_target: bool = True):
    from utils.preprocessing import validate_columns

    if not TRAIN_UPLOAD_PATH.exists():
        return None, "No training dataset uploaded yet."
    try:
        df = read_csv_flexible(TRAIN_UPLOAD_PATH)
        ok, missing = validate_columns(df, require_target=require_target)
        if not ok:
            return None, f"Missing required columns in training dataset: {', '.join(missing)}"
        return df, None
    except Exception as e:
        return None, f"Error reading training dataset: {str(e)}"


def get_prediction_df():
    from utils.preprocessing import validate_columns

    if not PREDICT_UPLOAD_PATH.exists():
        return None, "No prediction dataset uploaded yet."
    try:
        df = read_csv_flexible(PREDICT_UPLOAD_PATH)
        ok, missing = validate_columns(df, require_target=False)
        if not ok:
            return None, f"Missing required columns in prediction dataset: {', '.join(missing)}"
        return df, None
    except Exception as e:
        return None, f"Error reading prediction dataset: {str(e)}"

def get_actual_results_df():
    if not ACTUAL_RESULTS_PATH.exists():
        return None, "No actual results dataset uploaded yet."
    try:
        df = read_csv_flexible(ACTUAL_RESULTS_PATH)
        required_cols = ["student_id", "target"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return None, f"Missing required columns in actual results dataset: {', '.join(missing)}"
        return df, None
    except Exception as e:
        return None, f"Error reading actual results dataset: {str(e)}"


def home():
    metrics = load_metrics(str(MODELS_DIR))
    comparison = load_model_comparison(str(MODELS_DIR))
    return render_template("home.html", metrics=metrics, comparison=comparison)


def upload_train():
    from utils.preprocessing import DISPLAY_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN

    preview = None
    columns = None

    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or not file.filename:
                flash("Please select a training CSV file.", "danger")
                return redirect(url_for("upload_train"))

            if not file.filename.lower().endswith(".csv"):
                flash("Only CSV files are allowed for training upload.", "danger")
                return redirect(url_for("upload_train"))

            RAW_DIR.mkdir(parents=True, exist_ok=True)
            file.save(TRAIN_UPLOAD_PATH)
            flash("Training dataset uploaded successfully.", "success")
            return redirect(url_for("upload_train"))

        except Exception as e:
            flash(f"Upload failed: {str(e)}", "danger")
            return redirect(url_for("upload_train"))

    if TRAIN_UPLOAD_PATH.exists():
        try:
            df = read_csv_flexible(TRAIN_UPLOAD_PATH)
            preview = df.head(10).to_dict(orient="records")
            columns = list(df.columns)
        except Exception as e:
            flash(f"Could not preview training dataset: {str(e)}", "danger")

    required = DISPLAY_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]
    return render_template(
        "upload_train.html",
        preview=preview,
        columns=columns,
        required=required
    )


def upload_predict():
    from utils.preprocessing import DISPLAY_COLUMNS, FEATURE_COLUMNS

    preview = None
    columns = None

    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or not file.filename:
                flash("Please select a prediction CSV file.", "danger")
                return redirect(url_for("upload_predict"))

            if not file.filename.lower().endswith(".csv"):
                flash("Only CSV files are allowed for prediction upload.", "danger")
                return redirect(url_for("upload_predict"))

            RAW_DIR.mkdir(parents=True, exist_ok=True)
            file.save(PREDICT_UPLOAD_PATH)
            flash("Prediction dataset uploaded successfully.", "success")
            return redirect(url_for("upload_predict"))

        except Exception as e:
            flash(f"Upload failed: {str(e)}", "danger")
            return redirect(url_for("upload_predict"))

    if PREDICT_UPLOAD_PATH.exists():
        try:
            df = read_csv_flexible(PREDICT_UPLOAD_PATH)
            preview = df.head(10).to_dict(orient="records")
            columns = list(df.columns)
        except Exception as e:
            flash(f"Could not preview prediction dataset: {str(e)}", "danger")

    required = DISPLAY_COLUMNS + FEATURE_COLUMNS
    return render_template(
        "upload_predict.html",
        preview=preview,
        columns=columns,
        required=required
    )


def train():
    metrics = load_metrics(str(MODELS_DIR))
    importances = load_importances(str(MODELS_DIR))
    comparison = load_model_comparison(str(MODELS_DIR))

    if request.method == "POST":
        from utils.train_model import train_and_select_best

        df, err = get_training_df(require_target=True)
        if err:
            flash(err, "danger")
            return redirect(url_for("upload_train"))

        try:
            artifacts = train_and_select_best(df, str(MODELS_DIR))
            metrics = {"best_model": artifacts.model_name, **artifacts.metrics}
            importances = artifacts.feature_importances
            comparison = artifacts.model_comparison
            flash(f"Training complete. Best model: {artifacts.model_name}", "success")
        except Exception as e:
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

    if not MODEL_PATH.exists():
        flash("Train the model first.", "warning")
        return redirect(url_for("train"))

    df, err = get_prediction_df()
    if err:
        flash(err, "danger")
        return redirect(url_for("upload_predict"))

    try:
        res = predict_dataframe(df, str(MODEL_PATH))
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

    if not MODEL_PATH.exists():
        flash("Train the model first.", "warning")
        return redirect(url_for("train"))

    df, err = get_prediction_df()
    if err:
        flash(err, "danger")
        return redirect(url_for("upload_predict"))

    try:
        res = predict_dataframe(df, str(MODEL_PATH))
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
    importances = load_importances(str(MODELS_DIR))
    metrics = load_metrics(str(MODELS_DIR))
    if not importances:
        flash("Train the model first to generate explanations.", "warning")
        return redirect(url_for("train"))
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    return render_template("explain.html", ranked=ranked, metrics=metrics)


def about():
    return render_template("about.html")

def upload_actual():
    preview = None
    columns = None

    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or not file.filename:
                flash("Please select an actual results CSV file.", "danger")
                return redirect(url_for("upload_actual"))

            if not file.filename.lower().endswith(".csv"):
                flash("Only CSV files are allowed for actual results upload.", "danger")
                return redirect(url_for("upload_actual"))

            RAW_DIR.mkdir(parents=True, exist_ok=True)
            file.save(ACTUAL_RESULTS_PATH)
            flash("Actual results dataset uploaded successfully.", "success")
            return redirect(url_for("upload_actual"))

        except Exception as e:
            flash(f"Upload failed: {str(e)}", "danger")
            return redirect(url_for("upload_actual"))

    if ACTUAL_RESULTS_PATH.exists():
        try:
            df = read_csv_flexible(ACTUAL_RESULTS_PATH)
            preview = df.head(10).to_dict(orient="records")
            columns = list(df.columns)
        except Exception as e:
            flash(f"Could not preview actual results dataset: {str(e)}", "danger")

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

    if not MODEL_PATH.exists():
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
        predicted_results = predict_dataframe(pred_df, str(MODEL_PATH))
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
    _register_routes(app)
    return app


if __name__ == "__main__":
    from wsgi import app as application  # exits with a clear message if misconfigured

    application.run(debug=os.environ.get("FLASK_DEBUG", "").lower() == "true")
