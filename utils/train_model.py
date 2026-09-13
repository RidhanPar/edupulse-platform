from __future__ import annotations

from dataclasses import dataclass
import io

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from utils.preprocessing import build_preprocessing_pipeline, split_features_target, FEATURE_COLUMNS


@dataclass
class TrainingArtifacts:
    model_name: str
    metrics: dict
    feature_importances: dict
    model_comparison: list[dict]
    model: Pipeline


def _candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced"),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42),
    }


def _extract_feature_importances(model_obj, feature_names):
    if hasattr(model_obj, "feature_importances_"):
        vals = model_obj.feature_importances_
    elif hasattr(model_obj, "coef_"):
        vals = np.abs(model_obj.coef_[0])
    else:
        vals = np.zeros(len(feature_names))
    return {col: round(float(v), 4) for col, v in zip(feature_names, vals)}


def train_and_select_best(df: pd.DataFrame) -> TrainingArtifacts:
    X, y, _ = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    best_name = None
    best_pipe = None
    best_metrics = None
    best_importances = None
    best_score = -1.0
    comparison_rows = []

    for name, model in _candidate_models().items():
        pipe = Pipeline([
            ("prep", build_preprocessing_pipeline()),
            ("model", model),
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe.named_steps["model"], "predict_proba") else None

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4) if y_prob is not None else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        comparison_rows.append({
            "model": name,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        })

        score = metrics["f1"]
        if metrics["roc_auc"] is not None:
            score = (metrics["f1"] + metrics["roc_auc"]) / 2

        if score > best_score:
            best_score = score
            best_name = name
            best_pipe = pipe
            best_metrics = metrics
            best_importances = _extract_feature_importances(pipe.named_steps["model"], FEATURE_COLUMNS)

    return TrainingArtifacts(
        model_name=best_name,
        metrics=best_metrics,
        feature_importances=best_importances,
        model_comparison=comparison_rows,
        model=best_pipe,
    )


def serialize_model(model: Pipeline) -> bytes:
    """Bytes for object storage; utils.predict.load_model reverses this."""
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()
