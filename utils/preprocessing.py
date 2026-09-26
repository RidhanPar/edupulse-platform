from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DISPLAY_COLUMNS = ["student_id", "student_name"]

FEATURE_COLUMNS = [
    "attendance",
    "assignment_score",
    "quiz_score",
    "study_time",
    "lms_activity",
    "previous_grade",
    "missed_submissions",
]

TARGET_COLUMN = "target"


def available_features(df: pd.DataFrame) -> list[str]:
    """The expected feature columns this file actually has, in the canonical order."""
    return [c for c in FEATURE_COLUMNS if c in df.columns]


def missing_features(df: pd.DataFrame) -> list[str]:
    """Expected feature columns this file does not have. Training and prediction go on without them."""
    return [c for c in FEATURE_COLUMNS if c not in df.columns]


def validate_columns(df: pd.DataFrame, require_target: bool = True) -> tuple[bool, list[str]]:
    """Check the columns a file must have: the display columns, the target when
    training, and at least one feature. Features that are absent are reported
    separately by missing_features(); a model is trained on what is present.
    """
    required = DISPLAY_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if not available_features(df):
        missing.append(f"at least one feature column ({', '.join(FEATURE_COLUMNS)})")
    return len(missing) == 0, missing


def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])


def split_features_target(df: pd.DataFrame, features: list[str] | None = None):
    X = df[features if features is not None else available_features(df)].copy()
    y = df[TARGET_COLUMN].copy()
    meta = df[DISPLAY_COLUMNS].copy() if all(c in df.columns for c in DISPLAY_COLUMNS) else None
    return X, y, meta


def risk_from_probability(prob_fail: float) -> str:
    if prob_fail < 0.30:
        return "Low"
    if prob_fail < 0.60:
        return "Medium"
    return "High"


def recommendation_from_row(row: pd.Series) -> str:
    prediction = str(row.get("prediction", "")).strip()
    risk_level = str(row.get("risk_level", "")).strip()
    fail_probability = float(row.get("fail_probability", 0.0))

    issues = []

    attendance = float(row.get("attendance", 100))
    assignment_score = float(row.get("assignment_score", 100))
    quiz_score = float(row.get("quiz_score", 100))
    study_time = float(row.get("study_time", 10))
    lms_activity = float(row.get("lms_activity", 100))
    previous_grade = float(row.get("previous_grade", 100))
    missed_submissions = float(row.get("missed_submissions", 0))

    if attendance < 60:
        issues.append("low attendance")
    if assignment_score < 10:
        issues.append("weak assignment performance")
    if quiz_score < 10:
        issues.append("low quiz performance")
    if study_time <= 1:
        issues.append("limited study time")
    if lms_activity < 40:
        issues.append("low engagement level")
    if previous_grade < 10:
        issues.append("weak previous academic performance")
    if missed_submissions >= 2:
        issues.append("repeated academic failures")

    # Positive case
    if prediction == "Pass" and risk_level == "Low" and fail_probability < 0.30:
        if not issues:
            return (
                "The student is currently predicted to perform satisfactorily. "
                "Regular academic monitoring is sufficient at this stage."
            )
        return (
            f"The student is currently predicted to pass, but some attention may still be useful due to "
            f"{', '.join(issues)}. Regular monitoring and light academic support are recommended."
        )

    # Medium risk case
    if risk_level == "Medium":
        if issues:
            return (
                f"The student shows moderate academic risk associated with {', '.join(issues)}. "
                "Targeted support, progress monitoring, and advisor follow up are recommended."
            )
        return (
            "The student shows moderate academic risk. Additional monitoring and early academic support are recommended."
        )

    # High risk or fail case
    if prediction == "Fail" or risk_level == "High" or fail_probability >= 0.60:
        if issues:
            return (
                f"The student is at elevated academic risk, mainly associated with {', '.join(issues)}. "
                "Immediate academic review, advisor intervention, and structured support are recommended."
            )
        return (
            "The student is at elevated academic risk. Immediate academic review and structured support are recommended."
        )

    # Fallback
    if issues:
        return (
            f"The student shows some academic concerns, including {', '.join(issues)}. "
            "Follow up and continued monitoring are recommended."
        )

    return "The student currently shows no major academic risk indicators. Regular monitoring is sufficient."