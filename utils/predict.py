from __future__ import annotations

import io

import joblib
import pandas as pd

from utils.preprocessing import FEATURE_COLUMNS, DISPLAY_COLUMNS, risk_from_probability, recommendation_from_row


def load_model(data: bytes):
    """Reverse utils.train_model.serialize_model.

    joblib.load unpickles, which can execute code: only pass bytes this application
    wrote to its own storage, fetched through TenantStorage.
    """
    return joblib.load(io.BytesIO(data))


def predict_dataframe(df: pd.DataFrame, model) -> pd.DataFrame:
    X = df[FEATURE_COLUMNS].copy()

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(df)

    result = df.copy()
    result["prediction"] = ["Fail" if p == 1 else "Pass" for p in preds]
    result["fail_probability"] = [round(float(p), 4) for p in probs]
    result["confidence"] = [round(float(max(p, 1 - p)), 4) for p in probs]
    result["risk_level"] = [risk_from_probability(float(p)) for p in probs]
    result["recommendation"] = result.apply(recommendation_from_row, axis=1)

    ordered_cols = DISPLAY_COLUMNS + FEATURE_COLUMNS + [
        "prediction", "fail_probability", "confidence", "risk_level", "recommendation"
    ]
    return result[ordered_cols]
