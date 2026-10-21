from __future__ import annotations

import io

import joblib
import pandas as pd

from utils.preprocessing import DISPLAY_COLUMNS, risk_from_probability, recommendation_from_row
from utils.signing import verify


def load_model(data: bytes, signing_keys: list[str]):
    """Reverse utils.train_model.serialize_model.

    joblib.load unpickles, which can execute code, so the signature is verified first
    and an unsigned or badly signed file raises ModelSignatureError without ever being
    unpickled. Write access to the bucket alone therefore cannot run code in the app.
    """
    return joblib.load(io.BytesIO(verify(data, signing_keys)))


def predict_dataframe(df: pd.DataFrame, model, features: list[str]) -> pd.DataFrame:
    """Predict using exactly the features the model was trained on.

    A model cannot be applied to a file that lacks one of them: filling the gap with a
    constant would be inventing data, so this refuses instead.
    """
    absent = [feature for feature in features if feature not in df.columns]
    if absent:
        raise ValueError(
            "the prediction dataset is missing columns this model was trained on: " + ", ".join(absent)
        )

    X = df[features].copy()

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(df)

    result = df.copy()
    result["prediction"] = ["Fail" if p == 1 else "Pass" for p in preds]
    result["fail_probability"] = [round(float(p), 4) for p in probs]
    result["confidence"] = [round(float(max(p, 1 - p)), 4) for p in probs]
    result["risk_level"] = [risk_from_probability(float(p)) for p in probs]
    result["recommendation"] = result.apply(recommendation_from_row, axis=1)

    ordered_cols = DISPLAY_COLUMNS + features + [
        "prediction", "fail_probability", "confidence", "risk_level", "recommendation"
    ]
    return result[ordered_cols]
