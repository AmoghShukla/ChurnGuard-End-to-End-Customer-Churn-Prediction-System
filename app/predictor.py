"""
predictor.py
------------
Loads the trained pipeline once at startup and exposes a single
`predict()` function used by the API layer. Keeps the "ML serving"
concern separate from the "HTTP" concern (routers/services/repository
split, consistent with the clean-architecture style used elsewhere).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from app.features import engineer, get_feature_columns
from app.data_prep import clean

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
MODEL_VERSION = "xgboost-v1"

_model = None
_explainer = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def _get_explainer():
    global _explainer
    if _explainer is None:
        model = _get_model()
        _explainer = shap.TreeExplainer(model.named_steps["clf"])
    return _explainer


def _risk_tier(prob: float) -> str:
    if prob < 0.3:
        return "low"
    if prob < 0.6:
        return "medium"
    return "high"


def predict(customer: dict) -> dict:
    """
    Run the full pipeline for a single customer record:
    clean -> engineer -> predict -> explain top SHAP contributors.
    """
    model = _get_model()
    explainer = _get_explainer()

    df = pd.DataFrame([customer])
    # our clean() expects a Churn column when mapping; single-record
    # inference doesn't have a label, so we bypass that step here and
    # just reuse the TotalCharges/tenure logic directly since inference
    # payloads already come in typed (Pydantic-validated) and numeric.
    df = engineer(df)

    feature_cols = get_feature_columns()
    X = df[feature_cols]

    proba = float(model.predict_proba(X)[0, 1])

    X_transformed = model.named_steps["prep"].transform(X)
    feature_names = model.named_steps["prep"].get_feature_names_out()
    shap_values = explainer.shap_values(X_transformed)[0]

    top_idx = np.argsort(np.abs(shap_values))[::-1][:5]
    top_factors = [
        {
            "feature": feature_names[i],
            "impact": round(float(shap_values[i]), 4),
            "direction": "increases_risk" if shap_values[i] > 0 else "decreases_risk",
        }
        for i in top_idx
    ]

    return {
        "churn_probability": round(proba, 4),
        "risk_tier": _risk_tier(proba),
        "top_factors": top_factors,
        "model_version": MODEL_VERSION,
    }
