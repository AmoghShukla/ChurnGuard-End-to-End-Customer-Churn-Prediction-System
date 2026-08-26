"""
features.py
-----------
Feature engineering logic, packaged as a scikit-learn ColumnTransformer
so the exact same transformation is used at training time and at
inference time (no train/serve skew).

Engineered features (beyond the raw columns):
- tenure_bucket: coarse tenure groups (new / established / loyal) —
  captures the well-documented non-linear relationship between tenure
  and churn risk (risk drops sharply after the first ~12 months).
- num_services: count of subscribed add-on services — a proxy for
  "stickiness" (more services -> more switching friction).
- avg_monthly_spend_ratio: TotalCharges / (tenure + 1) vs MonthlyCharges
  — flags customers whose historical average diverges from their
  current bill (e.g. recent price hikes), a known churn trigger.
- contract_risk: ordinal encoding of contract length as a explicit
  risk-ordered feature, since month-to-month is known to dominate churn.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_bucket",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_services",
    "avg_monthly_spend_ratio",
    "contract_risk",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a cleaned dataframe. Pure, no fitting."""
    df = df.copy()

    # tenure buckets (months)
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4y+"],
    ).astype(str)

    # count of "Yes" service subscriptions across service columns
    def count_services(row):
        return sum(
            1
            for col in SERVICE_COLUMNS
            if str(row[col]).lower() not in ("no", "no internet service", "no phone service")
        )

    df["num_services"] = df.apply(count_services, axis=1)

    # historical average monthly spend vs current monthly charge
    # ratio > 1 => currently paying more than historical average (risk signal)
    hist_avg = df["TotalCharges"] / (df["tenure"] + 1)
    df["avg_monthly_spend_ratio"] = (
        df["MonthlyCharges"] / hist_avg.replace(0, np.nan)
    ).fillna(1.0).clip(0, 5)

    # explicit ordinal risk encoding for contract type
    contract_risk_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
    df["contract_risk"] = df["Contract"].map(contract_risk_map).fillna(2)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the sklearn ColumnTransformer used inside the model Pipeline."""
    categorical_pipe = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    numeric_pipe = Pipeline(steps=[("scale", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, CATEGORICAL_COLUMNS),
            ("num", numeric_pipe, NUMERIC_COLUMNS),
        ]
    )
    return preprocessor


def get_feature_columns() -> list[str]:
    return CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
