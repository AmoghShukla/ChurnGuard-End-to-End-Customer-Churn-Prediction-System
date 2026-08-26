"""
data_prep.py
------------
Loading and cleaning logic for the Telco Customer Churn dataset.
Kept separate from feature engineering so raw cleaning is testable
and reusable across the training script, notebook, and API.
"""

import pandas as pd
import numpy as np


RAW_COLUMNS_TO_DROP = ["customerID"]


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe:
    - TotalCharges arrives as a string with some blank entries (11 rows
      where tenure == 0, i.e. brand-new customers who haven't been
      billed yet). We coerce to numeric and impute with 0, since a
      customer with 0 tenure genuinely has 0 total charges — this is
      not a random missing value, it's a structural one.
    - SeniorCitizen arrives as 0/1 int; we leave it as-is (it behaves
      as a binary flag either way) but document the choice.
    - Standardize Churn to a binary integer target.
    - Drop customerID (identifier, not a feature).
    """
    df = df.copy()

    # TotalCharges: blank strings -> NaN -> numeric
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Structural missingness: tenure == 0 means no charges yet accrued.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # Target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0}).astype(int)

    # Drop identifier columns
    df = df.drop(columns=[c for c in RAW_COLUMNS_TO_DROP if c in df.columns])

    return df


def load_clean(path: str) -> pd.DataFrame:
    """Convenience wrapper: load + clean in one call."""
    return clean(load_raw(path))
