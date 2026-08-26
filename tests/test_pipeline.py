import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.data_prep import clean
from app.features import engineer, get_feature_columns
from app.main import app


@pytest.fixture
def sample_raw_row():
    return pd.DataFrame([{
        "customerID": "0000-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.7,
        "TotalCharges": " ",  # blank string, as it appears for tenure==0 in raw data
        "Churn": "Yes",
    }])


def test_clean_handles_blank_total_charges(sample_raw_row):
    cleaned = clean(sample_raw_row)
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["Churn"].iloc[0] == 1
    assert "customerID" not in cleaned.columns


def test_engineer_adds_expected_columns(sample_raw_row):
    cleaned = clean(sample_raw_row)
    engineered = engineer(cleaned)
    for col in get_feature_columns():
        assert col in engineered.columns
    assert engineered["tenure_bucket"].iloc[0] == "0-6m"
    assert engineered["contract_risk"].iloc[0] == 2  # month-to-month = highest risk


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_endpoint_returns_valid_schema():
    client = TestClient(app)
    payload = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.7, "TotalCharges": 151.65,
    }
    resp = client.post("/predict-churn", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in ("low", "medium", "high")
    assert len(body["top_factors"]) == 5


def test_predict_endpoint_rejects_invalid_payload():
    client = TestClient(app)
    resp = client.post("/predict-churn", json={"gender": "Not A Gender"})
    assert resp.status_code == 422
