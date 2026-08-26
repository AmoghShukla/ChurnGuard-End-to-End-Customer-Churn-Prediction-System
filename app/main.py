"""
main.py
-------
ChurnGuard API — thin FastAPI serving layer around the trained model.

Endpoints:
  GET  /health              liveness check
  POST /predict-churn       predict churn risk for a single customer
  GET  /metrics             training-time model metrics (for the dashboard)
  GET  /predictions/recent  recent logged predictions (for the dashboard)
"""

from datetime import datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app import db
from app.predictor import predict
from app.schemas.customer import ChurnPrediction, CustomerFeatures

app = FastAPI(
    title="ChurnGuard API",
    description="Predicts customer churn risk and explains the top drivers via SHAP.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    db.init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict-churn", response_model=ChurnPrediction)
def predict_churn(customer: CustomerFeatures, session: Session = Depends(db.get_session)):
    try:
        payload = customer.model_dump()
        result = predict(payload)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    db.log_prediction(session, result, payload)
    return result


@app.get("/metrics")
def get_metrics():
    import json
    from pathlib import Path

    metrics_path = Path(__file__).resolve().parent.parent / "models" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not found. Run train.py first.")
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/predictions/recent")
def recent_predictions(limit: int = 50, session: Session = Depends(db.get_session)):
    rows = (
        session.query(db.PredictionLog)
        .order_by(db.PredictionLog.timestamp.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat(),
            "churn_probability": r.churn_probability,
            "risk_tier": r.risk_tier,
            "model_version": r.model_version,
        }
        for r in rows
    ]
