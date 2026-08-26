"""
db.py
-----
Prediction logging layer. Every call to /predict-churn is persisted so
the monitoring dashboard (dashboard/app.py) can track prediction volume
and score-distribution drift over time.

Defaults to a local SQLite file so the project runs with zero setup.
Set DATABASE_URL to a Postgres DSN to use Postgres instead (matches
the stack already used in QuickHeal/CertivaX/LeagueForge) — no code
changes required, SQLAlchemy handles the dialect switch.
"""

import datetime
import os

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./churnguard.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    churn_probability = Column(Float)
    risk_tier = Column(String)
    model_version = Column(String)
    input_payload = Column(JSON)


def init_db():
    Base.metadata.create_all(bind=engine)


def log_prediction(session, result: dict, payload: dict):
    entry = PredictionLog(
        churn_probability=result["churn_probability"],
        risk_tier=result["risk_tier"],
        model_version=result["model_version"],
        input_payload=payload,
    )
    session.add(entry)
    session.commit()


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
