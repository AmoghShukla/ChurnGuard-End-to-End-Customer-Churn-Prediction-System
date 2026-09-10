"""
dashboard/app.py
-----------------
Lightweight monitoring dashboard for ChurnGuard.

Shows:
  - Training-time model comparison metrics (from models/metrics.json)
  - Live prediction volume and score-distribution drift (from the
    prediction_logs table the API writes to on every request)

Run with: streamlit run dashboard/app.py
Expects the API to be running at API_BASE_URL (default http://localhost:8000).
"""

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="ChurnGuard Monitor", layout="wide")
st.title("ChurnGuard : Model Monitoring Dashboard")

# ---- Training metrics ----
st.header("Model Comparison (Training Time)")
try:
    metrics = requests.get(f"{API_BASE_URL}/metrics", timeout=5).json()
    model_names = [k for k in metrics.keys() if k not in ("top_shap_features", "business_impact")]

    cols = st.columns(len(model_names))
    for col, name in zip(cols, model_names):
        m = metrics[name]
        with col:
            st.subheader(name)
            st.metric("ROC-AUC", m["roc_auc"])
            st.metric("Recall (Churn) ", m["recall"])
            st.metric("Precision (Churn)", m["precision"])
            st.metric("F1", m["f1"])

    st.subheader("Business Impact (XGBoost, default threshold)")
    biz = metrics.get("business_impact", {})
    b1, b2, b3 = st.columns(3)
    b1.metric("Churners Correctly Flagged", f"{biz.get('churners_correctly_flagged', '-')} / {biz.get('actual_churners_in_test', '-')}")
    b2.metric("Recall at Threshold", biz.get("recall_at_default_threshold", "-"))
    b3.metric("Est. Annual Revenue-at-Risk Flagged", f"${biz.get('estimated_annual_revenue_at_risk_correctly_flagged', 0):,.0f}")

    st.subheader("Top Churn Drivers (mean |SHAP|)")
    shap_df = pd.DataFrame(metrics.get("top_shap_features", []), columns=["feature", "mean_abs_shap"])
    fig = px.bar(shap_df, x="mean_abs_shap", y="feature", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.warning(f"Could not load training metrics from API ({API_BASE_URL}/metrics). "
               f"Make sure the API is running and train.py has been run. Error: {e}")

st.divider()

# ---- Live prediction monitoring ----
st.header("Live Prediction Monitoring")
try:
    recent = requests.get(f"{API_BASE_URL}/predictions/recent", params={"limit": 200}, timeout=5).json()
    if not recent:
        st.info("No predictions logged yet. Call POST /predict-churn to generate traffic.")
    else:
        df = pd.DataFrame(recent)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions Logged", len(df))
        c2.metric("Avg Churn Probability", f"{df['churn_probability'].mean():.3f}")
        c3.metric("% Flagged High Risk", f"{(df['risk_tier'] == 'high').mean() * 100:.1f}%")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Prediction Score Distribution")
            fig2 = px.histogram(df, x="churn_probability", nbins=20)
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            st.subheader("Risk Tier Breakdown")
            tier_counts = df["risk_tier"].value_counts().reset_index()
            tier_counts.columns = ["risk_tier", "count"]
            fig3 = px.pie(tier_counts, names="risk_tier", values="count")
            st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Prediction Volume Over Time")
        df_sorted = df.sort_values("timestamp")
        fig4 = px.line(df_sorted, x="timestamp", y="churn_probability", markers=True)
        st.plotly_chart(fig4, use_container_width=True)

        with st.expander("Raw recent predictions"):
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)

except Exception as e:
    st.warning(f"Could not load live predictions from API ({API_BASE_URL}/predictions/recent). Error: {e}")
