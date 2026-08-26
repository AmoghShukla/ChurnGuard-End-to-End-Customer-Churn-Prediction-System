"""
train.py
--------
End-to-end training script for ChurnGuard.

Trains and compares three models:
  1. Logistic Regression (interpretable baseline)
  2. Random Forest
  3. XGBoost (final production model)

Handles class imbalance via class weighting (documented choice over
SMOTE — see README for the tradeoff discussion).

Outputs:
  - models/churn_model.joblib      (fitted sklearn Pipeline: preprocessor + XGBoost)
  - models/metrics.json            (evaluation metrics for all 3 models)
  - models/shap_summary.png        (SHAP feature importance plot)
  - models/model_comparison.png    (ROC curves for all 3 models)
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.data_prep import load_clean
from app.features import build_preprocessor, engineer, get_feature_columns

DATA_PATH = "data/telco_churn.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 42


def prepare_data():
    df = load_clean(DATA_PATH)
    df = engineer(df)
    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test


def evaluate(name, model, X_test, y_test, results: dict):
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    metrics = {
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "avg_precision": round(average_precision_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    results[name] = metrics
    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    print("ROC-AUC:", metrics["roc_auc"])
    return proba


def plot_roc_comparison(curves: dict, y_test):
    plt.figure(figsize=(7, 6))
    for name, proba in curves.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison — ChurnGuard Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {MODELS_DIR / 'model_comparison.png'}")


def main():
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.3f}, Test churn rate: {y_test.mean():.3f}")

    preprocessor = build_preprocessor()

    # class weighting to handle imbalance (~26.5% positive class)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("prep", preprocessor),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        scale_pos_weight=scale_pos_weight,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    results = {}
    roc_curves = {}
    fitted_models = {}

    for name, pipe in models.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        fit_time = time.time() - t0
        proba = evaluate(name, pipe, X_test, y_test, results)
        results[name]["fit_time_sec"] = round(fit_time, 2)
        roc_curves[name] = proba
        fitted_models[name] = pipe

    plot_roc_comparison(roc_curves, y_test)

    # Final model = XGBoost (best AUC/recall tradeoff for this problem —
    # see README for the justification vs. the other two)
    final_model = fitted_models["XGBoost"]

    # ---- SHAP explainability on the final model ----
    print("\nComputing SHAP values for the final model (XGBoost)...")
    X_test_transformed = final_model.named_steps["prep"].transform(X_test)
    feature_names = final_model.named_steps["prep"].get_feature_names_out()
    explainer = shap.TreeExplainer(final_model.named_steps["clf"])
    shap_values = explainer.shap_values(X_test_transformed)

    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {MODELS_DIR / 'shap_summary.png'}")

    # top drivers for README / business narrative
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:10]
    top_features = [(feature_names[i], round(float(mean_abs_shap[i]), 4)) for i in top_idx]
    results["top_shap_features"] = top_features
    print("\nTop churn drivers (mean |SHAP|):")
    for feat, val in top_features:
        print(f"  {feat}: {val}")

    # ---- business translation ----
    # At the chosen decision threshold (0.5 default), report revenue-at-risk framing
    proba_final = roc_curves["XGBoost"]
    preds_final = (proba_final >= 0.5).astype(int)
    caught = ((preds_final == 1) & (y_test == 1)).sum()
    total_churners = (y_test == 1).sum()
    avg_monthly_charge = X_test["MonthlyCharges"].mean()
    annualized_ltv_at_risk_caught = caught * avg_monthly_charge * 12

    business = {
        "total_test_customers": int(len(y_test)),
        "actual_churners_in_test": int(total_churners),
        "churners_correctly_flagged": int(caught),
        "recall_at_default_threshold": round(caught / total_churners, 4),
        "avg_monthly_charge": round(float(avg_monthly_charge), 2),
        "estimated_annual_revenue_at_risk_correctly_flagged": round(
            annualized_ltv_at_risk_caught, 2
        ),
    }
    results["business_impact"] = business
    print("\nBusiness impact summary:")
    print(json.dumps(business, indent=2))

    # ---- save everything ----
    joblib.dump(final_model, MODELS_DIR / "churn_model.joblib")
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model: {MODELS_DIR / 'churn_model.joblib'}")
    print(f"Saved metrics: {MODELS_DIR / 'metrics.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
