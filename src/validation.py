"""Final accuracy validation for RetailPulse models.

Recomputes the headline accuracy metrics from the committed prediction
artifacts and checks each against an acceptance gate, then writes a
consolidated report to reports/accuracy_validation.json.

This is intentionally lightweight (pandas / numpy / sklearn.metrics only) so it
runs in CI without loading the heavy training stack (xgboost, optuna, shap,
prophet). It validates *saved outputs*, not the training process.

Run via scripts/run_validation.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT           = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
REPORTS_DIR    = ROOT / "reports"

CHURN_PREDICTIONS = DATA_PROCESSED / "churn_predictions.csv"
DAILY_REVENUE_TS  = DATA_PROCESSED / "daily_revenue_ts.csv"
BASELINE_METRICS  = MODELS_DIR / "baseline_metrics.json"
REPORT_OUT        = REPORTS_DIR / "accuracy_validation.json"

# ── Acceptance gates ──────────────────────────────────────────────────────────
# Churn gates mirror the training-time gates in src/churn.py; the forecast gate
# is expressed as a normalised RMSE (RMSE / mean active-day revenue) so it is
# independent of absolute revenue scale.
AUC_ROC_GATE         = 0.88
PRECISION_TOP20_GATE = 0.75
FORECAST_NRMSE_GATE  = 0.75


def precision_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, k: float = 0.20) -> float:
    """Precision among the top-k fraction of customers ranked by churn probability."""
    n_top = max(1, int(len(y_true) * k))
    idx   = np.argsort(y_prob)[::-1][:n_top]
    return float(y_true[idx].mean())


def validate_churn(predictions_csv: Path = CHURN_PREDICTIONS) -> dict:
    """Recompute churn classification metrics from saved predictions."""
    df = pd.read_csv(predictions_csv)
    y_true = df["actual_churn"].to_numpy()
    y_prob = df["churn_probability"].to_numpy()
    y_pred = df["predicted_churn"].to_numpy()

    metrics = {
        "auc_roc":         round(float(roc_auc_score(y_true, y_prob)), 4),
        "precision_top20": round(precision_at_top_k(y_true, y_prob, k=0.20), 4),
        "accuracy":        round(float(accuracy_score(y_true, y_pred)), 4),
        "f1":              round(float(f1_score(y_true, y_pred)), 4),
    }
    gates = [
        ("auc_roc",         metrics["auc_roc"],         AUC_ROC_GATE),
        ("precision_top20", metrics["precision_top20"], PRECISION_TOP20_GATE),
    ]
    failures = [f"{name} {val:.4f} < {gate}" for name, val, gate in gates if val < gate]
    return {
        "model":     "churn_xgboost",
        "n_samples": int(len(df)),
        "metrics":   metrics,
        "gates":     {name: gate for name, _, gate in gates},
        "passed":    not failures,
        "failures":  failures,
    }


def validate_forecast(
    baseline_json: Path = BASELINE_METRICS,
    daily_ts_csv: Path = DAILY_REVENUE_TS,
) -> dict:
    """Validate recorded forecast error against a scale-normalised gate.

    The forecaster's holdout RMSE/MAE are recorded at train time in
    baseline_metrics.json. We normalise RMSE by the mean revenue on active
    trading days (revenue > 0) to get a unitless error the gate can bound.
    """
    baseline = json.loads(Path(baseline_json).read_text())
    rmse = float(baseline["forecast_rmse"])
    mae  = float(baseline["forecast_mae"])

    revenue   = pd.read_csv(daily_ts_csv, index_col=0)["Revenue"]
    mean_rev  = float(revenue[revenue > 0].mean())
    nrmse     = round(rmse / mean_rev, 4)
    nmae      = round(mae / mean_rev, 4)

    failures = []
    if nrmse >= FORECAST_NRMSE_GATE:
        failures.append(f"nRMSE {nrmse:.4f} >= {FORECAST_NRMSE_GATE}")
    return {
        "model": "hybrid_prophet_lstm",
        "metrics": {
            "rmse":             round(rmse, 4),
            "mae":              round(mae, 4),
            "mean_active_revenue": round(mean_rev, 2),
            "nrmse":            nrmse,
            "nmae":             nmae,
        },
        "gates":    {"nrmse": FORECAST_NRMSE_GATE},
        "passed":   not failures,
        "failures": failures,
    }


def run_validation(write: bool = True) -> dict:
    """Run all validations, optionally persist the report, and return it."""
    churn    = validate_churn()
    forecast = validate_forecast()
    report = {
        "passed":  churn["passed"] and forecast["passed"],
        "churn":   churn,
        "forecast": forecast,
    }
    if write:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_OUT.write_text(json.dumps(report, indent=2))
    return report


def format_summary(report: dict) -> str:
    """Render a human-readable summary table from a validation report."""
    lines = ["", "=== Final Accuracy Validation ===", ""]
    for section in ("churn", "forecast"):
        r = report[section]
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(f"[{status}] {r['model']}")
        for name, val in r["metrics"].items():
            gate = r["gates"].get(name)
            suffix = f"  (gate {gate})" if gate is not None else ""
            lines.append(f"    {name:<22} {val}{suffix}")
        for f in r["failures"]:
            lines.append(f"    FAILED: {f}")
        lines.append("")
    overall = "PASS" if report["passed"] else "FAIL"
    lines.append(f"Overall: {overall}")
    return "\n".join(lines)
