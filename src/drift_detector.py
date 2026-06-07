from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import joblib
import mlflow
import pandas as pd

from evidently.legacy.metric_preset.data_drift import DataDriftPreset
from evidently.legacy.metric_preset.data_quality import DataQualityPreset
from evidently.legacy.metric_preset.target_drift import TargetDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report.report import Report

warnings.filterwarnings("ignore")

ROOT           = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"
DRIFT_DIR      = ROOT / "reports" / "drift"

REF_SNAPSHOT = pd.Timestamp("2011-06-01")
CUR_SNAPSHOT = pd.Timestamp("2011-09-01")

FEATURE_COLS = [
    "recency", "frequency", "monetary", "tenure",
    "avg_order_value", "avg_qty_per_order", "n_unique_products",
    "avg_days_between_orders", "purchase_regularity", "recent_freq_90d",
    "revenue_last_30d", "n_months_active", "spend_trend_90d",
]

_CHURN_SEGMENTS = {"At Risk", "Lost"}


def _build_features(df: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Build 13 behavioral features for all customers active before snapshot.

    Mirrors churn.py:load_and_preprocess lines 48-87 exactly, parameterised
    on snapshot so the same logic produces both reference and current windows.
    """
    cutoff90 = snapshot - pd.Timedelta(days=90)
    cutoff30 = snapshot - pd.Timedelta(days=30)
    obs      = df[df["InvoiceDate"] < snapshot]

    records = []
    for cid, grp in obs.groupby("Customer ID"):
        inv_dates = grp.groupby("Invoice")["InvoiceDate"].min().sort_values()
        last_dt   = inv_dates.max()
        freq      = len(inv_dates)
        pos_rev   = grp.loc[grp["Revenue"] > 0, "Revenue"]
        monetary  = pos_rev.sum()
        recency   = (snapshot - last_dt).days
        inter     = inv_dates.diff().dt.days.dropna()  # type: ignore[union-attr]
        recent90  = grp[grp["InvoiceDate"] >= cutoff90]
        inv_qty   = grp.groupby("Invoice")["Quantity"].sum()

        records.append({
            "Customer ID":             cid,
            "recency":                 recency,
            "frequency":               freq,
            "monetary":                monetary,
            "tenure":                  max((last_dt - inv_dates.min()).days, 1),
            "avg_order_value":         monetary / max(freq, 1),
            "avg_qty_per_order":       inv_qty.mean(),
            "n_unique_products":       grp["StockCode"].nunique(),
            "avg_days_between_orders": float(inter.mean()) if freq > 1 else float(recency),
            "purchase_regularity":     float(inter.std(ddof=0)) if len(inter) > 0 else 0.0,
            "recent_freq_90d":         recent90["Invoice"].nunique(),
            "revenue_last_30d": (
                grp.loc[(grp["InvoiceDate"] >= cutoff30) & (grp["Revenue"] > 0), "Revenue"].sum()
            ),
            "n_months_active":         grp["InvoiceDate"].dt.to_period("M").nunique(),  # type: ignore[attr-defined]
            "spend_trend_90d": (
                grp.loc[(grp["InvoiceDate"] >= cutoff90) & (grp["Revenue"] > 0), "Revenue"].sum()
                / (grp.loc[(grp["InvoiceDate"] < cutoff90) & (grp["Revenue"] > 0), "Revenue"].sum() + 1.0)
            ),
        })

    return pd.DataFrame(records)


def build_churn_datasets(
    retail_path: Path = DATA_PROCESSED / "retail_clean.csv",
    rfm_path:    Path = DATA_PROCESSED / "rfm_scores.csv",
    model_path:  Path = MODELS_DIR     / "churn_xgboost.pkl",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reference, current) churn feature DataFrames.

    Reference: customers with activity before 2011-06-01 (matches training SNAPSHOT).
    Current:   same customers re-scored against a 2011-09-01 SNAPSHOT to simulate
               3 months of production drift.
    Both DataFrames include FEATURE_COLS + churn_probability + predicted_churn.
    rfm_path is used to filter to RFM-tracked customers and assign the churned label,
    mirroring churn.py:89-94.
    """
    df = pd.read_csv(
        retail_path,
        usecols=["Customer ID", "Invoice", "InvoiceDate", "Revenue", "StockCode", "Quantity"],
        parse_dates=["InvoiceDate"],
    )

    ref_raw = _build_features(df, REF_SNAPSHOT)
    cur_raw = _build_features(df, CUR_SNAPSHOT)

    seg = (
        pd.read_csv(rfm_path, usecols=["Customer ID", "Segment"])
        .set_index("Customer ID")["Segment"]
    )

    for feat_df in (ref_raw, cur_raw):
        feat_df["churned"] = feat_df["Customer ID"].map(
            lambda cid, s=seg: int(s.get(cid, "Unknown") in _CHURN_SEGMENTS)
        )

    ref_raw = ref_raw[ref_raw["Customer ID"].isin(seg.index)].reset_index(drop=True)
    cur_raw = cur_raw[cur_raw["Customer ID"].isin(seg.index)].reset_index(drop=True)

    model = joblib.load(model_path)
    for feat_df in (ref_raw, cur_raw):
        proba = model.predict_proba(feat_df[FEATURE_COLS].values)[:, 1]
        feat_df["churn_probability"] = proba
        feat_df["predicted_churn"]   = (proba >= 0.5).astype(int)

    return ref_raw, cur_raw


def build_revenue_datasets(
    rolling_path: Path = DATA_PROCESSED / "daily_revenue_rolling.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split daily revenue rolling stats at 2011-01-01 into reference and current."""
    df    = pd.read_csv(rolling_path, parse_dates=["Date"])
    df    = df.dropna(subset=["Revenue", "rolling_7d_mean", "rolling_30d_mean"])
    split = pd.Timestamp("2011-01-01")
    cols  = ["Revenue", "rolling_7d_mean", "rolling_30d_mean"]
    ref   = df[df["Date"] <  split][cols].reset_index(drop=True)
    cur   = df[df["Date"] >= split][cols].reset_index(drop=True)
    return ref, cur


def run_data_drift_report(
    ref_df:   pd.DataFrame,
    cur_df:   pd.DataFrame,
    out_path: Path,
) -> Tuple[Report, Path]:
    """DataDriftPreset over the 13 churn feature columns."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df[FEATURE_COLS], current_data=cur_df[FEATURE_COLS])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    return report, out_path


def run_target_drift_report(
    ref_df:   pd.DataFrame,
    cur_df:   pd.DataFrame,
    out_path: Path,
) -> Tuple[Report, Path]:
    """TargetDriftPreset with explicit ColumnMapping — required so Evidently does not
    guess wrong columns when the DataFrame contains both features and predictions."""
    column_mapping = ColumnMapping(
        target     = "predicted_churn",
        prediction = "churn_probability",
    )
    report = Report(metrics=[TargetDriftPreset()])
    report.run(
        reference_data=ref_df,
        current_data=cur_df,
        column_mapping=column_mapping,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    return report, out_path


def run_data_quality_report(
    ref_df:   pd.DataFrame,
    cur_df:   pd.DataFrame,
    out_path: Path,
) -> Tuple[Report, Path]:
    """DataQualityPreset over the 13 churn feature columns."""
    report = Report(metrics=[DataQualityPreset()])
    report.run(reference_data=ref_df[FEATURE_COLS], current_data=cur_df[FEATURE_COLS])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    return report, out_path


def run_revenue_drift_report(
    ref_rev:  pd.DataFrame,
    cur_rev:  pd.DataFrame,
    out_path: Path,
) -> Tuple[Report, Path]:
    """DataDriftPreset over Revenue + rolling mean columns."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_rev, current_data=cur_rev)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    return report, out_path


def _extract_drift_summary(report: Report) -> dict:
    """Extract share_of_drifted_columns and per-feature stats from a DataDrift report."""
    d     = report.as_dict()
    share = d["metrics"][0]["result"]["share_of_drifted_columns"]

    per_feature: dict = {}
    for m in d["metrics"]:
        if "drift_by_columns" in m.get("result", {}):
            for col, info in m["result"]["drift_by_columns"].items():
                per_feature[col] = {
                    "drift_detected": bool(info["drift_detected"]),
                    "drift_score":    float(info["drift_score"]),
                    "stattest":       info["stattest_name"],
                }

    return {"share_of_drifted_columns": float(share), "per_feature": per_feature}


def run_all(log_to_mlflow: bool = True) -> dict:
    """Orchestrate all four drift reports, print summary, and log to MLflow."""
    print("Building churn datasets  (reference: before 2011-06-01 | current: before 2011-09-01)...")
    ref_churn, cur_churn = build_churn_datasets()
    print(f"  reference: {len(ref_churn)} customers | current: {len(cur_churn)} customers")

    print("Building revenue datasets  (split at 2011-01-01)...")
    ref_rev, cur_rev = build_revenue_datasets()
    print(f"  reference: {len(ref_rev)} days | current: {len(cur_rev)} days")

    print("Running data drift report ...")
    r_data,    p_data    = run_data_drift_report(   ref_churn, cur_churn, DRIFT_DIR / "data_drift.html")

    print("Running target drift report ...")
    r_target,  p_target  = run_target_drift_report( ref_churn, cur_churn, DRIFT_DIR / "target_drift.html")

    print("Running data quality report ...")
    r_quality, p_quality = run_data_quality_report( ref_churn, cur_churn, DRIFT_DIR / "data_quality.html")

    print("Running revenue drift report ...")
    r_revenue, p_revenue = run_revenue_drift_report(ref_rev,   cur_rev,   DRIFT_DIR / "revenue_drift.html")

    data_summary   = _extract_drift_summary(r_data)
    rev_summary    = _extract_drift_summary(r_revenue)
    target_result  = r_target.as_dict()["metrics"][0]["result"]

    results = {
        "churn_feature_drift_share":   data_summary["share_of_drifted_columns"],
        "churn_target_drift_detected": int(target_result["drift_detected"]),
        "revenue_drift_share":         rev_summary["share_of_drifted_columns"],
        "per_feature_drift":           data_summary["per_feature"],
    }

    print("\n--- Drift Detection Summary ---")
    print(f"  {'churn_feature_drift_share':<38} {results['churn_feature_drift_share']:.4f}")
    print(f"  {'churn_target_drift_detected':<38} {results['churn_target_drift_detected']}")
    print(f"  {'revenue_drift_share':<38} {results['revenue_drift_share']:.4f}")
    print("\n  Per-feature churn drift:")
    for col, info in results["per_feature_drift"].items():
        flag = "DRIFT" if info["drift_detected"] else "ok   "
        print(f"    {flag}  {col:<32}  score={info['drift_score']:.4f}  test={info['stattest']}")

    if log_to_mlflow:
        mlflow.set_experiment("drift_detection")
        with mlflow.start_run(run_name="drift_monitoring_v1"):
            mlflow.log_metrics({
                "churn_feature_drift_share":   results["churn_feature_drift_share"],
                "churn_target_drift_detected": float(results["churn_target_drift_detected"]),
                "revenue_drift_share":         results["revenue_drift_share"],
            })
            for path in (p_data, p_target, p_quality, p_revenue):
                mlflow.log_artifact(str(path), artifact_path="drift_reports")

    return results
