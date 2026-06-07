"""Customer Lifetime Value prediction using BG/NBD + Gamma-Gamma models.

Pipeline:
  1. Aggregate retail_clean.csv to invoice-level transactions.
  2. Build lifetimes summary (frequency, recency, T, monetary_value).
  3. Fit BetaGeoFitter (BG/NBD) — models purchase frequency and "alive" probability.
  4. Fit GammaGammaFitter — models average transaction value for repeat buyers.
  5. Compute 12-month CLV = expected purchases × expected transaction value (discounted).
  6. Assign CLV segments (High / Medium / Low Value) and save predictions CSV.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

warnings.filterwarnings("ignore")

ROOT           = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR     = ROOT / "models"

MLFLOW_EXPERIMENT = "retailpulse-clv"
MLFLOW_RUN_NAME   = "bgf-ggf-v1"


# ── Step 1: Build transaction summary ────────────────────────────────────────
def build_transaction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line items -> invoice level, then build lifetimes summary table."""
    txn = (
        df[df["Revenue"] > 0]
        .groupby(["Customer ID", "Invoice", "InvoiceDate"])["Revenue"]
        .sum()
        .reset_index()
    )
    obs_end = txn["InvoiceDate"].max()
    summary = summary_data_from_transaction_data(
        txn,
        customer_id_col="Customer ID",
        datetime_col="InvoiceDate",
        monetary_value_col="Revenue",
        observation_period_end=obs_end,
        freq="D",
    )
    # Remove customers with zero or negative monetary value (returns/cancels artefacts)
    summary = summary[summary["monetary_value"] > 0]
    return summary


# ── Step 2: Train BG/NBD ─────────────────────────────────────────────────────
def train_bgf(summary: pd.DataFrame) -> BetaGeoFitter:
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])
    return bgf


# ── Step 3: Train Gamma-Gamma ─────────────────────────────────────────────────
def train_ggf(summary: pd.DataFrame) -> GammaGammaFitter:
    repeat = summary[summary["frequency"] > 0]
    ggf = GammaGammaFitter(penalizer_coef=0.0)
    ggf.fit(repeat["frequency"], repeat["monetary_value"])
    return ggf


# ── Step 4: Generate predictions ─────────────────────────────────────────────
def predict_clv(
    summary: pd.DataFrame,
    bgf: BetaGeoFitter,
    ggf: GammaGammaFitter,
    clv_months: int = 12,
    monthly_discount_rate: float = 0.01,
) -> pd.DataFrame:
    """Return a DataFrame of CLV predictions for every customer."""
    df = summary.copy()

    # Probability customer is still active
    df["prob_alive"] = bgf.conditional_probability_alive(
        df["frequency"], df["recency"], df["T"]
    )

    # Predicted transactions in 90 / 180 / 365 days
    for days in [90, 180, 365]:
        df[f"pred_purchases_{days}d"] = bgf.conditional_expected_number_of_purchases_up_to_time(
            days, df["frequency"], df["recency"], df["T"]
        )

    # CLV only for repeat buyers (Gamma-Gamma requirement)
    repeat_mask = df["frequency"] > 0
    clv_series = ggf.customer_lifetime_value(
        bgf,
        df.loc[repeat_mask, "frequency"],
        df.loc[repeat_mask, "recency"],
        df.loc[repeat_mask, "T"],
        df.loc[repeat_mask, "monetary_value"],
        time=clv_months,
        freq="D",
        discount_rate=monthly_discount_rate,
    )
    df["clv_12m"] = np.nan
    df.loc[repeat_mask, "clv_12m"] = clv_series.values

    # CLV segment by tertile (repeat buyers only)
    repeat_clv = df.loc[repeat_mask, "clv_12m"]
    q33 = repeat_clv.quantile(0.33)
    q67 = repeat_clv.quantile(0.67)
    df["clv_segment"] = pd.NA
    df.loc[repeat_mask, "clv_segment"] = pd.cut(
        repeat_clv,
        bins=[-np.inf, q33, q67, np.inf],
        labels=["Low Value", "Medium Value", "High Value"],
    ).astype(str)

    return df.reset_index()


# ── Step 5: Main pipeline ─────────────────────────────────────────────────────
def run() -> None:
    print("Loading retail data…")
    raw = pd.read_csv(
        DATA_PROCESSED / "retail_clean.csv",
        usecols=["Customer ID", "Invoice", "InvoiceDate", "Revenue"],
        parse_dates=["InvoiceDate"],
    )

    print("Building transaction summary…")
    summary = build_transaction_summary(raw)
    print(f"  {len(summary):,} customers in summary  "
          f"({(summary['frequency'] > 0).sum():,} repeat buyers)")

    print("Training BG/NBD model…")
    bgf = train_bgf(summary)

    print("Training Gamma-Gamma model…")
    ggf = train_ggf(summary)

    print("Computing CLV predictions…")
    predictions = predict_clv(summary, bgf, ggf)

    # ── Save predictions ──────────────────────────────────────────────────────
    out_path = DATA_PROCESSED / "clv_predictions.csv"
    predictions.to_csv(out_path, index=False)
    print(f"Saved predictions -> {out_path}  ({len(predictions):,} rows)")

    # ── MLflow logging ────────────────────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=MLFLOW_RUN_NAME):
        repeat = predictions[predictions["frequency"] > 0]
        mlflow.log_params({
            "bgf_penalizer":     0.01,
            "ggf_penalizer":     0.0,
            "clv_months":        12,
            "discount_rate":     0.01,
            "total_customers":   len(predictions),
            "repeat_buyers":     len(repeat),
        })
        mlflow.log_metrics({
            "mean_clv_12m":         repeat["clv_12m"].mean(),
            "median_clv_12m":       repeat["clv_12m"].median(),
            "total_projected_clv":  repeat["clv_12m"].sum(),
            "mean_prob_alive":      predictions["prob_alive"].mean(),
            "pct_high_value":       (predictions["clv_segment"] == "High Value").mean(),
        })
        mlflow.log_artifact(str(out_path))

    print("MLflow run logged.")
    print("\nCLV summary:")
    print(f"  Total projected 12m CLV : Rs {repeat['clv_12m'].sum():,.0f}")
    print(f"  Avg CLV per customer    : Rs {repeat['clv_12m'].mean():,.0f}")
    print(f"  High-value customers    : {(predictions['clv_segment'] == 'High Value').sum():,}")
    print(f"  Avg probability alive   : {predictions['prob_alive'].mean():.1%}")
