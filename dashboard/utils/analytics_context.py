"""Build a structured data context string for LLM-powered analytics queries."""
from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data(ttl=300)
def build_context() -> str:
    """Load all processed datasets and return a structured plain-text summary."""
    from utils.data_loader import (
        load_retail_clean,
        load_daily_revenue_rolling,
        load_rfm_scores,
        load_churn_predictions,
        load_inventory_recommendations,
    )

    sections: list[str] = []

    # --- Business Overview ---
    try:
        df = load_retail_clean()
        total_rev = df["TotalPrice"].sum()
        n_customers = df["Customer ID"].nunique()
        n_orders = df["Invoice"].nunique()
        date_min = df["InvoiceDate"].min().strftime("%Y-%m-%d")
        date_max = df["InvoiceDate"].max().strftime("%Y-%m-%d")
        avg_order = df.groupby("Invoice")["TotalPrice"].sum().mean()
        sections.append(
            f"--- BUSINESS OVERVIEW ---\n"
            f"Date Range: {date_min} to {date_max}\n"
            f"Total Revenue: Rs {total_rev:,.0f}\n"
            f"Total Unique Customers: {n_customers:,}\n"
            f"Total Orders: {n_orders:,}\n"
            f"Average Order Value: Rs {avg_order:,.2f}"
        )
    except Exception as exc:
        sections.append(f"--- BUSINESS OVERVIEW ---\n[Unavailable: {exc}]")

    # --- Revenue Trend ---
    try:
        rev = load_daily_revenue_rolling()
        # Find the revenue column (first numeric non-Date column)
        rev_col = next(
            (c for c in rev.columns if c != "Date" and pd.api.types.is_numeric_dtype(rev[c])),
            None,
        )
        if rev_col and "Date" in rev.columns:
            rev = rev.sort_values("Date")
            cutoff = rev["Date"].max()
            last30 = rev[rev["Date"] > cutoff - pd.Timedelta(days=30)][rev_col].sum()
            prior30 = rev[
                (rev["Date"] <= cutoff - pd.Timedelta(days=30))
                & (rev["Date"] > cutoff - pd.Timedelta(days=60))
            ][rev_col].sum()
            chg = (last30 - prior30) / prior30 * 100 if prior30 > 0 else 0
            avg_daily = rev.tail(30)[rev_col].mean()
            sections.append(
                f"--- REVENUE TREND (Last 30 Days) ---\n"
                f"Last 30-Day Revenue: Rs {last30:,.0f}\n"
                f"Prior 30-Day Revenue: Rs {prior30:,.0f}\n"
                f"30-Day Change: {chg:+.1f}%\n"
                f"Avg Daily Revenue (30d): Rs {avg_daily:,.0f}"
            )
    except Exception as exc:
        sections.append(f"--- REVENUE TREND ---\n[Unavailable: {exc}]")

    # --- Customer Segments ---
    try:
        rfm = load_rfm_scores()
        if "Segment" in rfm.columns:
            counts = rfm["Segment"].value_counts()
            total = len(rfm)
            lines = "\n".join(
                f"  {seg}: {n:,} customers ({n / total * 100:.1f}%)"
                for seg, n in counts.items()
            )
            # Avg RFM scores per segment if available
            score_info = ""
            if {"Recency", "Frequency", "Monetary"}.issubset(rfm.columns):
                top_seg = counts.index[0]
                seg_mean = rfm[rfm["Segment"] == top_seg][["Recency", "Frequency", "Monetary"]].mean()
                score_info = (
                    f"\nTop segment ({top_seg}) avg: "
                    f"Recency {seg_mean['Recency']:.0f}d, "
                    f"Frequency {seg_mean['Frequency']:.1f}, "
                    f"Monetary Rs {seg_mean['Monetary']:,.0f}"
                )
            sections.append(
                f"--- CUSTOMER SEGMENTS (RFM) ---\n"
                f"Total Segmented: {total:,}\n"
                f"Segments:\n{lines}{score_info}"
            )
    except Exception as exc:
        sections.append(f"--- CUSTOMER SEGMENTS ---\n[Unavailable: {exc}]")

    # --- Churn Analysis ---
    try:
        churn = load_churn_predictions()
        pred_col = next((c for c in ["Churn_Predicted", "churn_predicted"] if c in churn.columns), None)
        prob_col = next((c for c in ["Churn_Probability", "churn_probability"] if c in churn.columns), None)
        if pred_col:
            total = len(churn)
            rate = churn[pred_col].mean() * 100
            high_risk = (churn[prob_col] > 0.7).sum() if prob_col else churn[pred_col].sum()
            lines = [
                f"Total Analyzed: {total:,}",
                f"Predicted Churn Rate: {rate:.1f}%",
                f"High-Risk Customers (p > 70%): {high_risk:,} ({high_risk / total * 100:.1f}%)",
            ]
            if prob_col:
                avg_prob = churn[prob_col].mean()
                lines.append(f"Avg Churn Probability: {avg_prob:.1%}")
            sections.append("--- CHURN ANALYSIS ---\n" + "\n".join(lines))
    except Exception as exc:
        sections.append(f"--- CHURN ANALYSIS ---\n[Unavailable: {exc}]")

    # --- Inventory Status ---
    try:
        inv = load_inventory_recommendations()
        if "Status" in inv.columns:
            counts = inv["Status"].value_counts()
            total = len(inv)
            lines = "\n".join(
                f"  {status}: {n:,} SKUs ({n / total * 100:.1f}%)"
                for status, n in counts.items()
            )
            sections.append(
                f"--- INVENTORY STATUS ---\n"
                f"Total SKUs: {total:,}\n"
                f"Status Breakdown:\n{lines}"
            )
    except Exception as exc:
        sections.append(f"--- INVENTORY STATUS ---\n[Unavailable: {exc}]")

    # --- CLV Predictions ---
    try:
        from utils.data_loader import load_clv_predictions

        clv = load_clv_predictions()
        if "clv_12m" in clv.columns:
            total_clv = clv["clv_12m"].sum()
            avg_clv = clv["clv_12m"].mean()
            lines = [
                f"Customers with Predictions: {len(clv):,}",
                f"Total 12-Month CLV: Rs {total_clv:,.0f}",
                f"Avg CLV per Customer: Rs {avg_clv:,.2f}",
            ]
            if "clv_segment" in clv.columns:
                seg_counts = clv["clv_segment"].value_counts()
                for seg, n in seg_counts.items():
                    lines.append(f"  {seg}: {n:,} ({n / len(clv) * 100:.1f}%)")
            if "prob_alive" in clv.columns:
                lines.append(f"Avg P(Alive): {clv['prob_alive'].mean():.1%}")
            sections.append("--- CLV PREDICTIONS (12-Month) ---\n" + "\n".join(lines))
    except Exception:
        pass  # CLV pipeline may not have run yet

    return "=== RETAILPULSE ANALYTICS CONTEXT ===\n\n" + "\n\n".join(sections)
