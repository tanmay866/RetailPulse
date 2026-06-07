"""Export functionality: CSV downloads and PDF analytical reports."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import pandas as pd
import streamlit as st

from utils.auth import require_auth
from utils.audit_log import log_action
from utils.data_loader import (
    anonymize_customer_ids,
    load_churn_predictions,
    load_customer_segments,
    load_daily_revenue_rolling,
    load_inventory_recommendations,
    load_rfm_scores,
    load_retail_clean,
    load_segmentation_churn_merged,
)
from utils.pdf_report import (
    FPDF_OK as _FPDF_OK,
    build_customer_health_pdf,
    build_full_report_pdf,
    build_inventory_pdf,
    build_overview_pdf,
)


class _ReportDef(TypedDict):
    title: str
    desc: str
    bullets: list[str]
    fn: Callable[[], bytes]
    fname: str

ROOT        = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "reports" / "figures"

# ── Auth ──────────────────────────────────────────────────────────────────────
_user = require_auth()

# ── Page ─────────────────────────────────────────────────────────────────────
st.title("Export & Reports")
st.caption("Download processed datasets as CSV or generate analytical PDF reports.")


# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load() -> dict:
    return {
        "retail":  load_retail_clean(),
        "rolling": load_daily_revenue_rolling(),
        "rfm":     load_rfm_scores(),
        "seg":     load_customer_segments(),
        "churn":   load_churn_predictions(),
        "inv":     load_inventory_recommendations(),
        "merged":  load_segmentation_churn_merged(),
    }

with st.spinner("Loading datasets..."):
    _raw = _load()

# Anonymize Customer IDs in all customer-bearing datasets for non-admin users.
_CUSTOMER_KEYS = {"rfm", "seg", "churn", "merged"}
if _user["role"] == "admin":
    D = _raw
else:
    D = {
        k: anonymize_customer_ids(v) if k in _CUSTOMER_KEYS else v
        for k, v in _raw.items()
    }
    st.info("Customer IDs are anonymized. Contact an administrator for raw records.")

tab_csv, tab_pdf = st.tabs(["CSV Downloads", "PDF Reports"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — CSV Downloads
# ─────────────────────────────────────────────────────────────────────────────
with tab_csv:
    st.subheader("Processed Dataset Downloads")
    st.markdown(
        "Each file is the output of the latest pipeline run. "
        "Preview shows the first 3 rows."
    )

    _CSV_EXPORTS = [
        (
            "Customer Segments (RFM)",
            f"{len(D['rfm']):,} customers · RFM scores + segment labels",
            D["rfm"],
            "retailpulse_rfm_segments.csv",
        ),
        (
            "Churn Predictions",
            f"{len(D['churn']):,} customers · churn probability + risk flag",
            D["churn"],
            "retailpulse_churn_predictions.csv",
        ),
        (
            "Inventory Recommendations",
            f"{len(D['inv']):,} SKUs · EOQ, reorder point, days of stock, status",
            D["inv"],
            "retailpulse_inventory_recommendations.csv",
        ),
        (
            "Daily Revenue (rolling stats)",
            f"{len(D['rolling']):,} days · 7-day and 30-day rolling mean/std",
            D["rolling"],
            "retailpulse_daily_revenue.csv",
        ),
        (
            "Full Customer Profile",
            f"{len(D['merged']):,} customers · RFM + segments + churn merged",
            D["merged"],
            "retailpulse_customer_profile.csv",
        ),
    ]

    for i, (label, desc, df, fname) in enumerate(_CSV_EXPORTS):
        with st.container(border=True):
            c_info, c_prev, c_dl = st.columns([4, 4, 2])
            with c_info:
                st.markdown(f"**{label}**")
                st.caption(desc)
            with c_prev:
                st.dataframe(df.head(3), hide_index=True, height=112, use_container_width=True)
            with c_dl:
                raw = df.to_csv(index=False).encode("utf-8")
                if st.download_button(
                    "Download CSV",
                    data=raw,
                    file_name=fname,
                    mime="text/csv",
                    key=f"dl_csv_{i}",
                    use_container_width=True,
                ):
                    log_action(_user["username"], "csv_download", fname, f"{len(df)} rows")
                st.caption(f"{len(raw) / 1024:.0f} KB  ·  {len(df):,} rows  ·  {df.shape[1]} cols")

    st.divider()
    st.subheader("Custom Filtered Export")

    merged = D["merged"]
    cf1, cf2, cf3 = st.columns(3)
    with cf1:
        seg_opts = sorted(merged["Segment"].unique())
        seg_sel  = st.multiselect("Segments", seg_opts, default=seg_opts)
    with cf2:
        cp_min, cp_max = st.slider("Churn Probability Range", 0.0, 1.0, (0.0, 1.0), 0.05)
    with cf3:
        rec_max = st.slider(
            "Max Recency (days)", 1,
            int(merged["Recency"].max()),
            int(merged["Recency"].max()),
        )

    filt = merged[
        merged["Segment"].isin(seg_sel) &
        merged["churn_probability"].between(cp_min, cp_max) &
        (merged["Recency"] <= rec_max)
    ]
    st.info(f"**{len(filt):,}** customers match the current filters.")
    st.dataframe(filt.head(25), use_container_width=True, hide_index=True, height=260)
    if st.download_button(
        f"Download Filtered Export  ({len(filt):,} rows)",
        data=filt.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_custom_export.csv",
        mime="text/csv",
        use_container_width=True,
    ):
        log_action(
            _user["username"], "csv_download", "retailpulse_custom_export.csv",
            f"{len(filt)} rows · seg={seg_sel} churn=[{cp_min},{cp_max}]",
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PDF Reports
# ─────────────────────────────────────────────────────────────────────────────
with tab_pdf:
    if not _FPDF_OK:
        st.error(
            "PDF generation requires **fpdf2**. "
            "Install it with:  `pip install fpdf2`"
        )
        st.stop()

    _REPORTS: list[_ReportDef] = [
        {
            "title":   "Business Overview",
            "desc":    "Revenue KPIs, monthly trend, top 10 products, geographic breakdown.",
            "bullets": [
                "Total revenue, orders, customers, AOV",
                "Monthly revenue trend chart",
                "Top 10 products chart + table",
                "Revenue by country",
            ],
            "fn":    lambda: build_overview_pdf(D, FIGURES_DIR),
            "fname": "retailpulse_business_overview.pdf",
        },
        {
            "title":   "Customer Health",
            "desc":    "RFM segmentation, cluster profiles, churn predictions and SHAP analysis.",
            "bullets": [
                "Segment distribution & breakdown table",
                "RFM score distributions",
                "Churn rate & risk classification (XGBoost)",
                "SHAP feature importance plots",
            ],
            "fn":    lambda: build_customer_health_pdf(D, FIGURES_DIR),
            "fname": "retailpulse_customer_health.pdf",
        },
        {
            "title":   "Inventory Status",
            "desc":    "Stock status breakdown, EOQ analysis, and critical SKU alert table.",
            "bullets": [
                "Status breakdown (Optimal / Stockout / Overstock)",
                "Days of stock distribution",
                "Stockout risk heatmap",
                "Critical SKU table (top 20)",
            ],
            "fn":    lambda: build_inventory_pdf(D, FIGURES_DIR),
            "fname": "retailpulse_inventory_status.pdf",
        },
        {
            "title":   "Full Analytics Report",
            "desc":    "All four domains in a single multi-section PDF with a cover page.",
            "bullets": [
                "Cover page with report summary",
                "Business performance section",
                "Customer segmentation & churn section",
                "Inventory management section",
            ],
            "fn":    lambda: build_full_report_pdf(D, FIGURES_DIR),
            "fname": "retailpulse_full_report.pdf",
        },
    ]

    st.subheader("Generate PDF Reports")
    cols = st.columns(2)
    for i, rpt in enumerate(_REPORTS):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"### {rpt['title']}")
                st.caption(rpt["desc"])
                st.markdown("\n".join(f"- {b}" for b in rpt["bullets"]))
                if st.button(
                    f"Generate {rpt['title']} PDF",
                    key=f"gen_pdf_{i}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Building {rpt['title']} report..."):
                        try:
                            pdf_bytes = rpt["fn"]()
                            st.session_state[f"pdf_{i}"] = pdf_bytes
                            log_action(
                                _user["username"], "pdf_generate",
                                rpt["fname"], rpt["title"],
                            )
                        except Exception as exc:
                            st.error(f"PDF generation failed: {exc}")
                            st.session_state.pop(f"pdf_{i}", None)

                if f"pdf_{i}" in st.session_state:
                    pdf_bytes = st.session_state[f"pdf_{i}"]
                    if st.download_button(
                        label=f"Download {rpt['title']} PDF",
                        data=pdf_bytes,
                        file_name=rpt["fname"],
                        mime="application/pdf",
                        key=f"dl_pdf_{i}",
                        use_container_width=True,
                    ):
                        log_action(
                            _user["username"], "pdf_download",
                            rpt["fname"], f"{len(pdf_bytes) // 1024} KB",
                        )
                    st.caption(f"Ready  -  {len(pdf_bytes) / 1024:.0f} KB")
