"""Day 20 - Export functionality: CSV downloads and PDF analytical reports."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import pandas as pd
import streamlit as st

from utils.data_loader import (
    load_churn_predictions,
    load_customer_segments,
    load_daily_revenue_rolling,
    load_inventory_recommendations,
    load_rfm_scores,
    load_retail_clean,
    load_segmentation_churn_merged,
)

ROOT        = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "reports" / "figures"

try:
    from fpdf import FPDF
    _FPDF_OK = True
except ImportError:
    _FPDF_OK = False

# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Export & Reports", layout="wide")
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

with st.spinner("Loading datasets…"):
    D = _load()

tab_csv, tab_pdf = st.tabs(["CSV Downloads", "PDF Reports"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 - CSV Downloads
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
                st.download_button(
                    "Download CSV",
                    data=raw,
                    file_name=fname,
                    mime="text/csv",
                    key=f"dl_csv_{i}",
                    use_container_width=True,
                )
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
    st.download_button(
        f"Download Filtered Export  ({len(filt):,} rows)",
        data=filt.to_csv(index=False).encode("utf-8"),
        file_name="retailpulse_custom_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 - PDF Reports
# ─────────────────────────────────────────────────────────────────────────────
with tab_pdf:
    if not _FPDF_OK:
        st.error(
            "PDF generation requires **fpdf2**. "
            "Install it with:  `pip install fpdf2`"
        )
        st.stop()

    # ── PDF helper class ──────────────────────────────────────────────────────
    class _PDF(FPDF):
        _BLUE  = (30, 136, 229)
        _DARK  = (33,  33,  33)
        _GRAY  = (110, 110, 110)
        _LIGHT = (245, 245, 245)

        def header(self):
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*self._GRAY)
            self.cell(130, 6, "RetailPulse Analytics Platform", new_x="RIGHT", new_y="TOP")
            self.cell(0, 6, f"Generated {pd.Timestamp.now().strftime('%d %b %Y')}", align="R", new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*self._BLUE)
            self.set_line_width(0.4)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.set_draw_color(0, 0, 0)
            self.set_line_width(0.2)
            self.ln(3)
            self.set_text_color(0, 0, 0)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(*self._GRAY)
            self.cell(0, 8, f"Page {self.page_no()}  |  RetailPulse - Confidential", align="C")

        def report_title(self, title: str, subtitle: str = ""):
            self.set_font("Helvetica", "B", 20)
            self.set_text_color(*self._BLUE)
            self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
            if subtitle:
                self.set_font("Helvetica", "", 9)
                self.set_text_color(*self._GRAY)
                self.cell(0, 6, subtitle, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.ln(4)

        def section(self, title: str):
            if self.get_y() > self.h - self.b_margin - 20:
                self.add_page()
            self.set_font("Helvetica", "B", 11)
            self.set_fill_color(*self._BLUE)
            self.set_text_color(255, 255, 255)
            self.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)
            self.ln(3)

        def body(self, text: str):
            self.set_font("Helvetica", "", 9)
            self.set_text_color(60, 60, 60)
            self.multi_cell(0, 5, text)
            self.set_text_color(0, 0, 0)
            self.ln(2)

        def kpi_row(self, items: list[tuple[str, str]]):
            n     = len(items)
            gap   = 2
            avail = self.w - self.l_margin - self.r_margin
            cw    = (avail - gap * (n - 1)) / n
            h_box = 16
            y0    = self.get_y()
            self.set_draw_color(200, 200, 200)
            for i, (label, value) in enumerate(items):
                x = self.l_margin + i * (cw + gap)
                self.set_fill_color(*self._LIGHT)
                self.rect(x, y0, cw, h_box, style="FD")
                self.set_xy(x + 2, y0 + 1)
                self.set_font("Helvetica", "", 7)
                self.set_text_color(*self._GRAY)
                self.cell(cw - 4, 5, label)
                self.set_xy(x + 2, y0 + 7)
                self.set_font("Helvetica", "B", 10)
                self.set_text_color(*self._DARK)
                self.cell(cw - 4, 7, value)
            self.set_xy(self.l_margin, y0 + h_box + 3)
            self.set_draw_color(0, 0, 0)
            self.set_text_color(0, 0, 0)

        def data_table(
            self,
            headers: list[str],
            rows: list[list],
            col_widths: list[float],
        ):
            self.set_font("Helvetica", "B", 8)
            self.set_fill_color(*self._BLUE)
            self.set_text_color(255, 255, 255)
            for h, cw in zip(headers, col_widths):
                self.cell(cw, 6, str(h), fill=True)
            self.ln()
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*self._DARK)
            for j, row in enumerate(rows):
                if self.get_y() > self.h - self.b_margin - 12:
                    self.add_page()
                self.set_fill_color(248, 248, 248) if j % 2 == 0 else self.set_fill_color(255, 255, 255)
                for val, cw in zip(row, col_widths):
                    self.cell(cw, 5, str(val)[:45], fill=True)
                self.ln()
            self.set_text_color(0, 0, 0)
            self.ln(4)

        def chart(self, path: Path, caption: str = "", w_pct: float = 0.95):
            if not path.exists():
                return
            avail = self.w - self.l_margin - self.r_margin
            img_w = avail * w_pct
            if self.get_y() + 120 > self.h - self.b_margin:
                self.add_page()
            self.image(str(path), x=self.l_margin, w=img_w)
            if caption:
                self.set_font("Helvetica", "I", 7)
                self.set_text_color(*self._GRAY)
                self.cell(0, 5, caption, align="C", new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(0, 0, 0)
            self.ln(3)

    # ── Report builder: Business Overview ────────────────────────────────────
    def _overview_pdf() -> bytes:
        retail  = D["retail"]
        rolling = D["rolling"]
        pdf = _PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        dr = (
            f"{rolling['Date'].min().strftime('%d %b %Y')} "
            f"- {rolling['Date'].max().strftime('%d %b %Y')}"
        )
        pdf.report_title("Business Overview Report", f"Data period: {dr}")

        pdf.section("Key Performance Indicators")
        total_rev    = retail["Revenue"].sum()
        uniq_cust    = retail["Customer ID"].nunique()
        total_orders = retail["Invoice"].nunique()
        aov          = retail.groupby("Invoice")["Revenue"].sum().mean()
        pdf.kpi_row([
            ("Total Revenue",      f"GBP {total_rev:,.0f}"),
            ("Unique Customers",   f"{uniq_cust:,}"),
            ("Total Orders",       f"{total_orders:,}"),
            ("Avg Order Value",    f"GBP {aov:,.2f}"),
        ])
        pdf.kpi_row([
            ("Transaction Lines",  f"{len(retail):,}"),
            ("Avg Qty / Line",     f"{retail['Quantity'].mean():.1f}"),
            ("Countries",          f"{retail['Country'].nunique()}"),
            ("Date Range",         dr),
        ])

        pdf.section("Revenue Trend")
        pdf.chart(FIGURES_DIR / "monthly_sales_trend.png", "Fig 1 - Monthly Revenue Trend")
        pdf.chart(FIGURES_DIR / "rolling_statistics.png",  "Fig 2 - Daily Revenue with 7-Day and 30-Day Rolling Statistics")

        pdf.section("Top Products by Revenue")
        pdf.chart(FIGURES_DIR / "top_products.png", "Fig 3 - Top 10 Products by Revenue")
        top_prods = (
            retail.groupby("Description")["Revenue"]
            .sum().nlargest(10).reset_index()
        )
        pdf.data_table(
            ["#", "Product", "Revenue (GBP)"],
            [[i + 1, row["Description"][:55], f"{row['Revenue']:,.0f}"]
             for i, (_, row) in enumerate(top_prods.iterrows())],
            col_widths=[10, 135, 45],
        )

        pdf.section("Revenue by Country")
        pdf.chart(FIGURES_DIR / "top_countries.png", "Fig 4 - Top Countries by Revenue")

        return bytes(pdf.output())

    # ── Report builder: Customer Health ──────────────────────────────────────
    def _customer_health_pdf() -> bytes:
        rfm   = D["rfm"]
        churn = D["churn"]
        pdf   = _PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.report_title("Customer Health Report", f"{len(rfm):,} customers analysed")

        # RFM segmentation
        pdf.section("Customer Segmentation (RFM)")
        pdf.kpi_row([
            ("Total Customers",     f"{len(rfm):,}"),
            ("Segments",            f"{rfm['Segment'].nunique()}"),
            ("Avg Recency (days)",  f"{rfm['Recency'].mean():.0f}"),
            ("Avg Monetary (GBP)",  f"{rfm['Monetary'].mean():,.0f}"),
        ])
        pdf.chart(FIGURES_DIR / "rfm_segments.png",      "Fig 1 - Customer Segment Distribution")
        pdf.chart(FIGURES_DIR / "rfm_distributions.png", "Fig 2 - RFM Score Distributions")
        pdf.chart(FIGURES_DIR / "cluster_profiles.png",  "Fig 3 - Segment Profiles")

        seg_counts = rfm["Segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        seg_counts["Share"] = (seg_counts["Count"] / len(rfm) * 100).round(1)
        pdf.section("Segment Breakdown")
        pdf.data_table(
            ["Segment", "Customers", "Share (%)"],
            [[r["Segment"], f"{r['Count']:,}", f"{r['Share']:.1f}%"]
             for _, r in seg_counts.iterrows()],
            col_widths=[90, 55, 45],
        )

        # Churn analysis
        pdf.section("Churn Prediction (XGBoost + Optuna)")
        churn_rate = (churn["predicted_churn"] == 1).mean() * 100
        high_risk  = int((churn["churn_probability"] > 0.80).sum())
        med_risk   = int(
            ((churn["churn_probability"] > 0.50) & (churn["churn_probability"] <= 0.80)).sum()
        )
        pdf.kpi_row([
            ("Predicted Churn Rate", f"{churn_rate:.1f}%"),
            ("High Risk  (p > 80%)", f"{high_risk:,}"),
            ("Med Risk   (p > 50%)", f"{med_risk:,}"),
            ("Model AUC-ROC",        "0.9278"),
        ])
        pdf.chart(FIGURES_DIR / "churn_roc_curve.png",        "Fig 4 - ROC Curve")
        pdf.chart(FIGURES_DIR / "churn_confusion_matrix.png", "Fig 5 - Confusion Matrix")
        pdf.chart(FIGURES_DIR / "churn_shap_bar.png",         "Fig 6 - Feature Importance (SHAP)")
        pdf.chart(FIGURES_DIR / "churn_shap_summary.png",     "Fig 7 - SHAP Summary Plot")

        return bytes(pdf.output())

    # ── Report builder: Inventory Status ─────────────────────────────────────
    def _inventory_pdf() -> bytes:
        inv = D["inv"]
        pdf = _PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.report_title("Inventory Status Report", f"{len(inv):,} SKUs analysed")

        status_counts = inv["status"].value_counts()
        stockout_n    = int(status_counts.get("STOCKOUT_RISK", 0))
        optimal_n     = int(status_counts.get("OPTIMAL",       0))
        overstock_n   = int(status_counts.get("OVERSTOCK",     0))
        critical_n    = int((inv["days_of_stock"] < 1).sum())

        pdf.section("Inventory Status Overview")
        pdf.kpi_row([
            ("Total SKUs",         f"{len(inv):,}"),
            ("Optimal",            f"{optimal_n:,}"),
            ("Stockout Risk",      f"{stockout_n:,}"),
            ("Overstock",          f"{overstock_n:,}"),
        ])
        pdf.kpi_row([
            ("Critical (< 1 day)", f"{critical_n:,}"),
            ("Stockout Rate",      f"{stockout_n / len(inv) * 100:.0f}%"),
            ("Avg Days of Stock",  f"{inv['days_of_stock'].mean():.1f}"),
            ("Avg EOQ",            f"{inv['eoq'].mean():.0f}"),
        ])

        pdf.section("Status Charts")
        pdf.chart(FIGURES_DIR / "inventory_status_breakdown.png", "Fig 1 - Status Breakdown")
        pdf.chart(FIGURES_DIR / "inventory_days_of_stock.png",    "Fig 2 - Days of Stock Distribution")
        pdf.chart(FIGURES_DIR / "inventory_stockout_heatmap.png", "Fig 3 - Stockout Risk Heatmap")
        pdf.chart(FIGURES_DIR / "inventory_eoq_vs_ordered.png",   "Fig 4 - EOQ vs Current Order Quantity")

        pdf.section("Critical SKUs  (Days of Stock < 7)")
        critical_df = (
            inv[inv["days_of_stock"] < 7]
            .sort_values("days_of_stock")
            .head(20)
        )
        if critical_df.empty:
            pdf.body("No SKUs with fewer than 7 days of stock.")
        else:
            pdf.data_table(
                ["Store", "Product", "Category", "Days Left", "Status", "To Order"],
                [
                    [
                        r["store_id"],
                        r["product_id"],
                        r["category"],
                        f"{r['days_of_stock']:.1f}",
                        r["status"],
                        f"{r['units_to_order']:.0f}",
                    ]
                    for _, r in critical_df.iterrows()
                ],
                col_widths=[24, 28, 35, 24, 40, 30],
            )

        return bytes(pdf.output())

    # ── Report builder: Full Analytics Report ────────────────────────────────
    def _full_report_pdf() -> bytes:
        retail  = D["retail"]
        rolling = D["rolling"]
        rfm     = D["rfm"]
        churn   = D["churn"]
        inv     = D["inv"]
        pdf     = _PDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        dr = (
            f"{rolling['Date'].min().strftime('%d %b %Y')} "
            f"- {rolling['Date'].max().strftime('%d %b %Y')}"
        )

        # Cover page
        pdf.add_page()
        pdf.ln(30)
        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(30, 136, 229)
        pdf.cell(0, 14, "RetailPulse", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(33, 33, 33)
        pdf.cell(0, 10, "Full Analytics Report", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(110, 110, 110)
        pdf.cell(0, 7, f"Data period: {dr}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(
            0, 7,
            f"Generated: {pd.Timestamp.now().strftime('%d %B %Y')}",
            align="C", new_x="LMARGIN", new_y="NEXT",
        )
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(80, 80, 80)
        for line in [
            "This report covers four analytical domains:",
            "  1. Business Performance - revenue KPIs, trends, top products",
            "  2. Customer Segmentation - RFM scoring, segment profiles",
            "  3. Churn Prediction - XGBoost model, SHAP feature importance",
            "  4. Inventory Management - stock status, EOQ, reorder recommendations",
        ]:
            pdf.cell(0, 6, line, align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        # Section 1 - Business
        pdf.add_page()
        pdf.report_title("1  -  Business Performance")
        total_rev    = retail["Revenue"].sum()
        uniq_cust    = retail["Customer ID"].nunique()
        total_orders = retail["Invoice"].nunique()
        aov          = retail.groupby("Invoice")["Revenue"].sum().mean()
        pdf.kpi_row([
            ("Total Revenue",    f"GBP {total_rev:,.0f}"),
            ("Unique Customers", f"{uniq_cust:,}"),
            ("Total Orders",     f"{total_orders:,}"),
            ("Avg Order Value",  f"GBP {aov:,.2f}"),
        ])
        pdf.section("Revenue Trend")
        pdf.chart(FIGURES_DIR / "monthly_sales_trend.png", "Monthly Revenue Trend")
        pdf.chart(FIGURES_DIR / "rolling_statistics.png",  "Daily Revenue with Rolling Statistics")
        pdf.section("Top Products & Geography")
        pdf.chart(FIGURES_DIR / "top_products.png",  "Top Products by Revenue")
        pdf.chart(FIGURES_DIR / "top_countries.png", "Revenue by Country")

        # Section 2 - Segmentation
        pdf.add_page()
        pdf.report_title("2  -  Customer Segmentation")
        pdf.kpi_row([
            ("Total Customers",    f"{len(rfm):,}"),
            ("Segments",           f"{rfm['Segment'].nunique()}"),
            ("Avg Recency",        f"{rfm['Recency'].mean():.0f} days"),
            ("Avg Monetary (GBP)", f"{rfm['Monetary'].mean():,.0f}"),
        ])
        pdf.section("Segment Charts")
        pdf.chart(FIGURES_DIR / "rfm_segments.png",    "Segment Distribution")
        pdf.chart(FIGURES_DIR / "cluster_profiles.png", "Cluster Profiles")
        pdf.chart(FIGURES_DIR / "kmeans_scatter.png",   "K-Means Cluster Scatter")

        # Section 3 - Churn
        pdf.add_page()
        pdf.report_title("3  -  Churn Prediction")
        churn_rate = (churn["predicted_churn"] == 1).mean() * 100
        high_risk  = int((churn["churn_probability"] > 0.80).sum())
        pdf.kpi_row([
            ("Predicted Churn Rate", f"{churn_rate:.1f}%"),
            ("High Risk (p > 80%)",  f"{high_risk:,}"),
            ("Model",                "XGBoost + Optuna"),
            ("AUC-ROC",              "0.9278"),
        ])
        pdf.section("Model Performance")
        pdf.chart(FIGURES_DIR / "churn_roc_curve.png",        "ROC Curve")
        pdf.chart(FIGURES_DIR / "churn_confusion_matrix.png", "Confusion Matrix")
        pdf.section("Feature Importance (SHAP)")
        pdf.chart(FIGURES_DIR / "churn_shap_bar.png",     "SHAP Feature Importance")
        pdf.chart(FIGURES_DIR / "churn_shap_summary.png", "SHAP Summary Plot")

        # Section 4 - Inventory
        pdf.add_page()
        pdf.report_title("4  -  Inventory Management")
        status_counts = inv["status"].value_counts()
        stockout_n = int(status_counts.get("STOCKOUT_RISK", 0))
        optimal_n  = int(status_counts.get("OPTIMAL",       0))
        pdf.kpi_row([
            ("Total SKUs",     f"{len(inv):,}"),
            ("Optimal",        f"{optimal_n:,}"),
            ("Stockout Risk",  f"{stockout_n:,}"),
            ("Stockout Rate",  f"{stockout_n / len(inv) * 100:.0f}%"),
        ])
        pdf.section("Inventory Charts")
        pdf.chart(FIGURES_DIR / "inventory_status_breakdown.png", "Status Breakdown")
        pdf.chart(FIGURES_DIR / "inventory_days_of_stock.png",    "Days of Stock Distribution")
        pdf.chart(FIGURES_DIR / "inventory_eoq_vs_ordered.png",   "EOQ vs Ordered Quantity")

        return bytes(pdf.output())

    # ── Report card UI ────────────────────────────────────────────────────────
    st.subheader("Generate PDF Reports")

    _REPORTS = [
        {
            "title":   "Business Overview",
            "icon":    "📊",
            "desc":    "Revenue KPIs, monthly trend, top 10 products, geographic breakdown.",
            "bullets": [
                "Total revenue, orders, customers, AOV",
                "Monthly revenue trend chart",
                "Top 10 products chart + table",
                "Revenue by country",
            ],
            "fn":    _overview_pdf,
            "fname": "retailpulse_business_overview.pdf",
        },
        {
            "title":   "Customer Health",
            "icon":    "👥",
            "desc":    "RFM segmentation, cluster profiles, churn predictions and SHAP analysis.",
            "bullets": [
                "Segment distribution & breakdown table",
                "RFM score distributions",
                "Churn rate & risk classification (XGBoost)",
                "SHAP feature importance plots",
            ],
            "fn":    _customer_health_pdf,
            "fname": "retailpulse_customer_health.pdf",
        },
        {
            "title":   "Inventory Status",
            "icon":    "📦",
            "desc":    "Stock status breakdown, EOQ analysis, and critical SKU alert table.",
            "bullets": [
                "Status breakdown (Optimal / Stockout / Overstock)",
                "Days of stock distribution",
                "Stockout risk heatmap",
                "Critical SKU table (top 20)",
            ],
            "fn":    _inventory_pdf,
            "fname": "retailpulse_inventory_status.pdf",
        },
        {
            "title":   "Full Analytics Report",
            "icon":    "📋",
            "desc":    "All four domains in a single multi-section PDF with a cover page.",
            "bullets": [
                "Cover page with report summary",
                "Business performance section",
                "Customer segmentation & churn section",
                "Inventory management section",
            ],
            "fn":    _full_report_pdf,
            "fname": "retailpulse_full_report.pdf",
        },
    ]

    cols = st.columns(2)
    for i, rpt in enumerate(_REPORTS):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"### {rpt['icon']}  {rpt['title']}")
                st.caption(rpt["desc"])
                for b in rpt["bullets"]:
                    st.markdown(f"- {b}")
                if st.button(
                    f"Generate  {rpt['title']}  PDF",
                    key=f"gen_pdf_{i}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Building {rpt['title']} report…"):
                        try:
                            pdf_bytes = rpt["fn"]()
                            st.download_button(
                                label=f"Download  {rpt['title']}  PDF",
                                data=pdf_bytes,
                                file_name=rpt["fname"],
                                mime="application/pdf",
                                key=f"dl_pdf_{i}",
                                use_container_width=True,
                            )
                            st.success(f"Ready - {len(pdf_bytes) / 1024:.0f} KB")
                        except Exception as exc:
                            st.error(f"PDF generation failed: {exc}")
