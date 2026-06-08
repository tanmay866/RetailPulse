"""PDF report builders for RetailPulse — plain module, no Streamlit magic."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from fpdf import FPDF
    FPDF_OK = True
except ImportError:
    FPDF_OK = False
    FPDF = object  # type: ignore[assignment,misc]


class _PDF(FPDF):  # type: ignore[misc]
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

    def report_title(self, title: str, subtitle: str = "") -> None:
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(*self._BLUE)
        self.cell(0, 12, title, new_x="LMARGIN", new_y="NEXT")
        if subtitle:
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*self._GRAY)
            self.cell(0, 6, subtitle, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def section(self, title: str) -> None:
        if self.get_y() > self.h - self.b_margin - 20:
            self.add_page()
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(*self._BLUE)
        self.set_text_color(255, 255, 255)
        self.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", "", 9)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5, text)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def kpi_row(self, items: list) -> None:
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

    def data_table(self, headers: list, rows: list, col_widths: list) -> None:
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
            fill_color = (248, 248, 248) if j % 2 == 0 else (255, 255, 255)
            self.set_fill_color(*fill_color)
            for val, cw in zip(row, col_widths):
                self.cell(cw, 5, str(val)[:45], fill=True)
            self.ln()
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def chart(self, path: Path, caption: str = "", w_pct: float = 0.95) -> None:
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


def build_overview_pdf(D: dict, figures_dir: Path) -> bytes:
    retail  = D["retail"]
    rolling = D["rolling"]
    kpis    = D.get("kpis", {})
    pdf = _PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    dr = (
        f"{rolling['Date'].min().strftime('%d %b %Y')} "
        f"- {rolling['Date'].max().strftime('%d %b %Y')}"
    )
    pdf.report_title("Business Overview Report", f"Data period: {dr}")

    pdf.section("Key Performance Indicators")
    if not retail.empty:
        total_rev    = retail["Revenue"].sum()
        uniq_cust    = retail["Customer ID"].nunique()
        total_orders = retail["Invoice"].nunique()
        aov          = retail.groupby("Invoice")["Revenue"].sum().mean()
        pdf.kpi_row([
            ("Transaction Lines",  f"{len(retail):,}"),
            ("Avg Qty / Line",     f"{retail['Quantity'].mean():.1f}"),
            ("Countries",          f"{retail['Country'].nunique()}"),
            ("Date Range",         dr),
        ])
    else:
        total_rev    = kpis.get("total_revenue",    0)
        uniq_cust    = kpis.get("unique_customers", 0)
        total_orders = kpis.get("total_orders",     0)
        aov          = kpis.get("avg_order_value",  0)
    pdf.kpi_row([
        ("Total Revenue",      f"GBP {total_rev:,.0f}"),
        ("Unique Customers",   f"{uniq_cust:,}"),
        ("Total Orders",       f"{total_orders:,}"),
        ("Avg Order Value",    f"GBP {aov:,.2f}"),
    ])

    pdf.section("Revenue Trend")
    pdf.chart(figures_dir / "monthly_sales_trend.png", "Fig 1 - Monthly Revenue Trend")
    pdf.chart(figures_dir / "rolling_statistics.png",  "Fig 2 - Daily Revenue with Rolling Statistics")

    pdf.section("Top Products by Revenue")
    pdf.chart(figures_dir / "top_products.png", "Fig 3 - Top 10 Products by Revenue")
    if not retail.empty:
        top_prods = (
            retail.groupby("Description")["Revenue"]
            .sum().nlargest(10).reset_index()
        )
        pdf.data_table(
            ["#", "Product", "Revenue (GBP)"],
            [
                [i + 1, str(row["Description"])[:55], f"{row['Revenue']:,.0f}"]
                for i, (_, row) in enumerate(top_prods.iterrows())
            ],
            col_widths=[10, 135, 45],
        )

    pdf.section("Revenue by Country")
    pdf.chart(figures_dir / "top_countries.png", "Fig 4 - Top Countries by Revenue")

    return bytes(pdf.output())


def build_customer_health_pdf(D: dict, figures_dir: Path) -> bytes:
    rfm   = D["rfm"]
    churn = D["churn"]
    pdf   = _PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.report_title("Customer Health Report", f"{len(rfm):,} customers analysed")

    pdf.section("Customer Segmentation (RFM)")
    pdf.kpi_row([
        ("Total Customers",     f"{len(rfm):,}"),
        ("Segments",            f"{rfm['Segment'].nunique()}"),
        ("Avg Recency (days)",  f"{rfm['Recency'].mean():.0f}"),
        ("Avg Monetary (GBP)",  f"{rfm['Monetary'].mean():,.0f}"),
    ])
    pdf.chart(figures_dir / "rfm_segments.png",      "Fig 1 - Customer Segment Distribution")
    pdf.chart(figures_dir / "rfm_distributions.png", "Fig 2 - RFM Score Distributions")
    pdf.chart(figures_dir / "cluster_profiles.png",  "Fig 3 - Segment Profiles")

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
    pdf.chart(figures_dir / "churn_roc_curve.png",        "Fig 4 - ROC Curve")
    pdf.chart(figures_dir / "churn_confusion_matrix.png", "Fig 5 - Confusion Matrix")
    pdf.chart(figures_dir / "churn_shap_bar.png",         "Fig 6 - Feature Importance (SHAP)")
    pdf.chart(figures_dir / "churn_shap_summary.png",     "Fig 7 - SHAP Summary Plot")

    return bytes(pdf.output())


def build_inventory_pdf(D: dict, figures_dir: Path) -> bytes:
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
    pdf.chart(figures_dir / "inventory_status_breakdown.png", "Fig 1 - Status Breakdown")
    pdf.chart(figures_dir / "inventory_days_of_stock.png",    "Fig 2 - Days of Stock Distribution")
    pdf.chart(figures_dir / "inventory_stockout_heatmap.png", "Fig 3 - Stockout Risk Heatmap")
    pdf.chart(figures_dir / "inventory_eoq_vs_ordered.png",   "Fig 4 - EOQ vs Current Order Quantity")

    pdf.section("Critical SKUs  (Days of Stock < 7)")
    critical_df = inv[inv["days_of_stock"] < 7].sort_values("days_of_stock").head(20)
    if critical_df.empty:
        pdf.body("No SKUs with fewer than 7 days of stock.")
    else:
        pdf.data_table(
            ["Store", "Product", "Category", "Days Left", "Status", "To Order"],
            [
                [
                    r["store_id"], r["product_id"], r["category"],
                    f"{r['days_of_stock']:.1f}", r["status"],
                    f"{r['units_to_order']:.0f}",
                ]
                for _, r in critical_df.iterrows()
            ],
            col_widths=[24, 28, 35, 24, 40, 30],
        )

    return bytes(pdf.output())


def build_full_report_pdf(D: dict, figures_dir: Path) -> bytes:
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

    kpis = D.get("kpis", {})

    # Section 1 - Business
    pdf.add_page()
    pdf.report_title("1  -  Business Performance")
    if not retail.empty:
        total_rev    = retail["Revenue"].sum()
        uniq_cust    = retail["Customer ID"].nunique()
        total_orders = retail["Invoice"].nunique()
        aov          = retail.groupby("Invoice")["Revenue"].sum().mean()
    else:
        total_rev    = kpis.get("total_revenue",    0)
        uniq_cust    = kpis.get("unique_customers", 0)
        total_orders = kpis.get("total_orders",     0)
        aov          = kpis.get("avg_order_value",  0)
    pdf.kpi_row([
        ("Total Revenue",    f"GBP {total_rev:,.0f}"),
        ("Unique Customers", f"{uniq_cust:,}"),
        ("Total Orders",     f"{total_orders:,}"),
        ("Avg Order Value",  f"GBP {aov:,.2f}"),
    ])
    pdf.section("Revenue Trend")
    pdf.chart(figures_dir / "monthly_sales_trend.png", "Monthly Revenue Trend")
    pdf.chart(figures_dir / "rolling_statistics.png",  "Daily Revenue with Rolling Statistics")
    pdf.section("Top Products & Geography")
    pdf.chart(figures_dir / "top_products.png",  "Top Products by Revenue")
    pdf.chart(figures_dir / "top_countries.png", "Revenue by Country")

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
    pdf.chart(figures_dir / "rfm_segments.png",    "Segment Distribution")
    pdf.chart(figures_dir / "cluster_profiles.png", "Cluster Profiles")
    pdf.chart(figures_dir / "kmeans_scatter.png",   "K-Means Cluster Scatter")

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
    pdf.chart(figures_dir / "churn_roc_curve.png",        "ROC Curve")
    pdf.chart(figures_dir / "churn_confusion_matrix.png", "Confusion Matrix")
    pdf.section("Feature Importance (SHAP)")
    pdf.chart(figures_dir / "churn_shap_bar.png",     "SHAP Feature Importance")
    pdf.chart(figures_dir / "churn_shap_summary.png", "SHAP Summary Plot")

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
    pdf.chart(figures_dir / "inventory_status_breakdown.png", "Status Breakdown")
    pdf.chart(figures_dir / "inventory_days_of_stock.png",    "Days of Stock Distribution")
    pdf.chart(figures_dir / "inventory_eoq_vs_ordered.png",   "EOQ vs Ordered Quantity")

    return bytes(pdf.output())
