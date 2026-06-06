import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import plotly.express as px
import streamlit as st

from utils.data_loader import load_daily_revenue_rolling, load_retail_clean

st.header("Business Overview")

retail   = load_retail_clean()
rolling  = load_daily_revenue_rolling()

# ── KPI cards ────────────────────────────────────────────────────────────────
total_revenue      = retail["Revenue"].sum()
unique_customers   = retail["Customer ID"].nunique()
total_orders       = retail["Invoice"].nunique()
avg_order_value    = retail.groupby("Invoice")["Revenue"].sum().mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue",      f"₹{total_revenue:,.0f}")
c2.metric("Unique Customers",   f"{unique_customers:,}")
c3.metric("Total Orders",       f"{total_orders:,}")
c4.metric("Avg Order Value",    f"₹{avg_order_value:,.2f}")

st.divider()

# ── Daily revenue + rolling means ────────────────────────────────────────────
st.subheader("Daily Revenue Trend")

rolling_clean = rolling.dropna(subset=["rolling_7d_mean"])

fig = px.line(
    rolling,
    x="Date",
    y=["Revenue", "rolling_7d_mean", "rolling_30d_mean"],
    labels={"value": "Revenue (₹)", "variable": "Series"},
    color_discrete_map={
        "Revenue":          "#a8c8e8",
        "rolling_7d_mean":  "#2196F3",
        "rolling_30d_mean": "#FF5722",
    },
)
fig.update_traces(selector=dict(name="Revenue"), line_width=1, opacity=0.5)
fig.update_layout(legend_title_text="", hovermode="x unified", height=360)
st.plotly_chart(fig, width="stretch")

st.divider()

# ── Top 10 products by revenue ────────────────────────────────────────────────
st.subheader("Top 10 Products by Revenue")

top_products = (
    retail.groupby("Description")["Revenue"]
    .sum()
    .nlargest(10)
    .reset_index()
    .sort_values("Revenue")
)

fig2 = px.bar(
    top_products,
    x="Revenue",
    y="Description",
    orientation="h",
    labels={"Revenue": "Revenue (₹)", "Description": "Product"},
    color="Revenue",
    color_continuous_scale="Blues",
)
fig2.update_layout(coloraxis_showscale=False, height=360)
st.plotly_chart(fig2, width="stretch")
