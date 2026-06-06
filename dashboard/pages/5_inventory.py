import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import plotly.express as px
import streamlit as st

from utils.data_loader import figure_path, load_inventory_recommendations

st.header("Inventory Optimization")

inv = load_inventory_recommendations()

# ── KPI cards ─────────────────────────────────────────────────────────────────
stockout_risk  = (inv["status"] == "STOCKOUT_RISK").sum()
avg_days_stock = inv["days_of_stock"].mean()
total_skus     = inv["product_id"].nunique()

c1, c2, c3 = st.columns(3)
c1.metric("Total SKUs",          f"{total_skus:,}")
c2.metric("At Stockout Risk",    f"{stockout_risk:,}")
c3.metric("Avg Days of Stock",   f"{avg_days_stock:.1f}")

st.divider()

# ── Inventory charts ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Stockout Risk Heatmap")
    st.image(str(figure_path("inventory_stockout_heatmap.png")), width="stretch")
with col2:
    st.subheader("Status Breakdown")
    st.image(str(figure_path("inventory_status_breakdown.png")), width="stretch")

st.divider()

# ── Status breakdown (interactive) ───────────────────────────────────────────
st.subheader("Status Distribution by Category")

status_cat = (
    inv.groupby(["category", "status"])
    .size()
    .reset_index(name="count")
)
fig = px.bar(
    status_cat,
    x="category",
    y="count",
    color="status",
    barmode="stack",
    labels={"count": "SKU Count", "category": "Category", "status": "Status"},
    color_discrete_map={
        "STOCKOUT_RISK": "#F44336",
        "LOW":           "#FF9800",
        "ADEQUATE":      "#4CAF50",
    },
)
fig.update_layout(height=340, legend_title_text="Status")
st.plotly_chart(fig, width="stretch")

st.divider()

# ── Filterable recommendations table ─────────────────────────────────────────
st.subheader("Inventory Recommendations")

status_options = ["All"] + sorted(inv["status"].unique().tolist())
selected_status = st.selectbox("Filter by Status", status_options)

display_inv = inv if selected_status == "All" else inv[inv["status"] == selected_status]

st.dataframe(
    display_inv[[
        "store_id", "product_id", "category", "region",
        "current_inventory", "safety_stock", "rop", "eoq",
        "days_of_stock", "units_to_order", "status",
    ]]
    .sort_values("days_of_stock")
    .reset_index(drop=True),
    width="stretch",
    height=420,
)
st.caption(f"Showing {len(display_inv):,} of {len(inv):,} records")
