"""Inventory Optimization Recommendations — Day 18.

Interactive UI with:
- What-if parameter explorer (service level, lead time, ordering cost)
- Urgency-tiered priority action queue
- Multi-dimensional filters (category, region, store)
- Interactive Plotly charts (scatter, heatmap, bar)
- CSV export of filtered recommendations
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from utils.auth import require_auth

require_auth()  # block access by unauthenticated users

from utils.data_loader import load_inventory_recommendations

# ── Constants matching inventory_optimizer.py defaults ────────────────────────
_Z_ORIGINAL  = float(norm.ppf(0.95))   # default service level
_LT_ORIGINAL = 1                        # default lead time (days)
_OC_ORIGINAL = 50.0                     # ordering cost
_HCR_ORIGINAL = 0.25                   # holding cost rate

STATUS_STOCKOUT = "STOCKOUT_RISK"
STATUS_OPTIMAL  = "OPTIMAL"

URGENCY_CRITICAL = "Critical (<1d)"
URGENCY_HIGH     = "High (1–3d)"
URGENCY_MEDIUM   = "Medium (3–7d)"
URGENCY_OK       = "OK (>7d)"

URGENCY_COLOR = {
    URGENCY_CRITICAL: "#F44336",
    URGENCY_HIGH:     "#FF9800",
    URGENCY_MEDIUM:   "#FFC107",
    URGENCY_OK:       "#4CAF50",
}

STATUS_COLOR = {
    STATUS_STOCKOUT: "#F44336",
    STATUS_OPTIMAL:  "#4CAF50",
}


# ── Helper: reconstruct demand std from pre-computed safety_stock ─────────────
def _reconstruct_std(safety_stock: pd.Series) -> pd.Series:
    """Back-calculate daily demand std using original optimizer parameters."""
    denominator = _Z_ORIGINAL * np.sqrt(_LT_ORIGINAL)
    return (safety_stock / denominator).clip(lower=0)


def _recompute_with_params(
    df: pd.DataFrame,
    service_level: float,
    lead_time: int,
    ordering_cost: float,
) -> pd.DataFrame:
    """Return a copy of df with recalculated safety_stock, rop, status, units_to_order."""
    out = df.copy()
    z_new = float(norm.ppf(service_level))

    std_demand = _reconstruct_std(out["safety_stock"])
    out["safety_stock"] = (z_new * std_demand * np.sqrt(lead_time)).round(2)
    out["rop"]          = (out["mean_daily_demand"] * lead_time + out["safety_stock"]).round(2)

    # Recompute EOQ with new ordering cost (using back-calculated holding cost)
    # holding_cost per unit ≈ HCR × avg_price; avg_price ≈ eoq² × HCR / (2 × D × OC)
    # Simplified: scale EOQ by √(new_OC / original_OC)
    oc_scale = np.sqrt(ordering_cost / _OC_ORIGINAL)
    out["eoq"] = (out["eoq"] * oc_scale).round(2)

    def _classify(row: pd.Series) -> str:
        if row["current_inventory"] < row["rop"]:
            return STATUS_STOCKOUT
        return STATUS_OPTIMAL

    out["status"] = out.apply(_classify, axis=1)
    out["units_to_order"] = np.where(
        out["status"] == STATUS_STOCKOUT,
        (out["rop"] + out["eoq"] - out["current_inventory"]).clip(lower=0).round(0),
        0.0,
    )
    out["days_of_stock"] = np.where(
        out["mean_daily_demand"] > 0,
        (out["current_inventory"] / out["mean_daily_demand"]).round(1),
        np.inf,
    )
    return out


def _assign_urgency(days: float) -> str:
    if days < 1:
        return URGENCY_CRITICAL
    if days < 3:
        return URGENCY_HIGH
    if days < 7:
        return URGENCY_MEDIUM
    return URGENCY_OK


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Inventory Optimization", layout="wide")
st.title("Inventory Optimization Recommendations")

# ── Load base data ─────────────────────────────────────────────────────────────
raw = load_inventory_recommendations()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Optimizer Parameters")

    service_level = st.slider(
        "Service Level",
        min_value=0.85,
        max_value=0.99,
        value=0.95,
        step=0.01,
        help="Target in-stock probability. Higher = more safety stock.",
        format="%.2f",
    )
    lead_time = st.slider(
        "Lead Time (days)",
        min_value=1,
        max_value=14,
        value=_LT_ORIGINAL,
        step=1,
        help="Days between placing and receiving an order.",
    )
    ordering_cost = st.slider(
        "Ordering Cost ($)",
        min_value=10.0,
        max_value=200.0,
        value=_OC_ORIGINAL,
        step=10.0,
        help="Fixed cost per purchase order (affects EOQ).",
    )

    params_changed = (
        service_level != 0.95
        or lead_time != _LT_ORIGINAL
        or ordering_cost != _OC_ORIGINAL
    )

    if params_changed:
        st.info("What-if mode active — metrics reflect new parameters.")

    st.divider()
    st.header("Filters")

    all_categories = sorted(raw["category"].unique())
    sel_categories = st.multiselect("Category", all_categories, default=all_categories)

    all_regions = sorted(raw["region"].unique())
    sel_regions = st.multiselect("Region", all_regions, default=all_regions)

    all_stores = sorted(raw["store_id"].unique())
    sel_stores = st.multiselect("Store", all_stores, default=all_stores)

    status_opts = ["All", STATUS_STOCKOUT, STATUS_OPTIMAL]
    sel_status = st.radio("Status", status_opts, index=0)

    st.divider()
    st.header("Display Options")
    top_n_queue = st.slider("Priority queue size", 5, 30, 10, step=5)

# ── Apply parameters + filters ────────────────────────────────────────────────
df = _recompute_with_params(raw, service_level, lead_time, ordering_cost)
df["urgency"] = df["days_of_stock"].apply(_assign_urgency)

mask = (
    df["category"].isin(sel_categories)
    & df["region"].isin(sel_regions)
    & df["store_id"].isin(sel_stores)
)
if sel_status != "All":
    mask &= df["status"] == sel_status

df_filtered = df[mask].copy()

# ── Compute baseline (original params) for delta display ─────────────────────
df_baseline = raw.copy()
df_baseline["urgency"] = df_baseline["days_of_stock"].apply(_assign_urgency)

baseline_stockout  = (df_baseline[mask]["status"] == STATUS_STOCKOUT).sum()
baseline_units     = int(df_baseline[mask]["units_to_order"].sum())
baseline_avg_days  = df_baseline[mask]["days_of_stock"].replace(np.inf, np.nan).mean()

current_stockout   = (df_filtered["status"] == STATUS_STOCKOUT).sum()
current_units      = int(df_filtered["units_to_order"].sum())
current_avg_days   = df_filtered["days_of_stock"].replace(np.inf, np.nan).mean()

# ── KPI Section ───────────────────────────────────────────────────────────────
st.subheader("Executive Summary")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric(
    "Total Records",
    f"{len(df_filtered):,}",
    help="Rows after current filters.",
)
col2.metric(
    "Stockout Risk",
    f"{current_stockout:,}",
    delta=current_stockout - baseline_stockout if params_changed else None,
    delta_color="inverse",
)
col3.metric(
    "Optimal",
    f"{len(df_filtered) - current_stockout:,}",
    delta=(len(df_filtered) - current_stockout) - (len(df_filtered) - baseline_stockout)
    if params_changed else None,
)
col4.metric(
    "Units to Order",
    f"{current_units:,}",
    delta=current_units - baseline_units if params_changed else None,
    delta_color="inverse",
)
col5.metric(
    "Avg Days of Stock",
    f"{current_avg_days:.1f}",
    delta=round(current_avg_days - baseline_avg_days, 1) if params_changed else None,
    delta_color="normal",
)

st.divider()

# ── Priority Action Queue ─────────────────────────────────────────────────────
st.subheader(f"Priority Action Queue — Top {top_n_queue} Most Urgent")

urgency_order = {
    URGENCY_CRITICAL: 0,
    URGENCY_HIGH:     1,
    URGENCY_MEDIUM:   2,
    URGENCY_OK:       3,
}

queue = (
    df_filtered[df_filtered["status"] == STATUS_STOCKOUT]
    .copy()
    .assign(urgency_rank=lambda d: d["urgency"].map(urgency_order))
    .sort_values(["urgency_rank", "days_of_stock"])
    .head(top_n_queue)
    .drop(columns=["urgency_rank"])
)

if queue.empty:
    st.success("No stockout-risk SKUs match the current filters.")
else:
    urgency_counts = queue["urgency"].value_counts()
    uc1, uc2, uc3 = st.columns(3)
    uc1.metric("Critical (<1d)", urgency_counts.get(URGENCY_CRITICAL, 0))
    uc2.metric("High (1–3d)",    urgency_counts.get(URGENCY_HIGH, 0))
    uc3.metric("Medium (3–7d)",  urgency_counts.get(URGENCY_MEDIUM, 0))

    def _row_color(urgency: str) -> str:
        colors = {
            URGENCY_CRITICAL: "background-color: #FFCDD2",
            URGENCY_HIGH:     "background-color: #FFE0B2",
            URGENCY_MEDIUM:   "background-color: #FFF9C4",
            URGENCY_OK:       "background-color: #C8E6C9",
        }
        return colors.get(urgency, "")

    display_queue = queue[[
        "store_id", "product_id", "category", "region",
        "current_inventory", "rop", "eoq",
        "days_of_stock", "units_to_order", "urgency",
    ]].rename(columns={
        "store_id": "Store", "product_id": "Product", "category": "Category",
        "region": "Region", "current_inventory": "Inventory", "rop": "ROP",
        "eoq": "EOQ", "days_of_stock": "Days Left",
        "units_to_order": "Units to Order", "urgency": "Urgency",
    })

    styled = display_queue.style.apply(
        lambda row: [_row_color(row["Urgency"])] * len(row), axis=1
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

st.divider()

# ── Charts — Row 1 ─────────────────────────────────────────────────────────────
st.subheader("Inventory Status Analysis")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Status Distribution by Category**")
    cat_status = (
        df_filtered.groupby(["category", "status"])
        .size()
        .reset_index(name="count")
    )
    fig_bar = px.bar(
        cat_status,
        x="category",
        y="count",
        color="status",
        barmode="stack",
        labels={"count": "Record Count", "category": "Category", "status": "Status"},
        color_discrete_map=STATUS_COLOR,
    )
    fig_bar.update_layout(height=350, legend_title_text="Status", margin=dict(t=10))
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col2:
    st.markdown("**Urgency Distribution by Category**")
    cat_urgency = (
        df_filtered.groupby(["category", "urgency"])
        .size()
        .reset_index(name="count")
    )
    urgency_order_list = [URGENCY_CRITICAL, URGENCY_HIGH, URGENCY_MEDIUM, URGENCY_OK]
    fig_urg = px.bar(
        cat_urgency,
        x="category",
        y="count",
        color="urgency",
        barmode="stack",
        labels={"count": "Record Count", "category": "Category", "urgency": "Urgency"},
        color_discrete_map=URGENCY_COLOR,
        category_orders={"urgency": urgency_order_list},
    )
    fig_urg.update_layout(height=350, legend_title_text="Urgency", margin=dict(t=10))
    st.plotly_chart(fig_urg, use_container_width=True)

# ── Regional Risk Heatmap ──────────────────────────────────────────────────────
st.subheader("Regional Risk Heatmap")
st.caption("Percentage of records at stockout risk per Region × Category")

region_cat = df_filtered.groupby(["region", "category"])
total_rc   = region_cat.size().reset_index(name="total")
stockout_rc = (
    df_filtered[df_filtered["status"] == STATUS_STOCKOUT]
    .groupby(["region", "category"])
    .size()
    .reset_index(name="stockout")
)
heatmap_df = total_rc.merge(stockout_rc, on=["region", "category"], how="left").fillna(0)
heatmap_df["pct_stockout"] = (heatmap_df["stockout"] / heatmap_df["total"] * 100).round(1)

heatmap_pivot = heatmap_df.pivot(index="region", columns="category", values="pct_stockout").fillna(0)

fig_heat = go.Figure(
    data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale="Reds",
        zmin=0,
        zmax=100,
        text=heatmap_pivot.values.round(1),
        texttemplate="%{text}%",
        colorbar=dict(title="% Stockout Risk"),
    )
)
fig_heat.update_layout(
    height=320,
    xaxis_title="Category",
    yaxis_title="Region",
    margin=dict(t=10, b=10),
)
st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# ── Charts — Row 2 ─────────────────────────────────────────────────────────────
st.subheader("Demand & Restock Analysis")
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    st.markdown("**Days of Stock vs Units to Order**")
    scatter_df = df_filtered[df_filtered["status"] == STATUS_STOCKOUT].copy()
    scatter_df["days_of_stock_plot"] = scatter_df["days_of_stock"].clip(upper=30)
    fig_scatter = px.scatter(
        scatter_df,
        x="days_of_stock_plot",
        y="units_to_order",
        color="category",
        symbol="region",
        size="eoq",
        size_max=18,
        hover_data=["store_id", "product_id", "current_inventory", "rop"],
        labels={
            "days_of_stock_plot": "Days of Stock Remaining",
            "units_to_order": "Units to Order",
            "category": "Category",
        },
    )
    fig_scatter.add_vline(
        x=lead_time, line_dash="dash", line_color="red",
        annotation_text=f"Lead time ({lead_time}d)", annotation_position="top right",
    )
    fig_scatter.update_layout(height=370, margin=dict(t=10))
    st.plotly_chart(fig_scatter, use_container_width=True)

with chart_col4:
    st.markdown("**Units to Order by Store**")
    store_units = (
        df_filtered[df_filtered["status"] == STATUS_STOCKOUT]
        .groupby(["store_id", "category"])["units_to_order"]
        .sum()
        .reset_index()
    )
    fig_store = px.bar(
        store_units,
        x="store_id",
        y="units_to_order",
        color="category",
        barmode="stack",
        labels={"units_to_order": "Total Units to Order", "store_id": "Store", "category": "Category"},
    )
    fig_store.update_layout(height=370, legend_title_text="Category", margin=dict(t=10))
    st.plotly_chart(fig_store, use_container_width=True)

# ── Safety Stock Distribution ─────────────────────────────────────────────────
st.subheader("Safety Stock & Reorder Points")
ss_col1, ss_col2 = st.columns(2)

with ss_col1:
    st.markdown("**Safety Stock Distribution by Category**")
    fig_ss = px.box(
        df_filtered,
        x="category",
        y="safety_stock",
        color="category",
        points="outliers",
        labels={"safety_stock": "Safety Stock (units)", "category": "Category"},
    )
    fig_ss.update_layout(height=350, showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig_ss, use_container_width=True)

with ss_col2:
    st.markdown("**Inventory vs Reorder Point**")
    fig_inv_rop = px.scatter(
        df_filtered,
        x="rop",
        y="current_inventory",
        color="status",
        facet_col="category",
        facet_col_wrap=3,
        color_discrete_map=STATUS_COLOR,
        labels={"rop": "Reorder Point", "current_inventory": "Current Inventory"},
        opacity=0.6,
        size_max=8,
    )
    # Diagonal reference line per facet
    max_val = max(df_filtered["rop"].max(), df_filtered["current_inventory"].max())
    fig_inv_rop.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="gray", dash="dot"), row="all", col="all",
    )
    fig_inv_rop.update_layout(height=380, margin=dict(t=30), showlegend=True)
    st.plotly_chart(fig_inv_rop, use_container_width=True)

st.divider()

# ── What-If Summary ───────────────────────────────────────────────────────────
if params_changed:
    st.subheader("What-If Parameter Impact")
    st.caption(
        f"Comparing **baseline** (SL=0.95, LT={_LT_ORIGINAL}d, OC=${_OC_ORIGINAL:.0f}) "
        f"vs **current** (SL={service_level:.2f}, LT={lead_time}d, OC=${ordering_cost:.0f})"
    )

    baseline_mask = df_baseline[mask]
    comparison = pd.DataFrame({
        "Metric": ["Stockout Records", "Optimal Records", "Total Units to Order", "Avg Days of Stock"],
        "Baseline": [
            (baseline_mask["status"] == STATUS_STOCKOUT).sum(),
            (baseline_mask["status"] == STATUS_OPTIMAL).sum(),
            baseline_mask["units_to_order"].sum(),
            round(baseline_mask["days_of_stock"].replace(np.inf, np.nan).mean(), 1),
        ],
        "What-If": [
            current_stockout,
            len(df_filtered) - current_stockout,
            current_units,
            round(current_avg_days, 1),
        ],
    })
    comparison["Change"] = comparison["What-If"] - comparison["Baseline"]
    st.dataframe(comparison.set_index("Metric"), use_container_width=True)
    st.divider()

# ── Detailed Recommendations Table ────────────────────────────────────────────
st.subheader("Detailed Recommendations")

col_sort, col_export = st.columns([3, 1])
with col_sort:
    sort_by = st.selectbox(
        "Sort by",
        ["days_of_stock", "units_to_order", "current_inventory", "rop", "eoq", "safety_stock"],
        index=0,
    )
    sort_asc = st.checkbox("Ascending", value=True)

with col_export:
    st.write("")  # vertical spacer
    csv_bytes = (
        df_filtered[[
            "store_id", "product_id", "category", "region",
            "current_inventory", "mean_daily_demand", "safety_stock",
            "rop", "eoq", "days_of_stock", "units_to_order", "status", "urgency",
        ]]
        .sort_values(sort_by, ascending=sort_asc)
        .to_csv(index=False)
        .encode("utf-8")
    )
    st.download_button(
        label="Export CSV",
        data=csv_bytes,
        file_name="inventory_recommendations.csv",
        mime="text/csv",
        use_container_width=True,
    )

display_df = (
    df_filtered[[
        "store_id", "product_id", "category", "region",
        "current_inventory", "mean_daily_demand", "safety_stock",
        "rop", "eoq", "days_of_stock", "units_to_order", "status", "urgency",
    ]]
    .sort_values(sort_by, ascending=sort_asc)
    .reset_index(drop=True)
)

st.dataframe(
    display_df.style.apply(
        lambda row: [
            f"background-color: {URGENCY_COLOR.get(row['urgency'], '#fff')}22"
        ] * len(row),
        axis=1,
    ),
    use_container_width=True,
    height=420,
)
st.caption(f"Showing {len(display_df):,} records — SL={service_level:.2f}, LT={lead_time}d, OC=${ordering_cost:.0f}")
