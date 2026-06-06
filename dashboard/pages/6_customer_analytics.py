import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import load_segmentation_churn_merged

st.header("Customer Analytics — Segmentation × Churn Risk")

df = load_segmentation_churn_merged()

SEGMENT_COLORS = {
    "Champions":           "#2196F3",
    "Loyal Customers":     "#4CAF50",
    "Potential Loyalists": "#8BC34A",
    "At Risk":             "#FF9800",
    "Lost":                "#F44336",
}

df["risk_tier"] = pd.cut(
    df["churn_probability"],
    bins=[-0.001, 0.30, 0.70, 1.001],
    labels=["Low (<30%)", "Medium (30–70%)", "High (>70%)"],
)

TIER_COLORS = {
    "Low (<30%)":      "#4CAF50",
    "Medium (30–70%)": "#FF9800",
    "High (>70%)":     "#F44336",
}

# ── KPI row ────────────────────────────────────────────────────────────────────
total_customers = len(df)
high_risk       = int((df["churn_probability"] >= 0.70).sum())
avg_prob        = df["churn_probability"].mean()
top_risk_seg    = df.groupby("Segment")["churn_probability"].mean().idxmax()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers Analyzed",    f"{total_customers:,}")
c2.metric("High-Risk (prob ≥ 70%)", f"{high_risk:,}")
c3.metric("Avg Churn Probability",  f"{avg_prob:.1%}")
c4.metric("Highest-Risk Segment",   top_risk_seg)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 1 — Churn Rate by RFM Segment
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Churn Rate by RFM Segment")

seg_stats = (
    df.groupby("Segment", observed=True)
    .agg(
        customer_count    = ("Customer ID", "count"),
        avg_churn_prob    = ("churn_probability", "mean"),
        predicted_churners= ("predicted_churn", "sum"),
    )
    .reset_index()
)
seg_stats["churn_rate_pct"] = (
    seg_stats["predicted_churners"] / seg_stats["customer_count"] * 100
)
seg_stats = seg_stats.sort_values("avg_churn_prob", ascending=False)

col1, col2 = st.columns(2)

with col1:
    fig_bar = px.bar(
        seg_stats,
        x="Segment",
        y="churn_rate_pct",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        text=seg_stats["churn_rate_pct"].map(lambda v: f"{v:.1f}%"),
        labels={
            "churn_rate_pct": "Predicted Churn Rate (%)",
            "Segment":        "RFM Segment",
        },
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        showlegend=False,
        height=340,
        yaxis_title="Churn Rate (%)",
        margin=dict(t=20, b=60, l=10, r=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    fig_violin = px.violin(
        df,
        x="Segment",
        y="churn_probability",
        color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        box=True,
        points=False,
        labels={
            "churn_probability": "Churn Probability",
            "Segment":           "RFM Segment",
        },
    )
    fig_violin.update_layout(
        showlegend=False,
        height=340,
        yaxis_tickformat=".0%",
        margin=dict(t=20, b=60, l=10, r=10),
    )
    st.plotly_chart(fig_violin, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 2 — Risk Tier Breakdown
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Churn Risk Tier Breakdown")

col3, col4 = st.columns(2)

with col3:
    tier_counts = (
        df["risk_tier"]
        .value_counts()
        .reset_index()
    )
    tier_counts.columns = ["Risk Tier", "Count"]
    tier_counts["Risk Tier"] = tier_counts["Risk Tier"].astype(str)

    fig_pie = px.pie(
        tier_counts,
        names="Risk Tier",
        values="Count",
        hole=0.45,
        color="Risk Tier",
        color_discrete_map=TIER_COLORS,
    )
    fig_pie.update_traces(textinfo="percent+label")
    fig_pie.update_layout(
        showlegend=False,
        height=300,
        margin=dict(t=20, b=20, l=10, r=10),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col4:
    tier_seg = (
        df.groupby(["Segment", "risk_tier"], observed=True)
        .size()
        .reset_index(name="count")
    )
    tier_seg["risk_tier"] = tier_seg["risk_tier"].astype(str)

    fig_stacked = px.bar(
        tier_seg,
        x="Segment",
        y="count",
        color="risk_tier",
        barmode="stack",
        color_discrete_map=TIER_COLORS,
        labels={
            "count":     "Customers",
            "risk_tier": "Risk Tier",
            "Segment":   "RFM Segment",
        },
        category_orders={
            "risk_tier": ["Low (<30%)", "Medium (30–70%)", "High (>70%)"]
        },
    )
    fig_stacked.update_layout(
        height=300,
        legend_title_text="Risk Tier",
        margin=dict(t=20, b=60, l=10, r=10),
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — Customer Risk Map
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Customer Risk Map (Recency × Monetary)")
st.caption(
    "Each point is a customer. Color shows churn probability (red = high risk). "
    "Point size scales with purchase frequency."
)

fig_scatter = px.scatter(
    df,
    x="Recency",
    y="Monetary",
    color="churn_probability",
    size="Frequency",
    size_max=18,
    hover_data={
        "Customer ID":       True,
        "Segment":           True,
        "churn_probability": ":.1%",
        "Recency":           True,
        "Monetary":          ":,.0f",
        "Frequency":         True,
    },
    color_continuous_scale="RdYlGn_r",
    range_color=[0, 1],
    labels={
        "Recency":           "Recency (days since last purchase)",
        "Monetary":          "Monetary (₹ total spend)",
        "churn_probability": "Churn Prob",
        "Frequency":         "Order Count",
    },
    opacity=0.72,
)
fig_scatter.update_layout(
    height=460,
    coloraxis_colorbar=dict(
        title="Churn Prob",
        tickformat=".0%",
    ),
    margin=dict(t=10, b=40, l=10, r=10),
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 4 — Segment Summary Table
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Segment Summary")

summary = (
    df.groupby("Segment", observed=True)
    .agg(
        Customers          = ("Customer ID", "count"),
        Avg_Recency        = ("Recency",           "mean"),
        Avg_Monetary       = ("Monetary",          "mean"),
        Avg_Frequency      = ("Frequency",         "mean"),
        Avg_Churn_Prob     = ("churn_probability", "mean"),
        Predicted_Churners = ("predicted_churn",   "sum"),
    )
    .reset_index()
)
summary["Churn_Rate_%"] = (
    summary["Predicted_Churners"] / summary["Customers"] * 100
).round(1)
summary = summary.sort_values("Avg_Churn_Prob", ascending=False)
summary["Avg_Churn_Prob"]  = summary["Avg_Churn_Prob"].map(lambda v: f"{v:.1%}")
summary["Avg_Recency"]     = summary["Avg_Recency"].round(0).astype(int)
summary["Avg_Monetary"]    = summary["Avg_Monetary"].map(lambda v: f"₹{v:,.0f}")
summary["Avg_Frequency"]   = summary["Avg_Frequency"].round(1)

st.dataframe(
    summary.rename(columns={
        "Avg_Recency":        "Avg Recency (d)",
        "Avg_Monetary":       "Avg Spend",
        "Avg_Frequency":      "Avg Orders",
        "Avg_Churn_Prob":     "Avg Churn Prob",
        "Predicted_Churners": "Pred. Churners",
        "Churn_Rate_%":       "Churn Rate %",
    }),
    use_container_width=True,
    hide_index=True,
)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 5 — Customer Drill-Down
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Customer Detail")

f_col1, f_col2 = st.columns(2)
with f_col1:
    seg_options  = ["All"] + sorted(df["Segment"].unique().tolist())
    selected_seg = st.selectbox("Filter by RFM Segment", seg_options)

with f_col2:
    tier_options  = ["All", "High (>70%)", "Medium (30–70%)", "Low (<30%)"]
    selected_tier = st.selectbox("Filter by Risk Tier", tier_options)

display_df = df.copy()
if selected_seg != "All":
    display_df = display_df[display_df["Segment"] == selected_seg]
if selected_tier != "All":
    display_df = display_df[display_df["risk_tier"].astype(str) == selected_tier]

display_df = (
    display_df
    .sort_values("churn_probability", ascending=False)
    .reset_index(drop=True)
)

st.dataframe(
    display_df[[
        "Customer ID", "Segment", "Recency", "Frequency", "Monetary",
        "churn_probability", "predicted_churn", "actual_churn", "risk_tier",
    ]].rename(columns={
        "churn_probability": "Churn Prob",
        "predicted_churn":   "Pred. Churned",
        "actual_churn":      "Actual Churned",
        "risk_tier":         "Risk Tier",
    }),
    use_container_width=True,
    height=420,
    hide_index=True,
)
st.caption(
    f"Showing {len(display_df):,} of {total_customers:,} model-evaluated customers "
    f"(20% holdout from churn model training)"
)
