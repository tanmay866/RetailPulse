"""Customer Lifetime Value (CLV) Analysis — BG/NBD + Gamma-Gamma model."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data_loader import (
    load_clv_predictions,
    load_rfm_scores,
    load_churn_predictions,
)

# ── Page ──────────────────────────────────────────────────────────────────────
st.header("Customer Lifetime Value")
st.caption(
    "BG/NBD model forecasts future purchase frequency. "
    "Gamma-Gamma model estimates expected spend per transaction. "
    "Combined to compute 12-month discounted CLV per customer."
)

# ── Load & merge data ─────────────────────────────────────────────────────────
clv   = load_clv_predictions()
rfm   = load_rfm_scores()
churn = load_churn_predictions()

# Merge CLV with RFM segment labels
clv = clv.merge(
    rfm[["Customer ID", "Segment"]],
    on="Customer ID",
    how="left",
)

# Merge with churn probability
churn_slim = churn[["Customer_ID", "churn_probability"]].rename(
    columns={"Customer_ID": "Customer ID"}
)
clv = clv.merge(churn_slim, on="Customer ID", how="left")

repeat = clv[clv["frequency"] > 0].copy()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("CLV Filters")
    seg_opts = ["All"] + sorted(clv["Segment"].dropna().unique())
    seg_sel  = st.selectbox("RFM Segment", seg_opts)

    clv_segs = ["All", "High Value", "Medium Value", "Low Value"]
    clv_seg_sel = st.selectbox("CLV Tier", clv_segs)

    horizon = st.radio(
        "Purchase Horizon",
        ["90 days", "180 days", "365 days"],
        index=1,
    )
    horizon_col = f"pred_purchases_{horizon.split()[0]}d"

# Apply filters
view = repeat.copy()
if seg_sel != "All":
    view = view[view["Segment"] == seg_sel]
if clv_seg_sel != "All":
    view = view[view["clv_segment"] == clv_seg_sel]

# ── KPI strip ─────────────────────────────────────────────────────────────────
total_clv    = repeat["clv_12m"].sum()
avg_clv      = repeat["clv_12m"].mean()
high_val_n   = int((repeat["clv_segment"] == "High Value").sum())
avg_alive    = clv["prob_alive"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total 12m CLV",          f"₹{total_clv:,.0f}")
k2.metric("Avg CLV / Customer",      f"₹{avg_clv:,.0f}")
k3.metric("High-Value Customers",    f"{high_val_n:,}",
          f"{high_val_n / len(repeat) * 100:.1f}% of repeat buyers")
k4.metric("Avg Probability Alive",   f"{avg_alive:.1%}",
          help="Probability the customer has not permanently churned (BG/NBD)")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dist, tab_seg, tab_quad, tab_top = st.tabs([
    "CLV Distribution",
    "Segment Breakdown",
    "CLV × Churn Quadrant",
    "Top Customers",
])


# ════════════════════════════════════════════════════════════════════════════
# Tab 1 — CLV Distribution
# ════════════════════════════════════════════════════════════════════════════
with tab_dist:
    col_hist, col_pie = st.columns([6, 4])

    with col_hist:
        st.markdown("**CLV Distribution by Tier**")
        _SEG_COLOR = {
            "High Value":   "#1565C0",
            "Medium Value": "#43A047",
            "Low Value":    "#EF5350",
        }
        fig_hist = px.histogram(
            repeat,
            x="clv_12m",
            color="clv_segment",
            color_discrete_map=_SEG_COLOR,
            nbins=60,
            labels={"clv_12m": "12-Month CLV (₹)", "clv_segment": "CLV Tier"},
            barmode="overlay",
            opacity=0.75,
        )
        fig_hist.update_layout(
            height=340,
            legend_title_text="CLV Tier",
            margin=dict(t=10, b=40, l=10, r=10),
            xaxis_tickprefix="₹",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_pie:
        st.markdown("**Tier Distribution**")
        tier_ct = repeat["clv_segment"].value_counts().reset_index()
        tier_ct.columns = ["Tier", "Customers"]
        tier_ct["Total CLV"] = tier_ct["Tier"].map(
            repeat.groupby("clv_segment")["clv_12m"].sum()
        )
        fig_pie = px.pie(
            tier_ct,
            names="Tier",
            values="Customers",
            hole=0.45,
            color="Tier",
            color_discrete_map=_SEG_COLOR,
        )
        fig_pie.update_traces(textinfo="percent+label")
        fig_pie.update_layout(
            height=340, showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**Predicted Purchases by Horizon**")
    hor_cols = ["pred_purchases_90d", "pred_purchases_180d", "pred_purchases_365d"]
    hor_labels = ["90 days", "180 days", "365 days"]
    avg_purchases = [repeat[c].mean() for c in hor_cols]

    fig_hor = go.Figure(go.Bar(
        x=hor_labels,
        y=avg_purchases,
        marker_color=["#90CAF9", "#42A5F5", "#1565C0"],
        text=[f"{v:.2f}" for v in avg_purchases],
        textposition="outside",
    ))
    fig_hor.update_layout(
        height=260, yaxis_title="Avg Expected Purchases",
        showlegend=False, margin=dict(t=10, b=40, l=10, r=10),
    )
    st.plotly_chart(fig_hor, use_container_width=True)

    col_prob, col_mono = st.columns(2)

    with col_prob:
        st.markdown("**Probability Alive Distribution**")
        fig_alive = px.histogram(
            clv,
            x="prob_alive",
            nbins=40,
            color_discrete_sequence=["#42A5F5"],
            labels={"prob_alive": "P(Alive)"},
        )
        fig_alive.update_layout(
            height=240,
            xaxis_tickformat=".0%",
            margin=dict(t=10, b=40, l=10, r=10),
        )
        st.plotly_chart(fig_alive, use_container_width=True)

    with col_mono:
        st.markdown("**Avg Transaction Value Distribution**")
        fig_mono = px.histogram(
            repeat,
            x="monetary_value",
            nbins=40,
            color_discrete_sequence=["#66BB6A"],
            labels={"monetary_value": "Avg Transaction Value (₹)"},
        )
        fig_mono.update_layout(
            height=240,
            xaxis_tickprefix="₹",
            margin=dict(t=10, b=40, l=10, r=10),
        )
        st.plotly_chart(fig_mono, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# Tab 2 — Segment Breakdown
# ════════════════════════════════════════════════════════════════════════════
with tab_seg:
    seg_clv = (
        repeat.groupby("Segment")
        .agg(
            Customers    = ("Customer ID",   "count"),
            Total_CLV    = ("clv_12m",        "sum"),
            Avg_CLV      = ("clv_12m",        "mean"),
            Avg_Alive    = ("prob_alive",     "mean"),
            Avg_Purchases= (horizon_col,      "mean"),
        )
        .reset_index()
        .sort_values("Total_CLV", ascending=False)
    )

    col_bar, col_seg = st.columns(2)

    with col_bar:
        st.markdown(f"**Total 12m CLV by RFM Segment**")
        fig_seg = px.bar(
            seg_clv.sort_values("Total_CLV"),
            x="Total_CLV",
            y="Segment",
            orientation="h",
            text=seg_clv.sort_values("Total_CLV")["Total_CLV"].map(
                lambda v: f"₹{v:,.0f}"
            ),
            color="Avg_CLV",
            color_continuous_scale="Blues",
            labels={"Total_CLV": "Total CLV (₹)", "Avg_CLV": "Avg CLV"},
        )
        fig_seg.update_traces(textposition="outside")
        fig_seg.update_layout(
            height=360, coloraxis_showscale=False,
            xaxis_tickprefix="₹",
            margin=dict(t=10, b=40, l=10, r=10),
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_seg:
        st.markdown(f"**Avg Expected Purchases ({horizon}) by Segment**")
        fig_pur = px.bar(
            seg_clv.sort_values("Avg_Purchases"),
            x="Avg_Purchases",
            y="Segment",
            orientation="h",
            text=seg_clv.sort_values("Avg_Purchases")["Avg_Purchases"].map(
                lambda v: f"{v:.2f}"
            ),
            color="Avg_Alive",
            color_continuous_scale="Greens",
            labels={"Avg_Purchases": f"Avg Predicted Purchases", "Avg_Alive": "P(Alive)"},
        )
        fig_pur.update_traces(textposition="outside")
        fig_pur.update_layout(
            height=360,
            coloraxis_colorbar=dict(title="P(Alive)", tickformat=".0%"),
            margin=dict(t=10, b=40, l=10, r=10),
        )
        st.plotly_chart(fig_pur, use_container_width=True)

    st.markdown("**Segment Summary Table**")
    display_seg = seg_clv.copy()
    display_seg["Total_CLV"]     = display_seg["Total_CLV"].map("₹{:,.0f}".format)
    display_seg["Avg_CLV"]       = display_seg["Avg_CLV"].map("₹{:,.0f}".format)
    display_seg["Avg_Alive"]     = display_seg["Avg_Alive"].map("{:.1%}".format)
    display_seg["Avg_Purchases"] = display_seg["Avg_Purchases"].map("{:.2f}".format)
    st.dataframe(
        display_seg.rename(columns={
            "Customers":     "Customers",
            "Total_CLV":     "Total 12m CLV",
            "Avg_CLV":       "Avg CLV",
            "Avg_Alive":     "P(Alive)",
            "Avg_Purchases": f"Avg Purchases ({horizon})",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# Tab 3 — CLV × Churn Quadrant
# ════════════════════════════════════════════════════════════════════════════
with tab_quad:
    st.markdown(
        "**Strategic Quadrant — CLV vs Churn Risk**\n\n"
        "Each bubble is a customer. "
        "Top-left = Protect (high value, low risk). "
        "Top-right = Urgent (high value, high risk). "
        "Bubble size = predicted purchases next 180 days."
    )

    quad_df = repeat.dropna(subset=["churn_probability", "clv_12m"]).copy()
    clv_mid   = quad_df["clv_12m"].median()
    churn_mid = 0.5

    fig_quad = px.scatter(
        quad_df,
        x="churn_probability",
        y="clv_12m",
        color="clv_segment",
        color_discrete_map=_SEG_COLOR,
        size="pred_purchases_180d",
        size_max=18,
        hover_data={
            "Customer ID":          True,
            "Segment":              True,
            "clv_12m":              ":,.0f",
            "churn_probability":    ":.1%",
            "prob_alive":           ":.1%",
            "pred_purchases_180d":  ":.2f",
        },
        labels={
            "churn_probability":   "Churn Probability",
            "clv_12m":             "12-Month CLV (₹)",
            "clv_segment":         "CLV Tier",
            "pred_purchases_180d": "Pred. Purchases (180d)",
        },
        opacity=0.70,
    )
    fig_quad.add_vline(x=churn_mid, line_dash="dot", line_color="#aaa", line_width=1)
    fig_quad.add_hline(y=clv_mid,   line_dash="dot", line_color="#aaa", line_width=1)

    # Quadrant labels
    y_max = quad_df["clv_12m"].quantile(0.97)
    for label, cx, cy in [
        ("PROTECT & GROW",  0.05, y_max * 0.90),
        ("URGENT ACTION",   0.72, y_max * 0.90),
        ("MONITOR",         0.05, clv_mid * 0.15),
        ("LOW PRIORITY",    0.72, clv_mid * 0.15),
    ]:
        fig_quad.add_annotation(
            x=cx, y=cy, text=label, showarrow=False,
            font=dict(size=9, color="#999"),
            xref="x", yref="y",
        )

    fig_quad.update_layout(
        height=480,
        xaxis=dict(tickformat=".0%", title="Churn Probability"),
        yaxis=dict(tickprefix="₹", tickformat=",.0f", title="12-Month CLV (₹)"),
        legend_title_text="CLV Tier",
        margin=dict(t=10, b=40, l=10, r=10),
    )
    st.plotly_chart(fig_quad, use_container_width=True)

    # Quadrant count summary
    urgent = quad_df[(quad_df["churn_probability"] > churn_mid) & (quad_df["clv_12m"] > clv_mid)]
    protect = quad_df[(quad_df["churn_probability"] <= churn_mid) & (quad_df["clv_12m"] > clv_mid)]
    qa, qb, qc, qd = st.columns(4)
    qa.metric("Urgent Action",   f"{len(urgent):,}",
              f"₹{urgent['clv_12m'].sum():,.0f} CLV at risk",
              delta_color="inverse")
    qb.metric("Protect & Grow",  f"{len(protect):,}",
              f"₹{protect['clv_12m'].sum():,.0f} secured CLV")
    qc.metric("CLV at Risk",
              f"₹{urgent['clv_12m'].sum():,.0f}",
              "from high-churn, high-value customers",
              delta_color="inverse")
    qd.metric("Median CLV Threshold", f"₹{clv_mid:,.0f}")


# ════════════════════════════════════════════════════════════════════════════
# Tab 4 — Top Customers
# ════════════════════════════════════════════════════════════════════════════
with tab_top:
    st.markdown("**Top Customers by 12-Month CLV**")

    n_top = st.slider("Show top N customers", 10, 100, 25, step=5)

    top_df = (
        view
        .sort_values("clv_12m", ascending=False)
        .head(n_top)
        .reset_index(drop=True)
    )

    display_top = top_df[[
        "Customer ID", "Segment", "clv_segment", "clv_12m",
        "prob_alive", "pred_purchases_90d", "pred_purchases_180d",
        "pred_purchases_365d", "monetary_value", "frequency", "recency",
    ]].copy()

    display_top["clv_12m"]              = display_top["clv_12m"].map("₹{:,.0f}".format)
    display_top["monetary_value"]       = display_top["monetary_value"].map("₹{:,.0f}".format)
    display_top["prob_alive"]           = display_top["prob_alive"].map("{:.1%}".format)
    display_top["pred_purchases_90d"]   = display_top["pred_purchases_90d"].map("{:.2f}".format)
    display_top["pred_purchases_180d"]  = display_top["pred_purchases_180d"].map("{:.2f}".format)
    display_top["pred_purchases_365d"]  = display_top["pred_purchases_365d"].map("{:.2f}".format)
    display_top["recency"]              = display_top["recency"].astype(int)

    st.dataframe(
        display_top.rename(columns={
            "clv_segment":          "CLV Tier",
            "clv_12m":              "12m CLV",
            "prob_alive":           "P(Alive)",
            "pred_purchases_90d":   "Pred Buys 90d",
            "pred_purchases_180d":  "Pred Buys 180d",
            "pred_purchases_365d":  "Pred Buys 365d",
            "monetary_value":       "Avg Txn Value",
            "frequency":            "Past Purchases",
            "recency":              "Days Since Last Buy",
        }),
        use_container_width=True,
        hide_index=True,
        height=480,
    )

    st.caption(
        f"Filters active: Segment={seg_sel} · CLV Tier={clv_seg_sel} · "
        f"Showing {len(top_df):,} of {len(view):,} filtered customers."
    )
