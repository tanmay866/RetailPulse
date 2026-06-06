"""Command Center — Fully Interactive Unified Dashboard.

Unified executive dashboard combining:
- Domain health scorecards (Revenue, Churn Risk, Inventory, RFM Quality)
- Revenue-at-risk cross-domain metric
- Interactive multi-tab analytics (Revenue & Forecast, Customer Intelligence,
  Inventory Pulse, Cross-Domain Insights)
- Smart recommendation engine driven by live data thresholds
"""
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
    load_churn_predictions,
    load_daily_revenue_rolling,
    load_daily_revenue_ts,
    load_inventory_recommendations,
    load_prophet_model,
    load_rfm_scores,
    load_segmentation_churn_merged,
)

# ── Palette ──────────────────────────────────────────────────────────────────
SEG_COLOR = {
    "Champions":           "#1565C0",
    "Loyal Customers":     "#2E7D32",
    "Potential Loyalists": "#558B2F",
    "At Risk":             "#E65100",
    "Lost":                "#C62828",
}

# ── Page ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Command Center",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Command Center")
st.caption(
    "Unified executive dashboard — revenue, customer health, "
    "inventory, and demand forecast in one interactive view."
)

# ── Load data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load() -> dict:
    return {
        "rolling": load_daily_revenue_rolling(),
        "ts":      load_daily_revenue_ts(),
        "rfm":     load_rfm_scores(),
        "churn":   load_churn_predictions(),
        "inv":     load_inventory_recommendations(),
        "merged":  load_segmentation_churn_merged(),
    }

@st.cache_data(show_spinner=False)
def _forecast(horizon: int) -> pd.DataFrame:
    m = load_prophet_model()
    future = m.make_future_dataframe(periods=horizon, freq="D")
    return m.predict(future)

with st.spinner("Loading analytics…"):
    D = _load()

rolling = D["rolling"]
rfm     = D["rfm"]
churn   = D["churn"]
inv     = D["inv"]
merged  = D["merged"]

# ── Sidebar global filters ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Global Filters")

    min_date = rolling["Date"].min().date()
    max_date = rolling["Date"].max().date()
    date_range = st.date_input(
        "Revenue Date Range",
        value=(pd.Timestamp(max_date) - pd.Timedelta(days=180), max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.divider()
    seg_opts   = sorted(rfm["Segment"].unique())
    seg_filter = st.multiselect("Customer Segments", seg_opts, default=seg_opts)

    st.divider()
    churn_thr = st.slider(
        "High-Risk Churn Threshold", 0.0, 1.0, 0.70, 0.05,
        help="Customers above this probability are flagged high-risk.",
    )

    st.divider()
    fc_horizon = st.radio(
        "Forecast Horizon", [7, 14, 30], index=2,
        format_func=lambda x: f"{x} days",
    )

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    d_start = pd.Timestamp(date_range[0])
    d_end   = pd.Timestamp(date_range[1])
else:
    d_start = pd.Timestamp(max_date) - pd.Timedelta(days=180)
    d_end   = pd.Timestamp(max_date)

# ── Health score computation ──────────────────────────────────────────────────
def _compute_scores() -> dict:
    # Revenue: compare last-30d total vs prior-30d total
    last_30 = rolling.tail(30)["Revenue"].sum()
    prev_30 = rolling.tail(60).head(30)["Revenue"].sum()
    pct_chg = (last_30 / max(prev_30, 1) - 1) * 100
    rev_score = float(np.clip(50 + pct_chg * 0.8, 0, 100))

    # Churn: penalise high- and medium-risk customers
    pct_hi  = (churn["churn_probability"] > churn_thr).mean()
    pct_med = ((churn["churn_probability"] > 0.40) &
               (churn["churn_probability"] <= churn_thr)).mean()
    churn_score = float(np.clip(100 - pct_hi * 200 - pct_med * 60, 0, 100))

    # Inventory: weighted optimal vs stockout
    pct_opt   = (inv["status"] == "OPTIMAL").mean()
    pct_stock = (inv["status"] == "STOCKOUT_RISK").mean()
    inv_score = float(np.clip(pct_opt * 80 + (1 - pct_stock) * 20, 0, 100))

    # RFM quality: weighted segment composition
    seg_w = {"Champions": 100, "Loyal Customers": 80,
              "Potential Loyalists": 60, "At Risk": 20, "Lost": 5}
    seg_pct   = rfm["Segment"].value_counts(normalize=True)
    rfm_score = float(np.clip(
        sum(seg_pct.get(s, 0) * w for s, w in seg_w.items()), 0, 100
    ))

    return {
        "revenue":   rev_score,
        "churn":     churn_score,
        "inventory": inv_score,
        "rfm":       rfm_score,
    }

scores = _compute_scores()


def _score_color(s: float) -> str:
    return "#2E7D32" if s >= 70 else ("#E65100" if s >= 45 else "#C62828")


def _score_label(s: float) -> str:
    return "HEALTHY" if s >= 70 else ("CAUTION" if s >= 45 else "CRITICAL")


# ── Section 1: Health Scorecard ───────────────────────────────────────────────
st.subheader("Domain Health Scorecard")

def _gauge_card(col: st.delta_generator.DeltaGenerator,
                title: str, score: float, detail: str) -> None:
    color = _score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 30, "color": color}},
        gauge={
            "axis":    {"range": [0, 100], "tickwidth": 1, "tickcolor": "#aaa",
                        "tickvals": [0, 25, 50, 75, 100]},
            "bar":     {"color": color, "thickness": 0.28},
            "bgcolor": "white",
            "steps":   [
                {"range": [0,  45], "color": "#FFEBEE"},
                {"range": [45, 70], "color": "#FFF8E1"},
                {"range": [70, 100], "color": "#E8F5E9"},
            ],
        },
        title={"text": f"<b>{title}</b>", "font": {"size": 13}},
    ))
    fig.update_layout(height=200, margin=dict(t=35, b=0, l=15, r=15))
    col.plotly_chart(fig, use_container_width=True)
    col.markdown(
        f"<div style='text-align:center;margin-top:-14px;'>"
        f"<span style='font-weight:700;color:{color};font-size:13px'>"
        f"{_score_label(score)}</span>"
        f"<br><span style='color:#777;font-size:11px'>{detail}</span></div>",
        unsafe_allow_html=True,
    )

last_30_rev = rolling.tail(30)["Revenue"].sum()
prev_30_rev = rolling.tail(60).head(30)["Revenue"].sum()
rev_delta   = (last_30_rev / max(prev_30_rev, 1) - 1) * 100
high_risk_n = int((churn["churn_probability"] > churn_thr).sum())
pct_opt     = (inv["status"] == "OPTIMAL").mean() * 100
champions_n = int((rfm["Segment"] == "Champions").sum())

g1, g2, g3, g4 = st.columns(4)
_gauge_card(g1, "Revenue Health",    scores["revenue"],
            f"30d trend: {rev_delta:+.1f}%")
_gauge_card(g2, "Churn Risk Score",  scores["churn"],
            f"{high_risk_n:,} high-risk customers")
_gauge_card(g3, "Inventory Health",  scores["inventory"],
            f"{pct_opt:.0f}% SKUs optimal")
_gauge_card(g4, "RFM Quality Score", scores["rfm"],
            f"{champions_n:,} Champions")

st.divider()

# ── Section 2: KPI Strip ──────────────────────────────────────────────────────
st.subheader("Key Performance Indicators")

total_rev   = float(merged["Monetary"].sum()) + float(
    D["rolling"].tail(1)["Revenue"].iloc[0])   # approx from available data
total_rev   = float(rolling["Revenue"].sum())
uniq_cust   = rfm["Customer ID"].nunique()
churn_rate  = (churn["predicted_churn"] == 1).mean() * 100
rev_at_risk = float((merged["Monetary"] * merged["churn_probability"]).sum())
stockout_n  = int((inv["status"] == "STOCKOUT_RISK").sum())
critical_n  = int((inv["days_of_stock"] < 1).sum())
avg_dos     = inv["days_of_stock"].mean()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Revenue",        f"₹{total_rev:,.0f}")
k2.metric("Last 30d Revenue",     f"₹{last_30_rev:,.0f}",
          f"{rev_delta:+.1f}% vs prior 30d",
          delta_color="normal" if rev_delta >= 0 else "inverse")
k3.metric("Customers Analysed",   f"{uniq_cust:,}")
k4.metric(f"High Risk (>{churn_thr:.0%})", f"{high_risk_n:,}",
          f"{high_risk_n/len(churn)*100:.1f}% of base",
          delta_color="inverse" if high_risk_n > 100 else "off")
k5.metric("Stockout SKUs",        f"{stockout_n:,}",
          f"{critical_n} critical (<1d)",
          delta_color="inverse" if critical_n > 0 else "off")
k6.metric("Revenue at Risk",      f"₹{rev_at_risk:,.0f}",
          help="Sum(Monetary x churn_probability) across all modelled customers")

k7, k8, k9, k10, k11, k12 = st.columns(6)
k7.metric("Champions",            f"{champions_n:,}",
          f"{champions_n/uniq_cust*100:.1f}% of base")
k8.metric("Predicted Churn Rate", f"{churn_rate:.1f}%")
k9.metric("Avg Days of Stock",    f"{avg_dos:.1f}",
          delta_color="inverse" if avg_dos < 5 else "off")
k10.metric("At Risk Customers",
           f"{int((rfm['Segment']=='At Risk').sum()):,}")
k11.metric("Lost Customers",
           f"{int((rfm['Segment']=='Lost').sum()):,}")
k12.metric("Inventory Optimal",   f"{pct_opt:.0f}%",
           delta_color="inverse" if pct_opt < 50 else "normal")

st.divider()

# ── Section 3: Tabs ──────────────────────────────────────────────────────────
tab_rev, tab_cust, tab_inv, tab_cross = st.tabs([
    "Revenue & Forecast",
    "Customer Intelligence",
    "Inventory Pulse",
    "Cross-Domain Insights",
])

# ── Tab 1: Revenue & Forecast ────────────────────────────────────────────────
with tab_rev:
    rev_filtered = rolling[
        (rolling["Date"] >= d_start) & (rolling["Date"] <= d_end)
    ].copy()

    st.markdown("**Revenue Trend** — filtered by sidebar date range")
    fig_rev = go.Figure()

    # ±1σ band
    fig_rev.add_trace(go.Scatter(
        x=pd.concat([rev_filtered["Date"], rev_filtered["Date"].iloc[::-1]]),
        y=pd.concat([
            rev_filtered["rolling_30d_mean"] + rev_filtered["rolling_30d_std"],
            (rev_filtered["rolling_30d_mean"] - rev_filtered["rolling_30d_std"]).iloc[::-1],
        ]),
        fill="toself", fillcolor="rgba(33,150,243,0.07)",
        line=dict(color="rgba(0,0,0,0)"), name="±1σ", hoverinfo="skip",
    ))

    fig_rev.add_trace(go.Bar(
        x=rev_filtered["Date"], y=rev_filtered["Revenue"],
        name="Daily Revenue", marker_color="#90CAF9", opacity=0.75,
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>₹%{y:,.0f}<extra></extra>",
    ))
    fig_rev.add_trace(go.Scatter(
        x=rev_filtered["Date"], y=rev_filtered["rolling_7d_mean"],
        name="7d Mean", line=dict(color="#E65100", width=1.5, dash="dot"),
    ))
    fig_rev.add_trace(go.Scatter(
        x=rev_filtered["Date"], y=rev_filtered["rolling_30d_mean"],
        name="30d Mean", line=dict(color="#1565C0", width=2),
    ))

    fig_rev.update_layout(
        height=370, hovermode="x unified",
        yaxis_title="Revenue (₹)", bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        margin=dict(t=10, b=10),
    )
    fig_rev.update_yaxes(tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig_rev, use_container_width=True)

    # Period stats
    ps1, ps2, ps3, ps4 = st.columns(4)
    period_rev = rev_filtered["Revenue"].sum()
    period_days = len(rev_filtered)
    ps1.metric("Period Revenue", f"₹{period_rev:,.0f}")
    ps2.metric("Days in Period",  f"{period_days}")
    ps3.metric("Avg Daily",       f"₹{period_rev/max(period_days,1):,.0f}")
    rev_hi = rev_filtered.loc[rev_filtered["Revenue"].idxmax()]
    ps4.metric("Peak Day",
               f"₹{rev_hi['Revenue']:,.0f}",
               f"{rev_hi['Date'].strftime('%d %b %Y')}")

    st.markdown("**30-Day Demand Forecast**")
    with st.spinner("Running Prophet forecast…"):
        try:
            fc = _forecast(fc_horizon)
            model = load_prophet_model()
            train_end = model.history["ds"].max()
            fc_fut = fc[fc["ds"] > train_end].copy()
            ts     = D["ts"]

            fig_fc = go.Figure()
            plot_hist = ts[ts["Date"] >= train_end - pd.Timedelta(days=60)]
            fig_fc.add_trace(go.Scatter(
                x=plot_hist["Date"], y=plot_hist["Revenue"],
                name="Actual (last 60d)", line=dict(color="#2196F3", width=1.5),
            ))
            fig_fc.add_trace(go.Scatter(
                x=pd.concat([fc_fut["ds"], fc_fut["ds"].iloc[::-1]]),
                y=pd.concat([fc_fut["yhat_upper"], fc_fut["yhat_lower"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(76,175,80,0.15)",
                line=dict(color="rgba(0,0,0,0)"), name="80% CI", hoverinfo="skip",
            ))
            fig_fc.add_trace(go.Scatter(
                x=fc_fut["ds"], y=fc_fut["yhat"].clip(lower=0),
                name=f"Prophet forecast ({fc_horizon}d)",
                line=dict(color="#4CAF50", width=2.5),
            ))
            proj_total = float(fc_fut["yhat"].clip(lower=0).sum())
            fig_fc.update_layout(
                height=330, hovermode="x unified",
                yaxis_title="Revenue (₹)",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
                margin=dict(t=10, b=10),
            )
            fig_fc.update_yaxes(tickprefix="₹", tickformat=",.0f")
            st.plotly_chart(fig_fc, use_container_width=True)
            fa, fb, fc_ = st.columns(3)
            fa.metric(f"Projected {fc_horizon}d Revenue", f"₹{proj_total:,.0f}")
            fb.metric("Projected Daily Avg",              f"₹{proj_total/fc_horizon:,.0f}")
            fc_.metric("vs Last 30d Daily Avg",
                       f"{(proj_total/fc_horizon) / max(last_30_rev/30, 1) * 100 - 100:+.1f}%")
        except Exception as exc:
            st.warning(f"Forecast unavailable: {exc}")

# ── Tab 2: Customer Intelligence ─────────────────────────────────────────────
with tab_cust:
    merged_filt = merged[merged["Segment"].isin(seg_filter)].copy()
    merged_filt["risk_tier"] = pd.cut(
        merged_filt["churn_probability"],
        bins=[-0.001, 0.30, churn_thr, 1.001],
        labels=["Low", "Medium", "High"],
    )

    st.markdown(
        f"Showing **{len(merged_filt):,}** customers across "
        f"**{len(seg_filter)}** selected segments."
    )

    col_tm, col_sc = st.columns([5, 6])

    with col_tm:
        st.markdown("**Segment Value Map**")
        seg_agg = (
            merged_filt.groupby("Segment")
            .agg(
                Customers       = ("Customer ID", "count"),
                Total_Revenue   = ("Monetary",           "sum"),
                Avg_Churn       = ("churn_probability",  "mean"),
                Rev_at_Risk     = ("churn_probability",
                                   lambda x: (x * merged_filt.loc[x.index, "Monetary"]).sum()),
            )
            .reset_index()
        )
        seg_agg["color_label"] = seg_agg["Avg_Churn"].map(lambda v: f"{v:.0%} avg churn")

        fig_tm = px.treemap(
            seg_agg,
            path=["Segment"],
            values="Total_Revenue",
            color="Avg_Churn",
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 1],
            custom_data=["Customers", "Rev_at_Risk", "Avg_Churn"],
            labels={"Total_Revenue": "Total Revenue (₹)", "Avg_Churn": "Avg Churn Prob"},
        )
        fig_tm.update_traces(
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Revenue: ₹%{value:,.0f}<br>"
                "Customers: %{customdata[0]:,}<br>"
                "Rev at Risk: ₹%{customdata[1]:,.0f}<br>"
                "Avg Churn: %{customdata[2]:.1%}"
                "<extra></extra>"
            )
        )
        fig_tm.update_layout(
            height=360,
            coloraxis_colorbar=dict(title="Avg Churn", tickformat=".0%"),
            margin=dict(t=10, b=5, l=5, r=5),
        )
        st.plotly_chart(fig_tm, use_container_width=True)

    with col_sc:
        st.markdown("**Customer Risk Map (Recency × Spend)**")
        fig_sc = px.scatter(
            merged_filt,
            x="Recency",
            y="Monetary",
            color="churn_probability",
            size="Frequency",
            size_max=15,
            hover_data={
                "Customer ID":       True,
                "Segment":           True,
                "churn_probability": ":.1%",
                "Recency":           True,
                "Monetary":          ":,.0f",
            },
            color_continuous_scale="RdYlGn_r",
            range_color=[0, 1],
            labels={
                "Recency":           "Recency (days)",
                "Monetary":          "Lifetime Spend (₹)",
                "churn_probability": "Churn Prob",
            },
            opacity=0.70,
        )
        fig_sc.update_layout(
            height=360,
            coloraxis_colorbar=dict(title="Churn Prob", tickformat=".0%"),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("**Top 20 High-Value At-Risk Customers** (highest Revenue-at-Risk)")
    rar_df = merged_filt.copy()
    rar_df["rev_at_risk"] = rar_df["Monetary"] * rar_df["churn_probability"]
    top_rar = (
        rar_df[rar_df["churn_probability"] >= churn_thr]
        .sort_values("rev_at_risk", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    if top_rar.empty:
        st.info("No customers above the current churn threshold.")
    else:
        display_top = top_rar[[
            "Customer ID", "Segment", "Recency", "Frequency",
            "Monetary", "churn_probability", "rev_at_risk",
        ]].rename(columns={
            "churn_probability": "Churn Prob",
            "rev_at_risk":       "Rev at Risk (₹)",
            "Monetary":          "Lifetime Spend (₹)",
        })
        display_top["Churn Prob"]       = display_top["Churn Prob"].map("{:.1%}".format)
        display_top["Lifetime Spend (₹)"] = display_top["Lifetime Spend (₹)"].map("₹{:,.0f}".format)
        display_top["Rev at Risk (₹)"]  = display_top["Rev at Risk (₹)"].map("₹{:,.0f}".format)
        st.dataframe(display_top, use_container_width=True, hide_index=True, height=300)

# ── Tab 3: Inventory Pulse ────────────────────────────────────────────────────
with tab_inv:
    inv_agg = inv.groupby(["category", "region"])["status"].apply(
        lambda s: (s == "STOCKOUT_RISK").mean() * 100
    ).reset_index()
    inv_agg.columns = ["Category", "Region", "Stockout %"]

    col_stat, col_heat = st.columns([4, 7])

    with col_stat:
        st.markdown("**Status Distribution**")
        status_ct = inv["status"].value_counts().reset_index()
        status_ct.columns = ["Status", "Count"]
        status_ct["Pct"] = (status_ct["Count"] / len(inv) * 100).round(1)

        _inv_color = {"STOCKOUT_RISK": "#EF5350", "OPTIMAL": "#66BB6A"}
        fig_pie = go.Figure(go.Pie(
            labels=status_ct["Status"],
            values=status_ct["Count"],
            hole=0.45,
            marker_colors=[_inv_color.get(s, "#90A4AE") for s in status_ct["Status"]],
            textinfo="label+percent",
        ))
        fig_pie.update_layout(
            height=280, showlegend=False,
            margin=dict(t=20, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        i1, i2 = st.columns(2)
        i1.metric("Total SKUs",   f"{len(inv):,}")
        i2.metric("Critical",     f"{critical_n:,}", "< 1 day left",
                  delta_color="inverse" if critical_n > 0 else "off")
        i1.metric("Avg Days Stock", f"{avg_dos:.1f}")
        i2.metric("Avg EOQ",       f"{inv['eoq'].mean():.0f}")

    with col_heat:
        st.markdown("**Stockout Rate — Category × Region**")
        heat_pivot = inv_agg.pivot(index="Category", columns="Region", values="Stockout %").fillna(0)
        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale="RdYlGn_r",
            zmin=0, zmax=100,
            text_auto=".0f",
            labels={"color": "Stockout %"},
            aspect="auto",
        )
        fig_heat.update_layout(
            height=280,
            margin=dict(t=20, b=10, l=10, r=10),
            coloraxis_colorbar=dict(title="Stockout %", ticksuffix="%"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("**Critical SKUs — Days of Stock < 5**")
    crit_df = (
        inv[inv["days_of_stock"] < 5]
        .sort_values("days_of_stock")
        .head(30)
        [["store_id", "product_id", "category", "region",
          "current_inventory", "days_of_stock", "units_to_order", "status"]]
        .rename(columns={
            "store_id":          "Store",
            "product_id":        "Product",
            "category":          "Category",
            "region":            "Region",
            "current_inventory": "Current Inv",
            "days_of_stock":     "Days Left",
            "units_to_order":    "To Order",
            "status":            "Status",
        })
    )
    if crit_df.empty:
        st.success("No SKUs with fewer than 5 days of stock remaining.")
    else:
        st.dataframe(
            crit_df.style.map(
                lambda v: "color:#C62828;font-weight:700"
                if isinstance(v, str) and v == "STOCKOUT_RISK" else "",
                subset=["Status"],
            ),
            use_container_width=True,
            hide_index=True,
            height=300,
        )

# ── Tab 4: Cross-Domain Insights ──────────────────────────────────────────────
with tab_cross:
    # Revenue-at-risk by segment
    seg_rar = (
        merged.groupby("Segment")
        .apply(
            lambda g: pd.Series({
                "total_rev":    g["Monetary"].sum(),
                "rev_at_risk":  (g["Monetary"] * g["churn_probability"]).sum(),
                "customers":    len(g),
                "avg_churn":    g["churn_probability"].mean(),
            }),
            include_groups=False,
        )
        .reset_index()
        .sort_values("rev_at_risk", ascending=False)
    )
    seg_rar["pct_at_risk"] = seg_rar["rev_at_risk"] / seg_rar["total_rev"] * 100

    col_rar, col_mat = st.columns([5, 6])

    with col_rar:
        st.markdown("**Revenue at Risk by Segment**")
        fig_rar = go.Figure()
        fig_rar.add_trace(go.Bar(
            x=seg_rar["Segment"], y=seg_rar["total_rev"],
            name="Total Revenue", marker_color="#90CAF9", opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Total: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_rar.add_trace(go.Bar(
            x=seg_rar["Segment"], y=seg_rar["rev_at_risk"],
            name="Revenue at Risk", marker_color="#EF5350",
            hovertemplate="<b>%{x}</b><br>At Risk: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_rar.update_layout(
            barmode="overlay", height=320,
            yaxis_title="Revenue (₹)",
            yaxis=dict(tickprefix="₹", tickformat=",.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1),
            margin=dict(t=10, b=60, l=10, r=10),
        )
        st.plotly_chart(fig_rar, use_container_width=True)
        st.caption(
            f"Total revenue at risk across all segments: "
            f"**₹{seg_rar['rev_at_risk'].sum():,.0f}**"
        )

    with col_mat:
        st.markdown("**Segment Performance Matrix**")
        fig_mat = px.scatter(
            seg_rar,
            x="avg_churn",
            y="total_rev",
            size="customers",
            color="Segment",
            color_discrete_map=SEG_COLOR,
            text="Segment",
            custom_data=["rev_at_risk", "pct_at_risk", "customers"],
            labels={
                "avg_churn":  "Avg Churn Probability",
                "total_rev":  "Total Historical Revenue (₹)",
                "Segment":    "Segment",
            },
            size_max=55,
        )
        fig_mat.update_traces(
            textposition="top center", textfont_size=10,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Avg Churn: %{x:.1%}<br>"
                "Total Rev: ₹%{y:,.0f}<br>"
                "Rev at Risk: ₹%{customdata[0]:,.0f} (%{customdata[1]:.1f}%)<br>"
                "Customers: %{customdata[2]:,}"
                "<extra></extra>"
            ),
        )
        fig_mat.update_layout(
            height=320, showlegend=False,
            xaxis=dict(tickformat=".0%", title="Avg Churn Probability"),
            yaxis=dict(tickprefix="₹", tickformat=",.0f",
                       title="Total Revenue (₹)"),
            margin=dict(t=10, b=40, l=10, r=10),
        )
        # Quadrant labels
        x_mid = seg_rar["avg_churn"].mean()
        y_mid = seg_rar["total_rev"].mean()
        for label, ax, ay in [
            ("Protect & Grow", 0.05, y_mid * 1.9),
            ("Urgent Action",  0.75, y_mid * 1.9),
            ("Monitor",        0.05, y_mid * 0.1),
            ("Low Priority",   0.75, y_mid * 0.1),
        ]:
            fig_mat.add_annotation(
                x=ax, y=ay, text=label, showarrow=False,
                font=dict(size=9, color="#888"),
                xref="x", yref="y",
            )
        fig_mat.add_vline(x=x_mid, line_dash="dot",
                          line_color="#ccc", line_width=1)
        fig_mat.add_hline(y=y_mid, line_dash="dot",
                          line_color="#ccc", line_width=1)
        st.plotly_chart(fig_mat, use_container_width=True)

    # Revenue-at-risk summary table
    st.markdown("**Revenue at Risk — Segment Detail**")
    rar_display = seg_rar.copy()
    rar_display["total_rev"]   = rar_display["total_rev"].map("₹{:,.0f}".format)
    rar_display["rev_at_risk"] = rar_display["rev_at_risk"].map("₹{:,.0f}".format)
    rar_display["avg_churn"]   = rar_display["avg_churn"].map("{:.1%}".format)
    rar_display["pct_at_risk"] = rar_display["pct_at_risk"].map("{:.1f}%".format)
    st.dataframe(
        rar_display.rename(columns={
            "total_rev":   "Total Revenue",
            "rev_at_risk": "Revenue at Risk",
            "customers":   "Customers",
            "avg_churn":   "Avg Churn Prob",
            "pct_at_risk": "% Revenue at Risk",
        }),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Section 4: Smart Recommendations ────────────────────────────────────────
st.subheader("Smart Recommendations")
st.caption("Auto-generated from live data thresholds — prioritised by business impact.")

recs: list[tuple[str, str, str, str]] = []

# Rule 1: Champions at churn risk
champ_ids = set(rfm.loc[rfm["Segment"] == "Champions", "Customer ID"])
prob_map  = churn.set_index("Customer_ID")["churn_probability"].to_dict()
champs_at_risk = [cid for cid in champ_ids if prob_map.get(cid, 0) > churn_thr]
if champs_at_risk:
    recs.append((
        "CRITICAL",
        "Retain Champion Customers",
        f"{len(champs_at_risk)} Champion-tier customer(s) have churn probability "
        f"> {churn_thr:.0%}. These are your highest-value accounts.",
        "Launch targeted win-back campaign with personalised offers. "
        "Prioritise accounts with Monetary > ₹10,000.",
    ))

# Rule 2: High stockout rate
stockout_pct = (inv["status"] == "STOCKOUT_RISK").mean() * 100
if stockout_pct > 50:
    recs.append((
        "CRITICAL",
        "Reorder Stockout SKUs Immediately",
        f"{stockout_pct:.0f}% of SKUs ({stockout_n:,}) are below reorder point.",
        f"Place emergency reorders for the {critical_n} SKUs with < 1 day of stock first, "
        "then work through remaining STOCKOUT_RISK items by category.",
    ))
elif stockout_pct > 30:
    recs.append((
        "HIGH",
        "Elevated Stockout Rate",
        f"{stockout_pct:.0f}% of SKUs are at stockout risk.",
        "Review reorder frequency and safety stock levels. "
        "Consider increasing EOQ for high-velocity SKUs.",
    ))

# Rule 3: Revenue trend
if rev_delta < -15:
    recs.append((
        "HIGH",
        "Revenue Decline Detected",
        f"Last 30-day revenue is {rev_delta:.1f}% below prior 30 days.",
        "Investigate demand drivers: check product mix, pricing changes, "
        "and at-risk customer activity. Run targeted promotions.",
    ))
elif rev_delta > 20:
    recs.append((
        "LOW",
        "Revenue Growth Opportunity",
        f"Strong +{rev_delta:.1f}% revenue growth vs prior 30 days.",
        "Ensure sufficient inventory to meet elevated demand. "
        "Consider upsell campaigns to Champions and Loyal Customers.",
    ))

# Rule 4: At-Risk segment
at_risk_n = int((rfm["Segment"] == "At Risk").sum())
at_risk_pct = at_risk_n / len(rfm) * 100
if at_risk_pct > 15:
    recs.append((
        "HIGH",
        "Large At-Risk Segment Requires Attention",
        f"{at_risk_n:,} customers ({at_risk_pct:.1f}%) are in the 'At Risk' segment.",
        "Trigger re-engagement emails with discount codes. "
        "Customers inactive > 90 days should receive win-back sequences.",
    ))

# Rule 5: Potential Loyalists opportunity
pot_loyal_n = int((rfm["Segment"] == "Potential Loyalists").sum())
if pot_loyal_n > 100:
    pot_rar = float(
        (merged.loc[merged["Segment"] == "Potential Loyalists", "Monetary"] *
         merged.loc[merged["Segment"] == "Potential Loyalists", "churn_probability"]).sum()
    )
    recs.append((
        "MEDIUM",
        "Convert Potential Loyalists",
        f"{pot_loyal_n:,} Potential Loyalists with ₹{pot_rar:,.0f} revenue at risk.",
        "This segment responds well to loyalty programmes, product recommendations, "
        "and consistent engagement. Reducing their churn would significantly reduce total revenue at risk.",
    ))

# Rule 6: Low customer health
if scores["churn"] < 45:
    recs.append((
        "HIGH",
        "Customer Health Score Critical",
        f"Churn Risk Score is {scores['churn']:.0f}/100 — many customers at risk.",
        "Review churn model outputs, identify common attributes of high-risk customers, "
        "and implement proactive retention workflows.",
    ))

if not recs:
    st.success("All systems healthy — no critical recommendations at current thresholds.")
else:
    SEV_COLOR_MAP = {
        "CRITICAL": ("#C62828", "#FFEBEE"),
        "HIGH":     ("#E65100", "#FFF3E0"),
        "MEDIUM":   ("#F57F17", "#FFFDE7"),
        "LOW":      ("#1565C0", "#E3F2FD"),
    }
    SEV_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    recs.sort(key=lambda r: SEV_RANK.get(r[0], 99))

    for sev, title, context, action in recs:
        fc_text, bg = SEV_COLOR_MAP.get(sev, ("#555", "#f9f9f9"))
        st.markdown(
            f"""<div style="background:{bg};border-left:4px solid {fc_text};
                padding:12px 16px;border-radius:4px;margin-bottom:8px;">
            <span style="color:{fc_text};font-weight:700;font-size:13px">{sev}</span>
            &ensp;<strong style="font-size:14px">{title}</strong>
            <br><span style="color:#444;font-size:0.88em">{context}</span>
            <br><span style="color:#555;font-size:0.85em"><b>Action:</b> {action}</span>
            </div>""",
            unsafe_allow_html=True,
        )
