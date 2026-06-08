import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(_PROJECT_ROOT))   # for src.metrics

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data_loader import load_daily_revenue_ts, load_prophet_model

# ─────────────────────────────────────────────────────────────────────────────
st.header("Demand Forecasting")

from src.metrics import FORECAST_REQUESTS
FORECAST_REQUESTS.inc()

ts    = load_daily_revenue_ts()
model = load_prophet_model()


@st.cache_data(ttl=3600)
def _get_forecast(horizon: int) -> pd.DataFrame:
    m = load_prophet_model()
    future = m.make_future_dataframe(periods=horizon, freq="D")
    return m.predict(future)


# ── Sidebar: all controls ────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Forecast Settings")
    horizon = st.radio(
        "Forecast horizon",
        [7, 14, 30],
        index=1,
        format_func=lambda x: f"{x} days",
    )

    st.divider()
    st.subheader("What-If Analysis")
    demand_shock = st.slider(
        "Demand change (%)",
        min_value=-50,
        max_value=100,
        value=0,
        step=5,
        help="Simulate a sustained or time-limited change in demand over the forecast window.",
    )
    scenario_mode = st.radio(
        "Applies to",
        ["Entire forecast window", "First N days only"],
        horizontal=True,
    )
    if scenario_mode == "First N days only":
        event_days = st.slider("Event duration (days)", 1, horizon, min(7, horizon))
    else:
        event_days = horizon


# ── Generate forecast ────────────────────────────────────────────────────────
fc = _get_forecast(horizon)
train_end = model.history["ds"].max()
fc_hist   = fc[fc["ds"] <= train_end].copy()
fc_fut    = fc[fc["ds"] > train_end].copy().reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# Section 1 — Interactive Forecast Chart
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Live Forecast")

plot_start    = train_end - pd.Timedelta(days=180)
ts_plot       = ts[ts["Date"] >= plot_start]
fc_hist_plot  = fc_hist[fc_hist["ds"] >= plot_start]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ts_plot["Date"], y=ts_plot["Revenue"],
    name="Actual Revenue",
    line=dict(color="#2196F3", width=1.5),
))

fig.add_trace(go.Scatter(
    x=fc_hist_plot["ds"], y=fc_hist_plot["yhat"],
    name="Prophet fit",
    line=dict(color="#FF9800", width=1, dash="dot"),
    opacity=0.75,
))

# 80% CI band (future only)
fig.add_trace(go.Scatter(
    x=pd.concat([fc_fut["ds"], fc_fut["ds"].iloc[::-1]]),
    y=pd.concat([fc_fut["yhat_upper"], fc_fut["yhat_lower"].iloc[::-1]]),
    fill="toself",
    fillcolor="rgba(76,175,80,0.15)",
    line=dict(color="rgba(0,0,0,0)"),
    name="80% CI",
))

fig.add_trace(go.Scatter(
    x=fc_fut["ds"], y=fc_fut["yhat"].clip(lower=0),
    name=f"Baseline forecast ({horizon}d)",
    line=dict(color="#4CAF50", width=2.5),
))

if demand_shock != 0:
    scenario_y = fc_fut["yhat"].copy()
    scenario_y.iloc[:event_days] = np.clip(
        scenario_y.iloc[:event_days] * (1 + demand_shock / 100), 0, None
    )
    fig.add_trace(go.Scatter(
        x=fc_fut["ds"], y=scenario_y,
        name=f"What-if: {demand_shock:+d}%",
        line=dict(color="#E91E63", width=2.5, dash="dash"),
    ))

fig.update_layout(
    height=430,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Revenue (₹)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=10, b=40, l=10, r=10),
)
fig.update_yaxes(tickprefix="₹", tickformat=",.0f")
st.plotly_chart(fig, use_container_width=True)


# ── What-if impact metrics ───────────────────────────────────────────────────
if demand_shock != 0:
    base_rev     = float(fc_fut["yhat"].iloc[:event_days].clip(lower=0).sum())
    scenario_rev = float(
        (fc_fut["yhat"].iloc[:event_days] * (1 + demand_shock / 100)).clip(lower=0).sum()
    )
    delta     = scenario_rev - base_rev
    delta_pct = delta / base_rev * 100 if base_rev else 0

    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline Revenue", f"₹{base_rev:,.0f}")
    m2.metric("Scenario Revenue", f"₹{scenario_rev:,.0f}", f"₹{delta:+,.0f}")
    m3.metric("Revenue Impact",   f"₹{delta:+,.0f}",       f"{delta_pct:+.1f}%")


st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 2 — Trend & Seasonality Components
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Trend & Seasonality Components")

comp_cols = [c for c in ("trend", "weekly", "yearly") if c in fc.columns]
fig2 = make_subplots(
    rows=len(comp_cols),
    cols=1,
    shared_xaxes=True,
    subplot_titles=[c.title() for c in comp_cols],
    vertical_spacing=0.10,
)
palette = ["#9C27B0", "#00BCD4", "#FF5722"]
for i, (col, color) in enumerate(zip(comp_cols, palette), start=1):
    subset = fc_hist
    fig2.add_trace(
        go.Scatter(
            x=subset["ds"], y=subset[col],
            line=dict(color=color, width=1.5),
            name=col.title(),
            showlegend=False,
        ),
        row=i, col=1,
    )
    fig2.update_yaxes(title_text="Value", row=i, col=1)

fig2.update_layout(
    height=max(350, 130 * len(comp_cols) + 80),
    margin=dict(t=50, b=20, l=10, r=10),
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 3 — Changepoint Explorer
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Changepoint Explorer")
st.caption(
    "Dashed lines mark where Prophet detected significant trend shifts. "
    "Red = upward shift, blue = downward shift. Only the top 50% by magnitude are shown."
)

deltas    = model.params["delta"].mean(axis=0)
cps       = pd.to_datetime(model.changepoints.values)
cp_df     = pd.DataFrame({"date": cps, "delta": deltas})
cp_df["abs_delta"] = np.abs(cp_df["delta"])
threshold = np.percentile(cp_df["abs_delta"], 50)
significant = cp_df[cp_df["abs_delta"] >= threshold]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=ts["Date"], y=ts["Revenue"],
    name="Revenue",
    line=dict(color="#2196F3", width=1),
    opacity=0.55,
))

for _, row in significant.iterrows():
    color = "rgba(244,67,54,0.65)" if row["delta"] > 0 else "rgba(33,150,243,0.65)"
    fig3.add_vline(
        x=row["date"].isoformat(),
        line_width=1.5,
        line_dash="dash",
        line_color=color,
    )

fig3.update_layout(
    height=340,
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Revenue (₹)",
    margin=dict(t=10, b=40, l=10, r=10),
)
fig3.update_yaxes(tickprefix="₹", tickformat=",.0f")
st.plotly_chart(fig3, use_container_width=True)

top10 = cp_df.nlargest(10, "abs_delta")[["date", "delta"]].copy()
top10["Date"]      = top10["date"].apply(lambda d: d.strftime("%Y-%m-%d"))
top10["Delta"]     = top10["delta"].round(4)
top10["Direction"] = top10["delta"].apply(lambda d: "▲ Upward" if d > 0 else "▼ Downward")
st.dataframe(top10[["Date", "Delta", "Direction"]], use_container_width=True, hide_index=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 4 — Forecast Accuracy by Day of Week
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Forecast Accuracy by Day of Week")
st.caption("Mean percentage error (actual − Prophet fit) grouped by day. "
           "Red bars = model over-predicts; blue = under-predicts.")

fc_aligned = fc[fc["ds"].isin(ts["Date"])][["ds", "yhat"]].copy()
merged     = ts.merge(fc_aligned, left_on="Date", right_on="ds", how="inner")
merged["error_pct"] = (
    (merged["Revenue"] - merged["yhat"])
    / merged["Revenue"].replace(0, np.nan) * 100
).fillna(0)
merged["day_name"] = merged["Date"].apply(lambda d: d.day_name())

dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow_err   = merged.groupby("day_name")["error_pct"].mean().reindex(dow_order).fillna(0)

fig4 = go.Figure(go.Bar(
    x=dow_order,
    y=dow_err.values,
    marker_color=["#EF5350" if v > 0 else "#42A5F5" for v in dow_err.values],
    text=[f"{v:+.1f}%" for v in dow_err.values],
    textposition="outside",
))
fig4.update_layout(
    height=310,
    yaxis_title="Mean Error %",
    showlegend=False,
    margin=dict(t=10, b=40, l=10, r=10),
)
fig4.add_hline(y=0, line_width=1, line_color="black")
st.plotly_chart(fig4, use_container_width=True)

st.divider()


# ════════════════════════════════════════════════════════════════════════════
# Section 5 — Weekly & Monthly Revenue Patterns
# ════════════════════════════════════════════════════════════════════════════
st.subheader("Revenue Patterns")

col_a, col_b = st.columns(2)

with col_a:
    avg_dow = (
        ts[ts["Revenue"] > 0]
        .assign(day=lambda d: d["Date"].dt.day_name())
        .groupby("day")["Revenue"]
        .mean()
        .reindex(dow_order)
        .fillna(0)
    )
    fig5a = go.Figure(go.Bar(
        x=dow_order,
        y=avg_dow.values,
        marker_color="#42A5F5",
        text=[f"₹{v:,.0f}" for v in avg_dow.values],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig5a.update_layout(
        title="Avg Daily Revenue by Day of Week",
        height=310,
        showlegend=False,
        yaxis_title="Avg Revenue (₹)",
        margin=dict(t=40, b=40, l=10, r=10),
    )
    fig5a.update_yaxes(tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig5a, use_container_width=True)

with col_b:
    avg_month = (
        ts[ts["Revenue"] > 0]
        .assign(month=lambda d: d["Date"].dt.month)
        .groupby("month")["Revenue"]
        .mean()
        .reset_index()
    )
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    avg_month["month_name"] = avg_month["month"].map(month_names)

    fig5b = go.Figure(go.Bar(
        x=avg_month["month_name"],
        y=avg_month["Revenue"],
        marker_color="#66BB6A",
        text=[f"₹{v:,.0f}" for v in avg_month["Revenue"]],
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig5b.update_layout(
        title="Avg Daily Revenue by Month",
        height=310,
        showlegend=False,
        yaxis_title="Avg Revenue (₹)",
        margin=dict(t=40, b=40, l=10, r=10),
    )
    fig5b.update_yaxes(tickprefix="₹", tickformat=",.0f")
    st.plotly_chart(fig5b, use_container_width=True)
