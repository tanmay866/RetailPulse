import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import plotly.express as px
import streamlit as st

from utils.data_loader import figure_path, load_daily_revenue_ts

st.header("Demand Forecasting")

ts = load_daily_revenue_ts()

# ── Horizon selector (static UI — live inference wired in Day 16) ─────────────
st.subheader("Forecast Horizon")
horizon = st.radio("Select horizon", [7, 14, 30], horizontal=True, format_func=lambda x: f"{x} days")
st.info(f"Showing pre-computed forecast. Live {horizon}-day re-forecast available in next iteration.")

st.divider()

# ── Hybrid forecast chart ─────────────────────────────────────────────────────
st.subheader("Hybrid Forecast (Prophet + LSTM Residual)")
st.image(str(figure_path("hybrid_forecast.png")), width="stretch")

st.divider()

# ── Historical revenue (interactive) ─────────────────────────────────────────
st.subheader("Historical Daily Revenue")

fig = px.line(
    ts,
    x="Date",
    y="Revenue",
    labels={"Revenue": "Revenue (£)", "Date": "Date"},
    color_discrete_sequence=["#2196F3"],
)
fig.update_layout(hovermode="x unified", height=320)
st.plotly_chart(fig, width="stretch")

st.divider()

# ── Prophet components ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Prophet Components")
    st.image(str(figure_path("prophet_components.png")), width="stretch")
with col2:
    st.subheader("Cross-Validation Metrics")
    st.image(str(figure_path("prophet_cv_metrics.png")), width="stretch")
