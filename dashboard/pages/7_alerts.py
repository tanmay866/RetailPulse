"""Day 19 — Real-Time Metrics & Alerts.

Simulated real-time monitoring dashboard with:
- Configurable auto-refresh with live countdown
- Data simulation: replay last 90 days of revenue history
- Rule-based alert engine (CRITICAL / HIGH / MEDIUM / LOW)
- Dismissible active alerts persisted in session_state
- Alert history feed with severity filter
- Live revenue chart with ±1σ/±2σ bands and anomaly markers
- Inventory and churn snapshot panels
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "dashboard"))
sys.path.insert(0, str(_PROJECT_ROOT))   # for src.metrics

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.auth import require_auth

require_auth(page="Real-Time Alerts")  # require login + role permission

from utils.data_loader import (
    load_churn_predictions,
    load_daily_revenue_rolling,
    load_inventory_recommendations,
    load_rfm_scores,
)

# ── Session state defaults (initialised once per browser session) ─────────────
_SS_DEFAULTS: dict = {
    "sim_idx":       513,   # start ~90 days from the end of the data
    "playing":       False,
    "auto_refresh":  False,
    "refresh_secs":  30,
    "alert_history": [],    # list[dict] – cumulative, survives reruns
    "dismissed_ids": set(), # alert IDs the user dismissed this session
    "refresh_count": 0,
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Data ───────────────────────────────────────────────────────────────────────
rolling = load_daily_revenue_rolling()
churn   = load_churn_predictions()
inv     = load_inventory_recommendations()
rfm     = load_rfm_scores()

MAX_IDX = len(rolling) - 1
MIN_IDX = 30  # need ≥30 rows of history for rolling stats

# ── Severity styling ──────────────────────────────────────────────────────────
SEV_COLOR = {"CRITICAL": "#D32F2F", "HIGH": "#E65100", "MEDIUM": "#F57F17", "LOW": "#1565C0"}
SEV_BG    = {"CRITICAL": "#FFEBEE", "HIGH": "#FFF3E0", "MEDIUM": "#FFFDE7", "LOW": "#E3F2FD"}


def _sev_dot(sev: str) -> str:
    """Return an inline HTML colored dot for severity (no emoji)."""
    return (
        f'<span style="display:inline-block;width:9px;height:9px;'
        f'border-radius:50%;background:{SEV_COLOR[sev]};'
        f'vertical-align:middle;margin-right:5px;"></span>'
    )
SEV_RANK  = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


# ── Metric computation ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _precompute_rfm_churn() -> dict:
    """Pre-join Champions × churn probabilities (static, compute once)."""
    champion_ids = set(rfm.loc[rfm["Segment"] == "Champions", "Customer ID"])
    prob_map     = churn.set_index("Customer_ID")["churn_probability"].to_dict()
    champ_at_risk = sum(1 for cid in champion_ids if prob_map.get(cid, 0) > 0.70)
    high_churn_ct = (churn["churn_probability"] > 0.80).sum()
    return {"champ_at_risk": champ_at_risk, "high_churn_ct": high_churn_ct}


_churn_snapshot = _precompute_rfm_churn()


def compute_metrics(idx: int) -> dict:
    row  = rolling.iloc[idx]
    rev  = float(row["Revenue"])
    mean_30 = float(row["rolling_30d_mean"]) if not pd.isna(row["rolling_30d_mean"]) else rev
    std_30  = float(row["rolling_30d_std"])  if not pd.isna(row["rolling_30d_std"])  else 1.0
    zscore  = (rev - mean_30) / max(std_30, 1.0)

    lo, hi   = max(0, idx - 6), max(0, idx - 13)
    rev_7d   = float(rolling.iloc[lo : idx + 1]["Revenue"].sum())
    rev_prev = float(rolling.iloc[hi : lo]["Revenue"].sum())
    pct_7d   = (rev_7d / max(rev_prev, 1) - 1) * 100

    stockout_n   = (inv["status"] == "STOCKOUT_RISK").sum()
    critical_n   = (inv["days_of_stock"] < 1).sum()
    stockout_pct = stockout_n / len(inv) * 100

    return {
        "date":           row["Date"],
        "revenue":        rev,
        "mean_30d":       mean_30,
        "std_30d":        std_30,
        "zscore":         zscore,
        "pct_7d":         pct_7d,
        "rev_7d_sum":     rev_7d,
        "stockout_count": stockout_n,
        "stockout_pct":   stockout_pct,
        "critical_sku_n": critical_n,
        **_churn_snapshot,
    }


class _RuleDef(TypedDict):
    id_key: str
    severity: str
    rule: str
    check: Callable[[dict, dict], bool]
    msg: Callable[[dict], str]


# ── Alert rules ───────────────────────────────────────────────────────────────
_RULES: list[_RuleDef] = [
    {
        "id_key":   "rev_crash",
        "severity": "CRITICAL",
        "rule":     "Revenue Crash",
        "check":    lambda m, t: m["zscore"] < -t["z_low"],
        "msg":      lambda m: (
            f"Daily revenue ₹{m['revenue']:,.0f} is {m['zscore']:.1f}σ below the "
            f"30-day mean (₹{m['mean_30d']:,.0f}). Investigate demand loss."
        ),
    },
    {
        "id_key":   "stockout_critical",
        "severity": "CRITICAL",
        "rule":     "Critical Stockout",
        "check":    lambda m, t: m["critical_sku_n"] > 0,
        "msg":      lambda m: (
            f"{m['critical_sku_n']} SKU(s) with <1 day of stock remaining. "
            "Immediate reorder required."
        ),
    },
    {
        "id_key":   "rev_decline_high",
        "severity": "HIGH",
        "rule":     "7-Day Revenue Decline",
        "check":    lambda m, t: m["pct_7d"] < -t["rev_drop_high"],
        "msg":      lambda m: (
            f"7-day revenue {m['pct_7d']:+.1f}% vs prior 7 days "
            f"(₹{m['rev_7d_sum']:,.0f} this week)."
        ),
    },
    {
        "id_key":   "rev_spike",
        "severity": "HIGH",
        "rule":     "Revenue Spike",
        "check":    lambda m, t: m["zscore"] > t["z_high"],
        "msg":      lambda m: (
            f"Daily revenue ₹{m['revenue']:,.0f} is +{m['zscore']:.1f}σ above the "
            "30-day mean. Verify demand source and stock sufficiency."
        ),
    },
    {
        "id_key":   "stockout_high",
        "severity": "HIGH",
        "rule":     "High Stockout Rate",
        "check":    lambda m, t: m["stockout_pct"] > t["stockout_high"],
        "msg":      lambda m: (
            f"{m['stockout_count']} SKUs ({m['stockout_pct']:.0f}%) are below reorder point."
        ),
    },
    {
        "id_key":   "champion_churn",
        "severity": "HIGH",
        "rule":     "Champion Segment at Churn Risk",
        "check":    lambda m, t: m["champ_at_risk"] > 0,
        "msg":      lambda m: (
            f"{m['champ_at_risk']} Champion-tier customer(s) have churn probability >70%. "
            "High-value retention actions recommended."
        ),
    },
    {
        "id_key":   "rev_decline_med",
        "severity": "MEDIUM",
        "rule":     "7-Day Revenue Softness",
        "check":    lambda m, t: -t["rev_drop_high"] <= m["pct_7d"] < -t["rev_drop_med"],
        "msg":      lambda m: (
            f"7-day revenue {m['pct_7d']:+.1f}% vs prior period — early warning."
        ),
    },
    {
        "id_key":   "stockout_med",
        "severity": "MEDIUM",
        "rule":     "Elevated Stockout Rate",
        "check":    lambda m, t: t["stockout_med"] < m["stockout_pct"] <= t["stockout_high"],
        "msg":      lambda m: (
            f"{m['stockout_count']} SKUs ({m['stockout_pct']:.0f}%) are at stockout risk."
        ),
    },
    {
        "id_key":   "high_churn",
        "severity": "MEDIUM",
        "rule":     "High Churn Volume",
        "check":    lambda m, t: m["high_churn_ct"] > t["churn_count"],
        "msg":      lambda m: (
            f"{m['high_churn_ct']} customers have churn probability >80%."
        ),
    },
]


def evaluate_alerts(metrics: dict, thr: dict, date_str: str) -> list[dict]:
    active = []
    for rule in _RULES:
        try:
            if rule["check"](metrics, thr):
                active.append({
                    "id":       f"{rule['id_key']}_{date_str}",
                    "rule":     rule["rule"],
                    "severity": rule["severity"],
                    "message":  rule["msg"](metrics),
                    "date":     date_str,
                })
        except Exception:
            pass
    return active


# ── Page header ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Real-Time Alerts", layout="wide")
st.title("Real-Time Metrics & Alerts")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation Controls")

    play_col, step_col = st.columns(2)
    with play_col:
        play_label = "Pause" if st.session_state.playing else "Play"
        if st.button(play_label, use_container_width=True):
            st.session_state.playing = not st.session_state.playing
            if st.session_state.playing:
                st.session_state.auto_refresh = True
    with step_col:
        if st.button("Step", use_container_width=True,
                     disabled=st.session_state.playing):
            st.session_state.sim_idx = min(st.session_state.sim_idx + 1, MAX_IDX)
            st.rerun()

    if st.button("Jump to End", use_container_width=True):
        st.session_state.sim_idx = MAX_IDX
        st.session_state.playing = False
        st.rerun()

    # Slider is bound to session_state["sim_idx"] via key=
    st.slider(
        "Simulation Date",
        min_value=MIN_IDX,
        max_value=MAX_IDX,
        key="sim_idx",
    )
    st.caption(f"Sim date: {rolling.iloc[st.session_state.sim_idx]['Date'].date()}")

    st.divider()
    st.header("Auto-Refresh")

    auto_toggle = st.toggle("Enable", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_toggle

    _INTERVALS = {"15 s": 15, "30 s": 30, "1 min": 60, "5 min": 300}
    sel_interval = st.selectbox(
        "Interval",
        list(_INTERVALS.keys()),
        index=1,
        disabled=not auto_toggle,
    )
    st.session_state.refresh_secs = _INTERVALS[sel_interval]

    st.divider()
    st.header("Alert Thresholds")
    with st.expander("Configure", expanded=False):
        t_z_low    = st.slider("Revenue crash σ",   -4.0, -1.0, -2.0, 0.1)
        t_z_high   = st.slider("Revenue spike σ",    1.0,  4.0,  2.0, 0.1)
        t_drop_hi  = st.slider("7d decline HIGH %",  5.0, 40.0, 15.0, 1.0)
        t_drop_med = st.slider("7d decline MED %",   2.0, 15.0,  5.0, 0.5)
        t_so_hi    = st.slider("Stockout HIGH %",   30.0, 90.0, 60.0, 5.0)
        t_so_med   = st.slider("Stockout MED %",    10.0, 60.0, 40.0, 5.0)
        t_churn_n  = st.slider("High churn count",     0,  200,   50,  10)

    thresholds = {
        "z_low":         abs(t_z_low),
        "z_high":        t_z_high,
        "rev_drop_high": t_drop_hi,
        "rev_drop_med":  t_drop_med,
        "stockout_high": t_so_hi,
        "stockout_med":  t_so_med,
        "churn_count":   t_churn_n,
    }

    st.divider()
    if st.button("Clear History", use_container_width=True):
        st.session_state.alert_history = []
        st.session_state.dismissed_ids = set()
        st.rerun()

# ── Compute current-frame metrics ─────────────────────────────────────────────
idx      = st.session_state.sim_idx
metrics  = compute_metrics(idx)
date_str = str(metrics["date"].date())

# Evaluate alert rules
active_alerts = evaluate_alerts(metrics, thresholds, date_str)
undismissed   = [
    a for a in active_alerts
    if a["id"] not in st.session_state.dismissed_ids
]

# Append new alerts to history and toast CRITICAL/HIGH
history_ids = {h["id"] for h in st.session_state.alert_history}
from src.metrics import ALERTS_FIRED
for a in active_alerts:
    if a["id"] not in history_ids:
        a["fired_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.alert_history.append(a)
        ALERTS_FIRED.labels(severity=a["severity"]).inc()
        if a["severity"] in ("CRITICAL", "HIGH"):
            st.toast(
                f"[{a['severity']}] {a['rule']}: {a['message'][:100]}",
            )

crit_n = sum(1 for a in undismissed if a["severity"] == "CRITICAL")
high_n = sum(1 for a in undismissed if a["severity"] == "HIGH")
med_n  = sum(1 for a in undismissed if a["severity"] == "MEDIUM")

# ── Status bar ─────────────────────────────────────────────────────────────────
sb1, sb2, sb3, sb4 = st.columns([3, 3, 3, 1])

with sb1:
    _state_color = "#16A34A" if st.session_state.playing else "#64748B"
    _state_label = "PLAYING" if st.session_state.playing else "PAUSED"
    st.markdown(
        f'<span style="display:inline-flex;align-items:center;gap:6px;">'
        f'<span style="width:9px;height:9px;border-radius:50%;background:{_state_color};'
        f'display:inline-block;"></span>'
        f'<strong>{_state_label}</strong></span><br>'
        f'<span style="font-size:0.82em;color:#64748B;">Sim date: <code>{date_str}</code></span>',
        unsafe_allow_html=True,
    )

with sb2:
    refresh_info = (
        f"Auto-refresh: every {sel_interval}"
        if auto_toggle else "Auto-refresh: OFF"
    )
    st.markdown(f"{refresh_info}  \nRefreshes: {st.session_state.refresh_count}")

with sb3:
    st.markdown(
        f'<span style="color:#D32F2F;font-weight:700;">{crit_n} Critical</span>'
        f'&ensp;<span style="color:#E65100;font-weight:700;">{high_n} High</span>'
        f'&ensp;<span style="color:#F57F17;font-weight:700;">{med_n} Medium</span>',
        unsafe_allow_html=True,
    )

with sb4:
    st.metric("Frame", f"{idx}/{MAX_IDX}")

st.divider()

# ── KPI cards — row 1 ─────────────────────────────────────────────────────────
st.subheader("Key Metrics")

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Today's Revenue",
    f"₹{metrics['revenue']:,.0f}",
    delta=f"₹{metrics['revenue'] - metrics['mean_30d']:+,.0f} vs 30d avg",
)
k2.metric(
    "7-Day Trend",
    f"{metrics['pct_7d']:+.1f}%",
    delta="vs prior 7 days",
    delta_color="normal" if metrics["pct_7d"] >= 0 else "inverse",
)
k3.metric(
    "Revenue Z-Score",
    f"{metrics['zscore']:+.2f}σ",
    delta="vs 30-day baseline",
    delta_color="off",
    help="|z|>2 = anomaly. Red=crash, orange=spike.",
)
k4.metric(
    "Active Alerts",
    len(undismissed),
    delta=f"{crit_n} critical",
    delta_color="inverse" if crit_n > 0 else "off",
)

k5, k6, k7, k8 = st.columns(4)
k5.metric(
    "30-Day Avg Revenue",
    f"₹{metrics['mean_30d']:,.0f}",
    delta=f"σ = ₹{metrics['std_30d']:,.0f}",
    delta_color="off",
)
k6.metric(
    "Stockout SKUs",
    f"{metrics['stockout_count']:,}",
    delta=f"{metrics['stockout_pct']:.0f}% of inventory",
    delta_color="inverse",
)
k7.metric(
    "Critical Stockout",
    f"{metrics['critical_sku_n']:,}",
    delta="SKUs with <1 day left",
    delta_color="inverse" if metrics["critical_sku_n"] > 0 else "off",
)
k8.metric(
    "High Churn Risk",
    f"{metrics['high_churn_ct']:,}",
    delta="customers (p>80%)",
    delta_color="inverse" if metrics["high_churn_ct"] > t_churn_n else "off",
)

st.divider()

# ── Active Alerts ─────────────────────────────────────────────────────────────
st.subheader("Active Alerts")

if not undismissed:
    st.success("**All systems normal** — no active alerts at the current thresholds.")
else:
    sorted_alerts = sorted(undismissed, key=lambda a: SEV_RANK.get(a["severity"], 99))
    for alert in sorted_alerts:
        sev = alert["severity"]
        left, right = st.columns([11, 1])
        with left:
            st.markdown(
                f"""<div style="
                    background:{SEV_BG[sev]};
                    border-left:4px solid {SEV_COLOR[sev]};
                    padding:10px 14px;
                    border-radius:4px;
                    margin-bottom:6px;
                ">
                <span style="color:{SEV_COLOR[sev]};font-weight:700;display:inline-flex;align-items:center;gap:5px;">
                    <span style="width:8px;height:8px;border-radius:50%;background:{SEV_COLOR[sev]};display:inline-block;"></span>{sev}
                </span>
                &ensp;<strong>{alert['rule']}</strong>
                &ensp;<span style="color:#666;font-size:0.82em;">({alert['date']})</span>
                <br><span style="font-size:0.9em;color:#333;">{alert['message']}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        with right:
            st.write("")
            if st.button("X", key=f"dismiss_{alert['id']}", help="Dismiss"):
                st.session_state.dismissed_ids.add(alert["id"])
                st.rerun()

st.divider()

# ── Live revenue chart ────────────────────────────────────────────────────────
st.subheader("Revenue Monitor — Last 30 Days")

win_start = max(0, idx - 29)
cdf = rolling.iloc[win_start : idx + 1].copy()
cdf["zscore"] = (
    (cdf["Revenue"] - cdf["rolling_30d_mean"])
    / cdf["rolling_30d_std"].replace(0, np.nan)
).fillna(0)
cdf["anomaly"] = cdf["zscore"].abs() > 2.0

fig = go.Figure()

# ±2σ shaded band
fig.add_trace(go.Scatter(
    x=pd.concat([cdf["Date"], cdf["Date"].iloc[::-1]]),
    y=pd.concat([
        cdf["rolling_30d_mean"] + 2 * cdf["rolling_30d_std"],
        (cdf["rolling_30d_mean"] - 2 * cdf["rolling_30d_std"]).iloc[::-1],
    ]),
    fill="toself", fillcolor="rgba(33,150,243,0.06)",
    line=dict(color="rgba(0,0,0,0)"), name="±2σ", hoverinfo="skip",
))

# ±1σ shaded band
fig.add_trace(go.Scatter(
    x=pd.concat([cdf["Date"], cdf["Date"].iloc[::-1]]),
    y=pd.concat([
        cdf["rolling_30d_mean"] + cdf["rolling_30d_std"],
        (cdf["rolling_30d_mean"] - cdf["rolling_30d_std"]).iloc[::-1],
    ]),
    fill="toself", fillcolor="rgba(33,150,243,0.12)",
    line=dict(color="rgba(0,0,0,0)"), name="±1σ", hoverinfo="skip",
))

# Daily revenue bars (colour by anomaly direction)
bar_colors = []
for _, row in cdf.iterrows():
    if row["anomaly"] and row["zscore"] < 0:
        bar_colors.append("#EF5350")
    elif row["anomaly"] and row["zscore"] > 0:
        bar_colors.append("#FF9800")
    else:
        bar_colors.append("#90CAF9")

fig.add_trace(go.Bar(
    x=cdf["Date"], y=cdf["Revenue"],
    name="Daily Revenue",
    marker_color=bar_colors,
    opacity=0.85,
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>₹%{y:,.0f}<extra></extra>",
))

# Rolling means
fig.add_trace(go.Scatter(
    x=cdf["Date"], y=cdf["rolling_30d_mean"],
    name="30d Mean", line=dict(color="#1565C0", width=2),
))
fig.add_trace(go.Scatter(
    x=cdf["Date"], y=cdf["rolling_7d_mean"],
    name="7d Mean", line=dict(color="#E64A19", width=1.5, dash="dot"),
))

# Anomaly diamond markers
adf = cdf[cdf["anomaly"]]
if not adf.empty:
    fig.add_trace(go.Scatter(
        x=adf["Date"], y=adf["Revenue"],
        mode="markers", name="Anomaly",
        marker=dict(
            color=["#D32F2F" if z < 0 else "#E65100" for z in adf["zscore"]],
            size=13, symbol="diamond",
            line=dict(color="white", width=1.5),
        ),
        text=[f"z={z:+.2f}" for z in adf["zscore"]],
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>₹%{y:,.0f}<br>z=%{text}<extra>Anomaly</extra>",
    ))

# Current-frame vertical line — add_vline does internal arithmetic that breaks on
# datetime x-values; use add_shape + add_annotation directly instead.
_now_iso = metrics["date"].isoformat()
fig.add_shape(
    type="line", xref="x", yref="paper",
    x0=_now_iso, x1=_now_iso, y0=0, y1=1,
    line=dict(dash="dash", color="#212121", width=1.5),
)
fig.add_annotation(
    xref="x", yref="paper",
    x=_now_iso, y=1.01,
    text="NOW", showarrow=False,
    font=dict(color="#212121", size=11),
    xanchor="left", yanchor="bottom",
)

# Threshold lines
lo_thr = metrics["mean_30d"] - abs(t_z_low) * metrics["std_30d"]
hi_thr = metrics["mean_30d"] + t_z_high * metrics["std_30d"]
fig.add_hline(
    y=lo_thr, line_dash="dot", line_color="#D32F2F", opacity=0.45,
    annotation_text=f"Crash ({t_z_low:.1f}σ)",
    annotation_position="bottom right", annotation_font_color="#D32F2F",
)
fig.add_hline(
    y=hi_thr, line_dash="dot", line_color="#E65100", opacity=0.45,
    annotation_text=f"Spike (+{t_z_high:.1f}σ)",
    annotation_position="top right", annotation_font_color="#E65100",
)

fig.update_layout(
    height=420,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis_title="Revenue (₹)",
    margin=dict(t=10, b=10),
    bargap=0.15,
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Alert History ─────────────────────────────────────────────────────────────
st.subheader("Alert History")

history = st.session_state.alert_history
if not history:
    st.info("No alerts have fired yet. Press Play or step through the simulation.")
else:
    hist_df = (
        pd.DataFrame(history)
        .drop_duplicates(subset=["id"])
        .sort_values("fired_at", ascending=False)
        .head(200)
    )

    sev_filter = st.multiselect(
        "Filter by severity",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        key="hist_sev_filter",
    )
    hist_df = hist_df[hist_df["severity"].isin(sev_filter)]

    display_hist = hist_df[["fired_at", "severity", "rule", "message", "date"]].rename(columns={
        "fired_at": "Fired At", "severity": "Severity",
        "rule": "Rule", "message": "Message", "date": "Sim Date",
    })

    st.dataframe(
        display_hist.style.map(
            lambda v: f"color:{SEV_COLOR.get(v,'#000')};font-weight:700",
            subset=["Severity"],
        ),
        use_container_width=True,
        height=300,
        hide_index=True,
    )
    st.caption(f"Showing {len(display_hist):,} of {len(history):,} total alerts in this session")

st.divider()

# ── Snapshot panels ───────────────────────────────────────────────────────────
st.subheader("Supporting Metrics Snapshot")

snap1, snap2 = st.columns(2)

with snap1:
    st.markdown("**Inventory Status**")
    inv_counts = inv["status"].value_counts().reset_index()
    inv_counts.columns = ["Status", "Count"]
    inv_counts["Pct"] = (inv_counts["Count"] / inv_counts["Count"].sum() * 100).round(1)

    _inv_colors = {
        "STOCKOUT_RISK": "#EF5350",
        "OPTIMAL":       "#66BB6A",
        "OVERSTOCK":     "#FFA726",
    }
    fig_inv = go.Figure(go.Bar(
        x=inv_counts["Status"],
        y=inv_counts["Count"],
        marker_color=[_inv_colors.get(s, "#90A4AE") for s in inv_counts["Status"]],
        text=[f"{p}%" for p in inv_counts["Pct"]],
        textposition="outside",
    ))
    fig_inv.update_layout(
        height=260, yaxis_title="SKU Count",
        showlegend=False, margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig_inv, use_container_width=True)

with snap2:
    st.markdown("**Churn Probability Distribution**")
    bins   = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    churn_copy = churn.copy()
    churn_copy["bucket"] = pd.cut(
        churn_copy["churn_probability"], bins=bins, labels=labels
    )
    cdist = churn_copy["bucket"].value_counts().sort_index().reset_index()
    cdist.columns = ["Bucket", "Count"]

    fig_ch = go.Figure(go.Bar(
        x=cdist["Bucket"],
        y=cdist["Count"],
        marker_color=["#66BB6A", "#AED581", "#FFF176", "#FFB74D", "#EF5350"],
        text=cdist["Count"],
        textposition="outside",
    ))
    fig_ch.update_layout(
        height=260, yaxis_title="Customers",
        showlegend=False, margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig_ch, use_container_width=True)

# ── Auto-refresh + simulation advance (must be last) ──────────────────────────
if st.session_state.auto_refresh:
    secs        = st.session_state.refresh_secs
    tick        = 1 if secs <= 60 else 5
    cdown_slot  = st.empty()

    for remaining in range(secs, 0, -tick):
        cdown_slot.caption(f"Next refresh in {remaining}s")
        time.sleep(tick)

    cdown_slot.empty()

    if st.session_state.playing:
        next_idx = min(st.session_state.sim_idx + 1, MAX_IDX)
        st.session_state.sim_idx = next_idx
        if next_idx >= MAX_IDX:
            st.session_state.playing   = False
            st.session_state.auto_refresh = False

    st.session_state.refresh_count += 1
    st.rerun()
