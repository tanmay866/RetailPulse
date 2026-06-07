"""Audit log viewer — restricted to admin role."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import pandas as pd
import streamlit as st

from utils.auth import require_auth
from utils.audit_log import read_log

user = require_auth()
if user["role"] != "admin":
    st.error("This page is restricted to administrators.")
    st.stop()

st.header("Audit Log")
st.caption("Security-relevant events logged by the system. Read-only.")

entries = read_log(limit=500)
if not entries:
    st.info("No audit entries recorded yet.")
    st.stop()

df = pd.DataFrame(entries)
df.columns = ["Timestamp", "User", "Action", "Resource", "Details"]

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Events",   len(df))
c2.metric("Unique Users",   df["User"].nunique())
c3.metric("Failed Logins",  (df["Action"] == "login_failed").sum())
c4.metric("Data Downloads", df["Action"].str.contains("download").sum())

st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
f1, f2 = st.columns(2)
with f1:
    user_opts = ["All"] + sorted(df["User"].unique())
    user_sel  = st.selectbox("Filter by user", user_opts)
with f2:
    action_opts = ["All"] + sorted(df["Action"].unique())
    action_sel  = st.selectbox("Filter by action", action_opts)

filt = df.copy()
if user_sel != "All":
    filt = filt[filt["User"] == user_sel]
if action_sel != "All":
    filt = filt[filt["Action"] == action_sel]

st.caption(f"Showing {len(filt):,} of {len(df):,} events")
st.dataframe(filt, use_container_width=True, hide_index=True, height=520)
