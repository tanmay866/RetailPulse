"""Email Alerts — send retail health alerts via Gmail."""
from __future__ import annotations

import datetime
import os

import pandas as pd
import streamlit as st

from utils.auth import require_auth
from utils.db import get_conn, execute as db_execute
from utils.alert_engine import (
    check_churn,
    check_stockout,
    check_revenue,
    build_plain_text,
    build_html_email,
    send_email,
)


# ── DB helpers ────────────────────────────────────────────────────────────────
def _ensure_table() -> None:
    """Create email_history table if it doesn't exist yet."""
    db_execute("""
        CREATE TABLE IF NOT EXISTS email_history (
            id              SERIAL PRIMARY KEY,
            sent_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            sent_by         TEXT NOT NULL,
            recipient       TEXT NOT NULL,
            subject         TEXT NOT NULL,
            churn_high_risk INTEGER,
            stockout_skus   INTEGER,
            revenue_change  TEXT,
            status          TEXT NOT NULL,
            error_msg       TEXT
        )
    """)


_HISTORY_KEY = "_email_history"  # st.session_state fallback key


def _save_email(
    sent_by: str,
    recipient: str,
    subject: str,
    churn_high_risk,
    stockout_skus,
    revenue_change: str,
    status: str,
    error_msg: str = "",
) -> None:
    record = {
        "sent_at":         datetime.datetime.now().strftime("%d %b %Y %H:%M"),
        "sent_by":         sent_by,
        "recipient":       recipient,
        "subject":         subject,
        "churn_high_risk": churn_high_risk,
        "stockout_skus":   stockout_skus,
        "revenue_change":  revenue_change,
        "status":          status,
        "error_msg":       error_msg,
    }
    with get_conn() as conn:
        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO email_history
                       (sent_by, recipient, subject, churn_high_risk,
                        stockout_skus, revenue_change, status, error_msg)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (sent_by, recipient, subject,
                     int(churn_high_risk) if str(churn_high_risk).isdigit() else None,
                     int(stockout_skus)   if str(stockout_skus).isdigit()   else None,
                     revenue_change, status, error_msg),
                )
            return
    # DB unavailable — store in session state
    if _HISTORY_KEY not in st.session_state:
        st.session_state[_HISTORY_KEY] = []
    st.session_state[_HISTORY_KEY].insert(0, record)


def _load_history(limit: int = 100) -> pd.DataFrame:
    with get_conn() as conn:
        if conn is not None:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT sent_at, sent_by, recipient, subject,
                              churn_high_risk, stockout_skus,
                              revenue_change, status, error_msg
                       FROM email_history
                       ORDER BY sent_at DESC
                       LIMIT %s""",
                    (limit,),
                )
                cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows, columns=cols)
            df["sent_at"] = pd.to_datetime(df["sent_at"]).dt.strftime("%d %b %Y %H:%M")
        else:
            # DB unavailable — read from session state
            rows = st.session_state.get(_HISTORY_KEY, [])[:limit]
            if not rows:
                return pd.DataFrame()
            df = pd.DataFrame(rows)

    df = df.rename(columns={
        "sent_at":         "Time",
        "sent_by":         "Sent By",
        "recipient":       "Recipient",
        "subject":         "Subject",
        "churn_high_risk": "Churn High-Risk",
        "stockout_skus":   "Stockout SKUs",
        "revenue_change":  "Revenue Change",
        "status":          "Status",
        "error_msg":       "Error",
    })
    if "Error" in df.columns and df["Error"].fillna("").eq("").all():
        df = df.drop(columns=["Error"])
    return df

# ── Auth (admin only) ─────────────────────────────────────────────────────────
user = require_auth()

st.title(":material/mark_email_unread: Email Alerts")
st.caption("Send automated retail health alerts to your inbox via Gmail")

# Ensure the table exists (idempotent)
_ensure_table()

# ── Sidebar — Gmail setup ─────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Gmail Setup")

    # Credentials come from secrets / environment only — never rendered as
    # editable inputs, so the App Password is not exposed on a deployed app.
    sender   = os.environ.get("ALERT_EMAIL_SENDER", "")
    password = os.environ.get("ALERT_EMAIL_PASSWORD", "")
    receiver = os.environ.get("ALERT_EMAIL_RECEIVER", "")

    if sender and password and receiver:
        st.success("Configured via secrets")
        st.caption(f"Sender:  {sender}")
        st.caption(f"Send to:  {receiver}")
    else:
        st.warning(
            "Email not configured. Set `ALERT_EMAIL_SENDER`, "
            "`ALERT_EMAIL_PASSWORD`, and `ALERT_EMAIL_RECEIVER` in your "
            "app secrets (use a Gmail **App Password**, not your login password)."
        )

    st.divider()
    if st.button("Send Test Email", use_container_width=True):
        if not sender or not password or not receiver:
            st.error("Email not configured — set the ALERT_EMAIL_* secrets first.")
        else:
            now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
            plain = (
                "RetailPulse Test Alert\n"
                "========================\n"
                "Email integration is working correctly.\n"
                "You will receive retail alerts here.\n\n"
                f"Sent by RetailPulse AI  •  {now}"
            )
            html = f"""
            <div style='font-family:Arial;padding:20px;background:#f5f5f5;'>
              <div style='background:#1976d2;padding:20px;border-radius:8px 8px 0 0;'>
                <h2 style='color:#fff;margin:0;'>✅ RetailPulse Test Alert</h2>
                <p style='color:rgba(255,255,255,0.8);margin:4px 0 0;'>{now}</p>
              </div>
              <div style='background:#fff;padding:20px;border-radius:0 0 8px 8px;'>
                <p>Email integration is working correctly.</p>
                <p>You will receive retail health alerts here.</p>
              </div>
            </div>"""
            with st.spinner("Sending…"):
                ok, msg = send_email(sender, password, receiver,
                                     "✅ RetailPulse — Test Email", plain, html)
            if ok:
                st.success("Test email sent! Check your inbox.")
                _save_email(
                    sent_by        = user["username"],
                    recipient      = receiver,
                    subject        = "✅ RetailPulse — Test Email",
                    churn_high_risk= "",
                    stockout_skus  = "",
                    revenue_change = "N/A",
                    status         = "Sent",
                )
            else:
                st.error(f"Failed: {msg}")
                _save_email(
                    sent_by        = user["username"],
                    recipient      = receiver,
                    subject        = "✅ RetailPulse — Test Email",
                    churn_high_risk= "",
                    stockout_skus  = "",
                    revenue_change = "N/A",
                    status         = "Failed",
                    error_msg      = msg,
                )

# ── Alert thresholds ──────────────────────────────────────────────────────────
st.subheader("Alert Thresholds")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Churn Risk**")
    churn_prob  = st.slider("Min churn probability", 0.5, 0.95, 0.7, 0.05, format="%.0f%%")
    churn_count = st.number_input("Min high-risk customers", 1, 500, 10, step=5)

with col2:
    st.markdown("**Stockout Risk**")
    stockout_count = st.number_input("Min at-risk SKUs", 1, 100, 5, step=1)

with col3:
    st.markdown("**Revenue Drop**")
    revenue_drop = st.slider("Min drop %", 5, 50, 10, 5, format="-%d%%")

st.divider()

# ── Live data status ──────────────────────────────────────────────────────────
st.subheader("Current Status")

with st.spinner("Checking live data…"):
    churn_result    = check_churn(churn_prob, churn_count)
    stockout_result = check_stockout(stockout_count)
    revenue_result  = check_revenue(revenue_drop)

chg = revenue_result.get("change_pct")


def _status_card(label: str, value: str, sub: str, triggered: bool) -> str:
    bg       = "#fff0f0" if triggered else "#f0fff4"
    border   = "#e53935" if triggered else "#2e7d32"
    badge_bg = "#e53935" if triggered else "#2e7d32"
    badge    = "⚠ Threshold Breached" if triggered else "✓ Within Threshold"
    return f"""
    <div style="background:{bg};border:1.5px solid {border};border-radius:10px;
                padding:18px 20px;height:130px;box-sizing:border-box;">
      <p style="margin:0 0 4px;font-size:13px;color:#555;">{label}</p>
      <p style="margin:0 0 4px;font-size:28px;font-weight:700;color:#1a1a2e;">{value}</p>
      <p style="margin:0 0 10px;font-size:12px;color:#777;">{sub}</p>
      <span style="background:{badge_bg};color:#fff;font-size:11px;
                   padding:3px 10px;border-radius:20px;">{badge}</span>
    </div>"""

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        _status_card(
            "High-Risk Customers",
            f"{churn_result.get('high_risk', 'N/A'):,}" if "high_risk" in churn_result else "N/A",
            f"Churn rate: {churn_result.get('rate', '?')}%",
            churn_result.get("triggered", False),
        ),
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        _status_card(
            "Stockout Risk SKUs",
            f"{stockout_result.get('at_risk', 'N/A'):,}" if "at_risk" in stockout_result else "N/A",
            f"Total SKUs: {stockout_result.get('total_skus', '?'):,}" if "total_skus" in stockout_result else "",
            stockout_result.get("triggered", False),
        ),
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        _status_card(
            "30-Day Revenue Change",
            f"{chg:+.1f}%" if chg is not None else "N/A",
            f"Last 30d: Rs {revenue_result.get('last30', 0):,.0f}" if chg is not None else "",
            revenue_result.get("triggered", False),
        ),
        unsafe_allow_html=True,
    )

st.divider()

# ── Email preview ─────────────────────────────────────────────────────────────
st.subheader("Email Preview")
custom_note = st.text_input("Optional note to include", placeholder="e.g. Weekly summary")

plain_preview = build_plain_text(churn_result, stockout_result, revenue_result, custom_note)
st.code(plain_preview, language=None)

st.divider()

# ── Send buttons ──────────────────────────────────────────────────────────────
any_triggered = (
    churn_result.get("triggered")
    or stockout_result.get("triggered")
    or revenue_result.get("triggered")
)

subject = (
    "🚨 RetailPulse Alert — Action Required"
    if any_triggered
    else "📊 RetailPulse Report — All Clear"
)

col_a, col_b = st.columns(2)

with col_a:
    btn_label = "🚨 Send Alert Email" if any_triggered else "📤 Send Report Email"
    if st.button(btn_label, use_container_width=True, type="primary"):
        if not sender or not password or not receiver:
            st.error("Email not configured — set the ALERT_EMAIL_* secrets first.")
        else:
            html_body = build_html_email(churn_result, stockout_result, revenue_result, custom_note)
            with st.spinner("Sending email…"):
                ok, resp_msg = send_email(sender, password, receiver,
                                          subject, plain_preview, html_body)
            if ok:
                st.success(f"Email sent to {receiver}")
                _save_email(
                    sent_by        = user["username"],
                    recipient      = receiver,
                    subject        = subject,
                    churn_high_risk= churn_result.get("high_risk", ""),
                    stockout_skus  = stockout_result.get("at_risk", ""),
                    revenue_change = f"{chg:+.1f}%" if chg is not None else "N/A",
                    status         = "Sent",
                )
            else:
                st.error(f"Failed: {resp_msg}")
                _save_email(
                    sent_by        = user["username"],
                    recipient      = receiver,
                    subject        = subject,
                    churn_high_risk= "",
                    stockout_skus  = "",
                    revenue_change = "N/A",
                    status         = "Failed",
                    error_msg      = resp_msg,
                )

with col_b:
    if st.button("Refresh Data", use_container_width=True):
        st.rerun()

# ── Alert history ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("Sent History")
history_df = _load_history()
if history_df.empty:
    st.info("No emails sent yet.")
else:
    st.dataframe(history_df, use_container_width=True, hide_index=True)
