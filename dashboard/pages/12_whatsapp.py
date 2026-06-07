"""Email Alerts — send retail health alerts via Gmail."""
from __future__ import annotations

import datetime
import os

import streamlit as st

from utils.auth import require_auth
from utils.alert_engine import (
    check_churn,
    check_stockout,
    check_revenue,
    build_plain_text,
    build_html_email,
    send_email,
)

# ── Auth (admin only) ─────────────────────────────────────────────────────────
user = require_auth()

st.title(":material/mark_email_unread: Email Alerts")
st.caption("Send automated retail health alerts to your inbox via Gmail")

# ── Session state ─────────────────────────────────────────────────────────────
if "alert_history" not in st.session_state:
    st.session_state.alert_history: list[dict] = []

# ── Sidebar — Gmail setup ─────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Gmail Setup")
    st.markdown(
        "**App Password (not your Gmail password):**\n"
        "1. Go to `myaccount.google.com`\n"
        "2. Security → 2-Step Verification → enable\n"
        "3. Security → **App Passwords**\n"
        "4. Create password for 'Mail'\n"
        "5. Copy the 16-character password"
    )
    st.divider()

    sender = st.text_input(
        "Sender Gmail",
        value=os.environ.get("ALERT_EMAIL_SENDER", ""),
        placeholder="youremail@gmail.com",
    )
    password = st.text_input(
        "App Password",
        value=os.environ.get("ALERT_EMAIL_PASSWORD", ""),
        placeholder="xxxx xxxx xxxx xxxx",
        type="password",
    )
    receiver = st.text_input(
        "Send Alerts To",
        value=os.environ.get("ALERT_EMAIL_RECEIVER", ""),
        placeholder="youremail@gmail.com",
    )

    st.divider()
    if st.button("Send Test Email", use_container_width=True):
        if not sender or not password or not receiver:
            st.error("Fill in all three email fields first.")
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
            else:
                st.error(f"Failed: {msg}")

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
            st.error("Fill in Gmail credentials in the sidebar first.")
        else:
            html_body = build_html_email(churn_result, stockout_result, revenue_result, custom_note)
            with st.spinner("Sending email…"):
                ok, resp_msg = send_email(sender, password, receiver,
                                          subject, plain_preview, html_body)
            if ok:
                st.success(f"Email sent to {receiver}")
                st.session_state.alert_history.append({
                    "Time": datetime.datetime.now().strftime("%d %b %Y %H:%M"),
                    "Subject": subject,
                    "Churn High-Risk": churn_result.get("high_risk", "N/A"),
                    "Stockout SKUs": stockout_result.get("at_risk", "N/A"),
                    "Revenue Change": f"{chg:+.1f}%" if chg is not None else "N/A",
                    "Status": "✅ Sent",
                })
            else:
                st.error(f"Failed: {resp_msg}")
                st.session_state.alert_history.append({
                    "Time": datetime.datetime.now().strftime("%d %b %Y %H:%M"),
                    "Subject": subject,
                    "Status": f"❌ {resp_msg}",
                })

with col_b:
    if st.button("Refresh Data", use_container_width=True):
        st.rerun()

# ── Alert history ─────────────────────────────────────────────────────────────
if st.session_state.alert_history:
    st.divider()
    st.subheader("Sent History (This Session)")
    st.dataframe(st.session_state.alert_history, use_container_width=True, hide_index=True)
