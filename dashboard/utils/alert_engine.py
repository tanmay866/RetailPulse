"""Check retail data thresholds and send email alert reports."""
from __future__ import annotations

import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


# ── Alert Checkers ────────────────────────────────────────────────────────────

def check_churn(prob_threshold: float = 0.7, min_count: int = 10) -> dict:
    from utils.data_loader import load_churn_predictions
    try:
        df = load_churn_predictions()
        prob_col = next((c for c in ["Churn_Probability", "churn_probability"] if c in df.columns), None)
        pred_col = next((c for c in ["Churn_Predicted", "churn_predicted"] if c in df.columns), None)

        if prob_col:
            high_risk = (df[prob_col] > prob_threshold).sum()
            rate = df[prob_col].mean() * 100
        elif pred_col:
            high_risk = int(df[pred_col].sum())
            rate = df[pred_col].mean() * 100
        else:
            return {"triggered": False, "reason": "No churn data"}

        triggered = high_risk >= min_count
        return {
            "triggered": triggered,
            "high_risk": high_risk,
            "rate": round(rate, 1),
            "total": len(df),
            "emoji": "🔴" if triggered else "🟢",
        }
    except Exception as exc:
        return {"triggered": False, "reason": str(exc)}


def check_stockout(min_count: int = 5) -> dict:
    from utils.data_loader import load_inventory_recommendations
    try:
        df = load_inventory_recommendations()
        status_col = next((c for c in df.columns if c.lower() == "status"), None)
        if not status_col:
            return {"triggered": False, "reason": "No status column"}

        at_risk = df[status_col].str.upper().str.contains("STOCKOUT|RISK", regex=True).sum()
        triggered = at_risk >= min_count
        return {
            "triggered": triggered,
            "at_risk": at_risk,
            "total_skus": len(df),
            "emoji": "🔴" if triggered else "🟢",
        }
    except Exception as exc:
        return {"triggered": False, "reason": str(exc)}


def check_revenue(drop_threshold_pct: float = 10.0) -> dict:
    import pandas as pd
    from utils.data_loader import load_daily_revenue_rolling
    try:
        df = load_daily_revenue_rolling()
        rev_col = next(
            (c for c in df.columns if c != "Date" and str(df[c].dtype).startswith(("float", "int"))),
            None,
        )
        if not rev_col or "Date" not in df.columns:
            return {"triggered": False, "reason": "No revenue data"}

        df = df.sort_values("Date")
        cutoff = df["Date"].max()
        last30 = df[df["Date"] > cutoff - pd.Timedelta(days=30)][rev_col].sum()
        prior30 = df[
            (df["Date"] <= cutoff - pd.Timedelta(days=30))
            & (df["Date"] > cutoff - pd.Timedelta(days=60))
        ][rev_col].sum()

        change_pct = ((last30 - prior30) / prior30 * 100) if prior30 > 0 else 0
        triggered = change_pct <= -drop_threshold_pct
        return {
            "triggered": triggered,
            "change_pct": round(change_pct, 1),
            "last30": round(last30, 0),
            "prior30": round(prior30, 0),
            "emoji": "🔴" if triggered else "🟢",
        }
    except Exception as exc:
        return {"triggered": False, "reason": str(exc)}


# ── Message Builders ──────────────────────────────────────────────────────────

def build_plain_text(churn: dict, stockout: dict, revenue: dict, note: str = "") -> str:
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    lines = ["RetailPulse Alert Report", "=" * 40, ""]

    if "high_risk" in churn:
        status = "ACTION REQUIRED" if churn["triggered"] else "OK"
        lines += [
            f"CHURN RISK [{status}]",
            f"  High-risk customers : {churn['high_risk']:,} / {churn['total']:,}",
            f"  Overall churn rate  : {churn['rate']}%",
            "",
        ]

    if "at_risk" in stockout:
        status = "ACTION REQUIRED" if stockout["triggered"] else "OK"
        lines += [
            f"INVENTORY [{status}]",
            f"  Products at stockout risk : {stockout['at_risk']:,} / {stockout['total_skus']:,}",
            "",
        ]

    if "change_pct" in revenue:
        status = "ACTION REQUIRED" if revenue["triggered"] else "OK"
        lines += [
            f"REVENUE — 30-DAY CHANGE [{status}]",
            f"  Change   : {revenue['change_pct']:+.1f}%",
            f"  Last 30d : Rs {revenue['last30']:,.0f}",
            f"  Prior 30d: Rs {revenue['prior30']:,.0f}",
            "",
        ]

    if note:
        lines += [f"Note: {note}", ""]

    lines += ["=" * 40, f"Sent by RetailPulse AI  •  {now}"]
    return "\n".join(lines)


def build_html_email(churn: dict, stockout: dict, revenue: dict, note: str = "") -> str:
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    any_triggered = churn.get("triggered") or stockout.get("triggered") or revenue.get("triggered")
    header_color = "#e53935" if any_triggered else "#1976d2"
    header_text = "Action Required" if any_triggered else "All Clear"

    def row(label: str, value: str, alert: bool = False) -> str:
        color = "#ffebee" if alert else "#ffffff"
        return (
            f"<tr style='background:{color}'>"
            f"<td style='padding:8px 12px;color:#555;'>{label}</td>"
            f"<td style='padding:8px 12px;font-weight:bold;'>{value}</td></tr>"
        )

    churn_rows = ""
    if "high_risk" in churn:
        churn_rows = (
            row("High-Risk Customers", f"{churn['high_risk']:,} / {churn['total']:,}", churn["triggered"])
            + row("Overall Churn Rate", f"{churn['rate']}%", churn["triggered"])
        )

    stockout_rows = ""
    if "at_risk" in stockout:
        stockout_rows = row(
            "Products at Stockout Risk",
            f"{stockout['at_risk']:,} / {stockout['total_skus']:,}",
            stockout["triggered"],
        )

    revenue_rows = ""
    if "change_pct" in revenue:
        revenue_rows = (
            row("30-Day Change", f"{revenue['change_pct']:+.1f}%", revenue["triggered"])
            + row("Last 30-Day Revenue", f"Rs {revenue['last30']:,.0f}")
            + row("Prior 30-Day Revenue", f"Rs {revenue['prior30']:,.0f}")
        )

    note_block = f"<p style='color:#555;font-style:italic;'>📝 {note}</p>" if note else ""

    return f"""
<!DOCTYPE html>
<html>
<body style="font-family:Arial,sans-serif;background:#f5f5f5;margin:0;padding:20px;">
  <div style="max-width:600px;margin:auto;background:#fff;border-radius:8px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.1);">

    <div style="background:{header_color};padding:24px 28px;">
      <h1 style="color:#fff;margin:0;font-size:22px;">🚨 RetailPulse Alert</h1>
      <p style="color:rgba(255,255,255,0.85);margin:6px 0 0;">{header_text} — {now}</p>
    </div>

    <div style="padding:24px 28px;">

      <h3 style="color:#333;border-bottom:1px solid #eee;padding-bottom:8px;">📊 Churn Risk</h3>
      <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
        {churn_rows or "<tr><td colspan='2' style='padding:8px 12px;color:#999;'>Data unavailable</td></tr>"}
      </table>

      <h3 style="color:#333;border-bottom:1px solid #eee;padding-bottom:8px;margin-top:20px;">📦 Inventory</h3>
      <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
        {stockout_rows or "<tr><td colspan='2' style='padding:8px 12px;color:#999;'>Data unavailable</td></tr>"}
      </table>

      <h3 style="color:#333;border-bottom:1px solid #eee;padding-bottom:8px;margin-top:20px;">📈 Revenue (30-Day)</h3>
      <table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">
        {revenue_rows or "<tr><td colspan='2' style='padding:8px 12px;color:#999;'>Data unavailable</td></tr>"}
      </table>

      {note_block}

    </div>

    <div style="background:#f5f5f5;padding:14px 28px;text-align:center;">
      <p style="color:#999;font-size:12px;margin:0;">RetailPulse AI Analytics Dashboard</p>
    </div>

  </div>
</body>
</html>
"""


# ── Email Sender ──────────────────────────────────────────────────────────────

def send_email(
    sender: str,
    password: str,
    receiver: str,
    subject: str,
    plain_text: str,
    html_body: str,
) -> tuple[bool, str]:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"RetailPulse AI <{sender}>"
        msg["To"] = receiver
        msg.attach(MIMEText(plain_text, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())

        return True, "Email sent successfully."
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Check your Gmail address and App Password."
    except smtplib.SMTPException as exc:
        return False, f"SMTP error: {exc}"
    except Exception as exc:
        return False, str(exc)
