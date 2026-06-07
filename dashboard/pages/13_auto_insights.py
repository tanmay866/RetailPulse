"""Auto-Insights — AI-generated business intelligence briefing from live retail data."""
from __future__ import annotations

import json
import os
import time

from groq import Groq
import streamlit as st

from utils.auth import require_auth
from utils.analytics_context import build_context

# ── Auth ──────────────────────────────────────────────────────────────────────
user = require_auth()

st.title(":material/auto_awesome: Auto-Insights")
st.caption("AI-generated business intelligence briefing — refreshed from live data")

# ── API key check ─────────────────────────────────────────────────────────────
_api_key = os.environ.get("GROQ_API_KEY", "")
if not _api_key:
    st.error("GROQ_API_KEY not set in `.env`.")
    st.stop()

client = Groq(api_key=_api_key)

# ── Session state ─────────────────────────────────────────────────────────────
if "ai_insights"       not in st.session_state:
    st.session_state.ai_insights    = None
if "ai_insights_ts"    not in st.session_state:
    st.session_state.ai_insights_ts = 0.0

# ── Helpers ───────────────────────────────────────────────────────────────────
_SEVERITY_STYLE = {
    "critical": ("#fff0f0", "#e53935", "🔴", "CRITICAL"),
    "warning":  ("#fff8e1", "#f57c00", "🟡", "WARNING"),
    "positive": ("#f0fff4", "#2e7d32", "🟢", "POSITIVE"),
    "neutral":  ("#f0f4ff", "#1565c0", "🔵", "INFO"),
}

def _insight_card(item: dict) -> str:
    sev = item.get("severity", "neutral")
    bg, border, icon, label = _SEVERITY_STYLE.get(sev, _SEVERITY_STYLE["neutral"])
    cat = item.get("category", "")
    return f"""
    <div style="background:{bg};border-left:4px solid {border};border-radius:8px;
                padding:16px 18px;margin-bottom:12px;min-height:160px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
        <span style="font-size:11px;font-weight:600;color:{border};letter-spacing:.5px;">{icon} {label} &nbsp;·&nbsp; {cat}</span>
      </div>
      <p style="margin:0 0 6px;font-size:15px;font-weight:700;color:#1a1a2e;">
        {item.get('title', '')}
      </p>
      <p style="margin:0 0 10px;font-size:13px;color:#444;line-height:1.5;">
        {item.get('insight', '')}
      </p>
      <p style="margin:0;font-size:12px;color:{border};font-style:italic;">
        💡 {item.get('recommendation', '')}
      </p>
    </div>"""


def _health_bar(score: int) -> str:
    if score >= 75:
        color = "#2e7d32"
        label = "Healthy"
    elif score >= 50:
        color = "#f57c00"
        label = "Needs Attention"
    else:
        color = "#e53935"
        label = "Critical"
    return f"""
    <div style="background:#f5f5f5;border-radius:10px;padding:20px 24px;
                display:flex;align-items:center;gap:24px;margin-bottom:8px;">
      <div style="text-align:center;">
        <div style="font-size:48px;font-weight:800;color:{color};line-height:1;">{score}</div>
        <div style="font-size:11px;color:#777;margin-top:2px;">/ 100</div>
      </div>
      <div style="flex:1;">
        <div style="font-size:13px;color:#555;margin-bottom:6px;">Business Health Score</div>
        <div style="background:#e0e0e0;border-radius:4px;height:10px;overflow:hidden;">
          <div style="background:{color};width:{score}%;height:100%;border-radius:4px;
                      transition:width .5s;"></div>
        </div>
        <div style="font-size:13px;font-weight:600;color:{color};margin-top:6px;">{label}</div>
      </div>
    </div>"""


def _generate_insights() -> dict | None:
    try:
        ctx = build_context()
    except Exception as exc:
        st.error(f"Could not load data context: {exc}")
        return None

    prompt = (
        "You are RetailPulse AI, a senior retail business analyst.\n"
        "Analyze the data below and return ONLY valid JSON — no explanation, no markdown, no code fences.\n\n"
        "Required JSON structure:\n"
        "{\n"
        '  "health_score": <integer 0-100>,\n'
        '  "summary": "<one sentence overall business health assessment>",\n'
        '  "insights": [\n'
        "    {\n"
        '      "title": "<short punchy title>",\n'
        '      "category": "<Churn | Revenue | Inventory | CLV | Segments | General>",\n'
        '      "severity": "<critical | warning | positive | neutral>",\n'
        '      "insight": "<2-3 sentences with specific numbers from the data>",\n'
        '      "recommendation": "<1-2 sentence actionable recommendation>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Generate exactly 6 insights covering different areas\n"
        "- Use specific numbers and percentages from the data\n"
        "- At least 1 critical or warning, at least 1 positive\n"
        "- Recommendations must be specific and actionable\n"
        "- Currency is Indian Rupees (Rs)\n\n"
        f"DATA:\n{ctx}"
    )

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        st.error(f"AI returned invalid JSON: {exc}")
        return None
    except Exception as exc:
        err = str(exc)
        if "401" in err or "invalid_api_key" in err.lower():
            st.error("Invalid GROQ_API_KEY — please check `.env`.")
        elif "429" in err or "rate_limit" in err.lower():
            st.error("Rate limit reached. Please wait a moment.")
        else:
            st.error(f"Error: {exc}")
        return None


# ── Controls ──────────────────────────────────────────────────────────────────
col_title, col_btn = st.columns([4, 1])
with col_btn:
    regenerate = st.button("Regenerate", use_container_width=True, type="primary")

# Auto-generate on first load or manual regenerate
if st.session_state.ai_insights is None or regenerate:
    with st.spinner("Analyzing your retail data with AI…"):
        result = _generate_insights()
    if result:
        st.session_state.ai_insights    = result
        st.session_state.ai_insights_ts = time.time()

data = st.session_state.ai_insights
if not data:
    st.info("Click **Regenerate** to generate insights.")
    st.stop()

# ── Last updated ──────────────────────────────────────────────────────────────
elapsed = int((time.time() - st.session_state.ai_insights_ts) / 60)
ts_label = "just now" if elapsed < 1 else f"{elapsed} min ago"
st.caption(f"Last generated: {ts_label}")

# ── Health score + summary ────────────────────────────────────────────────────
score   = data.get("health_score", 50)
summary = data.get("summary", "")

st.markdown(_health_bar(score), unsafe_allow_html=True)
st.info(f"**Executive Summary:** {summary}")

st.divider()

# ── Insight cards (2 columns) ─────────────────────────────────────────────────
st.subheader("Business Insights")

insights = data.get("insights", [])
if not insights:
    st.warning("No insights returned. Try regenerating.")
    st.stop()

left_col, right_col = st.columns(2)
for i, item in enumerate(insights):
    col = left_col if i % 2 == 0 else right_col
    with col:
        st.markdown(_insight_card(item), unsafe_allow_html=True)
