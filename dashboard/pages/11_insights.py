"""NLP Insights — Auto-generated briefing + natural-language Q&A."""
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

st.title(":material/psychology: NLP Insights")
st.caption("AI-powered analytics — auto briefing and natural-language Q&A")

# ── API key check ─────────────────────────────────────────────────────────────
_api_key = os.environ.get("GROQ_API_KEY", "")
if not _api_key:
    st.error("**GROQ_API_KEY** not set in `.env`.")
    st.stop()

client = Groq(api_key=_api_key)

# ── Session state ─────────────────────────────────────────────────────────────
if "nlp_messages"    not in st.session_state:
    st.session_state.nlp_messages    = []
if "ai_insights"     not in st.session_state:
    st.session_state.ai_insights     = None
if "ai_insights_ts"  not in st.session_state:
    st.session_state.ai_insights_ts  = 0.0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Chat Controls")
    if st.button("Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.nlp_messages = []
        st.rerun()
    st.divider()
    with st.expander("Live Data Context", expanded=False):
        st.caption("Business snapshot sent to AI on every query.")
        try:
            st.text(build_context())
        except Exception as exc:
            st.warning(f"Context unavailable: {exc}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_auto, tab_chat = st.tabs(["Auto-Insights", "Chat Assistant"])


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — AUTO-INSIGHTS                                          ║
# ╚══════════════════════════════════════════════════════════════════╝

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
      <div style="margin-bottom:6px;">
        <span style="font-size:11px;font-weight:600;color:{border};letter-spacing:.5px;">
          {icon} {label} &nbsp;·&nbsp; {cat}
        </span>
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
        color, label = "#2e7d32", "Healthy"
    elif score >= 50:
        color, label = "#f57c00", "Needs Attention"
    else:
        color, label = "#e53935", "Critical"
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
          <div style="background:{color};width:{score}%;height:100%;border-radius:4px;"></div>
        </div>
        <div style="font-size:13px;font-weight:600;color:{color};margin-top:6px;">{label}</div>
      </div>
    </div>"""


def _generate_insights() -> dict | None:
    try:
        ctx = build_context()
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        return None

    prompt = (
        "You are RetailPulse AI, a senior retail business analyst.\n"
        "Analyze the data and return ONLY valid JSON — no markdown, no explanation.\n\n"
        "Required JSON structure:\n"
        "{\n"
        '  "health_score": <integer 0-100>,\n'
        '  "summary": "<one sentence overall business health assessment>",\n'
        '  "insights": [\n'
        "    {\n"
        '      "title": "<short punchy title>",\n'
        '      "category": "<Churn|Revenue|Inventory|CLV|Segments|General>",\n'
        '      "severity": "<critical|warning|positive|neutral>",\n'
        '      "insight": "<2-3 sentences with specific numbers>",\n'
        '      "recommendation": "<1-2 sentence actionable recommendation>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules: exactly 6 insights, at least 1 critical/warning, at least 1 positive, "
        "use specific numbers, currency in Rs.\n\n"
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
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError as exc:
        st.error(f"AI returned invalid JSON: {exc}")
        return None
    except Exception as exc:
        err = str(exc)
        if "401" in err or "invalid_api_key" in err.lower():
            st.error("Invalid GROQ_API_KEY.")
        elif "429" in err or "rate_limit" in err.lower():
            st.error("Rate limit reached. Please wait and try again.")
        else:
            st.error(f"Error: {exc}")
        return None


with tab_auto:
    col_hdr, col_btn = st.columns([4, 1])
    with col_btn:
        regenerate = st.button("Regenerate", use_container_width=True, type="primary")

    if st.session_state.ai_insights is None or regenerate:
        with st.spinner("Analyzing your retail data with AI…"):
            result = _generate_insights()
        if result:
            st.session_state.ai_insights    = result
            st.session_state.ai_insights_ts = time.time()

    data = st.session_state.ai_insights
    if not data:
        st.info("Click **Regenerate** to generate insights.")
    else:
        elapsed = int((time.time() - st.session_state.ai_insights_ts) / 60)
        st.caption(f"Last generated: {'just now' if elapsed < 1 else f'{elapsed} min ago'}")

        st.markdown(_health_bar(data.get("health_score", 50)), unsafe_allow_html=True)
        st.info(f"**Executive Summary:** {data.get('summary', '')}")
        st.divider()
        st.subheader("Business Insights")

        insights = data.get("insights", [])
        left_col, right_col = st.columns(2)
        for i, item in enumerate(insights):
            col = left_col if i % 2 == 0 else right_col
            with col:
                st.markdown(_insight_card(item), unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — CHAT ASSISTANT                                         ║
# ╚══════════════════════════════════════════════════════════════════╝

with tab_chat:
    QUICK_QUESTIONS = [
        "Summarize the overall business health",
        "Which segment has the highest churn risk?",
        "How has revenue trended over the last 30 days?",
        "Which products are at stockout risk?",
        "Who are our most valuable customers by CLV?",
        "What actions should we take to reduce churn?",
    ]

    if not st.session_state.nlp_messages:
        st.subheader("Quick Questions")
        cols = st.columns(2)
        for i, q in enumerate(QUICK_QUESTIONS):
            if cols[i % 2].button(q, key=f"qq_{i}", use_container_width=True):
                st.session_state.nlp_messages.append({"role": "user", "content": q})
                st.rerun()
        st.divider()

    for msg in st.session_state.nlp_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about your retail data…"):
        st.session_state.nlp_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    if st.session_state.nlp_messages and st.session_state.nlp_messages[-1]["role"] == "user":
        try:
            ctx = build_context()
        except Exception as exc:
            ctx = f"[Data context unavailable: {exc}]"

        system_prompt = (
            "You are RetailPulse AI, an expert retail analytics assistant.\n"
            "Answer questions using ONLY the data provided in the context below.\n"
            "Be specific and cite exact numbers from the data.\n"
            "Format responses clearly — use bullet points or short tables when helpful.\n"
            "If a question cannot be answered from the data, say so honestly.\n"
            "All currency values are in Indian Rupees (Rs).\n\n"
            f"{ctx}"
        )

        api_messages = [{"role": "system", "content": system_prompt}] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.nlp_messages[-10:]
        ]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            try:
                stream = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=api_messages,
                    max_tokens=1500,
                    stream=True,
                )
                for chunk in stream:
                    text = chunk.choices[0].delta.content or ""
                    full_response += text
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                st.session_state.nlp_messages.append(
                    {"role": "assistant", "content": full_response}
                )
            except Exception as exc:
                err = str(exc)
                if "401" in err or "invalid_api_key" in err.lower():
                    placeholder.error("Invalid API key — please check GROQ_API_KEY.")
                elif "429" in err or "rate_limit" in err.lower():
                    placeholder.error("Rate limit reached. Please wait and try again.")
                else:
                    placeholder.error(f"Error: {exc}")
