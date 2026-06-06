"""Global UI theme injection for RetailPulse dashboard."""
import streamlit as st


def apply_theme() -> None:
    """Inject Inter font, global CSS design system, and sidebar dark theme.
    Call once per page — app.py calls it globally so individual pages don't need to.
    """
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ── Fonts ─────────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Base ───────────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* Page background */
.stApp { background: #F0F4F8; }

.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1440px;
}

/* ── Typography ─────────────────────────────────────────────────────────── */
h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    letter-spacing: -0.025em !important;
    line-height: 1.25 !important;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-size: 1.175rem !important;
    font-weight: 600 !important;
    color: #1E293B !important;
    letter-spacing: -0.01em !important;
}
h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
}
p, li { color: #334155; line-height: 1.6; }

[data-testid="stCaptionContainer"] p {
    color: #94A3B8 !important;
    font-size: 0.8rem !important;
}

/* ── Sidebar — dark navy ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #1E2A3B !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #1E2A3B !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Brand block (user sidebar content) above the nav links */
[data-testid="stSidebarContent"] {
    order: -1 !important;
}
[data-testid="stSidebarNav"],
[data-testid="stSidebarNavItems"] {
    order: 0 !important;
}

/* Sidebar text — base readable on dark navy */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #CBD5E1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F1F5F9 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #94A3B8 !important;
    font-size: 0.77rem !important;
}

/* Widget labels in sidebar — ensure bright enough */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
    color: #E2E8F0 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

/* Radio and toggle labels in sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] span,
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stToggle"] label,
[data-testid="stSidebar"] [data-testid="stToggle"] span {
    color: #CBD5E1 !important;
    font-weight: 500 !important;
}

/* Sidebar inputs — container */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div,
[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background: #263548 !important;
    border-color: #334155 !important;
    color: #E2E8F0 !important;
    border-radius: 8px !important;
}
/* Sidebar inputs — actual text inside input/textarea elements */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] input,
[data-testid="stSidebar"] [data-baseweb="base-input"] div {
    color: #E2E8F0 !important;
    -webkit-text-fill-color: #E2E8F0 !important;
    caret-color: #E2E8F0 !important;
}
/* Placeholder text */
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
}
/* Selected tags in multiselect */
[data-testid="stSidebar"] [data-baseweb="tag"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span {
    color: #FFFFFF !important;
}
/* Slider tick labels */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stTickBarMax"],
[data-testid="stSidebar"] [data-testid="stSlider"] p {
    color: #94A3B8 !important;
}
/* Slider current value label */
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderThumbValue"] {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] [role="slider"] {
    background: #2563EB !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #263548 !important;
    margin: 0.75rem 0 !important;
}

/* Sidebar nav items */
[data-testid="stSidebarNavItems"] li > a {
    border-radius: 8px !important;
    margin: 2px 4px !important;
    padding: 0.5rem 0.75rem !important;
    transition: background 0.15s ease;
    color: #94A3B8 !important;
}
[data-testid="stSidebarNavItems"] li > a:hover {
    background: #263548 !important;
    color: #CBD5E1 !important;
}
[data-testid="stSidebarNavItems"] li > a[aria-current="page"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
}
[data-testid="stSidebarNavItems"] li > a[aria-current="page"] span,
[data-testid="stSidebarNavItems"] li > a[aria-current="page"] p {
    color: #FFFFFF !important;
}

/* ── Metric cards ───────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(15,23,42,0.09) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    color: #64748B !important;
    text-transform: uppercase;
    letter-spacing: 0.05em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    letter-spacing: -0.02em !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
    font-weight: 500 !important;
}

/* ── Bordered containers / Cards ────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important;
    background: #FFFFFF !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.05) !important;
    overflow: hidden;
}

/* ── Tabs ───────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: #EEF2F7;
    border-radius: 10px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #E2E8F0;
    width: fit-content;
}
[data-testid="stTabs"] [role="tab"] {
    border-radius: 7px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    color: #64748B !important;
    padding: 0.45rem 1.1rem !important;
    border: none !important;
    transition: all 0.15s ease;
    background: transparent !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #FFFFFF !important;
    color: #1D4ED8 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(15,23,42,0.09) !important;
}
[data-testid="stTabs"] [data-testid="stTabContent"] {
    padding-top: 1rem;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.01em;
    box-shadow: 0 1px 3px rgba(37,99,235,0.25) !important;
}
.stButton > button:hover {
    background: #1D4ED8 !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.38) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: none !important;
}
.stButton > button:disabled {
    background: #CBD5E1 !important;
    box-shadow: none !important;
    transform: none !important;
}
/* Force white text on all children inside action buttons */
.stButton > button p,
.stButton > button span,
.stButton > button div,
.stButton > button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button:disabled p,
.stButton > button:disabled span,
.stButton > button:disabled * {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
}

/* ── Download buttons ───────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    background: #059669 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(5,150,105,0.25) !important;
    transition: all 0.18s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #047857 !important;
    box-shadow: 0 4px 14px rgba(5,150,105,0.35) !important;
    transform: translateY(-1px) !important;
}
/* Force white text on all children inside download buttons */
[data-testid="stDownloadButton"] > button p,
[data-testid="stDownloadButton"] > button span,
[data-testid="stDownloadButton"] > button div,
[data-testid="stDownloadButton"] > button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* ── Inputs (selectbox, multiselect, date, text) ────────────────────────── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] > div {
    border-radius: 8px !important;
    border-color: #CBD5E1 !important;
    background: #FFFFFF !important;
    transition: border-color 0.15s ease;
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}

/* ── Dataframes ─────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden;
    border: 1px solid #E2E8F0 !important;
}
[data-testid="stDataFrame"] thead th {
    background: #F8FAFC !important;
    color: #64748B !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

/* ── Plotly chart containers ────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border-radius: 12px;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 0.5rem 0.25rem;
}

/* ── Dividers ─────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #E2E8F0 !important;
    margin: 1.5rem 0 !important;
}

/* ── Alert / Info / Warning boxes ──────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.875rem !important;
    font-weight: 450;
}

/* ── Spinner ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {
    color: #64748B !important;
    font-size: 0.875rem !important;
}

/* ── Toggle (main content only — sidebar overrides above) ──────────────── */
.main [data-testid="stToggle"] label {
    color: #334155 !important;
    font-weight: 500;
}

/* ── Radio (main content only — sidebar overrides above) ───────────────── */
.main [data-testid="stRadio"] label {
    font-weight: 500 !important;
    color: #334155 !important;
}

/* ── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #334155 !important;
}

/* ── Image containers ─────────────────────────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid #E2E8F0;
}

/* ── Scrollbar (WebKit) ─────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }

/* ── Top header bar ─────────────────────────────────────────────────────── */
[data-testid="stHeader"] {
    background: rgba(240,244,248,0.85) !important;
    backdrop-filter: blur(8px);
    border-bottom: 1px solid #E2E8F0;
}
</style>
"""
