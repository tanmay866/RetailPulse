import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT))   # project root — needed for src.metrics

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

import streamlit as st
from utils.ui import apply_theme
from utils.auth import get_current_user, show_login_form, allowed_pages, logout
from utils import cache
from src.metrics import start_metrics_server, PAGE_VIEWS, PAGE_LOAD_SECONDS

st.set_page_config(
    page_title="RetailPulse",
    page_icon=":material/storefront:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

@st.cache_resource
def _start_metrics():
    start_metrics_server(port=8000)

_start_metrics()

# ── Auth gate — show login form and halt if not authenticated ─────────────────
# get_current_user() reads st.context.cookies (synchronous, Streamlit 1.37+)
# on the first render after a browser refresh, so no async workaround needed.
user = get_current_user()
if user is None:
    show_login_form()
    st.stop()

# ── Sidebar — brand + user info + logout ─────────────────────────────────────
_ROLE_LABEL = {
    "admin":          "Admin",
    "analyst":        "Analyst",
    "viewer":         "Viewer",
    "data_scientist": "Data Scientist",
}

with st.sidebar:
    st.title("RetailPulse")
    st.caption("AI-powered retail analytics")
    st.divider()
    st.caption(f"**{user['username']}**  ·  {_ROLE_LABEL.get(user['role'], user['role'])}")
    if st.button("Refresh data", use_container_width=True,
                 help="Reload the latest data, clearing the in-memory and Redis caches."):
        st.cache_data.clear()   # drop Streamlit's in-memory cache
        cache.flush_all()       # drop the Redis rp:* cache
        st.toast("Data refreshed.")
        st.rerun()
    if st.button("Logout", use_container_width=True):
        logout(user["username"])
        st.rerun()
    st.divider()

# ── Build navigation filtered by the user's role ─────────────────────────────
_ALL_PAGES = [
    ("Command Center",     "pages/0_checkpoint.py",         ":material/dashboard:"),
    ("Overview",           "pages/1_overview.py",           ":material/bar_chart:"),
    ("Segmentation",       "pages/2_segmentation.py",       ":material/group:"),
    ("Churn",              "pages/3_churn.py",              ":material/person_off:"),
    ("Customer Analytics", "pages/6_customer_analytics.py", ":material/analytics:"),
    ("Forecasting",        "pages/4_forecasting.py",        ":material/trending_up:"),
    ("Inventory",          "pages/5_inventory.py",          ":material/inventory_2:"),
    ("Real-Time Alerts",   "pages/7_alerts.py",             ":material/notifications_active:"),
    ("Export & Reports",   "pages/8_export.py",             ":material/file_download:"),
    ("Audit Log",          "pages/9_audit.py",              ":material/security:"),
    ("CLV Analysis",       "pages/10_clv.py",               ":material/monetization_on:"),
    ("NLP Insights",       "pages/11_insights.py",          ":material/psychology:"),
    ("Email Alerts",       "pages/12_email_alerts.py",      ":material/mark_email_unread:"),
]

_allowed = allowed_pages(user["role"])
pages = [
    st.Page(path, title=title, icon=icon)
    for title, path, icon in _ALL_PAGES
    if title in _allowed
]

pg = st.navigation(pages)
_t0 = time.perf_counter()
pg.run()
_elapsed = time.perf_counter() - _t0

PAGE_VIEWS.labels(page=pg.title).inc()
PAGE_LOAD_SECONDS.labels(page=pg.title).observe(_elapsed)
