import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from utils.ui import apply_theme
from utils.auth import get_current_user, show_login_form, allowed_pages, logout

st.set_page_config(
    page_title="RetailPulse",
    page_icon=":material/storefront:",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# ── Auth gate — show login form and halt if not authenticated ─────────────────
user = get_current_user()
if user is None:
    show_login_form()
    st.stop()

# ── Sidebar — brand + user info + logout ─────────────────────────────────────
_ROLE_LABEL = {"admin": "Admin", "analyst": "Analyst", "viewer": "Viewer"}

with st.sidebar:
    st.title("RetailPulse")
    st.caption("AI-powered retail analytics")
    st.divider()
    st.caption(f"**{user['username']}**  ·  {_ROLE_LABEL.get(user['role'], user['role'])}")
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
]

_allowed = allowed_pages(user["role"])
pages = [
    st.Page(path, title=title, icon=icon)
    for title, path, icon in _ALL_PAGES
    if title in _allowed
]

pg = st.navigation(pages)
pg.run()
