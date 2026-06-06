import streamlit as st

st.set_page_config(
    page_title="RetailPulse",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("pages/1_overview.py",     title="Overview",     icon="📊"),
        st.Page("pages/2_segmentation.py", title="Segmentation", icon="👥"),
        st.Page("pages/3_churn.py",        title="Churn",        icon="⚠️"),
        st.Page("pages/4_forecasting.py",  title="Forecasting",  icon="📈"),
        st.Page("pages/5_inventory.py",    title="Inventory",    icon="📦"),
    ]
)

with st.sidebar:
    st.title("RetailPulse")
    st.caption("AI-powered retail analytics")
    st.divider()

pg.run()
