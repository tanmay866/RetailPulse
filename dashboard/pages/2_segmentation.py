import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import plotly.express as px
import streamlit as st

from utils.auth import require_auth

require_auth(page="Segmentation")  # require login + role permission

from utils.data_loader import load_customer_segments, load_rfm_scores

st.header("Customer Segmentation")

rfm      = load_rfm_scores()
segments = load_customer_segments()

# ── Segment distribution ──────────────────────────────────────────────────────
st.subheader("RFM Segment Distribution")

seg_counts = rfm["Segment"].value_counts().reset_index()
seg_counts.columns = ["Segment", "Count"]

fig = px.pie(
    seg_counts,
    names="Segment",
    values="Count",
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.update_layout(showlegend=True, height=380)
st.plotly_chart(fig, width="stretch")

st.divider()

# ── Cluster scatter ───────────────────────────────────────────────────────────
st.subheader("Customer Clusters (Recency vs Monetary)")

fig2 = px.scatter(
    segments,
    x="Recency",
    y="Monetary",
    color="Business_Label",
    hover_data=["Customer ID", "Frequency"],
    labels={"Recency": "Recency (days)", "Monetary": "Monetary (₹)", "Business_Label": "Segment"},
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7,
)
fig2.update_layout(height=420, legend_title_text="Cluster")
st.plotly_chart(fig2, width="stretch")

st.divider()

# ── Filterable customer table ─────────────────────────────────────────────────
st.subheader("Customer RFM Table")

seg_options = ["All"] + sorted(rfm["Segment"].unique().tolist())
selected    = st.selectbox("Filter by Segment", seg_options)

display_df = rfm if selected == "All" else rfm[rfm["Segment"] == selected]

st.dataframe(
    display_df[["Customer ID", "Recency", "Frequency", "Monetary", "RFM_Total", "Segment"]]
    .sort_values("RFM_Total", ascending=False)
    .reset_index(drop=True),
    width="stretch",
    height=400,
)
st.caption(f"Showing {len(display_df):,} customers")
