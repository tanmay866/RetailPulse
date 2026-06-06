import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dashboard"))

import streamlit as st

from utils.data_loader import figure_path, load_churn_predictions

st.header("Churn Prediction")

churn = load_churn_predictions()

# ── KPI cards ─────────────────────────────────────────────────────────────────
at_risk    = churn["predicted_churn"].sum()
churn_rate = churn["predicted_churn"].mean() * 100

c1, c2, c3 = st.columns(3)
c1.metric("Total Customers",  f"{len(churn):,}")
c2.metric("Predicted Churners", f"{at_risk:,}")
c3.metric("Churn Rate",       f"{churn_rate:.1f}%")

st.divider()

# ── Evaluation charts ─────────────────────────────────────────────────────────
st.subheader("Model Evaluation")

col1, col2, col3 = st.columns(3)
with col1:
    st.image(str(figure_path("churn_roc_curve.png")),        caption="ROC Curve",          width="stretch")
with col2:
    st.image(str(figure_path("churn_confusion_matrix.png")), caption="Confusion Matrix",    width="stretch")
with col3:
    st.image(str(figure_path("churn_shap_summary.png")),     caption="SHAP Feature Impact", width="stretch")

st.divider()

# ── At-risk customer table ────────────────────────────────────────────────────
st.subheader("Top 50 At-Risk Customers")

top_risk = (
    churn[churn["predicted_churn"] == 1]
    .sort_values("churn_probability", ascending=False)
    .head(50)
    .reset_index(drop=True)
)

st.dataframe(
    top_risk[["Customer_ID", "churn_probability", "actual_churn"]]
    .rename(columns={
        "Customer_ID":       "Customer ID",
        "churn_probability": "Churn Probability",
        "actual_churn":      "Actual Churn",
    }),
    width="stretch",
    height=400,
)
