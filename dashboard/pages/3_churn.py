import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "dashboard"))
sys.path.insert(0, str(_PROJECT_ROOT))   # for src.metrics

import streamlit as st

from utils.data_loader import figure_path, load_churn_predictions

st.header("Churn Prediction")

churn = load_churn_predictions()

from src.metrics import CHURN_HIGH_RISK
CHURN_HIGH_RISK.set(int(churn["predicted_churn"].sum()))

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
_charts = [
    (col1, "churn_roc_curve.png",        "ROC Curve"),
    (col2, "churn_confusion_matrix.png", "Confusion Matrix"),
    (col3, "churn_shap_summary.png",     "SHAP Feature Impact"),
]
for col, fname, caption in _charts:
    with col:
        p = figure_path(fname)
        if Path(p).exists():
            st.image(str(p), caption=caption, width="stretch")
        else:
            st.caption(f"{caption} — chart not available")

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
