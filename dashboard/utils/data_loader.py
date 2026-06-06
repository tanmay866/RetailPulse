import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT           = Path(__file__).resolve().parents[2]   # dashboard/utils/ → project root
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES_DIR    = ROOT / "reports" / "figures"


@st.cache_data(ttl=3600)
def load_retail_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "retail_clean.csv", parse_dates=["InvoiceDate"])
    return df


@st.cache_data(ttl=3600)
def load_daily_revenue_rolling() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "daily_revenue_rolling.csv", parse_dates=["Date"])
    return df


@st.cache_data(ttl=3600)
def load_rfm_scores() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "rfm_scores.csv")


@st.cache_data(ttl=3600)
def load_customer_segments() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "customer_segments.csv")


@st.cache_data(ttl=3600)
def load_churn_predictions() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "churn_predictions.csv")


@st.cache_data(ttl=3600)
def load_daily_revenue_ts() -> pd.DataFrame:
    df = pd.read_csv(DATA_PROCESSED / "daily_revenue_ts.csv")
    df = df.rename(columns={"Unnamed: 0": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(ttl=3600)
def load_inventory_recommendations() -> pd.DataFrame:
    return pd.read_csv(DATA_PROCESSED / "inventory_recommendations.csv")


@st.cache_resource
def load_prophet_model():
    with open(ROOT / "models" / "prophet_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data(ttl=3600)
def load_segmentation_churn_merged() -> pd.DataFrame:
    rfm   = pd.read_csv(DATA_PROCESSED / "rfm_scores.csv")
    churn = pd.read_csv(DATA_PROCESSED / "churn_predictions.csv")
    churn = churn.rename(columns={"Customer_ID": "Customer ID"})
    merged = rfm.merge(churn, on="Customer ID", how="inner")
    return merged


def figure_path(filename: str) -> Path:
    return FIGURES_DIR / filename
