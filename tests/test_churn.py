import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.churn import (
    AUC_ROC_GATE,
    PRECISION_TOP20_GATE,
    load_and_preprocess,
    precision_at_top_k,
)

RETAIL_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "retail_clean.csv"
needs_data = pytest.mark.skipif(not RETAIL_CSV.exists(), reason="retail_clean.csv not available")


def test_precision_at_top_k_perfect():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert precision_at_top_k(y, p, k=0.30) == 1.0


def test_precision_at_top_k_zero():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    assert precision_at_top_k(y, p, k=0.30) == 0.0


def test_precision_at_top_k_all_positive():
    y = np.ones(10, dtype=int)
    p = np.linspace(0.1, 1.0, 10)
    assert precision_at_top_k(y, p, k=0.20) == 1.0


def test_precision_at_top_k_minimum_one_sample():
    y = np.array([1, 0])
    p = np.array([0.9, 0.1])
    result = precision_at_top_k(y, p, k=0.01)
    assert result in (0.0, 1.0)


def test_gate_constants():
    assert AUC_ROC_GATE == 0.88
    assert PRECISION_TOP20_GATE == 0.75


@needs_data
def test_load_and_preprocess_aligned_shapes():
    X, y, cids = load_and_preprocess(RETAIL_CSV)
    assert X.shape[0] == y.shape[0] == cids.shape[0]


@needs_data
def test_load_and_preprocess_expected_features():
    X, _, _ = load_and_preprocess(RETAIL_CSV)
    expected = {
        "recency", "frequency", "monetary", "tenure",
        "avg_order_value", "avg_qty_per_order", "n_unique_products",
        "avg_days_between_orders", "purchase_regularity",
        "recent_freq_90d", "revenue_last_30d", "n_months_active", "spend_trend_90d",
    }
    assert expected.issubset(set(X.columns))


@needs_data
def test_load_and_preprocess_all_numeric():
    X, _, _ = load_and_preprocess(RETAIL_CSV)
    object_cols = [c for c in X.columns if X[c].dtype == object]
    assert not object_cols, f"Non-numeric columns remain: {object_cols}"


@needs_data
def test_load_and_preprocess_target_binary():
    _, y, _ = load_and_preprocess(RETAIL_CSV)
    assert set(y.unique()).issubset({0, 1})


@needs_data
def test_load_and_preprocess_no_nulls():
    X, y, _ = load_and_preprocess(RETAIL_CSV)
    assert not X.isnull().any().any()
    assert not y.isnull().any()


@needs_data
def test_load_and_preprocess_recency_positive():
    X, _, _ = load_and_preprocess(RETAIL_CSV)
    assert (X["recency"] >= 0).all()
