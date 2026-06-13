import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.validation import (
    AUC_ROC_GATE,
    CHURN_PREDICTIONS,
    DAILY_REVENUE_TS,
    FORECAST_NRMSE_GATE,
    PRECISION_TOP20_GATE,
    precision_at_top_k,
    run_validation,
    validate_churn,
    validate_forecast,
)

needs_data = pytest.mark.skipif(
    not (CHURN_PREDICTIONS.exists() and DAILY_REVENUE_TS.exists()),
    reason="prediction artifacts not available",
)


# ── Unit test: precision_at_top_k is scale-free and uses ranking, not threshold ──

def test_precision_at_top_k_perfect_ranking():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])  # the two positives rank highest
    assert precision_at_top_k(y_true, y_prob, k=0.5) == 1.0


def test_precision_at_top_k_worst_ranking():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])  # positives rank lowest
    assert precision_at_top_k(y_true, y_prob, k=0.5) == 0.0


# ── Gate tests against the committed prediction artifacts ─────────────────────

@needs_data
def test_churn_gates_pass():
    result = validate_churn()
    assert result["passed"], result["failures"]
    assert result["metrics"]["auc_roc"] >= AUC_ROC_GATE
    assert result["metrics"]["precision_top20"] >= PRECISION_TOP20_GATE


@needs_data
def test_forecast_gate_passes():
    result = validate_forecast()
    assert result["passed"], result["failures"]
    assert result["metrics"]["nrmse"] < FORECAST_NRMSE_GATE


@needs_data
def test_run_validation_overall_pass_and_shape():
    report = run_validation(write=False)
    assert report["passed"]
    assert set(report) >= {"passed", "churn", "forecast"}
