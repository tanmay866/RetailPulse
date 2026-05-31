"""Unit tests for src/forecasting.py — Day 5 Prophet baseline pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.forecasting import (
    load_and_resample,
    fill_missing_dates,
    add_log_transform,
    add_time_features,
    add_lag_features,
    add_rolling_features,
    train_test_split_ts,
    evaluate_forecast,
    make_sequences,
    get_prophet_test_preds,
    train_prophet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def daily_df():
    """Load and resample once for all tests in this module."""
    return load_and_resample()


@pytest.fixture(scope='module')
def feature_df(daily_df):
    """Full feature-engineered DataFrame used for split/model tests."""
    df = fill_missing_dates(daily_df)
    df = add_log_transform(df)
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df.dropna()


@pytest.fixture(scope='module')
def split(feature_df):
    """Pre-computed train/test split shared across tests."""
    return train_test_split_ts(feature_df, test_pct=0.2)


@pytest.fixture(scope='module')
def prophet_model_and_preds(split):
    """Trained baseline Prophet model + test predictions (slow — runs once)."""
    train, test = split
    model = train_prophet(train, target='Revenue')
    preds = get_prophet_test_preds(model, test, target='Revenue')
    return model, preds, test


# ---------------------------------------------------------------------------
# Test 1: load_and_resample returns correct structure
# ---------------------------------------------------------------------------

def test_load_and_resample_aggregate(daily_df):
    assert isinstance(daily_df, pd.DataFrame), 'should return a DataFrame'
    assert isinstance(daily_df.index, pd.DatetimeIndex), 'index must be DatetimeIndex'
    assert 'Revenue' in daily_df.columns, 'must have Revenue column'
    assert 'Quantity' in daily_df.columns, 'must have Quantity column'
    assert len(daily_df) > 0, 'must not be empty'


# ---------------------------------------------------------------------------
# Test 2: fill_missing_dates produces contiguous daily index with no gaps
# ---------------------------------------------------------------------------

def test_fill_missing_dates_no_gaps(daily_df):
    filled = fill_missing_dates(daily_df)
    full_range = pd.date_range(start=filled.index.min(), end=filled.index.max(), freq='D')
    assert len(filled) == len(full_range), (
        f'index has gaps: expected {len(full_range)} rows, got {len(filled)}'
    )
    assert filled['Revenue'].isnull().sum() == 0, 'Revenue must have no nulls after fill'
    assert filled['Quantity'].isnull().sum() == 0, 'Quantity must have no nulls after fill'


# ---------------------------------------------------------------------------
# Test 3: add_log_transform produces correct log1p values
# ---------------------------------------------------------------------------

def test_add_log_transform(daily_df):
    filled = fill_missing_dates(daily_df)
    transformed = add_log_transform(filled)
    assert 'log_Revenue' in transformed.columns
    assert 'log_Quantity' in transformed.columns
    expected = np.log1p(transformed['Revenue'])
    np.testing.assert_allclose(
        transformed['log_Revenue'].values,
        expected.values,
        rtol=1e-6,
        err_msg='log_Revenue must equal log1p(Revenue) for all rows',
    )


# ---------------------------------------------------------------------------
# Test 4: train_test_split_ts is strictly chronological
# ---------------------------------------------------------------------------

def test_train_test_split_chronological(feature_df):
    train, test = train_test_split_ts(feature_df, test_pct=0.2)
    assert train.index.max() < test.index.min(), (
        'last train date must be strictly before first test date'
    )
    assert len(train) + len(test) == len(feature_df), (
        'train + test row count must equal total rows'
    )
    expected_test = int(len(feature_df) * 0.2)
    assert abs(len(test) - expected_test) <= 1, (
        f'test size should be ~20% of total, got {len(test)}'
    )


# ---------------------------------------------------------------------------
# Test 5: evaluate_forecast skips zero actuals in MAPE (no division by zero)
# ---------------------------------------------------------------------------

def test_evaluate_forecast_zero_actual_skipped():
    actual    = np.array([0.0, 100.0, 200.0, 0.0, 150.0])
    predicted = np.array([10.0, 90.0, 210.0, 5.0, 160.0])
    result = evaluate_forecast(actual, predicted)
    assert 'mae' in result and 'rmse' in result and 'mape' in result
    assert np.isfinite(result['mape']), 'MAPE must be finite even when actuals contain zeros'
    assert result['mae'] >= 0
    assert result['rmse'] >= 0


# ---------------------------------------------------------------------------
# Test 6: make_sequences returns correct shapes for LSTM
# ---------------------------------------------------------------------------

def test_make_sequences_shape():
    series  = np.arange(100, dtype=np.float32)
    seq_len = 30
    X, y = make_sequences(series, seq_len=seq_len)
    expected_n = len(series) - seq_len
    assert X.shape == (expected_n, seq_len), (
        f'X shape should be ({expected_n}, {seq_len}), got {X.shape}'
    )
    assert y.shape == (expected_n,), (
        f'y shape should be ({expected_n},), got {y.shape}'
    )
    # first window
    np.testing.assert_array_equal(X[0], series[:seq_len])
    assert y[0] == series[seq_len]


# ---------------------------------------------------------------------------
# Test 7: get_prophet_test_preds returns array aligned to test set
# ---------------------------------------------------------------------------

def test_prophet_returns_yhat(prophet_model_and_preds, split):
    _, preds, test = prophet_model_and_preds
    assert isinstance(preds, np.ndarray), 'predictions must be a numpy array'
    assert len(preds) == len(test), (
        f'prediction length {len(preds)} must match test length {len(test)}'
    )
    assert np.isfinite(preds).all(), 'all predictions must be finite (no NaN/Inf)'
