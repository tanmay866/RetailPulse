"""Prophet + LSTM residual-modeling hybrid for daily revenue forecasting.

Prophet captures trend and seasonality; an LSTM learns the residual series
(actual - prophet_yhat) that Prophet leaves behind. The hybrid forecast is
prophet_yhat + lstm_residual_forecast, clipped at zero (revenue can't be negative).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_residuals(prophet_model, train_df, target='Revenue'):
    """Return the training residual series: actual - Prophet yhat on the train dates.

    Builds the Prophet future frame directly from train_df.index and predicts on
    those exact dates, so the residuals align 1:1 with the training observations.
    Returns a pandas Series indexed by the original DatetimeIndex.
    """
    future = pd.DataFrame({'ds': train_df.index})
    yhat = prophet_model.predict(future)['yhat'].values
    residuals = train_df[target].values - yhat
    return pd.Series(residuals, index=train_df.index, name='residual')


def hybrid_forecast(prophet_test_preds, residual_preds):
    """Recombine Prophet forecast and LSTM residual forecast, clipped at zero."""
    combined = np.asarray(prophet_test_preds) + np.asarray(residual_preds)
    return np.clip(combined, 0, None)


def plot_residuals(residuals, save_path='reports/figures/hybrid_residuals.png'):
    """Plot the training residual series to inspect for learnable structure."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(residuals.index, residuals.values, color='firebrick', linewidth=0.9)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.set_title('Prophet Training Residuals (Actual - yhat)', fontweight='bold')
    ax.set_ylabel('Residual (£)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'saved {save_path}')


def plot_hybrid_forecast(test_series, prophet_preds, hybrid_preds,
                         save_path='reports/figures/hybrid_forecast.png'):
    """Plot actual vs Prophet-only vs Prophet+LSTM hybrid on the test window."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_series.index, test_series.values, label='Actual', color='steelblue')
    ax.plot(test_series.index, prophet_preds, label='Prophet only',
            color='darkorange', linestyle='--')
    ax.plot(test_series.index, hybrid_preds, label='Prophet + LSTM hybrid',
            color='seagreen', linestyle='--')
    ax.set_title('Hybrid Forecast - Prophet + LSTM Residual Ensemble', fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'saved {save_path}')
