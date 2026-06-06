"""Time-series feature engineering and stationarity analysis for demand forecasting.

Functions are designed to be called in sequence from prepare_forecast.py
or used individually inside notebooks/forecasting.ipynb.
"""
from __future__ import annotations

from typing import Literal, Union, overload

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


@overload
def load_and_resample(
    path: str = 'data/processed/retail_clean.csv',
    mode: Literal['aggregate'] = 'aggregate',
) -> pd.DataFrame: ...


@overload
def load_and_resample(
    path: str,
    mode: Literal['by_category'],
) -> dict[str, pd.DataFrame]: ...


def load_and_resample(
    path: str = 'data/processed/retail_clean.csv',
    mode: str = 'aggregate',
) -> Union[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load retail_clean.csv and resample to daily Revenue + Quantity totals.

    mode='aggregate' returns one DataFrame, mode='by_category' returns a dict
    keyed by product description.
    """
    df = pd.read_csv(path, low_memory=False)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()
    df['Revenue'] = df['Quantity'] * df['Price']

    if mode == 'aggregate':
        return df.set_index('InvoiceDate').resample('D')[['Revenue', 'Quantity']].sum()

    elif mode == 'by_category':
        result: dict[str, pd.DataFrame] = {}
        for category, group in df.groupby('Description'):
            result[str(category)] = (
                group.set_index('InvoiceDate')
                .resample('D')[['Revenue', 'Quantity']]
                .sum()
            )
        return result

    else:
        raise ValueError(f"mode must be 'aggregate' or 'by_category', got '{mode}'")


def fill_missing_dates(df):
    """Fill gaps in the daily index so every date in the range has a row.

    Revenue and Quantity get zero-filled (no sales = 0), everything else
    gets forward-filled.
    """
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_range)

    zero_cols = [c for c in df.columns if c in ('Revenue', 'Quantity')]
    ffill_cols = [c for c in df.columns if c not in zero_cols]

    df[zero_cols] = df[zero_cols].fillna(0)
    if ffill_cols:
        df[ffill_cols] = df[ffill_cols].ffill()

    return df


def add_log_transform(df):
    """Add log1p versions of Revenue and Quantity columns (originals kept)."""
    df = df.copy()
    if 'Revenue' in df.columns:
        df['log_Revenue'] = np.log1p(df['Revenue'])
    if 'Quantity' in df.columns:
        df['log_Quantity'] = np.log1p(df['Quantity'])
    return df


def add_time_features(df):
    """Add calendar features: day_of_week, month, quarter, is_weekend, is_month_end."""
    df = df.copy()
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['is_weekend'] = df.index.dayofweek >= 5
    df['is_month_end'] = df.index.is_month_end
    return df


def add_lag_features(df, target='Revenue'):
    """Add 1-day, 7-day, and 30-day lags of the target column."""
    df = df.copy()
    df['lag_1'] = df[target].shift(1)
    df['lag_7'] = df[target].shift(7)
    df['lag_30'] = df[target].shift(30)
    return df


def add_rolling_features(df, target='Revenue'):
    """Add 7/30-day rolling mean and 7-day rolling std of the target column.

    shift(1) before rolling so today's value doesn't leak into its own features.
    """
    df = df.copy()
    s = df[target].shift(1)
    df['rolling_7_mean'] = s.rolling(7, min_periods=1).mean()
    df['rolling_30_mean'] = s.rolling(30, min_periods=1).mean()
    df['rolling_7_std'] = s.rolling(7, min_periods=1).std()
    return df


def train_test_split_ts(df, test_pct=0.2):
    """Chronological train/test split - no shuffling."""
    split = int(len(df) * (1 - test_pct))
    return df.iloc[:split], df.iloc[split:]


def test_stationarity(series):
    """Run ADF + KPSS and return a combined verdict dict.

    ADF passes if p < 0.05 (rejects unit root).
    KPSS passes if p > 0.05 (fails to reject stationarity).

    Verdicts:
        both pass        -> 'stationary'
        both fail        -> 'non-stationary'
        ADF pass only    -> 'trend-stationary'
        KPSS pass only   -> 'difference-stationary'
    """
    clean = series.dropna()

    _, adf_p, _, _, _, _ = adfuller(clean, autolag='AIC')
    _, kpss_p, _, _ = kpss(clean, regression='c', nlags='auto')

    adf_pass = adf_p < 0.05
    kpss_pass = kpss_p > 0.05

    if adf_pass and kpss_pass:
        verdict = 'stationary'
        note = 'both tests agree - good to go'
    elif not adf_pass and not kpss_pass:
        verdict = 'non-stationary'
        note = 'both tests agree - try differencing'
    elif adf_pass and not kpss_pass:
        verdict = 'trend-stationary'
        note = 'detrending may be enough'
    else:
        verdict = 'difference-stationary'
        note = 'first difference should fix it'

    print(f'ADF  p={adf_p:.4f} ({"pass" if adf_pass else "fail"})')
    print(f'KPSS p={kpss_p:.4f} ({"pass" if kpss_pass else "fail"})')
    print(f'-> {verdict} - {note}')

    return {'adf_pvalue': adf_p, 'kpss_pvalue': kpss_p, 'verdict': verdict}


def decompose_series(series, model='additive', period=7):
    """Decompose series into trend/seasonal/residual. Returns DecomposeResult."""
    return seasonal_decompose(series, model=model, period=period)


def difference_series(series, order=1, seasonal_period=None):
    """Difference a series (regular first, seasonal second if given)."""
    s = series.copy()
    for _ in range(order):
        s = s.diff()
    if seasonal_period is not None:
        s = s.diff(seasonal_period)
    return s.dropna()


def plot_acf_pacf(series, lags=40, save_path='reports/figures/ts_acf_pacf.png'):
    """Plot ACF and PACF side by side and save to disk."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(series.dropna(), lags=lags, ax=ax1, title='ACF')
    plot_pacf(series.dropna(), lags=lags, ax=ax2, title='PACF')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'saved {save_path}')


def plot_decomposition(series, period=7, save_path='reports/figures/ts_decomposition.png'):
    """Decompose series and save a 4-panel plot (observed, trend, seasonal, residual)."""
    result = decompose_series(series, period=period)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    result.observed.plot(ax=axes[0], title='Observed', color='steelblue')
    result.trend.plot(ax=axes[1], title='Trend', color='darkorange')
    result.seasonal.plot(ax=axes[2], title='Seasonal', color='seagreen')
    result.resid.plot(ax=axes[3], title='Residual', color='firebrick')

    for ax in axes:
        ax.set_xlabel('')

    plt.suptitle('Time Series Decomposition', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {save_path}')


def log_dataset_stats(df, train, test, stationarity_result, missing_dates_filled=0):
    """Log dataset prep stats to MLflow under the retailpulse-forecasting experiment.

    Args:
        df: the final feature-engineered DataFrame (after dropna)
        train: training split DataFrame
        test: test split DataFrame
        stationarity_result: dict returned by test_stationarity()
        missing_dates_filled: number of dates added by fill_missing_dates()
    """
    mlflow.set_experiment('retailpulse-forecasting')

    with mlflow.start_run(run_name='ts-dataset-prep'):
        mlflow.log_params({
            'date_start': str(df.index.min().date()),
            'date_end': str(df.index.max().date()),
            'fill_mode': 'zero-fill',
            'verdict': stationarity_result['verdict'],
        })

        mlflow.log_metrics({
            'total_rows': len(df),
            'missing_dates_filled': missing_dates_filled,
            'adf_pvalue': stationarity_result['adf_pvalue'],
            'kpss_pvalue': stationarity_result['kpss_pvalue'],
            'train_size': len(train),
            'test_size': len(test),
            'test_pct': len(test) / len(df),
        })

    print('MLflow run logged: ts-dataset-prep')


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------

def train_prophet(train_df, target='Revenue', yearly_seasonality=True, weekly_seasonality=True):
    """Fit a Prophet model on the training DataFrame.

    Expects a DatetimeIndex. Converts to Prophet's (ds, y) format internally.
    """
    prophet_df = pd.DataFrame({
        'ds': train_df.index,
        'y': train_df[target].values,
    })
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
    )
    model.fit(prophet_df)
    return model


def forecast_prophet(model, periods=30, freq='D'):
    """Generate a Prophet forecast for the given number of future periods.

    Returns the full forecast DataFrame including history + future rows.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    return model.predict(future)


def evaluate_forecast(actual, predicted):
    """Compute MAE, RMSE, and MAPE between actual and predicted values."""
    actual = np.array(actual)
    predicted = np.array(predicted)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    # skip zero actuals to avoid division by zero in MAPE
    mask = actual != 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return {
        'mae': round(float(mae), 4),
        'rmse': round(float(rmse), 4),
        'mape': round(float(mape), 4),
    }


def get_prophet_test_preds(model, test_df, target='Revenue'):
    """Extract Prophet yhat values aligned to the test DataFrame's index.

    Runs predict on just the test dates so the returned array lines up
    with test_df[target] for evaluate_forecast().
    """
    future = pd.DataFrame({'ds': test_df.index})
    forecast = model.predict(future)
    return forecast['yhat'].values


def plot_prophet_forecast(model, forecast, save_path='reports/figures/prophet_forecast.png'):
    """Plot Prophet forecast with uncertainty intervals and save to disk."""
    fig = model.plot(forecast)
    fig.suptitle('Prophet Forecast - Daily Revenue', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {save_path}')


def tune_prophet(train_df, test_df, target='Revenue',
                 save_path='reports/figures/prophet_tuning.png'):
    """Grid-search over changepoint_prior_scale and seasonality_mode.

    Trains one Prophet model per config, evaluates on test_df, returns a
    DataFrame of results sorted by RMSE (best first).
    """
    grid = [
        {'changepoint_prior_scale': 0.05, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.30, 'seasonality_mode': 'additive'},
        {'changepoint_prior_scale': 0.05, 'seasonality_mode': 'multiplicative'},
        {'changepoint_prior_scale': 0.30, 'seasonality_mode': 'multiplicative'},
    ]

    prophet_df = pd.DataFrame({'ds': train_df.index, 'y': train_df[target].values})
    future_df  = pd.DataFrame({'ds': test_df.index})

    records = []
    for cfg in grid:
        label = f"cps={cfg['changepoint_prior_scale']}  mode={cfg['seasonality_mode']}"
        print(f'  training: {label}')
        m = Prophet(
            changepoint_prior_scale=cfg['changepoint_prior_scale'],
            seasonality_mode=cfg['seasonality_mode'],
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
        )
        m.fit(prophet_df)
        yhat = np.clip(m.predict(future_df)['yhat'].values, 0, None)  # type: ignore[arg-type]
        metrics = evaluate_forecast(test_df[target], yhat)
        records.append({
            'changepoint_prior_scale': cfg['changepoint_prior_scale'],
            'seasonality_mode':        cfg['seasonality_mode'],
            'mae':                     metrics['mae'],
            'rmse':                    metrics['rmse'],
            'mape':                    metrics['mape'],
            '_model':                  m,
            '_preds':                  yhat,
        })

    results = (
        pd.DataFrame(records)
        .sort_values('rmse')
        .reset_index(drop=True)
    )

    # bar chart comparison
    labels = [
        f"cps={r['changepoint_prior_scale']}\n{r['seasonality_mode']}"
        for _, r in results.iterrows()
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    axes[0].bar(x, results['mae'],  color=['seagreen' if i == 0 else 'steelblue' for i in x], edgecolor='white')
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_title('MAE by Config', fontweight='bold')
    axes[0].set_ylabel('MAE (₹)')
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(x, results['rmse'], color=['seagreen' if i == 0 else 'steelblue' for i in x], edgecolor='white')
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_title('RMSE by Config (sorted best→worst)', fontweight='bold')
    axes[1].set_ylabel('RMSE (₹)')
    axes[1].grid(axis='y', alpha=0.3)

    plt.suptitle('Prophet Hyperparameter Grid — Test-Set Metrics', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {save_path}')

    return results


def prophet_cross_validate(model, initial='450 days', period='30 days', horizon='90 days',
                           save_path='reports/figures/prophet_cv_metrics.png'):
    """Run Prophet rolling-window cross-validation and plot RMSE/MAE by horizon.

    initial is set to 450 days so yearly seasonality (365.25 days) fits inside
    the first training window. Returns (cv_df, metrics_df).
    """
    from prophet.diagnostics import cross_validation, performance_metrics

    cv_df = cross_validation(model, initial=initial, period=period, horizon=horizon, parallel=None)
    # disable_tqdm not available in all versions; mape is dropped when y=0 exists
    metrics_df = performance_metrics(cv_df, rolling_window=0.1)
    metrics_df['horizon_days'] = metrics_df['horizon'].dt.days

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(metrics_df['horizon_days'], metrics_df['rmse'],
                 color='darkorange', linewidth=1.5, marker='o', markersize=3)
    axes[0].set_title('RMSE by Forecast Horizon', fontweight='bold')
    axes[0].set_xlabel('Horizon (days ahead)')
    axes[0].set_ylabel('RMSE (₹)')
    axes[0].grid(alpha=0.3)

    axes[1].plot(metrics_df['horizon_days'], metrics_df['mae'],
                 color='steelblue', linewidth=1.5, marker='o', markersize=3)
    axes[1].set_title('MAE by Forecast Horizon', fontweight='bold')
    axes[1].set_xlabel('Horizon (days ahead)')
    axes[1].set_ylabel('MAE (₹)')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Prophet Cross-Validation — Error by Forecast Horizon', fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {save_path}')

    return cv_df, metrics_df


def plot_prophet_changepoints(model, df, target='Revenue', save_path='reports/figures/prophet_changepoints.png'):
    """Plot revenue series overlaid with Prophet changepoints and their trend-delta weights."""
    import matplotlib.dates as mdates

    deltas = model.params['delta'].mean(axis=0)
    changepoints = model.changepoints.values
    significant = np.abs(deltas) > np.percentile(np.abs(deltas), 50)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df.index, df[target], color='steelblue', linewidth=0.9, label='Actual Revenue')
    for cp, sig in zip(changepoints, significant):
        color = 'crimson' if sig else 'lightcoral'
        alpha = 0.8 if sig else 0.3
        ax1.axvline(cp, color=color, linestyle='--', linewidth=0.8, alpha=alpha)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='steelblue', label='Revenue'),
        Line2D([0], [0], color='crimson',   linestyle='--', label='Strong changepoint'),
        Line2D([0], [0], color='lightcoral',linestyle='--', label='Weak changepoint'),
    ]
    ax1.legend(handles=handles, fontsize=9)
    ax1.set_ylabel('Revenue (₹)')
    ax1.set_title('Revenue with Prophet Changepoints', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    colors = ['crimson' if d > 0 else 'steelblue' for d in deltas]
    ax2.bar(changepoints, deltas, color=colors, width=5, alpha=0.75)
    ax2.axhline(0, color='black', linewidth=0.7)
    ax2.set_ylabel('Trend delta')
    ax2.set_title('Changepoint Weights (+ = upward shift, - = downward)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.suptitle('Prophet Changepoint Analysis — Daily Revenue', fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {save_path}')


def plot_prophet_components(model, forecast, save_path='reports/figures/prophet_components.png'):
    """Plot Prophet trend + seasonality components and save to disk."""
    fig = model.plot_components(forecast)
    axes = fig.get_axes()
    titles = ['Trend', 'Weekly Seasonality', 'Yearly Seasonality']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Prophet Model Components — Daily Revenue', fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved {save_path}')


# ---------------------------------------------------------------------------
# LSTM (PyTorch)
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """Simple stacked LSTM that maps a sequence to a single next-step output."""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_sequences(series, seq_len=30):
    """Slide a window over the series to produce (X, y) pairs for LSTM training.

    Returns numpy arrays: X shape (n, seq_len), y shape (n,).
    """
    values = np.array(series, dtype=np.float32)
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def train_lstm(train_series, seq_len=30, hidden_size=64, num_layers=2,
               epochs=50, lr=0.001, dropout=0.2):
    """Scale, sequence, and train an LSTM on the training series.

    Returns (model, scaler) — keep both for forecasting and inverse transform.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()

    X, y = make_sequences(scaled, seq_len)
    X_t = torch.tensor(X).unsqueeze(-1)        # (batch, seq_len, 1)
    y_t = torch.tensor(y).unsqueeze(-1)        # (batch, 1)

    model = LSTMForecaster(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'  epoch {epoch + 1}/{epochs}  loss={loss.item():.6f}')

    return model, scaler


def forecast_lstm(model, train_series, steps, scaler, seq_len=30):
    """Roll the trained LSTM forward one step at a time for multi-step forecasting.

    Seeds from the last seq_len values of train_series. Returns predictions
    in the original (unscaled) revenue scale.
    """
    model.eval()
    scaled = scaler.transform(train_series.values.reshape(-1, 1)).flatten()
    window = list(scaled[-seq_len:])
    preds_scaled = []

    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            pred = model(x).item()
            preds_scaled.append(pred)
            window.append(pred)

    return scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Shared plotting + MLflow logging for model runs
# ---------------------------------------------------------------------------

def plot_forecast_comparison(test_series, prophet_preds, lstm_preds,
                              save_path='reports/figures/forecast_comparison.png'):
    """Plot actual vs Prophet vs LSTM on the test window and save to disk."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(test_series.index, test_series.values, label='Actual', color='steelblue')
    ax.plot(test_series.index, prophet_preds, label='Prophet', color='darkorange', linestyle='--')
    ax.plot(test_series.index, lstm_preds, label='LSTM', color='seagreen', linestyle='--')
    ax.set_title('Forecast Comparison - Prophet vs LSTM')
    ax.set_xlabel('')
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'saved {save_path}')


def save_lstm(model, scaler, model_path='models/lstm_forecaster.pt', scaler_path='models/lstm_scaler.pkl'):
    """Save LSTM weights and scaler to disk."""
    import pickle
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'saved {model_path}, {scaler_path}')


def load_lstm(model_path='models/lstm_forecaster.pt', scaler_path='models/lstm_scaler.pkl',
              hidden_size=64, num_layers=2):
    """Load LSTM weights and scaler from disk."""
    import pickle
    model = LSTMForecaster(hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def save_prophet(model, path='models/prophet_model.pkl'):
    """Serialize Prophet model to disk with pickle."""
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f'saved {path}')


def load_prophet(path='models/prophet_model.pkl'):
    """Load a pickled Prophet model from disk."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def log_model_metrics(model_name, metrics, params=None, run_name=None):
    """Log forecast evaluation metrics to MLflow under retailpulse-forecasting."""
    mlflow.set_experiment('retailpulse-forecasting')
    run_name = run_name or f'{model_name}-eval'
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    print(f'MLflow run logged: {run_name}')
