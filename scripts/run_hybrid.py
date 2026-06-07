"""CLI runner for the Prophet + LSTM residual-modeling hybrid forecaster."""
import os
import pickle

import mlflow
import pandas as pd

from src.forecasting import (
    train_test_split_ts,
    evaluate_forecast,
    load_prophet,
    get_prophet_test_preds,
)
from src.lstm_lightning import (
    LSTMLightning,
    train_lstm_lightning,
    forecast_lstm_lightning,
)
from src.hybrid_forecaster import (
    compute_residuals,
    hybrid_forecast,
    plot_residuals,
    plot_hybrid_forecast,
)

os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# ---------- config (matches the LSTM Lightning baseline) ----------
SEQ_LEN     = 30
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
LR          = 1e-3
BATCH_SIZE  = 32
MAX_EPOCHS  = 50
PATIENCE    = 10

EXPERIMENT  = 'retailpulse-forecasting'

# ---------- load data ----------
print('--- loading dataset ---')
df = pd.read_csv('data/processed/daily_revenue_ts.csv', index_col=0, parse_dates=True)
train, test = train_test_split_ts(df)
print(f'shape: {df.shape}  train: {len(train)}  test: {len(test)}')

# ---------- Prophet stage ----------
print('\n--- loading saved Prophet baseline ---')
prophet_model = load_prophet()

print('--- Prophet: yhat on test period ---')
prophet_test_preds = get_prophet_test_preds(prophet_model, test, target='Revenue')

print('--- Prophet: training residuals ---')
residuals = compute_residuals(prophet_model, train, target='Revenue')
print(f'residual mean={residuals.mean():.2f}  std={residuals.std():.2f}')
plot_residuals(residuals)

# ---------- LSTM-on-residuals + hybrid + MLflow ----------
print('\n--- training LSTM on residuals ---')
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name='prophet-lstm-hybrid') as run:
    mlflow.log_params({
        'ensemble_method': 'residual',
        'seq_len':         SEQ_LEN,
        'hidden_size':     HIDDEN_SIZE,
        'num_layers':      NUM_LAYERS,
        'dropout':         DROPOUT,
        'lr':              LR,
        'batch_size':      BATCH_SIZE,
        'max_epochs':      MAX_EPOCHS,
        'patience':        PATIENCE,
    })
    mlflow.set_tags({
        'model_type': 'hybrid',
        'framework':  'prophet+pytorch_lightning',
        'target':     'daily_revenue',
        'dataset':    'online_retail_II',
    })

    model, dm = train_lstm_lightning(
        residuals,
        seq_len=SEQ_LEN,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        lr=LR,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        mlflow_run_id=run.info.run_id,
        checkpoint_name='hybrid_residual_lstm',
        experiment_name=EXPERIMENT,
    )

    print('\n--- forecasting residuals on test period ---')
    residual_preds = forecast_lstm_lightning(
        model,
        seed_series=residuals,
        steps=len(test),
        scaler=dm.scaler,
        seq_len=SEQ_LEN,
    )

    hybrid_preds   = hybrid_forecast(prophet_test_preds, residual_preds)
    hybrid_metrics = evaluate_forecast(test['Revenue'], hybrid_preds)
    print(f'MAE  : {hybrid_metrics["mae"]}')
    print(f'RMSE : {hybrid_metrics["rmse"]}')
    print(f'MAPE : {hybrid_metrics["mape"]}%')

    mlflow.log_metrics(hybrid_metrics)
    plot_hybrid_forecast(test['Revenue'], prophet_test_preds, hybrid_preds)
    mlflow.log_artifact('reports/figures/hybrid_residuals.png', artifact_path='plots')
    mlflow.log_artifact('reports/figures/hybrid_forecast.png',  artifact_path='plots')

print('MLflow run logged: prophet-lstm-hybrid')

# ---------- comparison table (computed live, self-contained) ----------
print('\n--- comparison ---')
prophet_metrics = evaluate_forecast(test['Revenue'], prophet_test_preds)

lstm_model  = LSTMLightning.load_from_checkpoint('models/lstm_lightning_checkpoint.ckpt')
with open('models/lstm_lightning_scaler.pkl', 'rb') as f:
    lstm_scaler = pickle.load(f)
lstm_preds   = forecast_lstm_lightning(
    lstm_model, seed_series=train['Revenue'], steps=len(test),
    scaler=lstm_scaler, seq_len=SEQ_LEN,
)
lstm_metrics = evaluate_forecast(test['Revenue'], lstm_preds)

rows = [
    ('Prophet',         prophet_metrics),
    ('LSTM Lightning',  lstm_metrics),
    ('Prophet+LSTM',    hybrid_metrics),
]
print(f'{"Model":<16}{"MAE":>12}{"RMSE":>12}{"MAPE%":>10}')
print('-' * 50)
for name, m in rows:
    print(f'{name:<16}{m["mae"]:>12}{m["rmse"]:>12}{m["mape"]:>10}')

best = min(rows, key=lambda r: r[1]['rmse'])[0]
print(f'\nBest by RMSE: {best}')
print('\nHybrid done.')
