"""CLI runner for the PyTorch Lightning LSTM forecasting model."""
import os
import pickle

import mlflow
import mlflow.pytorch
import pandas as pd

from src.forecasting import (
    train_test_split_ts,
    evaluate_forecast,
    plot_forecast_comparison,
    load_prophet,
    get_prophet_test_preds,
)
from src.lstm_lightning import train_lstm_lightning, forecast_lstm_lightning

os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# ---------- config ----------
SEQ_LEN     = 30
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
LR          = 1e-3
BATCH_SIZE  = 32
MAX_EPOCHS  = 50
PATIENCE    = 10

# ---------- load data ----------
print('--- loading dataset ---')
df = pd.read_csv('data/processed/daily_revenue_ts.csv', index_col=0, parse_dates=True)
print(f'shape: {df.shape}')

train, test = train_test_split_ts(df)
print(f'train: {len(train)}  test: {len(test)}')

# ---------- train ----------
print('\n--- training LSTM (Lightning) ---')
model, dm = train_lstm_lightning(
    train['Revenue'],
    seq_len=SEQ_LEN,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    lr=LR,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
)

# ---------- forecast ----------
print('\n--- forecasting on test period ---')
lstm_preds = forecast_lstm_lightning(
    model,
    seed_series=train['Revenue'],
    steps=len(test),
    scaler=dm.scaler,
    seq_len=SEQ_LEN,
)

# ---------- evaluate ----------
metrics = evaluate_forecast(test['Revenue'], lstm_preds)
print(f'MAE  : {metrics["mae"]}')
print(f'RMSE : {metrics["rmse"]}')
print(f'MAPE : {metrics["mape"]}%')

# ---------- log metrics + model artifact to MLflow ----------
print('\n--- logging to MLflow ---')
mlflow.set_experiment('retailpulse-forecasting')
with mlflow.start_run(run_name='lstm-lightning-eval'):
    mlflow.log_params({
        'seq_len':     SEQ_LEN,
        'hidden_size': HIDDEN_SIZE,
        'num_layers':  NUM_LAYERS,
        'dropout':     DROPOUT,
        'lr':          LR,
        'batch_size':  BATCH_SIZE,
        'max_epochs':  MAX_EPOCHS,
        'patience':    PATIENCE,
    })
    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(model, artifact_path='lstm_lightning_model')
print('MLflow run logged: lstm-lightning-eval')

# ---------- save scaler ----------
scaler_path = 'models/lstm_lightning_scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(dm.scaler, f)
print(f'saved {scaler_path}')

# ---------- comparison plot ----------
print('\n--- comparison plot ---')
try:
    prophet_model = load_prophet()
    prophet_preds = get_prophet_test_preds(prophet_model, test, target='Revenue')
    plot_forecast_comparison(
        test['Revenue'],
        prophet_preds,
        lstm_preds,
        save_path='reports/figures/lstm_lightning_forecast.png',
    )
except FileNotFoundError:
    print('Prophet model not found — skipping comparison plot. Run run_models.py first.')

print('\nLSTM Lightning done.')
