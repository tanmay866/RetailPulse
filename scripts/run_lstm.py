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

SCALER_PATH = 'models/lstm_lightning_scaler.pkl'
CKPT_PATH   = 'models/lstm_lightning_checkpoint.ckpt'

# ---------- load data ----------
print('--- loading dataset ---')
df = pd.read_csv('data/processed/daily_revenue_ts.csv', index_col=0, parse_dates=True)
print(f'shape: {df.shape}')

train, test = train_test_split_ts(df)
print(f'train: {len(train)}  test: {len(test)}')

# ---------- train + evaluate + log ----------
print('\n--- training LSTM (Lightning) ---')
mlflow.set_experiment('retailpulse-forecasting')
with mlflow.start_run(run_name='lstm-lightning-baseline') as run:
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
    mlflow.set_tags({
        'model_type': 'lstm',
        'framework':  'pytorch_lightning',
        'target':     'daily_revenue',
        'dataset':    'online_retail_II',
    })

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
        mlflow_run_id=run.info.run_id,
    )

    print('\n--- forecasting on test period ---')
    lstm_preds = forecast_lstm_lightning(
        model,
        seed_series=train['Revenue'],
        steps=len(test),
        scaler=dm.scaler,
        seq_len=SEQ_LEN,
    )

    metrics = evaluate_forecast(test['Revenue'], lstm_preds)
    print(f'MAE  : {metrics["mae"]}')
    print(f'RMSE : {metrics["rmse"]}')
    print(f'MAPE : {metrics["mape"]}%')

    mlflow.log_metrics(metrics)
    mlflow.pytorch.log_model(model, artifact_path='lstm_lightning_model')
    if os.path.exists(CKPT_PATH):
        mlflow.log_artifact(CKPT_PATH, artifact_path='checkpoint')

print('MLflow run logged: lstm-lightning-baseline')

# ---------- save scaler ----------
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(dm.scaler, f)
print(f'saved {SCALER_PATH}')

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
