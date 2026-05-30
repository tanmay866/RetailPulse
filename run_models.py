import os
import pandas as pd

from src.forecasting import (
    train_test_split_ts,
    train_prophet,
    forecast_prophet,
    get_prophet_test_preds,
    plot_prophet_forecast,
    train_lstm,
    forecast_lstm,
    evaluate_forecast,
    plot_forecast_comparison,
    save_lstm,
    save_prophet,
    log_model_metrics,
)

os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

print('--- loading prepared dataset ---')
df = pd.read_csv('data/processed/daily_revenue_ts.csv', index_col=0, parse_dates=True)
print(f'shape: {df.shape}')

train, test = train_test_split_ts(df)
print(f'train: {len(train)}  test: {len(test)}')


# ---------- Prophet ----------

print('\n--- Prophet: training ---')
prophet_model = train_prophet(train, target='Revenue')

print('\n--- Prophet: forecast on test period ---')
prophet_preds = get_prophet_test_preds(prophet_model, test, target='Revenue')

prophet_metrics = evaluate_forecast(test['Revenue'], prophet_preds)
print(f'MAE  : {prophet_metrics["mae"]}')
print(f'RMSE : {prophet_metrics["rmse"]}')
print(f'MAPE : {prophet_metrics["mape"]}%')

print('\n--- Prophet: full forecast plot ---')
full_forecast = forecast_prophet(prophet_model, periods=len(test))
plot_prophet_forecast(prophet_model, full_forecast)

print('\n--- Prophet: saving model ---')
save_prophet(prophet_model)

log_model_metrics(
    'prophet',
    metrics=prophet_metrics,
    params={'yearly_seasonality': True, 'weekly_seasonality': True},
    run_name='prophet-eval',
)


# ---------- LSTM ----------

print('\n--- LSTM: training ---')
lstm_model, scaler = train_lstm(
    train['Revenue'],
    seq_len=30,
    hidden_size=64,
    num_layers=2,
    epochs=50,
    lr=0.001,
)

print('\n--- LSTM: forecast on test period ---')
lstm_preds = forecast_lstm(lstm_model, train['Revenue'], steps=len(test), scaler=scaler, seq_len=30)

lstm_metrics = evaluate_forecast(test['Revenue'], lstm_preds)
print(f'MAE  : {lstm_metrics["mae"]}')
print(f'RMSE : {lstm_metrics["rmse"]}')
print(f'MAPE : {lstm_metrics["mape"]}%')

print('\n--- LSTM: saving model ---')
save_lstm(lstm_model, scaler)

log_model_metrics(
    'lstm',
    metrics=lstm_metrics,
    params={'seq_len': 30, 'hidden_size': 64, 'num_layers': 2, 'epochs': 50, 'lr': 0.001},
    run_name='lstm-eval',
)


# ---------- Comparison ----------

print('\n--- comparison plot ---')
plot_forecast_comparison(test['Revenue'], prophet_preds, lstm_preds)

print('\n--- summary ---')
print(f'{"Model":<10} {"MAE":>10} {"RMSE":>10} {"MAPE%":>10}')
print('-' * 45)
print(f'{"Prophet":<10} {prophet_metrics["mae"]:>10} {prophet_metrics["rmse"]:>10} {prophet_metrics["mape"]:>10}')
print(f'{"LSTM":<10} {lstm_metrics["mae"]:>10} {lstm_metrics["rmse"]:>10} {lstm_metrics["mape"]:>10}')

print('\nModels done.')
