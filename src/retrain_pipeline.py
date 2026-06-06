from __future__ import annotations

import json
from pathlib import Path

ROOT          = Path(__file__).resolve().parents[1]
MODELS_DIR    = ROOT / "models"
BASELINE_PATH = MODELS_DIR / "baseline_metrics.json"

DRIFT_THRESHOLD = 0.30  # retrain if >30% of churn features have drifted


def get_drift_score() -> float:
    """Run Evidently drift detection and return churn feature drift share (0.0–1.0)."""
    from src.drift_detector import run_all
    results = run_all(log_to_mlflow=False)
    return float(results["churn_feature_drift_share"])


def retrain_churn() -> dict:
    """Retrain XGBoost churn model (20 Optuna trials). Logs to MLflow experiment
    'churn_prediction'. Calls sys.exit(1) if AUC or precision gates not met —
    Airflow marks the task failed, which is the intended behaviour.
    """
    from src.churn import run as churn_run
    metrics = churn_run(n_trials=20)
    return {"churn_auc": float(metrics["auc_roc"])}


def retrain_forecasting() -> dict:
    """Retrain Prophet from scratch, then retrain LSTM on Prophet residuals.
    Saves:
      models/prophet_model.pkl
      models/hybrid_residual_lstm.ckpt  (written by PyTorch Lightning)
    Logs to MLflow experiment 'retailpulse-forecasting'.
    """
    import mlflow
    import pandas as pd

    from src.forecasting import (
        evaluate_forecast,
        get_prophet_test_preds,
        save_prophet,
        train_prophet,
        train_test_split_ts,
    )
    from src.hybrid_forecaster import compute_residuals, hybrid_forecast
    from src.lstm_lightning import forecast_lstm_lightning, train_lstm_lightning

    df = pd.read_csv(
        ROOT / "data" / "processed" / "daily_revenue_ts.csv",
        index_col=0,
        parse_dates=True,
    )
    train, test = train_test_split_ts(df)

    prophet_model = train_prophet(train, target="Revenue")
    save_prophet(prophet_model, path=str(MODELS_DIR / "prophet_model.pkl"))

    residuals = compute_residuals(prophet_model, train, target="Revenue")

    mlflow.set_experiment("retailpulse-forecasting")
    with mlflow.start_run(run_name="prophet-lstm-hybrid") as run:
        model, dm = train_lstm_lightning(
            residuals,
            seq_len=30,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            lr=1e-3,
            batch_size=32,
            max_epochs=50,
            patience=10,
            mlflow_run_id=run.info.run_id,
            checkpoint_name="hybrid_residual_lstm",
            experiment_name="retailpulse-forecasting",
        )
        prophet_preds  = get_prophet_test_preds(prophet_model, test, target="Revenue")
        residual_preds = forecast_lstm_lightning(model, residuals, len(test), dm.scaler)
        hybrid_preds   = hybrid_forecast(prophet_preds, residual_preds)
        metrics        = evaluate_forecast(test["Revenue"], hybrid_preds)
        mlflow.log_metrics(metrics)

    return {
        "forecast_rmse": metrics["rmse"],
        "forecast_mae":  metrics["mae"],
    }


def validate_models(new_metrics: dict) -> bool:
    """Return True if new models beat or match the stored baseline.
    Returns True unconditionally on the first run (no baseline exists yet).
    """
    if not BASELINE_PATH.exists():
        return True
    baseline    = json.loads(BASELINE_PATH.read_text())
    churn_ok    = new_metrics.get("churn_auc", 1.0)     >= baseline.get("churn_auc", 0.0)
    forecast_ok = new_metrics.get("forecast_rmse", 0.0) <= baseline.get("forecast_rmse", float("inf"))
    return churn_ok and forecast_ok


def promote_models(metrics: dict) -> None:
    """Write models/baseline_metrics.json so the next retrain has a comparison target.
    Model artifact files (prophet_model.pkl, hybrid_residual_lstm.ckpt,
    churn_xgboost.pkl) are already written to models/ by their respective
    retrain functions — nothing to copy here.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Baseline updated -> {BASELINE_PATH}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
