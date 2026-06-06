from __future__ import annotations

import pendulum
from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator


@dag(
    dag_id="retailpulse_retrain",
    schedule="@weekly",
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    tags=["retailpulse", "mlops"],
    doc_md="""
## RetailPulse Automated Retraining Pipeline

Runs weekly. Checks for feature drift via Evidently AI and retrains the churn
(XGBoost) and forecasting (Prophet + LSTM) models if churn feature drift share
exceeds 30%.  New models are promoted only if they beat the stored baseline in
`models/baseline_metrics.json`.

### Task flow

```
check_drift → branch_on_drift
                ├─ [drift ≥ 0.30] → retrain_churn → retrain_forecasting
                │                        → validate_models → promote_if_better
                └─ [drift < 0.30] → skip_retrain
```

### MLflow experiments written
- `churn_prediction`       (run: churn_xgboost_v1)
- `retailpulse-forecasting` (run: prophet-lstm-hybrid)
    """,
)
def retailpulse_retrain() -> None:

    @task
    def check_drift() -> float:
        from src.retrain_pipeline import get_drift_score
        score = get_drift_score()
        print(f"Churn feature drift share: {score:.4f}")
        return score

    @task.branch
    def branch_on_drift(drift_score: float) -> str:
        from src.retrain_pipeline import DRIFT_THRESHOLD
        print(f"Drift score {drift_score:.4f} vs threshold {DRIFT_THRESHOLD}")
        if drift_score >= DRIFT_THRESHOLD:
            return "retrain_churn"
        return "skip_retrain"

    @task(task_id="retrain_churn")
    def retrain_churn_task() -> dict:
        from src.retrain_pipeline import retrain_churn
        metrics = retrain_churn()
        print(f"Churn retrain complete — AUC: {metrics['churn_auc']:.4f}")
        return metrics

    @task(task_id="retrain_forecasting")
    def retrain_forecasting_task() -> dict:
        from src.retrain_pipeline import retrain_forecasting
        metrics = retrain_forecasting()
        print(f"Forecasting retrain complete — RMSE: {metrics['forecast_rmse']:.4f}")
        return metrics

    @task(task_id="validate_models")
    def validate_models_task(churn_metrics: dict, forecast_metrics: dict) -> dict:
        from airflow.exceptions import AirflowSkipException
        from src.retrain_pipeline import validate_models
        combined = {**churn_metrics, **forecast_metrics}
        if not validate_models(combined):
            raise AirflowSkipException(
                "New models did not beat baseline — skipping promotion."
            )
        print("Validation passed — new models beat baseline.")
        return combined

    @task(task_id="promote_if_better")
    def promote_task(metrics: dict) -> None:
        from src.retrain_pipeline import promote_models
        promote_models(metrics)

    skip = EmptyOperator(task_id="skip_retrain")

    # ---- wire up the graph ----
    score      = check_drift()
    branch     = branch_on_drift(score)
    churn_m    = retrain_churn_task()
    forecast_m = retrain_forecasting_task()
    valid_m    = validate_models_task(churn_m, forecast_m)
    promote_task(valid_m)

    branch >> [churn_m, skip]
    churn_m >> forecast_m


dag_instance = retailpulse_retrain()
