# RetailPulse

AI-powered customer analytics and demand forecasting platform built on retail transaction data.
Internship project at Zidio Development.

---

## What This Project Does

- Customer segmentation using RFM scoring and clustering (K-Means, DBSCAN)
- Demand forecasting using Prophet + PyTorch Lightning LSTM hybrid
- Churn prediction using XGBoost tuned with Optuna
- Inventory optimization (EOQ, safety stock, reorder point)
- Feature importance analysis and multi-model tuning
- Data drift detection using Evidently AI
- Automated model retraining pipeline with Apache Airflow
- Experiment tracking with MLflow

---

## Project Structure

```
RetailPulse/
├── data/
│   ├── raw/                             ← place downloaded CSVs here
│   ├── interim/                         ← cleaned intermediate files
│   └── processed/
│       ├── retail_clean.csv             ← 779k cleaned transactions
│       ├── rfm_scores.csv               ← RFM scores for 5,878 customers
│       ├── customer_segments.csv        ← K-Means + DBSCAN segment labels
│       ├── daily_revenue_ts.csv         ← 709-row daily revenue series
│       ├── churn_predictions.csv        ← per-customer churn probabilities
│       └── inventory_recommendations.csv
├── models/
│   ├── prophet_model.pkl
│   ├── hybrid_residual_lstm.ckpt        ← Prophet residual LSTM checkpoint
│   ├── churn_xgboost.pkl                ← best Optuna-tuned XGBoost churn model
│   └── baseline_metrics.json           ← gate thresholds for retrain validation
├── notebooks/
│   ├── eda.ipynb
│   ├── cleaning.ipynb
│   ├── validation.ipynb
│   ├── segmentation.ipynb
│   ├── forecasting.ipynb
│   ├── lstm_lightning.ipynb
│   ├── churn_prediction.ipynb
│   ├── inventory_optimization.ipynb
│   ├── feature_importance.ipynb
│   └── drift_detection.ipynb
├── src/
│   ├── feature_engineering.py          ← load/clean sales, RFM, rolling stats
│   ├── segmentation.py                 ← K-Means, DBSCAN, evaluation
│   ├── forecasting.py                  ← Prophet training, evaluation, MLflow
│   ├── lstm_lightning.py               ← PyTorch Lightning LSTM + DataModule
│   ├── hybrid_forecaster.py            ← Prophet + LSTM residual hybrid
│   ├── churn.py                        ← XGBoost churn model + Optuna tuning
│   ├── inventory_optimizer.py          ← EOQ, safety stock, reorder logic
│   ├── model_tuner.py                  ← multi-model Optuna tuning + comparison
│   ├── drift_detector.py               ← Evidently drift reports + MLflow logging
│   └── retrain_pipeline.py             ← drift-triggered retrain + model gate
├── dags/
│   └── retailpulse_retrain_dag.py      ← Airflow DAG for scheduled retraining
├── docker/
│   ├── Dockerfile.airflow
│   └── docker-compose-airflow.yml
├── tests/
│   ├── test_forecasting.py
│   ├── test_lstm_lightning.py
│   ├── test_churn.py
│   └── test_inventory.py
├── reports/
│   ├── figures/                        ← model and analysis plots
│   └── week2_checkpoint.json           ← Week 2 verified metric snapshot
├── main.py                             ← segmentation pipeline entry point
├── prepare_forecast.py                 ← time-series data prep
├── run_hybrid.py                       ← Prophet + LSTM hybrid training
├── run_churn.py                        ← churn model training
├── run_inventory.py                    ← inventory optimization
├── run_model_tuner.py                  ← multi-model Optuna tuning
├── run_drift.py                        ← drift detection report
├── run_models.py                       ← Prophet + raw LSTM (baseline)
├── run_lstm.py                         ← PyTorch Lightning LSTM (baseline)
├── requirements.txt
└── .gitignore
```

---

## Datasets

Not included due to file size. Download from Kaggle and place in `data/raw/`:

1. **Online Retail II** — `online_retail_II.csv`
   https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

2. **Customer Churn** — `online_retail_customer_churn.csv`
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn

3. **Retail Store Inventory** — `retail_store_inventory.csv`
   https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset

---

## Setup

```bash
git clone https://github.com/tanmay866/RetailPulse.git
cd RetailPulse
pip install -r requirements.txt
```

Download the datasets (see above) and place them in `data/raw/`.

---

## How to Run

```bash
# Segmentation
python main.py

# Time-series data prep (run once before forecasting)
python prepare_forecast.py

# Hybrid forecasting (Prophet + LSTM)
python run_hybrid.py

# Churn prediction
python run_churn.py

# Inventory optimization
python run_inventory.py

# Multi-model tuning (Optuna)
python run_model_tuner.py

# Drift detection
python run_drift.py

# Tests
pytest tests/ -v
```

---

## Progress

| Task | Status |
|------|--------|
| EDA | Done |
| Data Cleaning & Feature Engineering | Done |
| Data Validation (Great Expectations) | Done |
| Customer Segmentation (K-Means + DBSCAN) | Done |
| Time-Series Forecasting (Prophet + LSTM hybrid) | Done |
| Churn Prediction (XGBoost + Optuna) | Done |
| Inventory Optimization (EOQ + safety stock) | Done |
| Feature Importance & Model Tuning | Done |
| Drift Detection (Evidently AI) | Done |
| Automated Retraining Pipeline (Airflow) | Done |
| Week 2 Checkpoint | Done |
| Streamlit Dashboard | Pending |
