# RetailPulse

AI-powered customer analytics and demand forecasting platform built on retail transaction data.
Internship project at Zidio Development.

---

## What This Project Does

- Customer segmentation using RFM scoring and clustering (K-Means, DBSCAN)
- Churn prediction using classification models
- Demand forecasting using time series models (Prophet, LSTM)
- Data validation with Great Expectations
- Experiment tracking with MLflow

---

## Project Structure

```
RetailPulse/
├── data/
│   ├── raw/                        ← place downloaded CSVs here
│   ├── interim/                    ← cleaned intermediate files
│   └── processed/
│       ├── retail_clean.csv        ← 779k cleaned transactions
│       ├── rfm_scores.csv          ← RFM scores for 5,878 customers
│       ├── customer_segments.csv   ← K-Means + DBSCAN segment labels
│       ├── daily_revenue_rolling.csv
│       └── daily_revenue_ts.csv    ← 709-row feature-engineered daily series
├── models/
│   ├── prophet_model.pkl           ← trained Prophet forecasting model
│   ├── lstm_forecaster.pt          ← Day 5 raw PyTorch LSTM weights
│   ├── lstm_scaler.pkl             ← scaler for Day 5 LSTM
│   ├── lstm_lightning_checkpoint.ckpt  ← best Lightning checkpoint (gitignored)
│   └── lstm_lightning_scaler.pkl   ← scaler for Lightning LSTM
├── notebooks/
│   ├── eda.ipynb
│   ├── cleaning.ipynb
│   ├── validation.ipynb
│   ├── segmentation.ipynb
│   ├── forecasting.ipynb           ← Prophet + Day 5 LSTM walkthrough
│   └── lstm_lightning.ipynb        ← Day 6 PyTorch Lightning LSTM walkthrough
├── reports/
│   └── figures/
│       ├── ts_decomposition.png
│       ├── ts_acf_pacf.png
│       ├── prophet_forecast.png
│       ├── prophet_components.png
│       ├── prophet_changepoints.png
│       ├── prophet_cv_metrics.png
│       ├── prophet_tuning.png
│       ├── forecast_comparison.png
│       ├── lstm_training_curve.png      ← train vs val loss per epoch
│       └── lstm_lightning_forecast.png  ← Actual vs Prophet vs LSTM-Lightning
├── src/
│   ├── feature_engineering.py      ← load/clean sales, RFM, rolling stats
│   ├── segmentation.py             ← K-Means, DBSCAN, evaluation, visualisation
│   ├── forecasting.py              ← Prophet + raw LSTM, evaluation, MLflow
│   └── lstm_lightning.py           ← PyTorch Lightning LSTM, DataModule, trainer
├── tests/
│   ├── test_forecasting.py         ← 7 pytest unit tests for Day 5 pipeline
│   └── test_lstm_lightning.py      ← 4 pytest unit tests for Day 6 Lightning LSTM
├── main.py                         ← segmentation pipeline CLI entry point
├── prepare_forecast.py             ← time-series data prep pipeline
├── run_models.py                   ← Prophet + raw LSTM training runner (Day 5)
├── run_lstm.py                     ← PyTorch Lightning LSTM runner (Day 6)
├── pyrefly.toml
├── requirements.txt
└── .gitignore
```

---

## Datasets

Datasets are not included in this repo due to file size.
Download them from Kaggle and place in `data/raw/`:

1. **Online Retail II**
   Link: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
   File: `online_retail_II.csv`

2. **Customer Churn**
   Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   File: `online_retail_customer_churn.csv`

3. **Retail Store Inventory**
   Link: https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset
   File: `retail_store_inventory.csv`

---

## Setup

```bash
git clone https://github.com/tanmay866/RetailPulse.git
cd RetailPulse
pip install -r requirements.txt
```

**Key dependencies:** pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels,
prophet, torch, pytorch-lightning, mlflow, great_expectations, kneed

Download the datasets (see above) and place them in `data/raw/`.

---

## How to Run

### Segmentation pipeline
```bash
python main.py
```

### Forecasting — data prep
```bash
python prepare_forecast.py
```

### Forecasting — Prophet + raw LSTM (Day 5)
```bash
python run_models.py
```

### Forecasting — PyTorch Lightning LSTM (Day 6)
```bash
python run_lstm.py
```

### Tests
```bash
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
| Time-Series Forecasting Prep | Done |
| Baseline Prophet Model | Done |
| LSTM Forecaster — PyTorch Lightning | Done |
| Churn Prediction | Pending |

---


