# RetailPulse

AI-powered customer analytics and demand forecasting platform built on retail transaction data.
Internship project at Zidio Development.

---

## What This Project Does

- Customer segmentation using RFM scoring and clustering (K-Means, DBSCAN)
- Churn prediction using classification models
- Demand forecasting using time series models
- Data validation with Great Expectations

---

## Project Structure

```
RetailPulse/
├── data/
│   ├── raw/              ← place downloaded CSVs here
│   ├── interim/          ← cleaned intermediate files
│   └── processed/        ← final output files (RFM scores, segments, etc.)
├── notebooks/
│   ├── eda.ipynb
│   ├── cleaning.ipynb
│   ├── validation.ipynb
│   ├── segmentation.ipynb
│   └── forecasting.ipynb       ← time-series forecasting prep (Day 4)
├── src/
│   ├── feature_engineering.py
│   ├── segmentation.py         ← clustering functions (K-Means, DBSCAN)
│   └── forecasting.py          ← time-series feature engineering and stationarity
├── models/               ← trained model files (not tracked by git)
├── reports/
│   └── figures/          ← generated charts
├── main.py               ← end-to-end segmentation pipeline
├── prepare_forecast.py   ← end-to-end forecasting prep pipeline (Day 4)
├── pyrefly.toml          ← Pyrefly type checker config
├── requirements.txt
└── .gitignore
```

---

## Datasets

Datasets are not included in this repo due to file size.
Download them from Kaggle and place in `data/raw/` folder:

1. **Online Retail II**
   Link: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
   File: `online_retail_II.csv` → place in `data/raw/`

2. **Customer Churn**
   Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   File: `online_retail_customer_churn.csv` → place in `data/raw/`

3. **Retail Store Inventory**
   Link: https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset
   File: `retail_store_inventory.csv` → place in `data/raw/`

---

## Setup

```bash
# clone the repo
git clone https://github.com/tanmay866/RetailPulse.git
cd RetailPulse

# install dependencies
pip install -r requirements.txt
```

**Key dependencies:** pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels, mlflow, great_expectations, kneed

Then download the datasets (see above) and place them in `data/raw/`.

---

## Progress

| Task | Status |
|------|--------|
| EDA | Done |
| Data Cleaning & Feature Engineering | Done |
| Data Validation (Great Expectations) | Done |
| Customer Segmentation (K-Means, DBSCAN) | Done |
| Churn Prediction | Pending |
| Demand Forecasting (prep) | Done |
