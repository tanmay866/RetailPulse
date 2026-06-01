"""Generate a profile card for all processed datasets and write it to reports/dataset_profile.json."""
import json
import os

import pandas as pd

OUT_PATH = 'reports/dataset_profile.json'


def _profile_df(df, name):
    missing = df.isnull().sum()
    missing = missing[missing > 0].to_dict()
    return {
        'name':           name,
        'rows':           int(len(df)),
        'columns':        int(len(df.columns)),
        'column_names':   list(df.columns),
        'dtypes':         {c: str(t) for c, t in df.dtypes.items()},
        'missing_values': {k: int(v) for k, v in missing.items()},
    }


def build_profile():
    os.makedirs('reports', exist_ok=True)

    # ---- retail_clean.csv ----
    clean = pd.read_csv('data/processed/retail_clean.csv', low_memory=False)
    clean['InvoiceDate'] = pd.to_datetime(clean['InvoiceDate'])
    clean['Revenue']     = clean['Quantity'] * clean['Price']

    clean_profile = _profile_df(clean, 'retail_clean')
    clean_profile.update({
        'date_range': {
            'start':     str(clean['InvoiceDate'].min().date()),
            'end':       str(clean['InvoiceDate'].max().date()),
            'span_days': int((clean['InvoiceDate'].max() - clean['InvoiceDate'].min()).days),
        },
        'revenue_stats': {
            'min':   round(float(clean['Revenue'].min()), 4),
            'max':   round(float(clean['Revenue'].max()), 4),
            'mean':  round(float(clean['Revenue'].mean()), 4),
            'std':   round(float(clean['Revenue'].std()), 4),
            'total': round(float(clean['Revenue'].sum()), 2),
        },
        'unique_customers': int(clean['Customer ID'].nunique()),
        'unique_products':  int(clean['Description'].nunique()),
        'unique_invoices':  int(clean['Invoice'].nunique()),
        'countries':        int(clean['Country'].nunique()),
    })

    # ---- rfm_scores.csv ----
    rfm         = pd.read_csv('data/processed/rfm_scores.csv')
    rfm_profile = _profile_df(rfm, 'rfm_scores')
    if 'Segment' in rfm.columns:
        rfm_profile['segment_distribution'] = rfm['Segment'].value_counts().to_dict()

    # ---- daily_revenue_ts.csv ----
    ts         = pd.read_csv('data/processed/daily_revenue_ts.csv', index_col=0, parse_dates=True)
    ts_profile = _profile_df(ts, 'daily_revenue_ts')
    ts_profile.update({
        'date_range': {
            'start':     str(ts.index.min().date()),
            'end':       str(ts.index.max().date()),
            'span_days': int(len(ts)),
        },
        'revenue_stats': {
            'min':  round(float(ts['Revenue'].min()), 2),
            'max':  round(float(ts['Revenue'].max()), 2),
            'mean': round(float(ts['Revenue'].mean()), 2),
            'std':  round(float(ts['Revenue'].std()), 2),
        },
    })

    profile = {
        'retail_clean':     clean_profile,
        'rfm_scores':       rfm_profile,
        'daily_revenue_ts': ts_profile,
    }

    with open(OUT_PATH, 'w') as f:
        json.dump(profile, f, indent=2, default=str)
    print(f'saved {OUT_PATH}')


if __name__ == '__main__':
    build_profile()
