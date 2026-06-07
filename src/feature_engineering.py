import pandas as pd
import os


# -- data loading & cleaning -------------------------------------------------
def load_sales(path):
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df


def clean_sales(df):
    # remove cancellations, zero prices, missing customer ids
    df = df[
        (df['Quantity'] > 0) &
        (df['Price'] > 0) &
        (df['Customer ID'].notna())
    ].copy()

    df['Customer ID'] = df['Customer ID'].astype(int)
    df['Description'] = df['Description'].fillna('Unknown')
    df['Revenue'] = df['Quantity'] * df['Price']

    before = len(df)
    df = df.drop_duplicates()
    print(f'removed {before - len(df)} duplicates, {len(df)} rows kept')

    return df.reset_index(drop=True)


# -- RFM scoring -------------------------------------------------------------
def build_rfm(df, ref_date=None):
    if ref_date is None:
        ref_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg(
        Recency   = ('InvoiceDate', lambda x: (ref_date - x.max()).days),
        Frequency = ('Invoice', 'nunique'),
        Monetary  = ('Revenue', 'sum')
    ).reset_index()

    # score 1-5 (recency reversed: lower days = higher score)
    rfm['R_score'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1,2,3,4,5]).astype(int)

    rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    rfm['RFM_Total'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    rfm['Segment']   = rfm['RFM_Total'].apply(_segment_label)

    print(f'{len(rfm)} customers segmented')
    print(rfm['Segment'].value_counts().to_string())

    return rfm


def _segment_label(score):
    if score >= 13:
        return 'Champions'
    elif score >= 10:
        return 'Loyal Customers'
    elif score >= 7:
        return 'Potential Loyalists'
    elif score >= 5:
        return 'At Risk'
    else:
        return 'Lost'


# -- rolling statistics ------------------------------------------------------
def build_rolling_stats(df, windows=[7, 30]):
    daily = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()
    daily.columns = ['Date', 'Revenue']
    daily['Date'] = pd.to_datetime(daily['Date'])
    daily = daily.sort_values('Date').reset_index(drop=True)

    for w in windows:
        daily[f'rolling_{w}d_mean'] = daily['Revenue'].rolling(window=w).mean()
        daily[f'rolling_{w}d_std']  = daily['Revenue'].rolling(window=w).std()

    print(f'{len(daily)} days processed, windows={windows}')
    return daily


# -- run as script -----------------------------------------------------------

if __name__ == '__main__':
    RAW_DIR       = 'data/raw'
    INTERIM_DIR   = 'data/interim'
    PROCESSED_DIR = 'data/processed'

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df_raw   = load_sales(f'{RAW_DIR}/online_retail_II.csv')
    df_clean = clean_sales(df_raw)
    df_clean.to_csv(f'{INTERIM_DIR}/retail_valid_transactions.csv', index=False)

    rfm = build_rfm(df_clean)
    rfm.to_csv(f'{PROCESSED_DIR}/rfm_scores.csv', index=False)

    daily = build_rolling_stats(df_clean)
    daily.to_csv(f'{PROCESSED_DIR}/daily_revenue_rolling.csv', index=False)

    df_clean.to_csv(f'{PROCESSED_DIR}/retail_clean.csv', index=False)

    print('done')
