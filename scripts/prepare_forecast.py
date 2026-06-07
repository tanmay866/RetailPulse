import os

from src.forecasting import (
    load_and_resample,
    fill_missing_dates,
    add_log_transform,
    add_time_features,
    add_lag_features,
    add_rolling_features,
    train_test_split_ts,
    test_stationarity,
    difference_series,
    plot_decomposition,
    plot_acf_pacf,
    log_dataset_stats,
)

os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# 1. load and resample to daily
print('--- step 1: load and resample ---')
df = load_and_resample()
print(f'shape: {df.shape}')
print(f'date range: {df.index.min().date()} -> {df.index.max().date()}')

# 2. fill missing dates
print('\n--- step 2: fill missing dates ---')
original_len = len(df)
df = fill_missing_dates(df)
filled = len(df) - original_len
print(f'filled {filled} missing dates ({original_len} -> {len(df)} rows)')

# 3. log transform
print('\n--- step 3: log transform ---')
df = add_log_transform(df)

# 4. time features
print('\n--- step 4: time features ---')
df = add_time_features(df)

# 5. lag features
print('\n--- step 5: lag features ---')
df = add_lag_features(df)

# 6. rolling features
print('\n--- step 6: rolling features ---')
df = add_rolling_features(df)

# 7. drop NaN rows introduced by lags/rolling
print('\n--- step 7: drop NaN rows ---')
before = len(df)
df = df.dropna()
print(f'dropped {before - len(df)} rows, final shape: {df.shape}')

# 8. train/test split
print('\n--- step 8: train/test split ---')
train, test = train_test_split_ts(df)
print(f'train: {len(train)} rows  |  test: {len(test)} rows')

# 9. save processed dataset
print('\n--- step 9: save ---')
df.to_csv('data/processed/daily_revenue_ts.csv')
print('saved data/processed/daily_revenue_ts.csv')

# 10. stationarity test on Revenue
print('\n--- step 10: stationarity test ---')
result = test_stationarity(df['Revenue'])
verdict = result['verdict']

# 11. decomposition plot
print('\n--- step 11: decomposition plot ---')
plot_decomposition(df['Revenue'])

# 12. ACF/PACF plot
print('\n--- step 12: acf/pacf plot ---')
plot_acf_pacf(df['Revenue'])

# 13. if not stationary, difference and re-test
if verdict != 'stationary':
    print('\n--- step 13: differencing + re-test ---')
    diff_series = difference_series(df['Revenue'], order=1)
    print('stationarity after first differencing:')
    result_diff = test_stationarity(diff_series)
    print(f'new verdict: {result_diff["verdict"]}')
else:
    print('\n--- step 13: skipped (already stationary) ---')

# 14. log to MLflow
print('\n--- step 14: mlflow logging ---')
log_dataset_stats(df, train, test, result, missing_dates_filled=filled)

# 15. done
print('\nForecasting prep done.')
