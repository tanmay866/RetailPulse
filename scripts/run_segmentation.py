import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.segmentation import (
    preprocess_rfm, select_best_k,
    run_kmeans, find_eps, run_dbscan,
    evaluate_clusters, reduce_to_2d,
    cluster_profiles, assign_business_labels
)

os.makedirs('reports/figures', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)


print('--- step 1: load data ---')
rfm = pd.read_csv('data/processed/rfm_scores.csv')
print(f'loaded rfm_scores.csv - {len(rfm)} customers')


print('\n--- step 2: preprocess ---')
X_scaled, rfm_clean = preprocess_rfm(rfm)
print('X_scaled shape:', X_scaled.shape)


print('\n--- step 3: dbscan - detect noise points ---')
eps = find_eps(X_scaled, min_samples=5)
dbscan_labels = run_dbscan(X_scaled, eps=eps, min_samples=5, random_state=42)
outlier_mask  = dbscan_labels == -1
print(f'noise points found: {outlier_mask.sum()}')


print('\n--- step 4: remove outliers ---')
X_clean          = X_scaled[~outlier_mask]
rfm_clean_filtered = rfm_clean[~outlier_mask].reset_index(drop=True)
print(f'X_clean shape: {X_clean.shape}')


print('\n--- step 5: k-means - find best k on clean data ---')
best_k = select_best_k(X_clean, max_k=10, min_cluster_pct=0.01)
print(f'best k = {best_k}')


print('\n--- step 6: k-means fit ---')
kmeans_labels, kmeans_model = run_kmeans(X_clean, best_k, random_state=42)
sizes = pd.Series(kmeans_labels).value_counts().sort_index()
print('cluster sizes:')
print(sizes.to_string())
print(f'min cluster size: {sizes.min()}')


print('\n--- step 7: evaluate ---')
# k-means uses X_clean (outliers removed), dbscan uses full X_scaled (filters -1 internally)
km_scores = evaluate_clusters(X_clean,  kmeans_labels)
db_scores = evaluate_clusters(X_scaled, dbscan_labels)

print(f'\n{"Metric":<22} | {"K-Means":>10} | {"DBSCAN":>10}')
print('-' * 48)
print(f'{"Silhouette":<22} | {km_scores["silhouette"]:>10.4f} | {db_scores["silhouette"]:>10.4f}')
print(f'{"Davies-Bouldin":<22} | {km_scores["davies_bouldin"]:>10.4f} | {db_scores["davies_bouldin"]:>10.4f}')
print(f'{"Calinski-Harabasz":<22} | {km_scores["calinski_harabasz"]:>10.2f} | {db_scores["calinski_harabasz"]:>10.2f}')

print('\n--- step 8: visualize ---')
coords_2d = reduce_to_2d(X_clean, random_state=42)

# plot 1 — k-means scatter
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.6, s=10)
ax.set_title(f'K-Means Clusters (k={best_k})', fontweight='bold')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
plt.colorbar(scatter, ax=ax)
plt.tight_layout()
plt.savefig('reports/figures/kmeans_scatter.png', dpi=150)
plt.close()
print('saved kmeans_scatter.png')

# plot 2 — dbscan scatter (noise in black, plotted separately)
fig, ax = plt.subplots(figsize=(10, 7))
mask_noise = dbscan_labels == -1
mask_valid = ~mask_noise
# dbscan uses full X_scaled so we need its own 2d coords
coords_2d_full = reduce_to_2d(X_scaled, random_state=42)
if mask_valid.any():
    ax.scatter(coords_2d_full[mask_valid, 0], coords_2d_full[mask_valid, 1], c=dbscan_labels[mask_valid], cmap='tab10', alpha=0.6, s=10)
if mask_noise.any():
    ax.scatter(coords_2d_full[mask_noise, 0], coords_2d_full[mask_noise, 1], c='black', alpha=0.5, s=10, label='noise')
ax.set_title('DBSCAN Clusters (black = noise)', fontweight='bold')
ax.set_xlabel('PCA Component 1')
ax.set_ylabel('PCA Component 2')
ax.legend()
plt.tight_layout()
plt.savefig('reports/figures/dbscan_scatter.png', dpi=150)
plt.close()
print('saved dbscan_scatter.png')


print('\n--- step 9: cluster profiles ---')
profiles = cluster_profiles(rfm_clean_filtered, kmeans_labels)
profiles.plot(kind='bar', figsize=(12, 5), colormap='Set2', edgecolor='white')
plt.title('Mean RFM per Cluster', fontweight='bold')
plt.xlabel('Cluster')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('reports/figures/cluster_profiles.png', dpi=150)
plt.close()
print('saved cluster_profiles.png')


print('\n--- step 10: business labels ---')
km_business = assign_business_labels(rfm_clean_filtered, kmeans_labels)
print('K-Means segments:')
print(km_business.value_counts().to_string())


print('\n--- step 11: re-attach outlier rows ---')
# get original rfm rows for the 38 outliers
rfm_outliers = rfm[outlier_mask].copy().reset_index(drop=True)
rfm_outliers['KMeans_Cluster'] = -1
rfm_outliers['DBSCAN_Cluster'] = -1
rfm_outliers['Business_Label'] = 'Outlier'
print(f'outlier rows: {len(rfm_outliers)}')


print('\n--- step 12: merge and save ---')
# build clean rows dataframe
rfm_segment = rfm[~outlier_mask].copy().reset_index(drop=True)
rfm_segment['KMeans_Cluster'] = kmeans_labels
rfm_segment['DBSCAN_Cluster'] = dbscan_labels[~outlier_mask]
rfm_segment['Business_Label'] = km_business.values

# combine clean + outlier rows
final = pd.concat([rfm_segment, rfm_outliers], ignore_index=True)

# keep only required columns
cols = ['Customer ID', 'Recency', 'Frequency', 'Monetary',
        'KMeans_Cluster', 'DBSCAN_Cluster', 'Business_Label']
# handle possible column name difference
if 'Customer ID' not in final.columns and 'CustomerID' in final.columns:
    cols[0] = 'CustomerID'
final = final[cols]

final.to_csv('data/processed/customer_segments.csv', index=False)
print('saved customer_segments.csv')
print(f'final shape: {final.shape}')
print('\nBusiness_Label distribution:')
print(final['Business_Label'].value_counts().to_string())

print('\ndone.')
