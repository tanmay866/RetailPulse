import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from kneed import KneeLocator


# -- preprocessing -----------------------------------------------------------

def preprocess_rfm(rfm):
    # log-transform Monetary to reduce right skew, then scale all 3
    rfm_clean = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    rfm_clean['Monetary'] = np.log1p(rfm_clean['Monetary'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_clean)

    # return both — X_scaled for clustering, rfm_clean for profiling
    return X_scaled, rfm_clean


# -- k-means helpers ---------------------------------------------------------

def elbow_analysis(X, max_k=10):
    # returns inertia per k only
    inertia = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia[k] = km.inertia_
    return inertia


def silhouette_analysis(X, max_k=10):
    # returns silhouette score per k only
    scores = {}
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    return scores


def select_best_k(X, max_k=10, min_cluster_pct=0.01, min_k=3):
    # picks best k where all clusters have at least 1% of total customers
    # min_k enforces a minimum number of clusters regardless of silhouette
    n = len(X)
    min_size = max(int(n * min_cluster_pct), 1)
    valid = {}

    for k in range(max(2, min_k), max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sizes = pd.Series(labels).value_counts()

        if sizes.min() >= min_size:
            score = silhouette_score(X, labels)
            valid[k] = score
            print(f'  k={k}: silhouette={score:.4f}, min_cluster={sizes.min()} [valid]')
        else:
            print(f'  k={k}: min_cluster={sizes.min()} < {min_size} [skipped - too small]')

    if not valid:
        print(f'no valid k found, defaulting to k={min_k}')
        return min_k

    best_k = max(valid, key=lambda k: valid[k])
    print(f'  -> best valid k = {best_k}')
    return best_k


def run_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return labels, km


# -- dbscan helpers ----------------------------------------------------------

def find_eps(X, min_samples=5, random_state=42):
    # fit nearest neighbors to get k-distances
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)

    # sort the distances to the k-th nearest neighbor
    distances = np.sort(distances[:, -1])

    # find knee point automatically
    knee = KneeLocator(
        x=range(len(distances)),
        y=distances,
        curve='convex',
        direction='increasing'
    )

    eps = distances[knee.knee] if knee.knee is not None else 0.5
    print(f'auto eps: {eps:.4f}')
    return eps


def run_dbscan(X, eps, min_samples=5, random_state=42):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    print(f'DBSCAN: {n_clusters} clusters, {n_noise} noise points')
    return labels


# -- evaluation --------------------------------------------------------------

def evaluate_clusters(X, labels):
    # filter out noise points (-1) before computing any metric
    mask = labels != -1
    X_clean = X[mask]
    labels_clean = labels[mask]

    if len(set(labels_clean)) < 2:
        print('not enough clusters to evaluate')
        return {}

    return {
        'silhouette':  round(silhouette_score(X_clean, labels_clean), 4),
        'davies_bouldin': round(davies_bouldin_score(X_clean, labels_clean), 4),
        'calinski_harabasz': round(calinski_harabasz_score(X_clean, labels_clean), 2),
    }


# -- visualization helpers ---------------------------------------------------

def reduce_to_2d(X_scaled, random_state=42):
    pca = PCA(n_components=2, random_state=random_state)
    return pca.fit_transform(X_scaled)


# -- profiling and labeling --------------------------------------------------

def cluster_profiles(rfm_clean, labels):
    # uses unscaled rfm_clean so mean values are real numbers, not z-scores
    df = rfm_clean.copy()
    df['Cluster'] = labels
    df = df[df['Cluster'] != -1]  # exclude noise
    return df.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)


def assign_business_labels(rfm_clean, labels):
    profiles = cluster_profiles(rfm_clean, labels)

    # rank clusters relatively — no hardcoded thresholds
    profiles['R_rank'] = profiles['Recency'].rank()           # lower recency = better
    profiles['F_rank'] = profiles['Frequency'].rank(ascending=False)
    profiles['M_rank'] = profiles['Monetary'].rank(ascending=False)
    profiles['score']  = profiles['R_rank'] + profiles['F_rank'] + profiles['M_rank']
    profiles = profiles.sort_values('score')

    n = len(profiles)
    label_map = {}

    for i, cluster_id in enumerate(profiles.index):
        if n == 1:
            label_map[cluster_id] = 'Champions'
        elif n == 2:
            # with only 2 clusters use top/bottom split
            label_map[cluster_id] = 'Champions' if i == 0 else 'Churned'
        else:
            # 4 labels spread across clusters by relative rank
            pct = i / (n - 1)
            if pct <= 0.25:
                label_map[cluster_id] = 'Champions'
            elif pct <= 0.55:
                label_map[cluster_id] = 'Loyal Customers'
            elif pct <= 0.80:
                label_map[cluster_id] = 'At Risk'
            else:
                label_map[cluster_id] = 'Churned'

    label_map[-1] = 'Outlier'  # for DBSCAN noise points

    series = pd.Series(labels).map(label_map)
    return series


# -- outlier removal ---------------------------------------------------------

def remove_outliers(X_scaled, threshold=3.0):
    # flag rows where any feature is more than threshold std devs from mean
    outlier_mask = np.any(np.abs(X_scaled) > threshold, axis=1)
    X_clean = X_scaled[~outlier_mask]
    return X_clean, outlier_mask
