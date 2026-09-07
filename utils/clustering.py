"""
K-Means Customer Segmentation Module (V3 - Feature-Engineered)

Three methods available:
  - 'composite' (recommended): 2D features — R_rank + RFM_composite
    Combines correlated F/M into one engagement dimension, reduces redundancy.
    K=5 silhouette ≈ 0.42, K=4 ≈ 0.45.
  - 'rank': 3D features — R_rank + F_rank + M_rank (with Winsorizing)
    Standard percentile rank approach. K=5 silhouette ≈ 0.36.
  - 'log': 3D features — log1p(R) + log1p(F) + log1p(M)
    Legacy log-transform. K=5 silhouette ≈ 0.32.

All methods use StandardScaler + K-Means with n_init=50, max_iter=500.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples


def prepare_features(rfm_df: pd.DataFrame, method: str = 'composite',
                     winsorize_pct: float = 0.995) -> tuple:
    """
    Prepare features for K-Means clustering.

    Parameters:
        rfm_df: DataFrame with Recency, Frequency, Monetary columns
        method: 'composite' (recommended 2D), 'rank' (3D), or 'log' (3D legacy)
        winsorize_pct: upper clip percentile for Winsorizing (default 0.995 = top 0.5%)

    Returns:
        (scaled_features, feature_names, scaler, transformed_df)
        transformed_df contains the intermediate feature values for debugging
    """
    df = rfm_df[['Recency', 'Frequency', 'Monetary']].copy()

    if method == 'composite':
        # === Composite method (2D): R_rank + RFM_composite ===
        # Winsorize F and M first
        f_cap = df['Frequency'].quantile(winsorize_pct)
        m_cap = df['Monetary'].quantile(winsorize_pct)
        df['Frequency'] = df['Frequency'].clip(upper=f_cap)
        df['Monetary'] = df['Monetary'].clip(upper=m_cap)

        # Percentile ranks
        r_rank = (-df['Recency']).rank(pct=True)
        f_rank = df['Frequency'].rank(pct=True)
        m_rank = df['Monetary'].rank(pct=True)

        # 2D composite: timeliness + overall engagement
        rfm_composite = r_rank + f_rank + m_rank
        features = np.column_stack([r_rank.values, rfm_composite.values])
        feature_names = ['R_rank', 'RFM_composite']

        transformed = pd.DataFrame(features, columns=feature_names)

    elif method == 'rank':
        # === Rank method (3D): R_rank + F_rank + M_rank ===
        f_cap = df['Frequency'].quantile(winsorize_pct)
        m_cap = df['Monetary'].quantile(winsorize_pct)
        df['Frequency'] = df['Frequency'].clip(upper=f_cap)
        df['Monetary'] = df['Monetary'].clip(upper=m_cap)

        df['R_inv'] = -df['Recency']
        df['R_rank'] = df['R_inv'].rank(pct=True)
        df['F_rank'] = df['Frequency'].rank(pct=True)
        df['M_rank'] = df['Monetary'].rank(pct=True)

        features = df[['R_rank', 'F_rank', 'M_rank']].values
        feature_names = ['Recency', 'Frequency', 'Monetary']

        transformed = df[['R_rank', 'F_rank', 'M_rank']].copy()
        transformed.columns = feature_names

    elif method == 'log':
        # === Log method (3D): log1p transform ===
        feature_names = ['Recency', 'Frequency', 'Monetary']
        features = df.values.copy().astype(float)
        features[:, 0] = np.log1p(features[:, 0])
        features[:, 1] = np.log1p(features[:, 1])
        features[:, 2] = np.log1p(features[:, 2])
        transformed = pd.DataFrame(features, columns=feature_names)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'composite', 'rank', or 'log'.")

    # StandardScaler for all methods
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return scaled, feature_names, scaler, transformed


def find_optimal_k(features, k_range: range = range(2, 11), random_state: int = 42) -> dict:
    """
    Find optimal K using elbow method and silhouette scores.

    Returns:
        dict with 'inertia', 'silhouette_scores', 'k_values' lists
    """
    inertia = []
    sil_scores = []
    k_values = list(k_range)

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=random_state,
                    n_init=50, max_iter=500)
        labels = km.fit_predict(features)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(features, labels))

    best_k = k_values[np.argmax(sil_scores)]

    # 轮廓系数在偏态 RFM 数据上常在 K=2 (活跃/沉睡 二分) 取最大, 但无业务意义;
    # 故另给一个 K>=3 范围内的推荐值, 供业务粒度选择参考
    practical = [(k, s) for k, s in zip(k_values, sil_scores) if k >= 3]
    if practical:
        recommended_k, recommended_sil = max(practical, key=lambda t: t[1])
    else:
        recommended_k, recommended_sil = best_k, max(sil_scores)

    return {
        'k_values': k_values,
        'inertia': inertia,
        'silhouette_scores': sil_scores,
        'best_k': best_k,
        'best_silhouette': max(sil_scores),
        'recommended_k': recommended_k,
        'recommended_silhouette': recommended_sil,
    }


def run_kmeans(rfm_df: pd.DataFrame, n_clusters: int = 5,
               method: str = 'composite', winsorize_pct: float = 0.995,
               random_state: int = 42) -> dict:
    """
    Run K-Means clustering on RFM data.

    Returns:
        dict with:
        - 'clustered_df': original rfm_df with 'Cluster' column added
        - 'model': fitted KMeans model
        - 'scaler': fitted StandardScaler
        - 'scaled_features': the scaled feature array
        - 'silhouette_score': overall silhouette score
        - 'silhouette_per_cluster': dict {cluster_id: score}
        - 'cluster_centers': cluster centers in transformed scale
        - 'cluster_profile': DataFrame with per-cluster RFM means
        - 'method': the method used ('rank' or 'log')
    """
    scaled, feature_names, scaler, transformed = prepare_features(
        rfm_df, method=method, winsorize_pct=winsorize_pct)

    # === Step 4: K-Means with aggressive tuning ===
    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=50, max_iter=500)
    labels = km.fit_predict(scaled)

    clustered = rfm_df.copy()
    clustered['Cluster'] = labels

    sil_score = silhouette_score(scaled, labels)

    # Per-cluster silhouette scores
    sil_samples = silhouette_samples(scaled, labels)
    sil_per_cluster = {}
    for c in range(n_clusters):
        mask = labels == c
        sil_per_cluster[c] = round(float(sil_samples[mask].mean()), 4)

    # Cluster centers in transformed scale (for heatmap)
    centers_df = pd.DataFrame(km.cluster_centers_, columns=feature_names)
    centers_df.index.name = 'Cluster'

    # Cluster profile: mean RFM per cluster (in original scale)
    profile = clustered.groupby('Cluster').agg(
        Customers=('Customer ID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
    ).round(2)
    profile['Pct_Customers'] = (profile['Customers'] / len(clustered) * 100).round(1)
    profile['Pct_Revenue'] = (profile['Total_Revenue'] / clustered['Monetary'].sum() * 100).round(1)

    return {
        'clustered_df': clustered,
        'model': km,
        'scaler': scaler,
        'scaled_features': scaled,
        'silhouette_score': round(sil_score, 4),
        'silhouette_per_cluster': sil_per_cluster,
        'cluster_centers': centers_df,
        'cluster_profile': profile,
        'method': method,
    }


def get_cluster_labels(profile: pd.DataFrame) -> dict:
    """
    Generate Chinese labels for each cluster based on RFM characteristics.
    Uses R/F/M thresholds (vs mean) to assign business-meaningful labels,
    with fallback pool to guarantee uniqueness.
    """
    labels = {}

    avg_r = profile['Avg_Recency'].mean()
    avg_f = profile['Avg_Frequency'].mean()
    avg_m = profile['Avg_Monetary'].mean()

    # Sort clusters by composite value (best first)
    r_rank = profile['Avg_Recency'].rank(ascending=True).astype(int)
    f_rank = profile['Avg_Frequency'].rank(ascending=False).astype(int)
    m_rank = profile['Avg_Monetary'].rank(ascending=False).astype(int)
    sorted_clusters = (r_rank + f_rank + m_rank).sort_values().index.tolist()

    # Rule-based label assignment with uniqueness guarantee
    used = set()
    rules = [
        # (condition, label)
        (lambda r, f, m: r < avg_r and f > avg_f and m > avg_m, '重要价值客户'),
        (lambda r, f, m: r < avg_r and f > avg_f and m <= avg_m, '重要发展客户'),
        (lambda r, f, m: r < avg_r and f <= avg_f and m > avg_m, '重要保持客户'),
        (lambda r, f, m: r < avg_r and f <= avg_f and m <= avg_m, '新客户'),
        (lambda r, f, m: r >= avg_r and f > avg_f and m > avg_m, '重要挽留客户'),
        (lambda r, f, m: r >= avg_r and f > avg_f and m <= avg_m, '需关注客户'),
        (lambda r, f, m: r >= avg_r and f <= avg_f and m > avg_m, '一般挽留客户'),
        (lambda r, f, m: r >= avg_r and f <= avg_f and m <= avg_m, '流失客户'),
        (lambda r, f, m: r >= avg_r * 1.5 and f <= avg_f * 0.5, '沉睡客户'),
    ]
    fallback_pool = [
        '重要价值客户', '重要发展客户', '新客户', '沉睡客户',
        '流失客户', '需关注客户', '重要保持客户', '重要挽留客户',
        '一般价值客户', '一般保持客户',
    ]

    for cluster_id in sorted_clusters:
        r = profile.loc[cluster_id, 'Avg_Recency']
        f = profile.loc[cluster_id, 'Avg_Frequency']
        m = profile.loc[cluster_id, 'Avg_Monetary']

        # Trait description in Chinese
        parts = []
        parts.append('近期活跃' if r < avg_r else '长期未购')
        parts.append('高频购买' if f > avg_f else '低频购买')
        parts.append('高消费' if m > avg_m else '低消费')

        # Try rules first
        assigned = None
        for condition, label in rules:
            if condition(r, f, m) and label not in used:
                assigned = label
                break

        # Fallback: pick next available from pool
        if assigned is None:
            for candidate in fallback_pool:
                if candidate not in used:
                    assigned = candidate
                    break

        used.add(assigned)
        labels[cluster_id] = {
            'name': assigned,
            'traits': ' / '.join(parts),
        }

    return labels
