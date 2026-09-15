"""
K-Means Customer Segmentation Module (V3 - Feature-Engineered)

Four methods available (K=4 时的轮廓系数, 仅 RFM / 加扩展特征):
  - 'composite': 2D features — R_rank + RFM_composite
    Combines correlated F/M into one engagement dimension.  0.4486 / 0.3034
  - 'rank': 3D features — R_rank + F_rank + M_rank (with Winsorizing)
    Standard percentile rank approach.                     0.3898 / 0.2857
  - 'log': 3D features — log1p(R) + log1p(F) + log1p(M)
    Legacy log-transform.                                  0.3326 / 0.2750
  - 'pca': 2D features — 对全部输入特征 (rank 口径) 标准化后取前 2 个主成分
    两个主成分**天然正交** (composite 的两维相关约 0.6, 并不正交), 且同时承载全部
    输入特征的信息 (前两主成分解释 82.1% 方差, 仅 RFM 时 94.3%)。
    加入扩展特征后该方法的轮廓系数最高:                0.4369 / 0.3967

All methods use StandardScaler + K-Means with n_init=10, max_iter=500.

n_init 取值说明: sklearn 默认为 10。曾用 50, 实测在这份 4,295 x 2~5 维、已标准化的平滑
特征空间上, n_init=10 与 50 给出的簇划分几乎一致 (K 扫描的轮廓系数仅第 4 位小数有
±0.003 的差异, 推荐 K 不变), 但 K 扫描耗时从 4.5s 降到约 2.6s —— 故采用 10。

可选的 extra_features (如 ['AOV', 'N_products']) 会以百分位排名追加到上述维度之后,
只在 R/F/M 之外增加输入维度, 不改动三种方法原有的 R/F/M 处理路径。
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples


def prepare_features(rfm_df: pd.DataFrame, method: str = 'composite',
                     winsorize_pct: float = 0.995,
                     extra_features: list = None,
                     random_state: int = 42) -> tuple:
    """
    Prepare features for K-Means clustering.

    Parameters:
        rfm_df: DataFrame with Recency, Frequency, Monetary columns
        method: 'composite' (recommended 2D), 'rank' (3D), 'log' (3D legacy),
                或 'pca' (对全部输入特征做正交降维后取前 2 个主成分)
        winsorize_pct: upper clip percentile for Winsorizing (default 0.995 = top 0.5%)
        extra_features: 额外纳入聚类的非 RFM 客户特征列名 (如 ['AOV', 'N_products']),
                        须已存在于 rfm_df。它们只在 R/F/M 之外**增加维度**, 不改动
                        三种方法原有的 R/F/M 处理路径; 统一走百分位排名 —— rank 只看
                        次序、天然抗离群 (AOV 偏度 11.6 也无需再截断), 且无量纲,
                        不会因量级差异压过其他维度
        random_state: 随机种子 (仅 PCA 方法用到; 本例数据量下 PCA 走完全 SVD, 结果确定)

    Returns:
        (scaled_features, feature_names, scaler, transformed_df, pca_model)
        transformed_df contains the intermediate feature values for debugging;
        pca_model 仅 method='pca' 时为拟合好的 PCA 对象, 其余方法为 None
    """
    extra_features = list(extra_features or [])
    _missing = [c for c in extra_features if c not in rfm_df.columns]
    if _missing:
        raise ValueError(f"extra_features 不存在于 rfm_df: {_missing}")

    df = rfm_df[['Recency', 'Frequency', 'Monetary'] + extra_features].copy()

    # 额外特征统一转百分位排名, 三种方法共用同一套列名 {原列名}_rank
    extra_cols = []
    for col in extra_features:
        df[f'{col}_rank'] = df[col].rank(pct=True)
        extra_cols.append(f'{col}_rank')

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
        features = np.column_stack([r_rank.values, rfm_composite.values]
                                   + [df[c].values for c in extra_cols])
        feature_names = ['R_rank', 'RFM_composite'] + extra_cols

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

        features = df[['R_rank', 'F_rank', 'M_rank'] + extra_cols].values
        feature_names = ['Recency', 'Frequency', 'Monetary'] + extra_cols

        transformed = df[['R_rank', 'F_rank', 'M_rank'] + extra_cols].copy()
        transformed.columns = feature_names

    elif method == 'log':
        # === Log method (3D): log1p transform ===
        feature_names = ['Recency', 'Frequency', 'Monetary'] + extra_cols
        base = np.log1p(df[['Recency', 'Frequency', 'Monetary']].values.astype(float))
        features = np.column_stack([base] + [df[c].values for c in extra_cols])
        transformed = pd.DataFrame(features, columns=feature_names)

    elif method == 'pca':
        # === PCA method (2D): 对全部输入特征做正交降维 ===
        # 输入取 rank 口径 (R/F/M 排名 + 扩展特征排名), 再取标准化的前 2 个主成分作为聚类空间。
        # 与 composite 的区别: composite 的两维共享 R_rank、相关约 0.6 并不正交; PCA 的两个主成分
        # 天然正交, 且能同时承载全部输入特征的信息 (实测前两主成分解释 82.1% 方差, 仅 RFM 时为 94.3%)。
        # 刻意不在 PCA 之后再标准化: 主成分按方差降序排列, PC1 承载的信息本就最多,
        # 强行拉平会丢掉这个权重 (实测不拉平的轮廓系数更高)。
        df['R_inv'] = -df['Recency']
        df['R_rank'] = df['R_inv'].rank(pct=True)
        df['F_rank'] = df['Frequency'].rank(pct=True)
        df['M_rank'] = df['Monetary'].rank(pct=True)

        scaler = StandardScaler()
        base_scaled = scaler.fit_transform(
            df[['R_rank', 'F_rank', 'M_rank'] + extra_cols].values)
        pca_model = PCA(n_components=2, random_state=random_state)
        pcs = pca_model.fit_transform(base_scaled)

        feature_names = ['PC1', 'PC2']
        return (pcs, feature_names, scaler,
                pd.DataFrame(pcs, columns=feature_names), pca_model)

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'composite', 'rank', 'log', or 'pca'.")

    # StandardScaler for all methods
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    return scaled, feature_names, scaler, transformed, None


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
                    n_init=10, max_iter=500)
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
               random_state: int = 42, extra_features: list = None) -> dict:
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
    scaled, feature_names, scaler, transformed, pca_model = prepare_features(
        rfm_df, method=method, winsorize_pct=winsorize_pct,
        extra_features=extra_features)

    # === Step 4: K-Means ===
    km = KMeans(n_clusters=n_clusters, random_state=random_state,
                n_init=10, max_iter=500)
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
    profile_aggs = dict(
        Customers=('Customer ID', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
    )
    # 扩展特征一并聚合进画像 (如 Avg_AOV / Avg_N_products), 让新维度落到业务解释里
    for _col in (extra_features or []):
        if _col in clustered.columns:
            profile_aggs[f'Avg_{_col}'] = (_col, 'mean')
    profile = clustered.groupby('Cluster').agg(**profile_aggs).round(2)
    profile['Pct_Customers'] = (profile['Customers'] / len(clustered) * 100).round(1)
    profile['Pct_Revenue'] = (profile['Total_Revenue'] / clustered['Monetary'].sum() * 100).round(1)

    return {
        'clustered_df': clustered,
        'model': km,
        'scaler': scaler,
        'pca': pca_model,
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
    # fallback_pool 必须提供 >= K 上限 (app.py 侧边栏 K 最大 15) 个唯一名称,
    # 且需覆盖 rules 里的全部标签 —— 否则 K 偏大时 used 耗尽, 标签会退化成 None
    fallback_pool = [
        '重要价值客户', '重要发展客户', '新客户', '沉睡客户',
        '流失客户', '需关注客户', '重要保持客户', '重要挽留客户',
        '一般价值客户', '一般保持客户', '一般挽留客户', '一般发展客户',
        '低价值客户', '待唤醒客户', '边缘客户', '长尾客户',
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

        # Last resort: pool 也耗尽时生成唯一序号名, 保证永不返回 None
        if assigned is None:
            _n = 1
            while f'细分客群 {_n}' in used:
                _n += 1
            assigned = f'细分客群 {_n}'

        used.add(assigned)
        labels[cluster_id] = {
            'name': assigned,
            'traits': ' / '.join(parts),
        }

    return labels
