"""
RFM (Recency, Frequency, Monetary) Calculation Module
Computes RFM features from cleaned transaction data for customer segmentation.
"""
import pandas as pd
import numpy as np


def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Calculate RFM metrics for each customer.

    Parameters:
        df: cleaned transaction DataFrame (must have CustomerID, InvoiceDate, Revenue)
        snapshot_date: reference date for Recency calculation.
                       If None, uses max(InvoiceDate) + 1 day.

    Returns:
        DataFrame with one row per customer, columns:
        CustomerID, Recency, Frequency, Monetary
    """
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('Customer ID').agg(
        Recency=('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency=('Invoice', 'nunique'),
        Monetary=('Revenue', 'sum')
    ).reset_index()

    # Round Monetary to 2 decimals
    rfm['Monetary'] = rfm['Monetary'].round(2)

    return rfm


def add_rfm_scores(rfm_df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Add R_score, F_score, M_score (1-5) using quantile-based binning.

    Recency is inverted: lower recency (more recent) = higher score.

    Parameters:
        rfm_df: DataFrame from calculate_rfm()
        n_bins: number of bins (default 5)

    Returns:
        DataFrame with added R_score, F_score, M_score, RFM_score columns
    """
    df = rfm_df.copy()

    # R_score: lower recency = higher score (invert)
    # 与 F/M 一致先做 rank(method='first'): Recency 是整数天数、重复值密集,
    # 直接 qcut 会让 bin 边界落在重复值上并被 duplicates='drop' 合并, 拿不到均匀五档,
    # 而下游客户分群规则依赖 R_score >= 4 / <= 2 这类阈值, 档位不均会导致分群偏移
    df['R_score'] = pd.qcut(df['Recency'].rank(method='first'), q=n_bins, labels=False, duplicates='drop')
    df['R_score'] = n_bins - df['R_score']  # Invert

    # F_score: higher frequency = higher score
    df['F_score'] = pd.qcut(df['Frequency'].rank(method='first'), q=n_bins, labels=False, duplicates='drop') + 1

    # M_score: higher monetary = higher score
    df['M_score'] = pd.qcut(df['Monetary'].rank(method='first'), q=n_bins, labels=False, duplicates='drop') + 1

    # Combined RFM score
    df['RFM_score'] = df['R_score'] * 100 + df['F_score'] * 10 + df['M_score']

    return df


def get_rfm_stats(rfm_df: pd.DataFrame) -> dict:
    """Get summary statistics of RFM data."""
    return {
        'num_customers': len(rfm_df),
        'avg_recency': round(rfm_df['Recency'].mean(), 1),
        'median_recency': round(rfm_df['Recency'].median(), 1),
        'avg_frequency': round(rfm_df['Frequency'].mean(), 1),
        'median_frequency': round(rfm_df['Frequency'].median(), 1),
        'avg_monetary': round(rfm_df['Monetary'].mean(), 2),
        'median_monetary': round(rfm_df['Monetary'].median(), 2),
        'total_revenue': round(rfm_df['Monetary'].sum(), 2),
    }


# 扩展特征的中文名与说明, 供页面与文档共用
EXTENDED_FEATURE_INFO = {
    'AOV': {
        'name': '平均客单价',
        'en': 'AOV (Average Order Value)',
        'formula': '总收入 / 发票数',
        'desc': '每次消费的平均金额, 衡量单次消费强度',
    },
    'N_products': {
        'name': '品类广度',
        'en': 'N_products',
        'formula': '不重复商品编码数',
        'desc': '买过多少种不同商品, 衡量消费宽度',
    },
}


def calculate_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 RFM 之外的两个客户级扩展特征。

    选型依据 (在完整清洗数据上实测, 与 R/F/M 的 Pearson 相关系数):
      - AOV (平均客单价): 与 Recency -0.05 / Frequency 0.10 / Monetary 0.33,
        是候选特征里最独立的"单次消费强度"维度, 能把"高频小单"与"低频大单"分开
      - N_products (品类广度): 与 Frequency 0.66 属中等相关但语义正交
        (F 是"来几次", 广度是"每次买得宽不宽"), 二者彼此相关仅 0.16
    被否决的候选: 活跃月数(与 F 相关 0.70)、退货次数(0.73 且 60% 为 0)、
    客户生命周期跨度(34.6% 结构性零值)、单次平均件数(偏度 45.7)、
    退货率(定义失真, 退货发票数可超过正向发票数)。

    Parameters:
        df: 清洗后交易数据 (需含 Customer ID, StockCode, Invoice, Revenue)

    Returns:
        DataFrame, 每客户一行: Customer ID, AOV, N_products
    """
    agg = df.groupby('Customer ID').agg(
        N_products=('StockCode', 'nunique'),
        _revenue=('Revenue', 'sum'),
        _invoices=('Invoice', 'nunique'),
    )
    # AOV 与 Monetary/Frequency 同源 (Monetary=客户收入, Frequency=发票数), 但这里用
    # 未取整的收入相除再取整, 故与「先给 Monetary 取两位再相除」相比可能有 0.01 的差异
    # (全量数据上 4295 人中 28 人差 0.01), 属更精确的一侧
    agg['AOV'] = (agg.pop('_revenue') / agg.pop('_invoices')).round(2)

    return agg.reset_index()
