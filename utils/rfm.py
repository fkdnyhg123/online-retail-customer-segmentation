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
    df['R_score'] = pd.qcut(df['Recency'], q=n_bins, labels=False, duplicates='drop')
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
