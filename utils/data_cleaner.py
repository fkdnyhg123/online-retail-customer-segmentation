"""
Data Cleaning Module
Handles all data quality issues in the Online Retail II dataset.
Independent module - can be called from Streamlit app or standalone scripts.

Actual column names in the Excel file:
  Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
"""
import pandas as pd
import numpy as np

# Column name constants for consistency
COL_INVOICE = 'Invoice'
COL_STOCK = 'StockCode'
COL_DESC = 'Description'
COL_QTY = 'Quantity'
COL_DATE = 'InvoiceDate'
COL_PRICE = 'Price'
COL_CUSTOMER = 'Customer ID'
COL_COUNTRY = 'Country'

SPECIAL_CODES = ['POST', 'BANK CHARGES', 'C2', 'D', 'DOT', 'M', 'PADS', 'CRUK']


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw Excel dataset."""
    df = pd.read_excel(filepath, engine='openpyxl')
    return df


def get_data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a comprehensive data quality report before cleaning."""
    report = {
        'total_rows': len(df),
        'cancelled_orders': df[df[COL_INVOICE].astype(str).str.startswith('C')].shape[0],
        'missing_customer_id': int(df[COL_CUSTOMER].isnull().sum()),
        'negative_quantity': df[df[COL_QTY] < 0].shape[0],
        'zero_neg_price': df[df[COL_PRICE] <= 0].shape[0],
        'exact_duplicates': int(df.duplicated().sum()),
        'special_stockcodes': df[df[COL_STOCK].isin(SPECIAL_CODES)].shape[0],
        'missing_description': int(df[COL_DESC].isnull().sum()),
    }
    for k, v in list(report.items()):
        if k != 'total_rows':
            report[f'{k}_pct'] = round(v / report['total_rows'] * 100, 2)
    return report


def clean_data(df: pd.DataFrame, drop_no_customer: bool = True) -> pd.DataFrame:
    """
    Full data cleaning pipeline.

    Steps:
    1. Remove cancelled orders (Invoice starts with 'C')
    2. Remove non-product stock codes
    3. Remove rows with missing Description
    4. Remove negative Quantity (returns)
    5. Remove zero/negative Price
    6. Remove exact duplicates
    7. Drop rows with missing Customer ID (for customer-level analysis)
    8. Convert InvoiceDate to datetime
    9. Add Revenue column

    Parameters:
        df: raw DataFrame
        drop_no_customer: if True, drop rows without Customer ID
                         (set False for product-level analysis)

    Returns:
        Cleaned DataFrame
    """
    cleaned = df.copy()

    # 1. Remove cancelled orders
    cleaned = cleaned[~cleaned[COL_INVOICE].astype(str).str.startswith('C')]

    # 2. Remove non-product stock codes
    cleaned = cleaned[~cleaned[COL_STOCK].isin(SPECIAL_CODES)]

    # 3. Remove missing Description
    cleaned = cleaned.dropna(subset=[COL_DESC])

    # 4. Remove negative Quantity (these are returns/cancellations)
    cleaned = cleaned[cleaned[COL_QTY] > 0]

    # 5. Remove zero/negative Price (gifts, adjustments)
    cleaned = cleaned[cleaned[COL_PRICE] > 0]

    # 6. Remove exact duplicates
    cleaned = cleaned.drop_duplicates()

    # 7. Drop rows without Customer ID (needed for RFM/clustering)
    if drop_no_customer:
        cleaned = cleaned.dropna(subset=[COL_CUSTOMER])
        cleaned[COL_CUSTOMER] = cleaned[COL_CUSTOMER].astype(int)

    # 8. Ensure InvoiceDate is datetime
    cleaned[COL_DATE] = pd.to_datetime(cleaned[COL_DATE])

    # 9. Add Revenue column
    cleaned['Revenue'] = cleaned[COL_QTY] * cleaned[COL_PRICE]

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def get_cleaning_summary(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """Summary of what was removed during cleaning."""
    return {
        'raw_rows': len(raw_df),
        'cleaned_rows': len(cleaned_df),
        'removed_rows': len(raw_df) - len(cleaned_df),
        'removal_pct': round((len(raw_df) - len(cleaned_df)) / len(raw_df) * 100, 2),
        'unique_customers': cleaned_df[COL_CUSTOMER].nunique() if COL_CUSTOMER in cleaned_df else 'N/A',
        'unique_products': cleaned_df[COL_STOCK].nunique(),
        'date_range_start': str(cleaned_df[COL_DATE].min().date()),
        'date_range_end': str(cleaned_df[COL_DATE].max().date()),
    }
