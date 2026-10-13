"""
Week 1 - Data Exploration
Course: Big Data Analysis Comprehensive Practice
Dataset: UCI Online Retail II (UK online retail store)
Objective: Load data, basic exploration, identify data quality issues
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

# ============================================================
# 1. Load Data
# ============================================================
DATA_PATH = "online_retail_II(1).xlsx"
print("=" * 60)
print("Loading dataset: Online Retail II")
print("=" * 60)

df = pd.read_excel(DATA_PATH, engine='openpyxl')
print(f"Data loaded successfully! Shape: {df.shape}")
print(f"  - Rows: {df.shape[0]:,}")
print(f"  - Columns: {df.shape[1]}")
print(f"  - Column names: {list(df.columns)}")

# ============================================================
# 2. Basic Information: df.info()
# ============================================================
print("\n" + "=" * 60)
print("df.info()")
print("=" * 60)
df.info()

# ============================================================
# 3. First 10 Rows: df.head(10)
# ============================================================
print("\n" + "=" * 60)
print("df.head(10)")
print("=" * 60)
print(df.head(10).to_string())

# ============================================================
# 4. Statistical Summary: df.describe()
# ============================================================
print("\n" + "=" * 60)
print("df.describe()")
print("=" * 60)
print(df.describe().to_string())

# ============================================================
# 5. Missing Values: df.isnull().sum()
# ============================================================
print("\n" + "=" * 60)
print("Missing values: df.isnull().sum()")
print("=" * 60)
null_counts = df.isnull().sum()
null_pct = (df.isnull().sum() / len(df) * 100).round(2)
null_df = pd.DataFrame({'Count': null_counts, 'Percentage(%)': null_pct})
print(null_df.to_string())
print(f"\nTotal rows with any missing value: {df.isnull().any(axis=1).sum():,}")

# ============================================================
# 6. Data Quality Issues Identification
# ============================================================
print("\n" + "=" * 60)
print("Data Quality Issues")
print("=" * 60)

# 6.1 Cancelled orders (Invoice starts with 'C')
cancelled = df[df['Invoice'].astype(str).str.startswith('C')]
print(f"\n[Issue 1] Cancelled orders (Invoice starts with 'C'): {len(cancelled):,} rows ({len(cancelled)/len(df)*100:.2f}%)")

# 6.2 Missing Customer ID
missing_cid = df['Customer ID'].isnull().sum()
print(f"[Issue 2] Missing Customer ID: {missing_cid:,} rows ({missing_cid/len(df)*100:.2f}%)")

# 6.3 Negative Quantity
neg_qty = df[df['Quantity'] < 0]
print(f"[Issue 3] Negative Quantity: {len(neg_qty):,} rows ({len(neg_qty)/len(df)*100:.2f}%)")

# 6.4 Zero or Negative Price
zero_price = df[df['Price'] <= 0]
print(f"[Issue 4] Zero/Negative Price: {len(zero_price):,} rows ({len(zero_price)/len(df)*100:.2f}%)")

# 6.5 Non-product StockCodes
special_codes = ['POST', 'BANK CHARGES', 'C2', 'D', 'DOT', 'M', 'PADS', 'CRUK']
special_codes_found = df[df['StockCode'].isin(special_codes)]
print(f"[Issue 5] Non-product StockCodes (POST, BANK CHARGES, etc.): {len(special_codes_found):,} rows")
print(f"         Found codes: {special_codes_found['StockCode'].unique().tolist()}")

# 6.6 Duplicate rows
duplicates = df.duplicated().sum()
print(f"[Issue 6] Exact duplicate rows: {duplicates:,} rows ({duplicates/len(df)*100:.2f}%)")

# 6.7 Missing Description
missing_desc = df['Description'].isnull().sum()
print(f"[Issue 7] Missing Description: {missing_desc:,} rows ({missing_desc/len(df)*100:.2f}%)")

# 6.8 Country distribution
print(f"\n[Info] Countries: {df['Country'].nunique()}")
print("Top 10 countries:")
print(df['Country'].value_counts().head(10).to_string())

# ============================================================
# 7. Preliminary Observations (3 required by homework)
# ============================================================
print("\n" + "=" * 60)
print("Preliminary Observations (Homework P1)")
print("=" * 60)
print("""
Observation 1 - Data Scale & Completeness:
  The dataset contains 525,461 transaction records with 8 fields.
  Customer ID has ~20.5% missing values (107,927 rows), which will
  significantly impact customer-level analysis like RFM segmentation.
  Description has 2,928 missing values (~0.56%).

Observation 2 - Data Quality Issues:
  Multiple quality problems exist: ~1.9% cancelled orders (Invoice
  starting with 'C'), 12,326 negative quantities, 3,690 zero/negative
  prices, 2,769 non-product stock codes (POST, BANK CHARGES etc.),
  and 6,865 exact duplicate rows. These need careful cleaning before
  any analysis.

Observation 3 - Geographic Concentration:
  The vast majority of orders come from United Kingdom, with a small
  number from Germany, France, and other European countries. This
  geographic imbalance should be considered in analysis.
""")

print("=" * 60)
print("Week 1 exploration complete!")
print("=" * 60)
