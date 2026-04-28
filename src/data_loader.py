# ================================================
# DATA LOADER
# ================================================
# Responsible for:
# 1. Loading raw CSV data
# 2. Cleaning it
# 3. Returning clean DataFrame
# ================================================

import pandas as pd
import sys
import os

# This line lets us import from config folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    DATA_PATH,
    ENCODING,
    DATE_COLUMN,
    CUSTOMER_ID_COLUMN,
    QUANTITY_COLUMN,
    PRICE_COLUMN
)


def load_data(path=None):
    """
    Load raw CSV data from the data folder.

    Args:
        path: Optional custom path. Uses config path if not provided.

    Returns:
        Raw DataFrame
    """
    if path is None:
        path = DATA_PATH

    print(f"⏳ Loading data from: {path}")
    df = pd.read_csv(path, encoding=ENCODING)
    print(f"✅ Loaded {len(df):,} rows and {df.shape[1]} columns")

    return df


def clean_data(df):
    """
    Clean raw retail data.

    Steps:
    1. Remove missing Customer IDs (guest checkouts)
    2. Remove missing Descriptions
    3. Remove cancelled orders (Invoice starts with 'C')
    4. Remove negative/zero quantities
    5. Remove negative/zero prices
    6. Convert dates to datetime
    7. Fix Customer ID type
    8. Add Revenue column

    Args:
        df: Raw DataFrame

    Returns:
        Clean DataFrame
    """
    print("\n⏳ Starting data cleaning...")
    initial_rows = len(df)

    # ----------------------------------------
    # Step 1: Remove missing Customer IDs
    # ----------------------------------------
    df = df.dropna(subset=[CUSTOMER_ID_COLUMN])
    print(f"   After removing missing Customer IDs: {len(df):,} rows")

    # ----------------------------------------
    # Step 2: Remove missing Descriptions
    # ----------------------------------------
    df = df.dropna(subset=['Description'])
    print(f"   After removing missing Descriptions: {len(df):,} rows")

    # ----------------------------------------
    # Step 3: Remove cancelled orders
    # ----------------------------------------
    df = df[~df['Invoice'].str.startswith('C')]
    print(f"   After removing cancellations:        {len(df):,} rows")

    # ----------------------------------------
    # Step 4: Remove negative/zero quantities
    # ----------------------------------------
    df = df[df[QUANTITY_COLUMN] > 0]
    print(f"   After removing bad quantities:       {len(df):,} rows")

    # ----------------------------------------
    # Step 5: Remove negative/zero prices
    # ----------------------------------------
    df = df[df[PRICE_COLUMN] > 0]
    print(f"   After removing bad prices:           {len(df):,} rows")

    # ----------------------------------------
    # Step 6: Convert dates to datetime
    # ----------------------------------------
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    print(f"   Dates converted to datetime ✅")

    # ----------------------------------------
    # Step 7: Fix Customer ID type
    # ----------------------------------------
    df[CUSTOMER_ID_COLUMN] = df[CUSTOMER_ID_COLUMN].astype(int).astype(str)
    print(f"   Customer ID converted to string ✅")

    # ----------------------------------------
    # Step 8: Add Revenue column
    # ----------------------------------------
    df['Revenue'] = df[QUANTITY_COLUMN] * df[PRICE_COLUMN]
    print(f"   Revenue column created ✅")

    # ----------------------------------------
    # Summary
    # ----------------------------------------
    removed = initial_rows - len(df)
    print(f"\n✅ Cleaning complete!")
    print(f"   Started with: {initial_rows:,} rows")
    print(f"   Removed:      {removed:,} rows")
    print(f"   Final:        {len(df):,} rows")
    print(f"   Unique customers: {df[CUSTOMER_ID_COLUMN].nunique():,}")

    return df


def load_and_clean(path=None):
    """
    Convenience function.
    Loads AND cleans data in one call.

    Args:
        path: Optional custom path

    Returns:
        Clean DataFrame
    """
    df = load_data(path)
    df = clean_data(df)
    return df