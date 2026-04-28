# ================================================
# RFM BUILDER
# ================================================
# Responsible for:
# 1. Calculating RFM metrics
# 2. Adding extra features (AOV, Tenure)
# 3. Defining churn target
# 4. Handling outliers
# 5. Returning model-ready DataFrame
# ================================================

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    DATE_COLUMN,
    CUSTOMER_ID_COLUMN,
    CHURN_THRESHOLD_DAYS,
    UPPER_PERCENTILE,
    LOWER_PERCENTILE
)


def get_snapshot_date(df):
    """
    Calculate snapshot date.
    Always 1 day after the last transaction in dataset.

    Args:
        df: Clean DataFrame

    Returns:
        snapshot_date (Timestamp)
    """
    snapshot_date = df[DATE_COLUMN].max() + pd.Timedelta(days=1)
    print(f"📅 Snapshot date: {snapshot_date.date()}")
    return snapshot_date


def calculate_rfm(df, snapshot_date):
    """
    Calculate core RFM metrics per customer.

    R = Recency   (days since last purchase)
    F = Frequency (number of unique invoices)
    M = Monetary  (total revenue)

    Args:
        df: Clean DataFrame
        snapshot_date: Reference date for recency calculation

    Returns:
        DataFrame with RFM columns
    """
    print("⏳ Calculating RFM metrics...")

    rfm = df.groupby(CUSTOMER_ID_COLUMN).agg({
        DATE_COLUMN: lambda x: (snapshot_date - x.max()).days,
        'Invoice': 'nunique',
        'Revenue': 'sum'
    })

    rfm.rename(columns={
        DATE_COLUMN: 'Recency',
        'Invoice': 'Frequency',
        'Revenue': 'Monetary'
    }, inplace=True)

    print(f"✅ RFM calculated for {len(rfm):,} customers")
    return rfm


def add_extra_features(rfm, df, snapshot_date):
    """
    Add behavioral features beyond basic RFM.

    AOV    = Average Order Value (Monetary / Frequency)
    Tenure = Days since first purchase

    Args:
        rfm: RFM DataFrame
        df: Clean original DataFrame (needed for first purchase date)
        snapshot_date: Reference date

    Returns:
        RFM DataFrame with extra features
    """
    print("⏳ Adding extra features...")

    # Average Order Value
    rfm['AOV'] = rfm['Monetary'] / rfm['Frequency']

    # Tenure (days since first purchase)
    first_purchase = df.groupby(CUSTOMER_ID_COLUMN)[DATE_COLUMN].min()
    rfm['Tenure'] = (snapshot_date - first_purchase).dt.days

    print("✅ Extra features added: AOV, Tenure")
    return rfm


def handle_outliers(rfm):
    """
    Winsorize outliers using percentile capping.

    Caps extreme values instead of removing them.
    Prevents outliers from skewing the model.

    Args:
        rfm: RFM DataFrame

    Returns:
        RFM DataFrame with outliers handled
    """
    print("⏳ Handling outliers...")

    # Calculate caps
    recency_cap = rfm['Recency'].quantile(UPPER_PERCENTILE)
    frequency_cap = rfm['Frequency'].quantile(UPPER_PERCENTILE)
    monetary_cap = rfm['Monetary'].quantile(UPPER_PERCENTILE)
    monetary_floor = rfm['Monetary'].quantile(LOWER_PERCENTILE)
    tenure_cap = rfm['Tenure'].quantile(UPPER_PERCENTILE)

    # Apply caps
    rfm['Recency'] = rfm['Recency'].clip(upper=recency_cap)
    rfm['Frequency'] = rfm['Frequency'].clip(upper=frequency_cap)
    rfm['Monetary'] = rfm['Monetary'].clip(
        upper=monetary_cap,
        lower=monetary_floor
    )
    rfm['Tenure'] = rfm['Tenure'].clip(upper=tenure_cap)

    print(f"✅ Outliers capped at {UPPER_PERCENTILE*100:.0f}th percentile")
    return rfm


def define_churn(rfm):
    """
    Define churn target variable.

    Churn = 1 if customer hasn't purchased
            in CHURN_THRESHOLD_DAYS days
    Churn = 0 if customer is still active

    Args:
        rfm: RFM DataFrame

    Returns:
        RFM DataFrame with Churn column
    """
    print(f"⏳ Defining churn (threshold: {CHURN_THRESHOLD_DAYS} days)...")

    rfm['Churn'] = (rfm['Recency'] > CHURN_THRESHOLD_DAYS).astype(int)

    churn_rate = rfm['Churn'].mean() * 100
    print(f"✅ Churn defined")
    print(f"   Active customers:  {(rfm['Churn']==0).sum():,}")
    print(f"   Churned customers: {(rfm['Churn']==1).sum():,}")
    print(f"   Churn rate:        {churn_rate:.1f}%")

    return rfm


def build_rfm(df):
    """
    Master function.
    Runs entire RFM pipeline in one call.

    Steps:
    1. Get snapshot date
    2. Calculate RFM
    3. Add extra features
    4. Handle outliers
    5. Define churn

    Args:
        df: Clean DataFrame

    Returns:
        Model-ready RFM DataFrame
    """
    print("\n" + "="*40)
    print("BUILDING RFM TABLE")
    print("="*40)

    snapshot_date = get_snapshot_date(df)
    rfm = calculate_rfm(df, snapshot_date)
    rfm = add_extra_features(rfm, df, snapshot_date)
    rfm = handle_outliers(rfm)
    rfm = define_churn(rfm)

    print("\n✅ RFM table ready!")
    print(f"   Shape: {rfm.shape}")
    print(f"   Columns: {rfm.columns.tolist()}")

    return rfm