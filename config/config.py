# ================================================
# CHURN PROJECT - CONFIGURATION FILE
# ================================================
# This is the control panel for the entire project.
# Change settings here and everything else updates.
# ================================================

import os

# ------------------------------------------------
# PATHS
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, 'data', 'sample_data.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'models', 'features.pkl')
OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'churn_risk_report.xlsx')

# ------------------------------------------------
# DATA SETTINGS
# ------------------------------------------------
# Encoding for reading CSV
ENCODING = 'unicode_escape'

# Date column name
DATE_COLUMN = 'InvoiceDate'

# Customer ID column name
CUSTOMER_ID_COLUMN = 'Customer ID'

# Revenue column calculation
QUANTITY_COLUMN = 'Quantity'
PRICE_COLUMN = 'Price'

# ------------------------------------------------
# CHURN SETTINGS
# ------------------------------------------------
# A customer is considered churned if they haven't
# purchased in this many days
CHURN_THRESHOLD_DAYS = 90

# ------------------------------------------------
# RFM SETTINGS
# ------------------------------------------------
# Winsorization caps (percentiles)
UPPER_PERCENTILE = 0.99
LOWER_PERCENTILE = 0.01

# ------------------------------------------------
# MODEL SETTINGS
# ------------------------------------------------
# Features used for training
# NOTE: Recency is excluded intentionally
# (it was used to define churn - circular logic)
FEATURES = ['Frequency', 'Monetary', 'AOV', 'Tenure']

# XGBoost parameters
MODEL_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.1,
    'random_state': 42,
    'eval_metric': 'logloss'
}

# Train/test split ratio
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ------------------------------------------------
# RISK SEGMENTATION SETTINGS
# ------------------------------------------------
# Churn probability thresholds for risk levels
LOW_RISK_THRESHOLD = 0.33
MEDIUM_RISK_THRESHOLD = 0.66

RISK_LABELS = {
    'low': '🟢 Low Risk',
    'medium': '🟡 Medium Risk',
    'high': '🔴 High Risk'
}

RISK_ACTIONS = {
    '🟢 Low Risk': 'No action needed - consider upsell',
    '🟡 Medium Risk': 'Send soft nudge email this week',
    '🔴 High Risk': 'Send aggressive retention offer immediately'
}

# ------------------------------------------------
# CLIENT SETTINGS (change per client)
# ------------------------------------------------
CLIENT_NAME = "UK Retail Demo"
CURRENCY_SYMBOL = "£"