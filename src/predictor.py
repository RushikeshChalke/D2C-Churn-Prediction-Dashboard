# ================================================
# PREDICTOR
# ================================================
# Responsible for:
# 1. Loading saved model
# 2. Scoring customers
# 3. Assigning risk levels
# 4. Adding recommended actions
# 5. Exporting final report
# ================================================

import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    FEATURES,
    LOW_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    RISK_LABELS,
    RISK_ACTIONS,
    OUTPUT_PATH,
    CURRENCY_SYMBOL,
    CLIENT_NAME
)
from src.model_trainer import load_model


def score_customers(rfm, model=None, features=None):
    """
    Score all customers with churn probability.

    Args:
        rfm: RFM DataFrame
        model: Optional pre-loaded model
        features: Optional pre-loaded features list

    Returns:
        RFM DataFrame with predictions added
    """
    print("⏳ Scoring customers...")

    # Load model if not provided
    if model is None or features is None:
        model, features = load_model()

    # Generate predictions
    rfm['Churn_Probability'] = model.predict_proba(
        rfm[features]
    )[:, 1]

    rfm['Churn_Predicted'] = model.predict(rfm[features])

    print(f"✅ {len(rfm):,} customers scored")
    return rfm


def assign_risk_levels(rfm):
    """
    Assign risk levels based on churn probability.

    🟢 Low Risk:    probability < 0.33
    🟡 Medium Risk: probability 0.33 - 0.66
    🔴 High Risk:   probability > 0.66

    Args:
        rfm: Scored RFM DataFrame

    Returns:
        RFM DataFrame with risk levels added
    """
    print("⏳ Assigning risk levels...")

    rfm['Risk_Level'] = pd.cut(
        rfm['Churn_Probability'],
        bins=[0, LOW_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD, 1.0],
        labels=[
            RISK_LABELS['low'],
            RISK_LABELS['medium'],
            RISK_LABELS['high']
        ]
    )

    rfm['Recommended_Action'] = rfm['Risk_Level'].map(RISK_ACTIONS)

    # Summary
    print("✅ Risk levels assigned")
    print(f"\n   RISK LEVEL SUMMARY ({CLIENT_NAME}):")
    print(f"   {'─'*40}")
    for label in RISK_LABELS.values():
        count = (rfm['Risk_Level'] == label).sum()
        pct = count / len(rfm) * 100
        print(f"   {label}: {count:,} customers ({pct:.1f}%)")
    print(f"   {'─'*40}")
    print(f"   Total: {len(rfm):,} customers")

    return rfm


def calculate_revenue_at_risk(rfm):
    """
    Calculate potential revenue at risk from churning customers.
    Useful for showing ROI to the client.

    Args:
        rfm: RFM DataFrame with risk levels

    Returns:
        Dictionary with revenue at risk figures
    """
    high_risk = rfm[rfm['Risk_Level'] == RISK_LABELS['high']]
    medium_risk = rfm[rfm['Risk_Level'] == RISK_LABELS['medium']]

    revenue_at_risk = {
        'high_risk_customers': len(high_risk),
        'medium_risk_customers': len(medium_risk),
        'high_risk_revenue': high_risk['Monetary'].sum(),
        'medium_risk_revenue': medium_risk['Monetary'].sum(),
        'total_at_risk_revenue': (
            high_risk['Monetary'].sum() +
            medium_risk['Monetary'].sum()
        )
    }

    print(f"\n   REVENUE AT RISK:")
    print(f"   {'─'*40}")
    print(f"   High Risk Revenue:   "
          f"{CURRENCY_SYMBOL}"
          f"{revenue_at_risk['high_risk_revenue']:,.2f}")
    print(f"   Medium Risk Revenue: "
          f"{CURRENCY_SYMBOL}"
          f"{revenue_at_risk['medium_risk_revenue']:,.2f}")
    print(f"   Total At Risk:       "
          f"{CURRENCY_SYMBOL}"
          f"{revenue_at_risk['total_at_risk_revenue']:,.2f}")

    return revenue_at_risk


def get_customer_profile(rfm, customer_id):
    """
    Get detailed profile for a single customer.
    Used in Streamlit dashboard for individual lookup.

    Args:
        rfm: Scored RFM DataFrame
        customer_id: Customer ID string

    Returns:
        Dictionary with customer details
    """
    if customer_id not in rfm.index:
        return None

    customer = rfm.loc[customer_id]

    profile = {
        'customer_id': customer_id,
        'recency': int(customer['Recency']),
        'frequency': int(customer['Frequency']),
        'monetary': round(customer['Monetary'], 2),
        'aov': round(customer['AOV'], 2),
        'tenure': int(customer['Tenure']),
        'churn_probability': round(customer['Churn_Probability'] * 100, 1),
        'risk_level': customer['Risk_Level'],
        'recommended_action': customer['Recommended_Action']
    }

    return profile


def export_report(rfm):
    """
    Export final churn risk report to Excel.
    This is the weekly deliverable for the client.

    Args:
        rfm: Scored RFM DataFrame with risk levels

    Returns:
        Path to exported file
    """
    print(f"\n⏳ Exporting report...")

    # Select and order columns for export
    export_df = rfm[[
        'Recency',
        'Frequency',
        'Monetary',
        'AOV',
        'Tenure',
        'Churn_Probability',
        'Risk_Level',
        'Recommended_Action'
    ]].copy()

    # Sort by highest risk first
    export_df = export_df.sort_values(
        'Churn_Probability',
        ascending=False
    )

    # Round probability to percentage
    export_df['Churn_Probability'] = (
        export_df['Churn_Probability'] * 100
    ).round(1)

    # Rename for readability
    export_df.rename(columns={
        'Churn_Probability': 'Churn_Probability_%'
    }, inplace=True)

    # Export
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    export_df.to_excel(OUTPUT_PATH, index=True)

    print(f"✅ Report exported to: {OUTPUT_PATH}")
    print(f"   Total customers: {len(export_df):,}")

    return OUTPUT_PATH


def run_prediction_pipeline(rfm, model=None, features=None):
    """
    Master function.
    Runs entire prediction pipeline in one call.

    Steps:
    1. Score customers
    2. Assign risk levels
    3. Calculate revenue at risk
    4. Export report

    Args:
        rfm: Model-ready RFM DataFrame
        model: Optional pre-loaded model
        features: Optional pre-loaded features

    Returns:
        Scored RFM DataFrame, revenue_at_risk dict
    """
    print("\n" + "="*40)
    print("PREDICTION PIPELINE")
    print("="*40)

    rfm = score_customers(rfm, model, features)
    rfm = assign_risk_levels(rfm)
    revenue_at_risk = calculate_revenue_at_risk(rfm)
    export_report(rfm)

    print("\n✅ Prediction pipeline complete!")
    return rfm, revenue_at_risk