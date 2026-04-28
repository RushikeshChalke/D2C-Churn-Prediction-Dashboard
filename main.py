# ================================================
# FULL PIPELINE RUNNER
# ================================================
# Run this file to:
# 1. Load and clean data
# 2. Build RFM table
# 3. Train model
# 4. Score customers
# 5. Export report
#
# This is what you run for every new client.
# Just change config.py settings first.
# ================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_and_clean
from src.rfm_builder import build_rfm
from src.model_trainer import run_training_pipeline
from src.predictor import run_prediction_pipeline

def main():
    print("\n" + "="*50)
    print("   CHURN PREDICTION PIPELINE")
    print("="*50)

    # Step 1: Load and clean data
    print("\n📦 STEP 1: LOADING DATA")
    df = load_and_clean()

    # Step 2: Build RFM
    print("\n📊 STEP 2: BUILDING RFM")
    rfm = build_rfm(df)

    # Step 3: Train model
    print("\n🤖 STEP 3: TRAINING MODEL")
    model, accuracy = run_training_pipeline(rfm)

    # Step 4: Score and export
    print("\n🎯 STEP 4: SCORING CUSTOMERS")
    rfm_scored, revenue_at_risk = run_prediction_pipeline(
        rfm, model
    )

    # Final Summary
    print("\n" + "="*50)
    print("   PIPELINE COMPLETE")
    print("="*50)
    print(f"   Accuracy:          {accuracy:.2%}")
    print(f"   Total customers:   {len(rfm_scored):,}")
    print(f"   High Risk:         "
          f"{(rfm_scored['Risk_Level'] == '🔴 High Risk').sum():,}")
    print(f"   Revenue at risk:   "
          f"£{revenue_at_risk['total_at_risk_revenue']:,.2f}")
    print(f"   Report saved to:   outputs/churn_risk_report.xlsx")
    print("="*50)

if __name__ == "__main__":
    main()