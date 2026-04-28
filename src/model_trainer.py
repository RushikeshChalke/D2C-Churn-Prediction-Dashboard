# ================================================
# MODEL TRAINER
# ================================================
# Responsible for:
# 1. Preparing data for ML
# 2. Training XGBoost model
# 3. Evaluating performance
# 4. Saving model to disk
# ================================================

import pickle
import os
import sys
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    FEATURES,
    MODEL_PARAMS,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_PATH,
    FEATURES_PATH
)


def prepare_data(rfm):
    """
    Split RFM data into train and test sets.

    Args:
        rfm: Model-ready RFM DataFrame

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("⏳ Preparing data for training...")

    X = rfm[FEATURES]
    y = rfm['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"✅ Data prepared")
    print(f"   Training set: {len(X_train):,} customers")
    print(f"   Testing set:  {len(X_test):,} customers")
    print(f"   Features:     {FEATURES}")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train XGBoost churn prediction model.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained XGBoost model
    """
    print("\n⏳ Training XGBoost model...")

    model = xgb.XGBClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    print("✅ Model trained successfully!")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    Prints full performance report.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels

    Returns:
        y_pred, accuracy
    """
    print("\n" + "="*40)
    print("MODEL PERFORMANCE REPORT")
    print("="*40)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy Score: {accuracy:.2%}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix breakdown
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"   True Negatives  (Active → Active):   {cm[0][0]}")
    print(f"   False Positives (Active → Churned):  {cm[0][1]}")
    print(f"   False Negatives (Churned → Active):  {cm[1][0]}")
    print(f"   True Positives  (Churned → Churned): {cm[1][1]}")
    print(f"\n   Correctly caught churners: {cm[1][1]}/{cm[1][0]+cm[1][1]}")

    return y_pred, accuracy


def save_model(model):
    """
    Save trained model and features list to disk.

    Args:
        model: Trained model

    Returns:
        None
    """
    print("\n⏳ Saving model...")

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

    # Save features
    with open(FEATURES_PATH, 'wb') as f:
        pickle.dump(FEATURES, f)

    print(f"✅ Model saved to:    {MODEL_PATH}")
    print(f"✅ Features saved to: {FEATURES_PATH}")


def load_model():
    """
    Load saved model and features from disk.

    Returns:
        model, features
    """
    print("⏳ Loading saved model...")

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, 'rb') as f:
        features = pickle.load(f)

    print("✅ Model loaded successfully!")
    print(f"   Features: {features}")

    return model, features


def run_training_pipeline(rfm):
    """
    Master function.
    Runs entire training pipeline in one call.

    Steps:
    1. Prepare data
    2. Train model
    3. Evaluate model
    4. Save model

    Args:
        rfm: Model-ready RFM DataFrame

    Returns:
        model, accuracy
    """
    print("\n" + "="*40)
    print("TRAINING PIPELINE")
    print("="*40)

    X_train, X_test, y_train, y_test = prepare_data(rfm)
    model = train_model(X_train, y_train)
    y_pred, accuracy = evaluate_model(model, X_test, y_test)
    save_model(model)

    print("\n✅ Training pipeline complete!")
    return model, accuracy