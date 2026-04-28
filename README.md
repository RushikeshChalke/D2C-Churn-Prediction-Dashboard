# 📊 Customer Churn Prediction & ROI Dashboard
> A Machine Learning solution to identify at-risk customers and protect revenue for D2C/Retail brands.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 🏢 Business Problem
The brand was losing customers silently. Without a predictive system, the marketing team was "spraying and praying"—sending discounts to everyone, which wasted budget on loyalists and ignored high-risk customers who were actually about to leave.

**Key Objective:** Identify high-risk customers *before* they churn and quantify the revenue at risk.

## 🚀 The Solution: AI-Driven Retention
I developed an end-to-end pipeline that transforms raw transaction data into an interactive retention dashboard.

### 1. Data Pipeline & Feature Engineering
- **RFM Analysis:** Processed 1M+ transactions to calculate Recency, Frequency, and Monetary value.
- **Behavioral Scoring:** Engineered features like **AOV (Average Order Value)** and **Customer Tenure** to capture loyalty signals.
- **Winsorization:** Handled extreme outliers to ensure model stability.

### 2. Machine Learning Model
- **Algorithm:** XGBoost Classifier.
- **Performance:** **81.8% Accuracy** on unseen test data.
- **Explainability:** Integrated **SHAP values** to explain *why* the model flags a customer (e.g., Short tenure + low frequency = high risk).

### 3. Interactive Dashboard
- **Executive Overview:** Real-time metrics on total revenue at risk.
- **Risk Segmentation:** Automated grouping into High, Medium, and Low-risk tiers.
- **Individual Lookup:** "Search by ID" tool for customer support teams to see a customer's risk profile and recommended action.

## 💰 Business Impact (ROI)
- **Revenue Protection:** Identified **£3.4M in Revenue at Risk**.
- **Efficiency:** Reduced marketing waste by targeting only the **38% of customers** truly at risk.
- **Precision:** Correctedly identified **82% of all churners** before they left.

## 🛠️ Tech Stack
- **Backend:** Python (Pandas, NumPy)
- **ML:** XGBoost, Scikit-Learn, SHAP
- **Dashboard:** Streamlit
- **Design:** Config-driven architecture (easy to swap datasets for different clients)

## 📦 Project Structure
```text
Churn_Project/
├── app/              # Streamlit Dashboard
├── config/           # Global Settings & Thresholds
├── src/              # Reusable Modular Scripts (DataLoader, Trainer, etc.)
├── models/           # Saved ML Models (Pickle)
└── data/             # Sample Transaction Data