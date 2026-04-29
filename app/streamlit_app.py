# ================================================
# CHURN PREDICTION DASHBOARD
# ================================================
# Run this with:
# streamlit run app/streamlit_app.py
# ================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import pickle

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_and_clean
from src.rfm_builder import build_rfm
from src.model_trainer import load_model
from src.predictor import (
    score_customers,
    assign_risk_levels,
    calculate_revenue_at_risk,
    get_customer_profile
)
from config.config import CURRENCY_SYMBOL, CLIENT_NAME

# ================================================
# PAGE CONFIGURATION
# ================================================
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ================================================
# LOAD DATA (cached so it doesn't reload every time)
# ================================================
@st.cache_data
def load_all_data():
    """
    Load and process all data.
    Cached so dashboard stays fast.
    """
    df = load_and_clean()
    rfm = build_rfm(df)
    model, features = load_model()
    rfm = score_customers(rfm, model, features)
    rfm = assign_risk_levels(rfm)
    revenue_at_risk = calculate_revenue_at_risk(rfm)
    return rfm, revenue_at_risk

# Show loading spinner while data loads
with st.spinner("Loading data... please wait."):
    rfm, revenue_at_risk = load_all_data()

# ================================================
# SIDEBAR NAVIGATION
# ================================================
st.sidebar.image(
    "https://img.icons8.com/color/96/analytics.png",
    width=80
)
st.sidebar.title("Churn Dashboard")
st.sidebar.markdown(f"**Client:** {CLIENT_NAME}")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "📊 Overview",
        "🎯 Risk Segments",
        "🔍 Customer Lookup",
        "🤖 Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Built with ❤️ using Python + XGBoost + Streamlit"
)

# ================================================
# PAGE 1: OVERVIEW
# ================================================
if page == "📊 Overview":

    st.title("📊 Customer Churn Overview")
    st.markdown(f"### {CLIENT_NAME} — Churn Risk Report")
    st.markdown("---")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Customers",
            value=f"{len(rfm):,}"
        )

    with col2:
        churn_rate = rfm['Churn'].mean() * 100
        st.metric(
            label="Overall Churn Rate",
            value=f"{churn_rate:.1f}%"
        )

    with col3:
        high_risk = (rfm['Risk_Level'] == '🔴 High Risk').sum()
        st.metric(
            label="High Risk Customers",
            value=f"{high_risk:,}",
            delta=f"{high_risk/len(rfm)*100:.1f}% of total",
            delta_color="inverse"
        )

    with col4:
        total_at_risk = revenue_at_risk['total_at_risk_revenue']
        st.metric(
            label="Revenue at Risk",
            value=f"{CURRENCY_SYMBOL}{total_at_risk:,.0f}",
            delta="Needs immediate action",
            delta_color="inverse"
        )

    st.markdown("---")

    # Two column layout for charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Level Distribution")
        risk_counts = rfm['Risk_Level'].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        risk_counts.plot(
            kind='bar',
            ax=ax,
            color=colors,
            edgecolor='white'
        )
        ax.set_title("Customers by Risk Level")
        ax.set_xlabel("")
        ax.set_ylabel("Number of Customers")
        ax.tick_params(axis='x', rotation=0)
        for i, v in enumerate(risk_counts):
            ax.text(i, v + 10, str(v), ha='center', fontweight='bold')
        st.pyplot(fig)

    with col2:
        st.subheader("Churn Probability Distribution")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            rfm['Churn_Probability'],
            bins=30,
            color='#3498db',
            edgecolor='white',
            alpha=0.8
        )
        ax.axvline(
            x=0.33, color='#f39c12',
            linestyle='--', label='Medium Risk Threshold'
        )
        ax.axvline(
            x=0.66, color='#e74c3c',
            linestyle='--', label='High Risk Threshold'
        )
        ax.set_title("Distribution of Churn Probabilities")
        ax.set_xlabel("Churn Probability")
        ax.set_ylabel("Number of Customers")
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")

    # RFM Summary Table
    st.subheader("RFM Summary Statistics")
    summary = rfm[['Recency', 'Frequency', 'Monetary',
                    'AOV', 'Tenure']].describe().round(2)
    st.dataframe(summary, use_container_width=True)


# ================================================
# PAGE 2: RISK SEGMENTS
# ================================================
elif page == "🎯 Risk Segments":

    st.title("🎯 Customer Risk Segments")
    st.markdown("---")

    # Filter by risk level
    risk_filter = st.selectbox(
        "Filter by Risk Level:",
        ["All", "🔴 High Risk", "🟡 Medium Risk", "🟢 Low Risk"]
    )

    if risk_filter == "All":
        filtered_rfm = rfm
    else:
        filtered_rfm = rfm[rfm['Risk_Level'] == risk_filter]

    # Metrics for filtered segment
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Customers", f"{len(filtered_rfm):,}")

    with col2:
        avg_prob = filtered_rfm['Churn_Probability'].mean() * 100
        st.metric("Avg Churn Probability", f"{avg_prob:.1f}%")

    with col3:
        segment_revenue = filtered_rfm['Monetary'].sum()
        st.metric(
            "Segment Revenue",
            f"{CURRENCY_SYMBOL}{segment_revenue:,.0f}"
        )

    st.markdown("---")

    # Scatter plot
    st.subheader("Frequency vs Monetary (by Risk Level)")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors_map = {
        '🟢 Low Risk': '#2ecc71',
        '🟡 Medium Risk': '#f39c12',
        '🔴 High Risk': '#e74c3c'
    }

    for risk_level, color in colors_map.items():
        mask = filtered_rfm['Risk_Level'] == risk_level
        ax.scatter(
            filtered_rfm[mask]['Frequency'],
            filtered_rfm[mask]['Monetary'],
            c=color,
            label=risk_level,
            alpha=0.6,
            s=50
        )

    ax.set_xlabel("Frequency (Number of Purchases)")
    ax.set_ylabel(f"Monetary (Total Spend {CURRENCY_SYMBOL})")
    ax.set_title("Customer Segments: Frequency vs Monetary Value")
    ax.legend()
    st.pyplot(fig)

    st.markdown("---")

    # Customer table
    st.subheader(f"Customer List ({len(filtered_rfm):,} customers)")

    display_df = filtered_rfm[[
        'Recency', 'Frequency', 'Monetary',
        'Churn_Probability', 'Risk_Level',
        'Recommended_Action'
    ]].copy()

    display_df['Churn_Probability'] = (
        display_df['Churn_Probability'] * 100
    ).round(1).astype(str) + '%'

    st.dataframe(
        display_df.sort_values('Churn_Probability', ascending=False),
        use_container_width=True
    )


# ================================================
# PAGE 3: CUSTOMER LOOKUP
# ================================================
elif page == "🔍 Customer Lookup":

    st.title("🔍 Individual Customer Lookup")
    st.markdown("---")

    # Search box
    customer_id = st.text_input(
        "Enter Customer ID:",
        placeholder="e.g. 12347"
    )

    if customer_id:
        profile = get_customer_profile(rfm, customer_id)

        if profile is None:
            st.error(f"Customer '{customer_id}' not found.")
        else:
            # Risk level color
            risk = profile['risk_level']
            if risk == '🔴 High Risk':
                st.error(f"Risk Level: {risk}")
            elif risk == '🟡 Medium Risk':
                st.warning(f"Risk Level: {risk}")
            else:
                st.success(f"Risk Level: {risk}")

            st.markdown("---")

            # Customer metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Recency",
                    f"{profile['recency']} days",
                    help="Days since last purchase"
                )
                st.metric(
                    "Frequency",
                    f"{profile['frequency']} orders",
                    help="Total number of purchases"
                )

            with col2:
                st.metric(
                    "Total Spend",
                    f"{CURRENCY_SYMBOL}{profile['monetary']:,.2f}",
                    help="Total revenue from this customer"
                )
                st.metric(
                    "Avg Order Value",
                    f"{CURRENCY_SYMBOL}{profile['aov']:,.2f}",
                    help="Average spend per order"
                )

            with col3:
                st.metric(
                    "Tenure",
                    f"{profile['tenure']} days",
                    help="Days since first purchase"
                )
                st.metric(
                    "Churn Probability",
                    f"{profile['churn_probability']}%",
                    help="Model's estimated churn probability"
                )

            st.markdown("---")

            # Recommended action
            st.subheader("💡 Recommended Action")
            st.info(profile['recommended_action'])

            # Churn probability gauge
            st.subheader("Churn Risk Meter")
            prob = profile['churn_probability'] / 100

            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(
                ['Risk'],
                [prob],
                color='#e74c3c' if prob > 0.66
                else '#f39c12' if prob > 0.33
                else '#2ecc71',
                height=0.5
            )
            ax.barh(
                ['Risk'],
                [1 - prob],
                left=[prob],
                color='#ecf0f1',
                height=0.5
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Churn Probability")
            ax.set_title(
                f"Churn Probability: {profile['churn_probability']}%"
            )
            ax.axvline(x=0.33, color='#f39c12', linestyle='--', alpha=0.7)
            ax.axvline(x=0.66, color='#e74c3c', linestyle='--', alpha=0.7)
            st.pyplot(fig)

    else:
        # Show sample customer IDs
        st.info("Enter a Customer ID above to see their churn profile.")
        st.markdown("**Sample Customer IDs to try:**")
        sample_ids = rfm.index[:10].tolist()
        st.write(", ".join(sample_ids))


# ================================================
# PAGE 4: MODEL PERFORMANCE
# ================================================
elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")
    st.markdown("---")

    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        accuracy_score
    )
    from src.model_trainer import load_model, prepare_data

    # Load model and prepare test data
    model, features = load_model()
    X_train, X_test, y_train, y_test = prepare_data(rfm)
    y_pred = model.predict(X_test)

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    with col1:
        st.metric("Accuracy", f"{accuracy:.2%}")

    with col2:
        st.metric(
            "Churners Caught",
            f"{cm[1][1]}/{cm[1][0]+cm[1][1]}"
        )

    with col3:
        precision = cm[1][1] / (cm[0][1] + cm[1][1])
        st.metric("Precision", f"{precision:.2%}")

    with col4:
        recall = cm[1][1] / (cm[1][0] + cm[1][1])
        st.metric("Recall", f"{recall:.2%}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Predicted Active', 'Predicted Churned'],
            yticklabels=['Actually Active', 'Actually Churned'],
            ax=ax
        )
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    with col2:
        # Feature Importance
        st.subheader("Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(
            importance_df['Feature'],
            importance_df['Importance'],
            color='#3498db',
            edgecolor='white'
        )
        ax.set_title("Feature Importance (XGBoost)")
        ax.set_xlabel("Importance Score")
        st.pyplot(fig)

    st.markdown("---")

    # Classification report
    st.subheader("Detailed Classification Report")
    report = classification_report(
        y_test, y_pred,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose().round(2)
    st.dataframe(report_df, use_container_width=True)