# ============================================
# Customer Churn Prediction App
# Streamlit Application
# ============================================

# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.genai_explainer import (
    generate_churn_explanation
)

# ============================================
# Streamlit Page Config
# ============================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ============================================
# Load Model and Training Columns
# ============================================

model = joblib.load(
    "models/best_churn_model.pkl"
)

training_columns = joblib.load(
    "models/training_columns.pkl"
)

# ============================================
# Retention Recommendation Engine
# ============================================

def generate_retention_recommendation(
    churn_probability,
    contract_type,
    tenure,
    monthly_charges,
    total_services
):

    recommendations = []

    # High Risk Customers
    if churn_probability >= 0.75:

        recommendations.append(
            "Priority retention outreach"
        )

        recommendations.append(
            "Assign customer success manager"
        )

    # Contract-based recommendation
    if contract_type == 'Month-to-month':

        recommendations.append(
            "Offer annual contract discount"
        )

    # New customer churn risk
    if tenure <= 12:

        recommendations.append(
            "Launch onboarding engagement campaign"
        )

    # Price sensitivity
    if monthly_charges >= 80:

        recommendations.append(
            "Provide personalized pricing offer"
        )

    # Low service adoption
    if total_services <= 2:

        recommendations.append(
            "Recommend bundled services"
        )

    # Fallback recommendation
    if len(recommendations) == 0:

        recommendations.append(
            "Standard retention monitoring"
        )

    return recommendations

# ============================================
# Tenure Group Function
# ============================================

def tenure_group(tenure):

    if tenure <= 12:
        return '0-1 Year'

    elif tenure <= 24:
        return '1-2 Years'

    elif tenure <= 48:
        return '2-4 Years'

    else:
        return '4+ Years'

# ============================================
# App Title
# ============================================

st.title("📊 Customer Churn Prediction")

st.write(
    """
    Predict whether a telecom customer
    is likely to churn based on
    customer behavior and subscription data.
    """
)

# ============================================
# User Inputs
# ============================================

st.header("Customer Information")

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

SeniorCitizen = st.selectbox(
    "Senior Citizen",
    [0, 1]
)

Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)

Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure (Months)",
    0,
    72,
    12
)

PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

MultipleLines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No"]
)

InternetService = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

OnlineSecurity = st.selectbox(
    "Online Security",
    ["Yes", "No"]
)

OnlineBackup = st.selectbox(
    "Online Backup",
    ["Yes", "No"]
)

DeviceProtection = st.selectbox(
    "Device Protection",
    ["Yes", "No"]
)

TechSupport = st.selectbox(
    "Tech Support",
    ["Yes", "No"]
)

StreamingTV = st.selectbox(
    "Streaming TV",
    ["Yes", "No"]
)

StreamingMovies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

Contract = st.selectbox(
    "Contract Type",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)

PaperlessBilling = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

# ============================================
# Prediction Button
# ============================================

if st.button("Predict Churn"):

    # ============================================
    # Create Input DataFrame
    # ============================================

    input_data = pd.DataFrame({

        'gender': [gender],
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'tenure': [tenure],
        'PhoneService': [PhoneService],
        'MultipleLines': [MultipleLines],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'TotalCharges': [TotalCharges]

    })

    # ============================================
    # Feature Engineering
    # ============================================

    # Tenure Group
    input_data['TenureGroup'] = (
        input_data['tenure']
        .apply(tenure_group)
    )

    # Total Services
    service_columns = [
        'PhoneService',
        'MultipleLines',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    input_data['TotalServices'] = 0

    for col in service_columns:

        input_data['TotalServices'] += np.where(
            input_data[col] == 'Yes',
            1,
            0
        )

    # Average Monthly Spend
    input_data['AvgMonthlySpend'] = (
        input_data['TotalCharges'] /
        (input_data['tenure'] + 1)
    )

    # High Value Customer
    input_data['IsHighValueCustomer'] = np.where(
        input_data['MonthlyCharges'] > 70,
        1,
        0
    )

    # Contract Risk
    risk_mapping = {
        'Month-to-month': 'High Risk',
        'One year': 'Medium Risk',
        'Two year': 'Low Risk'
    }

    input_data['ContractRisk'] = (
        input_data['Contract']
        .map(risk_mapping)
    )

    # ============================================
    # Encode Features
    # ============================================

    categorical_columns = (
        input_data.select_dtypes(
            include=['object', 'string']
        ).columns
    )

    input_data = pd.get_dummies(
        input_data,
        columns=categorical_columns,
        drop_first=True
    )

    # ============================================
    # Align Columns
    # ============================================

    for column in training_columns:

        if column not in input_data.columns:
            input_data[column] = 0

    input_data = input_data[
        training_columns
    ]

    # ============================================
    # Prediction
    # ============================================

    prediction = model.predict(
        input_data
    )[0]

    probability = (
        model.predict_proba(
            input_data
        )[0][1]
    )

    # ============================================
    # Generate Recommendations
    # ============================================

    recommendations = (
        generate_retention_recommendation(
            churn_probability=probability,
            contract_type=Contract,
            tenure=tenure,
            monthly_charges=MonthlyCharges,
            total_services=input_data.iloc[0].sum()
        )
    )

    # ============================================
    # Generate AI Explanation
    # ============================================

    explanation = generate_churn_explanation(
        customer_data=input_data.to_dict(),
        churn_probability=probability
    )

    # ============================================
    # Display Results
    # ============================================

    st.header("Prediction Result")

    if prediction == 1:

        st.error(
            f"""
            Customer is likely to CHURN

            Churn Probability:
            {probability:.2%}
            """
        )

    else:

        st.success(
            f"""
            Customer is likely to STAY

            Churn Probability:
            {probability:.2%}
            """
        )

    # ============================================
    # Risk Interpretation
    # ============================================

    st.subheader("Risk Interpretation")

    if probability >= 0.75:

        st.warning(
            "High churn risk customer"
        )

    elif probability >= 0.50:

        st.info(
            "Moderate churn risk customer"
        )

    else:

        st.success(
            "Low churn risk customer"
        )

    # ============================================
    # Retention Recommendations
    # ============================================

    st.subheader(
        "Retention Recommendations"
    )

    for recommendation in recommendations:

        st.write(
            f"• {recommendation}"
        )

    # ============================================
    # AI Explanation
    # ============================================

    st.subheader("AI Churn Explanation")

    st.write(explanation)