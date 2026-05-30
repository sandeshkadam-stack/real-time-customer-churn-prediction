# ============================================
# Customer Churn Prediction
# Prediction Script
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler


# ============================================
# Load Trained Model
# ============================================

def load_model(model_path):
    """
    Load trained ML model
    """
    
    model = joblib.load(model_path)
    
    return model


# ============================================
# Create Sample Customer Data
# ============================================

def create_sample_customer():
    """
    Create sample customer input
    """
    
    customer_data = {
        
        'gender': 'Male',
        
        'SeniorCitizen': 0,
        
        'Partner': 'Yes',
        
        'Dependents': 'No',
        
        'tenure': 5,
        
        'PhoneService': 'Yes',
        
        'MultipleLines': 'No',
        
        'InternetService': 'Fiber optic',
        
        'OnlineSecurity': 'No',
        
        'OnlineBackup': 'Yes',
        
        'DeviceProtection': 'No',
        
        'TechSupport': 'No',
        
        'StreamingTV': 'Yes',
        
        'StreamingMovies': 'Yes',
        
        'Contract': 'Month-to-month',
        
        'PaperlessBilling': 'Yes',
        
        'PaymentMethod':
            'Electronic check',
        
        'MonthlyCharges': 85.5,
        
        'TotalCharges': 450.2
    }
    
    return pd.DataFrame([customer_data])


# ============================================
# Feature Engineering
# ============================================

def apply_feature_engineering(df):
    """
    Apply same feature engineering
    used during training
    """
    
    # Tenure Group
    def tenure_group(tenure):
        
        if tenure <= 12:
            return '0-1 Year'
        
        elif tenure <= 24:
            return '1-2 Years'
        
        elif tenure <= 48:
            return '2-4 Years'
        
        else:
            return '4+ Years'
    
    
    df['TenureGroup'] = (
        df['tenure']
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
    
    df['TotalServices'] = 0
    
    for col in service_columns:
        
        df['TotalServices'] += np.where(
            df[col] == 'Yes',
            1,
            0
        )
    
    
    # Average Monthly Spend
    df['AvgMonthlySpend'] = (
        df['TotalCharges'] /
        (df['tenure'] + 1)
    )
    
    
    # High Value Customer
    df['IsHighValueCustomer'] = (
        np.where(
            df['MonthlyCharges'] > 70,
            1,
            0
        )
    )
    
    
    # Contract Risk
    risk_mapping = {
        'Month-to-month': 'High Risk',
        'One year': 'Medium Risk',
        'Two year': 'Low Risk'
    }
    
    df['ContractRisk'] = (
        df['Contract']
        .map(risk_mapping)
    )
    
    return df


# ============================================
# Encode Features
# ============================================

def encode_features(df):
    """
    Encode categorical features
    """
    
    categorical_columns = df.select_dtypes(
        include=['object', 'string']
    ).columns
    
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )
    
    return df


# ============================================
# Align Features
# ============================================

def align_features(df, training_columns):
    """
    Match prediction columns
    with training columns
    """
    
    for column in training_columns:
        
        if column not in df.columns:
            df[column] = 0
    
    
    df = df[training_columns]
    
    return df


# ============================================
# Predict Churn
# ============================================

def predict_churn(model, df):
    """
    Generate churn prediction
    """
    
    prediction = model.predict(df)[0]
    
    probability = (
        model.predict_proba(df)[0][1]
    )
    
    return prediction, probability


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    
    # Model path
    model_path = (
        "models/best_churn_model.pkl"
    )
    
    
    # Load model
    model = load_model(model_path)
    
    print("Model Loaded Successfully")
    
    
    # Create sample customer
    customer_df = create_sample_customer()
    
    
    # Feature engineering
    customer_df = apply_feature_engineering(
        customer_df
    )
    
    
    # Encode features
    customer_df = encode_features(
        customer_df
    )
    
    
    # IMPORTANT:
    # These columns must match
    # training dataset columns
    
    training_columns = joblib.load(
        "models/training_columns.pkl"
    )
    
    
    # Align features
    customer_df = align_features(
        customer_df,
        training_columns
    )
    
    
    # Prediction
    prediction, probability = (
        predict_churn(
            model,
            customer_df
        )
    )
    
    
    # Display result
    print("\n======================")
    print("CHURN PREDICTION")
    print("======================")
    
    if prediction == 1:
        
        print(
            "\nCustomer is likely to CHURN"
        )
    
    else:
        
        print(
            "\nCustomer is likely to STAY"
        )
    
    
    print(
        f"\nChurn Probability: "
        f"{probability:.2%}"
    )