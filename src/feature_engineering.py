# ============================================
# Customer Churn Prediction
# Feature Engineering Script
# ============================================

# Import Libraries
import pandas as pd
import numpy as np


# ============================================
# Load Dataset
# ============================================

def load_data(file_path):
    """
    Load dataset
    """
    
    df = pd.read_csv(file_path)
    
    return df


# ============================================
# Basic Cleaning
# ============================================

def basic_cleaning(df):
    """
    Perform basic cleaning
    """
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'],
        errors='coerce'
    )
    
    # Fill missing values
    df['TotalCharges'] = df['TotalCharges'].fillna(
        df['TotalCharges'].median()
    )
    
    return df


# ============================================
# Create Tenure Groups
# ============================================

def create_tenure_group(df):
    """
    Create customer tenure groups
    """
    
    def tenure_label(tenure):
        
        if tenure <= 12:
            return '0-1 Year'
        
        elif tenure <= 24:
            return '1-2 Years'
        
        elif tenure <= 48:
            return '2-4 Years'
        
        else:
            return '4+ Years'
    
    
    df['TenureGroup'] = df['tenure'].apply(
        tenure_label
    )
    
    return df


# ============================================
# Create Total Services Feature
# ============================================

def create_total_services(df):
    """
    Count total subscribed services
    """
    
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
            df[col].isin(['Yes']),
            1,
            0
        )
    
    return df


# ============================================
# Create Average Monthly Spend
# ============================================

def create_avg_monthly_spend(df):
    """
    Calculate average spend per tenure
    """
    
    df['AvgMonthlySpend'] = (
        df['TotalCharges'] /
        (df['tenure'] + 1)
    )
    
    return df


# ============================================
# High Value Customer Flag
# ============================================

def create_high_value_customer(df):
    """
    Identify high-value customers
    """
    
    median_charges = df['MonthlyCharges'].median()
    
    df['IsHighValueCustomer'] = np.where(
        df['MonthlyCharges'] > median_charges,
        1,
        0
    )
    
    return df


# ============================================
# Contract Risk Feature
# ============================================

def create_contract_risk(df):
    """
    Assign contract risk levels
    """
    
    risk_mapping = {
        'Month-to-month': 'High Risk',
        'One year': 'Medium Risk',
        'Two year': 'Low Risk'
    }
    
    df['ContractRisk'] = df['Contract'].map(
        risk_mapping
    )
    
    return df


# ============================================
# Main Feature Engineering Pipeline
# ============================================

def feature_engineering_pipeline(df):
    """
    Run complete feature engineering
    """
    
    # Basic cleaning
    df = basic_cleaning(df)
    
    # Feature creation
    df = create_tenure_group(df)
    
    df = create_total_services(df)
    
    df = create_avg_monthly_spend(df)
    
    df = create_high_value_customer(df)
    
    df = create_contract_risk(df)
    
    return df


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    
    file_path = "data/telco_churn.csv"
    
    # Load dataset
    df = load_data(file_path)
    
    # Apply feature engineering
    df = feature_engineering_pipeline(df)
    
    # Display sample output
    print(df.head())
    
    # Save engineered dataset
    output_path = "data/feature_engineered_churn.csv"
    
    df.to_csv(output_path, index=False)
    
    print("\nFeature Engineering Completed Successfully")
    
    print(f"\nDataset saved at: {output_path}")