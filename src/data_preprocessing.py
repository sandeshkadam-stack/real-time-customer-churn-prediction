# ============================================
# Customer Churn Prediction
# Data Preprocessing Script
# ============================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================
# Load Dataset
# ============================================

def load_data(file_path):
    """
    Load dataset from CSV file
    """
    
    df = pd.read_csv(file_path)
    
    return df


# ============================================
# Clean Dataset
# ============================================

def clean_data(df):
    """
    Perform basic data cleaning
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
    
    # Drop customerID column
    df.drop('customerID', axis=1, inplace=True)
    
    return df


# ============================================
# Encode Categorical Features
# ============================================

def encode_features(df):
    """
    Encode categorical columns using one-hot encoding
    """
    
    categorical_columns = df.select_dtypes(
        include=['object']
    ).columns
    
    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True
    )
    
    return df


# ============================================
# Split Features and Target
# ============================================

def split_features_target(df):
    """
    Split dataset into X and y
    """
    
    X = df.drop('Churn_Yes', axis=1)
    y = df['Churn_Yes']
    
    return X, y


# ============================================
# Scale Numerical Features
# ============================================

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    """
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


# ============================================
# Complete Preprocessing Pipeline
# ============================================

def preprocess_data(file_path):
    """
    Full preprocessing pipeline
    """
    
    # Load data
    df = load_data(file_path)
    
    # Clean data
    df = clean_data(df)
    
    # Encode categorical features
    df = encode_features(df)
    
    # Split features and target
    X, y = split_features_target(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test
    )
    
    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    
    file_path = "data/telco_churn.csv"
    
    X_train, X_test, y_train, y_test = preprocess_data(file_path)
    
    print("Data Preprocessing Completed Successfully")
    
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")