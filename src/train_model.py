# ============================================
# Customer Churn Prediction
# Model Training Script
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ML Models
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


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
# Encode Categorical Features
# ============================================

def encode_features(df):
    """
    One-hot encode categorical columns
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
# Prepare Features and Target
# ============================================

def prepare_data(df):
    """
    Split X and y
    """
    
    # Encode target variable
    df['Churn'] = df['Churn'].map({
        'Yes': 1,
        'No': 0
    })
    
    # Encode categorical features
    df = encode_features(df)
    
    # Features and target
    X = df.drop('Churn', axis=1)
    # Save training columns
    joblib.dump(
    X.columns.tolist(),
    "models/training_columns.pkl"
    )
    
    y = df['Churn']
    
    return X, y


# ============================================
# Scale Features
# ============================================

def scale_data(X_train, X_test):
    """
    Scale features
    """
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(
        X_train
    )
    
    X_test_scaled = scaler.transform(
        X_test
    )
    
    return X_train_scaled, X_test_scaled


# ============================================
# Evaluate Model
# ============================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    
    y_pred = model.predict(X_test)
    
    y_prob = model.predict_proba(X_test)[:,1]
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        
        'Precision': precision_score(y_test, y_pred),
        
        'Recall': recall_score(y_test, y_pred),
        
        'F1 Score': f1_score(y_test, y_pred),
        
        'ROC AUC': roc_auc_score(y_test, y_prob)
    }
    
    return metrics


# ============================================
# Train Models
# ============================================

def train_models(X_train, y_train):
    """
    Train multiple ML models
    """
    
    models = {
        'Logistic Regression':
            LogisticRegression(max_iter=1000),
        
        'Random Forest':
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
        
        'XGBoost':
            XGBClassifier(
                eval_metric='logloss',
                random_state=42
            )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        
        print(f"\nTraining {name}...")
        
        model.fit(X_train, y_train)
        
        trained_models[name] = model
    
    return trained_models


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    
    # Dataset path
    file_path = "data/feature_engineered_churn.csv"
    
    # Load dataset
    df = load_data(file_path)
    
    print("Dataset Loaded Successfully")
    
    
    # Prepare data
    X, y = prepare_data(df)
    
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    
    # Scale data
    X_train_scaled, X_test_scaled = scale_data(
        X_train,
        X_test
    )
    
    
    # Train models
    trained_models = train_models(
        X_train_scaled,
        y_train
    )
    
    
    # Evaluate models
    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")
    
    best_model = None
    best_roc_auc = 0
    
    
    for name, model in trained_models.items():
        
        metrics = evaluate_model(
            model,
            X_test_scaled,
            y_test
        )
        
        print(f"\n{name}")
        
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        
        # Select best model
        if metrics['ROC AUC'] > best_roc_auc:
            
            best_roc_auc = metrics['ROC AUC']
            
            best_model = model
            
            best_model_name = name
    
    
    # Save best model
    joblib.dump(
        best_model,
        "models/best_churn_model.pkl"
    )
    
    print("\n==============================")
    
    print(f"Best Model: {best_model_name}")
    
    print(f"Best ROC AUC: {best_roc_auc:.4f}")
    
    print("\nModel Saved Successfully")