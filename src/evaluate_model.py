# ============================================
# Customer Churn Prediction
# Model Evaluation Script
# ============================================

# Import Libraries
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


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
# Encode Features
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
# Prepare Data
# ============================================

def prepare_data(df):
    """
    Prepare features and target
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
    
    y = df['Churn']
    
    return X, y


# ============================================
# Scale Features
# ============================================

def scale_data(X_train, X_test):
    """
    Scale numerical features
    """
    
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled


# ============================================
# Plot Confusion Matrix
# ============================================

def plot_confusion_matrix(y_test, y_pred):
    """
    Plot confusion matrix
    """
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6,5))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    
    plt.title("Confusion Matrix")
    
    plt.xlabel("Predicted")
    
    plt.ylabel("Actual")
    
    # Save figure
    plt.savefig(
        "reports/confusion_matrix.png"
    )
    
    plt.show()


# ============================================
# Plot ROC Curve
# ============================================

def plot_roc_curve(y_test, y_prob):
    """
    Plot ROC curve
    """
    
    fpr, tpr, thresholds = roc_curve(
        y_test,
        y_prob
    )
    
    auc_score = roc_auc_score(
        y_test,
        y_prob
    )
    
    plt.figure(figsize=(7,5))
    
    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc_score:.4f}"
    )
    
    plt.plot(
        [0,1],
        [0,1],
        linestyle='--'
    )
    
    plt.xlabel("False Positive Rate")
    
    plt.ylabel("True Positive Rate")
    
    plt.title("ROC Curve")
    
    plt.legend()
    
    # Save figure
    plt.savefig(
        "../reports/roc_curve.png"
    )
    
    plt.show()


# ============================================
# Save Classification Report
# ============================================

def save_classification_report(report):
    """
    Save classification report
    """
    
    with open(
        "reports/classification_report.txt",
        "w"
    ) as file:
        
        file.write(report)


# ============================================
# Feature Importance
# ============================================

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance
    Supports tree models and logistic regression
    """
    
    # Tree-based models
    if hasattr(model, 'feature_importances_'):
        
        importance = model.feature_importances_
    
    
    # Logistic Regression
    elif hasattr(model, 'coef_'):
        
        importance = np.abs(model.coef_[0])
    
    
    else:
        
        print("Feature importance not available")
        
        return
    
    
    # Create dataframe
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    
    # Sort values
    feature_importance_df = (
        feature_importance_df
        .sort_values(
            by='Importance',
            ascending=False
        )
        .head(10)
    )
    
    
    # Plot
    plt.figure(figsize=(10,6))
    
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feature_importance_df
    )
    
    plt.title("Top 10 Feature Importances")
    
    
    # Save figure
    plt.savefig(
        "reports/feature_importance.png",
        bbox_inches='tight'
    )
    
    plt.show()
    
    print("\nFeature importance plot saved successfully")


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    
    # Dataset path
    data_path = (
        "data/feature_engineered_churn.csv"
    )
    
    # Model path
    model_path = (
        "models/best_churn_model.pkl"
    )
    
    
    # Load dataset
    df = load_data(data_path)
    
    print("Dataset Loaded Successfully")
    
    
    # Prepare data
    X, y = prepare_data(df)
    
    
    # Train-test split
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )
    
    
    # Scale data
    X_train_scaled, X_test_scaled = (
        scale_data(X_train, X_test)
    )
    
    
    # Load trained model
    model = joblib.load(model_path)
    
    print("Model Loaded Successfully")
    
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    y_prob = model.predict_proba(
        X_test_scaled
    )[:,1]
    
    
    # Classification report
    report = classification_report(
        y_test,
        y_pred
    )
    
    print("\n==========================")
    print("CLASSIFICATION REPORT")
    print("==========================\n")
    
    print(report)
    
    
    # Save report
    save_classification_report(report)
    
    
    # Confusion Matrix
    plot_confusion_matrix(
        y_test,
        y_pred
    )
    
    
    # ROC Curve
    plot_roc_curve(
        y_test,
        y_prob
    )
    
    
    # Feature Importance
    plot_feature_importance(
        model,
        X.columns
    )
    
    
    print("\nEvaluation Completed Successfully")
    
    print("\nReports saved in reports/ folder")