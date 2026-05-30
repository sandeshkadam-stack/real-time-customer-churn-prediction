import mlflow
import mlflow.sklearn

import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv(
    "../data/cleaned_customer_churn.csv"
)

# ============================================
# Preprocessing
# ============================================

df.columns = df.columns.str.strip()

X = df.drop(["customerID", "Churn", "Churn_Label"], axis=1)

X = pd.get_dummies(X, drop_first=True)

y = df["Churn_Label"]

# ============================================
# Train-Test Split
# ============================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

# ============================================
# MLflow Experiment
# ============================================

mlflow.set_experiment(
    "Customer_Churn_Experiment"
)

with mlflow.start_run():

    # ========================================
    # Hyperparameters
    # ========================================

    n_estimators = 100

    max_depth = 10

    # ========================================
    # Model
    # ========================================

    model = RandomForestClassifier(

        n_estimators=n_estimators,

        max_depth=max_depth,

        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # ========================================
    # Metrics
    # ========================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    # ========================================
    # Log Parameters
    # ========================================

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    # ========================================
    # Log Metrics
    # ========================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    # ========================================
    # Log Model
    # ========================================

    mlflow.sklearn.log_model(
        model,
        "random_forest_model"
    )

    # ========================================
    # Save Local Model
    # ========================================

    joblib.dump(
        model,
        "../models/best_churn_model.pkl"
    )

    joblib.dump(
        list(X_train.columns),
        "../models/training_columns.pkl"
    )

    print("MLflow run completed.")