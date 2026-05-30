from kafka import KafkaConsumer
import json
import joblib
import pandas as pd
import os
import time

from prometheus_client import (
    start_http_server,
    Counter,
    Gauge
)

from sqlalchemy import create_engine

# ============================================
# Load ML Model
# ============================================

model = joblib.load(
    "models/best_churn_model.pkl"
)

training_columns = joblib.load(
    "models/training_columns.pkl"
)

# ============================================
# Prometheus Metrics
# ============================================

prediction_counter = Counter(
    'predictions_total',
    'Total predictions made'
)

churn_gauge = Gauge(
    'churn_probability',
    'Current churn probability'
)

# ============================================
# Start Prometheus Metrics Server
# ============================================

start_http_server(8000)

print("Prometheus metrics running on port 8000")

# ============================================
# PostgreSQL Connection
# ============================================

engine = create_engine(
    "postgresql+psycopg2://postgres:Sandesh!1@postgres:5432/churn_monitoring"
)

print("Connected to PostgreSQL successfully.")

# ============================================
# Output JSON File
# ============================================

file_path = "data/churn_predictions.json"

# Create file if not exists

if not os.path.exists(file_path):

    os.makedirs(
        os.path.dirname(file_path),
        exist_ok=True
    )

    with open(file_path, "w") as f:
        json.dump([], f)

# ============================================
# Kafka Consumer
# ============================================

consumer = KafkaConsumer(

    'customer-churn',

    bootstrap_servers='kafka:9092',

    auto_offset_reset='earliest',

    enable_auto_commit=True,

    group_id='churn-group',

    value_deserializer=lambda x:
    json.loads(x.decode('utf-8'))
)

print("Listening for customer events...")

# ============================================
# Consume Messages
# ============================================

for message in consumer:

    try:

        # ============================================
        # Read Kafka Event
        # ============================================

        data = message.value

        print("\nReceived Event:")
        print(data)

        # ============================================
        # Convert to DataFrame
        # ============================================

        df = pd.DataFrame([data])

        # ============================================
        # One-Hot Encoding
        # ============================================

        df = pd.get_dummies(df)

        # ============================================
        # Match Training Columns
        # ============================================

        missing_cols = (
            set(training_columns)
            - set(df.columns)
        )

        for col in missing_cols:
            df[col] = 0

        df = df[training_columns]

        # ============================================
        # Prediction
        # ============================================

        prediction = model.predict(df)[0]

        probability = (
            model.predict_proba(df)[0][1]
        )

        result = (
            "Likely to Churn"
            if prediction == 1
            else "Will Stay"
        )

        print("Prediction:", result)

        print(
            "Probability:",
            round(probability, 2)
        )

        # ============================================
        # Update Prometheus Metrics
        # ============================================

        prediction_counter.inc()

        churn_gauge.set(probability)

        # ============================================
        # Create Output Record
        # ============================================

        output = {

            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "customerID": data.get(
                "customerID",
                "Unknown"
            ),

            "prediction": result,

            "probability": round(
                float(probability),
                2
            ),

            "customer_data": data
        }

        # ============================================
        # Save Prediction to PostgreSQL
        # ============================================

        try:

            sql_df = pd.DataFrame([{

                "timestamp": output["timestamp"],

                "customer_id": output["customerID"],

                "gender": data.get("gender"),

                "senior_citizen": data.get(
                    "SeniorCitizen"
                ),

                "tenure": data.get("tenure"),

                "monthly_charges": data.get(
                    "MonthlyCharges"
                ),

                "contract": data.get("Contract"),

                "prediction": output["prediction"],

                "probability": output["probability"]
            }])

            sql_df.to_sql(

                "churn_predictions",

                engine,

                if_exists="append",

                index=False
            )

            print(
                "Prediction saved to PostgreSQL."
            )

        except Exception as sql_error:

            print(
                f"SQL Error: {str(sql_error)}"
            )

        # ============================================
        # Read Existing JSON Predictions
        # ============================================

        try:

            if (
                os.path.exists(file_path)
                and os.path.getsize(file_path) > 0
            ):

                with open(file_path, "r") as f:

                    predictions = json.load(f)

            else:

                predictions = []

        except json.JSONDecodeError:

            print(
                "Invalid JSON detected. Resetting file..."
            )

            predictions = []

        # ============================================
        # Append New Prediction
        # ============================================

        predictions.append(output)

        # Keep only latest 100 records

        predictions = predictions[-100:]

        # ============================================
        # Save Updated JSON
        # ============================================

        with open(file_path, "w") as f:

            json.dump(
                predictions,
                f,
                indent=4
            )

        print(
            "Prediction saved to JSON backup."
        )

    except Exception as e:

        print(
            f"Error processing message: {str(e)}"
        )