import streamlit as st
import pandas as pd
import json
import time
import os
import matplotlib.pyplot as plt

# ============================================
# Streamlit Page Config
# ============================================

st.set_page_config(
    page_title="ML Monitoring Dashboard",
    layout="wide"
)

st.title("📈 Real-Time ML Monitoring Dashboard")

# ============================================
# File Paths
# ============================================

PREDICTION_FILE = "../data/churn_predictions.json"

BASELINE_FILE = "baseline_stats.json"

# ============================================
# Load Baseline Statistics
# ============================================

with open(BASELINE_FILE, "r") as f:

    baseline = json.load(f)

# ============================================
# Auto Refresh Loop
# ============================================

placeholder = st.empty()

while True:

    with placeholder.container():

        # ============================================
        # Check Prediction File
        # ============================================

        if not os.path.exists(PREDICTION_FILE):

            st.warning(
                "No prediction file found."
            )

            time.sleep(5)

            continue

        # ============================================
        # Load Predictions
        # ============================================

        with open(PREDICTION_FILE, "r") as f:

            predictions = json.load(f)

        if len(predictions) == 0:

            st.warning("No predictions available.")

            time.sleep(5)

            continue

        # ============================================
        # Extract Customer Data
        # ============================================

        customer_data = [
            x["customer_data"]
            for x in predictions
            if "customer_data" in x
        ]

        df = pd.DataFrame(customer_data)

        prediction_df = pd.DataFrame(predictions)

        # ============================================
        # Metrics Section
        # ============================================

        st.subheader("📊 Prediction Metrics")

        col1, col2, col3 = st.columns(3)

        churn_count = (
            prediction_df["prediction"]
            == "Likely to Churn"
        ).sum()

        stay_count = (
            prediction_df["prediction"]
            == "Will Stay"
        ).sum()

        churn_percent = (
            churn_count / len(prediction_df)
        ) * 100

        col1.metric(
            "Total Predictions",
            len(prediction_df)
        )

        col2.metric(
            "Likely To Churn",
            churn_count
        )

        col3.metric(
            "Churn Rate %",
            round(churn_percent, 2)
        )

        # ============================================
        # Drift Monitoring
        # ============================================

        st.subheader("🚨 Drift Monitoring")

        drift_results = []

        for col in [
            "tenure",
            "MonthlyCharges"
        ]:

            live_mean = df[col].mean()

            baseline_mean = baseline[col]["mean"]

            difference_percent = abs(
                live_mean - baseline_mean
            ) / baseline_mean * 100

            status = (
                "🚨 Drift Detected"
                if difference_percent > 20
                else "✅ Normal"
            )

            drift_results.append({

                "Feature": col,

                "Baseline Mean":
                round(baseline_mean, 2),

                "Live Mean":
                round(live_mean, 2),

                "Difference %":
                round(difference_percent, 2),

                "Status": status
            })

        drift_df = pd.DataFrame(drift_results)

        st.dataframe(drift_df)

        # ============================================
        # Prediction Distribution Chart
        # ============================================

        st.subheader("📉 Prediction Distribution")

        prediction_counts = (
            prediction_df["prediction"]
            .value_counts()
        )

        fig, ax = plt.subplots()

        prediction_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_ylabel("Count")

        st.pyplot(fig)

        # ============================================
        # Monthly Charges Trend
        # ============================================

        st.subheader("💰 Monthly Charges Trend")

        fig2, ax2 = plt.subplots()

        df["MonthlyCharges"].plot(
            kind="line",
            ax=ax2
        )

        ax2.set_ylabel("Monthly Charges")

        st.pyplot(fig2)

        # ============================================
        # Recent Predictions
        # ============================================

        st.subheader("🧾 Recent Predictions")

        st.dataframe(
            prediction_df.tail(10)
        )

    time.sleep(5)