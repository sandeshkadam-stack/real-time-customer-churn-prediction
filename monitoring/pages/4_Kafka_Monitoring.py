import streamlit as st
import pandas as pd

from db_connection import engine

st.title("🛰 Kafka Monitoring")

# ============================================
# Load Kafka-Processed Predictions
# ============================================

query = """
SELECT *
FROM churn_predictions
ORDER BY timestamp DESC
"""

df = pd.read_sql(query, engine)

# ============================================
# Summary Metrics
# ============================================

total_messages = len(df)

churn_predictions = len(
    df[df["prediction"] == "Likely to Churn"]
)

stay_predictions = len(
    df[df["prediction"] == "Will Stay"]
)

avg_probability = round(
    df["probability"].mean(),
    2
)

# ============================================
# Dashboard Metrics
# ============================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Messages Processed",
        total_messages
    )

with col2:
    st.metric(
        "Churn Predictions",
        churn_predictions
    )

with col3:
    st.metric(
        "Stay Predictions",
        stay_predictions
    )

with col4:
    st.metric(
        "Avg Churn Probability",
        avg_probability
    )

# ============================================
# Recent Kafka Events
# ============================================

st.subheader("Latest Processed Events")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# ============================================
# Prediction Distribution
# ============================================

st.subheader("Prediction Distribution")

prediction_counts = (
    df["prediction"]
    .value_counts()
)

st.bar_chart(
    prediction_counts
)

# ============================================
# Churn Probability Trend
# ============================================

st.subheader("Churn Probability Trend")

trend_df = (
    df.sort_values("timestamp")
)

st.line_chart(
    trend_df.set_index("timestamp")[
        "probability"
    ]
)