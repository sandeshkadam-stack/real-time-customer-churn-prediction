import streamlit as st
import pandas as pd

from db_connection import engine

st.title("📊 Model Metrics")

query = """

SELECT *
FROM churn_predictions

"""

df = pd.read_sql(query, engine)

avg_probability = (
    df["probability"].mean()
)

churn_rate = (
    (
        df["prediction"]
        == "Likely to Churn"
    ).mean()
) * 100

st.metric(
    "Average Churn Probability",
    round(avg_probability, 2)
)

st.metric(
    "Churn Rate %",
    round(churn_rate, 2)
)

st.metric(
    "Total Predictions",
    len(df)
)