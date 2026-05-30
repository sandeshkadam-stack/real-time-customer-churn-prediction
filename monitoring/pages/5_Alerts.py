import streamlit as st
import pandas as pd

from db_connection import engine

st.title("🚨 Alerts")

query = """

SELECT *
FROM churn_predictions
WHERE probability > 0.85
ORDER BY timestamp DESC

"""

df = pd.read_sql(query, engine)

st.metric(
    "High Risk Customers",
    len(df)
)

st.dataframe(df)