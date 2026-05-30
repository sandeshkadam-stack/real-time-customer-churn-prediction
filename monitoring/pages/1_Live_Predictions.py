import streamlit as st
import pandas as pd

from db_connection import engine

st.title("📡 Live Predictions")

query = """

SELECT *
FROM churn_predictions
ORDER BY timestamp DESC
LIMIT 20

"""

df = pd.read_sql(query, engine)

st.dataframe(df)