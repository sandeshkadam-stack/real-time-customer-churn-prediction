import streamlit as st
import pandas as pd
import json
import os
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Real-Time Churn Dashboard",
    layout="wide"
)

st.title("📊 Real-Time Customer Churn Dashboard")

# Auto refresh every 5 sec
st_autorefresh(interval=5000, key="refresh")

file_path = "../data/churn_predictions.json"

if os.path.exists(file_path):

    try:

        with open(file_path, "r") as f:
            data = json.load(f)

    except:
        data = []

else:
    data = []

if len(data) > 0:

    df = pd.DataFrame(data)

    # Latest records first
    df = df.iloc[::-1]

    st.dataframe(
        df,
        use_container_width=True
    )

    churn_count = (
        df["prediction"]
        == "Likely to Churn"
    ).sum()

    stay_count = (
        df["prediction"]
        == "Will Stay"
    ).sum()

    col1, col2 = st.columns(2)

    col1.metric(
        "Customers Likely To Churn",
        int(churn_count)
    )

    col2.metric(
        "Customers Staying",
        int(stay_count)
    )

else:

    st.warning("No predictions available yet.")