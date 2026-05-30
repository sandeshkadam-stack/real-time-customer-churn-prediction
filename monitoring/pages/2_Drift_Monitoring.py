import streamlit as st
import streamlit.components.v1 as components
import os

st.title("📈 Drift Monitoring")

report_path = (
    "monitoring/evidently_reports/drift_report.html"
)

if os.path.exists(report_path):

    with open(report_path, "r") as f:

        html_content = f.read()

    components.html(
        html_content,
        height=1000,
        scrolling=True
    )

else:

    st.warning(
        "Drift report not found. "
        "Please generate Evidently report first."
    )