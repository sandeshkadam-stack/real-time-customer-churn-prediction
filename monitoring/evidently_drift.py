import pandas as pd
import json
import os

from evidently.report import Report

from evidently.metric_preset import (
    DataDriftPreset
)

# ============================================
# Load Baseline Dataset
# ============================================

reference_data = pd.read_csv(
    "data/cleaned_customer_churn.csv"
)

# ============================================
# Load Live Predictions
# ============================================

prediction_file = (
    "data/churn_predictions.json"
)

if not os.path.exists(prediction_file):

    print("Prediction file not found.")
    exit()

with open(prediction_file, "r") as f:

    predictions = json.load(f)

customer_data = [
    x["customer_data"]
    for x in predictions
    if "customer_data" in x
]

current_data = pd.DataFrame(customer_data)

# ============================================
# Match Columns
# ============================================

common_columns = list(
    set(reference_data.columns)
    &
    set(current_data.columns)
)

reference_data = reference_data[
    common_columns
]

current_data = current_data[
    common_columns
]

# ============================================
# Generate Drift Report
# ============================================

report = Report(
    metrics=[
        DataDriftPreset()
    ]
)

report.run(
    reference_data=reference_data,
    current_data=current_data
)

# ============================================
# Save HTML Report
# ============================================

output_path = (
    "monitoring/evidently_reports/drift_report.html"
)

os.makedirs( "monitoring/evidently_reports", exist_ok=True )


report.save_html(output_path)

print(
    f"Drift report saved at {output_path}"
)