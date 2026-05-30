import json
import pandas as pd
import os

# ============================================
# Load Baseline Statistics
# ============================================

with open(
    "baseline_stats.json",
    "r"
) as f:

    baseline = json.load(f)

# ============================================
# Load Predictions
# ============================================

prediction_file = "../data/churn_predictions.json"

if not os.path.exists(prediction_file):

    print("No prediction data found.")
    exit()

with open(prediction_file, "r") as f:

    predictions = json.load(f)

# ============================================
# Extract Customer Data
# ============================================

customer_data = [
    x["customer_data"]
    for x in predictions
    if "customer_data" in x
]

df = pd.DataFrame(customer_data)

# ============================================
# Drift Detection
# ============================================

print("\n========== DRIFT REPORT ==========\n")

for col in ["tenure", "MonthlyCharges"]:

    live_mean = df[col].mean()

    baseline_mean = baseline[col]["mean"]

    difference_percent = abs(
        live_mean - baseline_mean
    ) / baseline_mean * 100

    print(f"{col}")

    print(
        f"Baseline Mean: {baseline_mean:.2f}"
    )

    print(
        f"Live Mean: {live_mean:.2f}"
    )

    print(
        f"Difference: {difference_percent:.2f}%"
    )

    # ============================================
    # Threshold-Based Drift Alert
    # ============================================

    if difference_percent > 20:

        print("🚨 DRIFT DETECTED")

    else:

        print("✅ Normal")

    print()

    # ============================================
# Prediction Distribution
# ============================================

prediction_df = pd.DataFrame(predictions)

churn_percent = (
    (
        prediction_df["prediction"]
        == "Likely to Churn"
    ).mean()
) * 100

print("\n========== PREDICTION MONITORING ==========\n")

print(
    f"Churn Prediction Rate: "
    f"{churn_percent:.2f}%"
)

# ============================================
# Alert if predictions become abnormal
# ============================================

if churn_percent > 80:

    print("🚨 Too many churn predictions")

elif churn_percent < 5:

    print("🚨 Too few churn predictions")

else:

    print("✅ Prediction distribution normal")