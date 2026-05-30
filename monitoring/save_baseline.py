import pandas as pd
import json

# ============================================
# Load Original Training Dataset
# ============================================

df = pd.read_csv(
    "../data/cleaned_customer_churn.csv"
)

# ============================================
# Select Numerical Columns
# ============================================

numerical_cols = [
    "tenure",
    "MonthlyCharges"
]

# ============================================
# Compute Baseline Statistics
# ============================================

baseline = {}

for col in numerical_cols:

    baseline[col] = {

        "mean": float(df[col].mean()),

        "std": float(df[col].std()),

        "min": float(df[col].min()),

        "max": float(df[col].max())
    }

# ============================================
# Save Baseline
# ============================================

with open(
    "baseline_stats.json",
    "w"
) as f:

    json.dump(
        baseline,
        f,
        indent=4
    )

print("Baseline statistics saved.")