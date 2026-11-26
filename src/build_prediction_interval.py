"""Build data/processed/prediction_interval.json.

Run from the project root:  python src/build_prediction_interval.py

For each predicted-value tier, stores the 10th and 90th percentile of
(actual / predicted) on the VALIDATION period (2023-2024). The app multiplies
a new prediction by these ratios to show a typical range. The test period is
only used to report how often the range held (coverage), never to build it.
"""

import json

import joblib
import numpy as np
import pandas as pd

VALIDATION_START, TEST_START = "2023-01-01", "2025-01-01"
EDGES = [0, 1e6, 5e6, 20e6, np.inf]
LABELS = ["under 1M", "1M-5M", "5M-20M", "over 20M"]

model = joblib.load("models/weighted_log_regression.joblib")
preprocessor = joblib.load("models/polynomial_preprocessor.joblib")

df = pd.read_csv(
    "data/processed/footmarket_final.csv",
    dtype={"last_club_id": str},
    parse_dates=["valuation_date"],
)
df["age_squared"] = df["age"] ** 2

df["predicted"] = np.maximum(
    np.expm1(model.predict(preprocessor.transform(df[preprocessor.feature_names_in_]))),
    0,
).clip(min=1)
df["ratio"] = df["market_value"] / df["predicted"]
df["tier"] = pd.cut(df["predicted"], EDGES, labels=LABELS)

validation = df[(df.valuation_date >= VALIDATION_START) & (df.valuation_date < TEST_START)]
test = df[df.valuation_date >= TEST_START]

bins = []
for i, label in enumerate(LABELS):
    low, high = validation.loc[validation.tier == label, "ratio"].quantile([0.1, 0.9])
    t = test.loc[test.tier == label, "ratio"]
    print(f"{label:>9}: {low:.2f}x - {high:.2f}x | test coverage {((t >= low) & (t <= high)).mean():.1%}")
    bins.append({
        "label": label,
        "upper": None if np.isinf(EDGES[i + 1]) else float(EDGES[i + 1]),
        "low_ratio": round(float(low), 4),
        "high_ratio": round(float(high), 4),
    })

with open("data/processed/prediction_interval.json", "w", encoding="utf-8") as f:
    json.dump(
        {"method": "10th/90th percentile of actual/predicted on validation (2023-2024)",
         "bins": bins},
        f, indent=2,
    )
print("Saved data/processed/prediction_interval.json")
