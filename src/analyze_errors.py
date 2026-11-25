import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load validation dataset
df = pd.read_csv(
    "data/processed/splits/validation.csv",
    dtype={"last_club_id": str}
)

# Create age squared feature
df["age_squared"] = df["age"] ** 2

# Load our V3 model
model = joblib.load(
    "models/polynomial_log_regression.joblib"
)

preprocessor = joblib.load(
    "models/polynomial_preprocessor.joblib"
)

# Generate predictions
X = df[preprocessor.feature_names_in_]
X_processed = preprocessor.transform(X)

df["predicted"] = np.maximum(
    np.expm1(model.predict(X_processed)), 0
)

# Calculate errors
df["absolute_error"] = abs(
    df["market_value"] - df["predicted"]
)

# Analyse performance by position
print("\nPERFORMANCE BY POSITION")

for position, group in df.groupby("position"):

    actual = group["market_value"]
    predicted = group["predicted"]

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{position}")
    print("Records:", len(group))
    print(f"Average value: EUR {actual.mean():,.2f}")
    print(f"MAE: EUR {mae:,.2f}")
    print(f"RMSE: EUR {rmse:,.2f}")
    print(f"R2: {r2:.4f}")
    print(f"MAE / Average value: {mae / actual.mean():.2%}")

# Analyse performance by market value
df["value_category"] = pd.cut(
    df["market_value"],
    bins=[
        0, 500000, 1000000, 5000000,
        10000000, 50000000, np.inf
    ],
    labels=[
        "Up to 500K",
        "500K - 1M",
        "1M - 5M",
        "5M - 10M",
        "10M - 50M",
        "Above 50M"
    ]
)

summary = df.groupby(
    "value_category",
    observed=True
).agg(
    records=("market_value", "count"),
    average_actual=("market_value", "mean"),
    average_predicted=("predicted", "mean"),
    mae=("absolute_error", "mean")
)

print("\nPERFORMANCE BY MARKET VALUE")
print(summary.round(2).to_string())

# Find biggest prediction errors
print("\nTOP 10 PREDICTION ERRORS")

print(
    df.nlargest(10, "absolute_error")[
        [
            "name",
            "position",
            "age",
            "market_value",
            "predicted",
            "absolute_error"
        ]
    ].round(2).to_string(index=False)
)