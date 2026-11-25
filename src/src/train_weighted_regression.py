import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load datasets
train = pd.read_csv(
    "data/processed/splits/train.csv",
    dtype={"last_club_id": str}
)

validation = pd.read_csv(
    "data/processed/splits/validation.csv",
    dtype={"last_club_id": str}
)

# Create age squared feature
train["age_squared"] = train["age"] ** 2
validation["age_squared"] = validation["age"] ** 2

# Load V3 preprocessor
preprocessor = joblib.load(
    "models/polynomial_preprocessor.joblib"
)

features = preprocessor.feature_names_in_

X_train = preprocessor.transform(train[features])
X_val = preprocessor.transform(validation[features])

y_train = train["market_value"]
y_val = validation["market_value"]

# Log transformation
y_train_log = np.log1p(y_train)

# Give higher-valued training records more weight
weights = (y_train / y_train.median()) ** 0.25

# Limit extreme weights
weights = np.clip(weights, 0.5, 5)

print("Minimum weight:", weights.min())
print("Maximum weight:", weights.max())

# Train weighted Linear Regression
model = LinearRegression()

print("\nTraining Weighted Linear Regression...")

model.fit(
    X_train,
    y_train_log,
    sample_weight=weights
)

# Make predictions
predictions = np.maximum(
    np.expm1(model.predict(X_val)),
    0
)

# Evaluate model
mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
r2 = r2_score(y_val, predictions)

print("\nWEIGHTED REGRESSION RESULTS")
print(f"MAE: EUR {mae:,.2f}")
print(f"RMSE: EUR {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# Evaluate expensive players
mask = y_val > 50000000

print("\nEXPENSIVE PLAYER ANALYSIS")
print("Records:", mask.sum())
print(f"Average actual: EUR {y_val[mask].mean():,.2f}")
print(f"Average predicted: EUR {predictions[mask].mean():,.2f}")

print(
    f"MAE: EUR "
    f"{mean_absolute_error(y_val[mask], predictions[mask]):,.2f}"
)

# Compare with previous models
results = pd.read_csv(
    "data/processed/regression_comparison.csv"
)

results = results[
    results["Model"] != "Weighted Log Linear Regression"
]

new_result = pd.DataFrame([{
    "Model": "Weighted Log Linear Regression",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2
}])

results = pd.concat(
    [results, new_result],
    ignore_index=True
)

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

# Save model
joblib.dump(
    model,
    "models/weighted_log_regression.joblib"
)

results.to_csv(
    "data/processed/regression_comparison.csv",
    index=False
)

print("\nWeighted Linear Regression completed!")