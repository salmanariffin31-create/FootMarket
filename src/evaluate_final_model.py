
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load untouched test dataset
test = pd.read_csv(
    "data/processed/splits/test.csv",
    dtype={"last_club_id": str}
)

print("Test dataset loaded:", test.shape)

# Create age squared feature
test["age_squared"] = test["age"] ** 2

# Load our selected V4 model
model = joblib.load(
    "models/weighted_log_regression.joblib"
)

# Load corresponding preprocessor
preprocessor = joblib.load(
    "models/polynomial_preprocessor.joblib"
)

# Prepare input features
features = preprocessor.feature_names_in_

X_test = test[features]
y_test = test["market_value"]

X_test_processed = preprocessor.transform(X_test)

# Generate predictions
predictions_log = model.predict(X_test_processed)

predictions = np.maximum(
    np.expm1(predictions_log), 0
)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

r2 = r2_score(y_test, predictions)

print("\nFINAL TEST RESULTS")
print("Test records:", len(test))
print(f"MAE: EUR {mae:,.2f}")
print(f"RMSE: EUR {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")
print("Zero predictions:", (predictions == 0).sum())

# Compare validation and test performance
results = pd.DataFrame([
    {
        "Dataset": "Validation",
        "MAE": 2782120.06,
        "RMSE": 6874155.06,
        "R2": 0.694651
    },
    {
        "Dataset": "Test",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
])

print("\nVALIDATION VS TEST")
print(results.to_string(index=False))

# Save player predictions
comparison = pd.DataFrame({
    "Player": test["name"],
    "Valuation Date": test["valuation_date"],
    "Position": test["position"],
    "Actual": y_test,
    "Predicted": predictions
})

comparison["Absolute Error"] = abs(
    comparison["Actual"] - comparison["Predicted"]
)

comparison.to_csv(
    "data/processed/final_test_predictions.csv",
    index=False
)

results.to_csv(
    "data/processed/final_evaluation.csv",
    index=False
)

# Create reports folder
Path("reports/figures").mkdir(
    parents=True,
    exist_ok=True
)

# Actual vs predicted scatter plot
plt.figure(figsize=(9, 7))

plt.scatter(
    y_test / 1000000,
    predictions / 1000000,
    alpha=0.25,
    s=10
)

maximum = max(y_test.max(), predictions.max()) / 1000000

plt.plot(
    [0, maximum],
    [0, maximum],
    "r--",
    label="Perfect Prediction"
)

plt.title("FootMarket - Actual vs Predicted Market Value")
plt.xlabel("Actual Market Value (EUR Millions)")
plt.ylabel("Predicted Market Value (EUR Millions)")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

plt.savefig(
    "reports/figures/actual_vs_predicted.png",
    dpi=300
)

print("\nEvaluation results saved successfully!")
print("Final model evaluation completed!")

plt.show()
