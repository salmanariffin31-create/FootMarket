
import pandas as pd
import numpy as np
import joblib
import time

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

print("Datasets loaded successfully!")

# Select features
features = [
    "age",
    "goals",
    "assists",
    "minutes",
    "appearances",
    "goals_per_90",
    "assists_per_90",
    "goal_contributions",
    "contributions_per_90",
    "minutes_per_appearance",
    "position",
    "last_club_id"
]

X_train = train[features]
y_train = train["market_value"]

X_val = validation[features]
y_val = validation["market_value"]

# Load fitted preprocessor
preprocessor = joblib.load(
    "models/preprocessor.joblib"
)

X_train = preprocessor.transform(X_train)
X_val = preprocessor.transform(X_val)

# Apply logarithmic transformation
y_train_log = np.log1p(y_train)

# Initialise Linear Regression
model = LinearRegression()

print("\nTraining Log-Transformed Linear Regression...")

start_time = time.time()

model.fit(X_train, y_train_log)

training_time = time.time() - start_time

# Predict logarithmic market values
predictions_log = model.predict(X_val)

# Convert predictions back into euros
predictions = np.expm1(predictions_log)

# Prevent negative predictions
predictions = np.maximum(predictions, 0)

# Evaluate model
mae = mean_absolute_error(y_val, predictions)

rmse = np.sqrt(
    mean_squared_error(y_val, predictions)
)

r2 = r2_score(y_val, predictions)

print("\nLOG LINEAR REGRESSION RESULTS")

print(f"MAE: EUR {mae:,.2f}")
print(f"RMSE: EUR {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"Training time: {training_time:.2f} seconds")

print("Zero predictions:", (predictions == 0).sum())

# Compare predictions
comparison = pd.DataFrame({
    "Player": validation["name"],
    "Actual": y_val,
    "Predicted": predictions
})

comparison["Error"] = abs(
    comparison["Actual"] - comparison["Predicted"]
)

print("\nSAMPLE PREDICTIONS")

print(
    comparison.head(15).round(2).to_string(index=False)
)

# Compare both models
results = pd.DataFrame([
    {
        "Model": "Original Linear Regression",
        "MAE": 3625109.15,
        "RMSE": 8150867.95,
        "R2": 0.5707
    },
    {
        "Model": "Log Linear Regression",
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
])

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

# Save trained model
joblib.dump(
    model,
    "models/log_linear_regression.joblib"
)

# Save predictions
comparison.to_csv(
    "data/processed/log_regression_predictions.csv",
    index=False
)

# Save comparison results
results.to_csv(
    "data/processed/regression_comparison.csv",
    index=False
)

print("\nModel saved successfully!")
print("Log Linear Regression completed!")
