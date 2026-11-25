
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

# Select input features
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

# Separate features and target
X_train = train[features]
y_train = train["market_value"]

X_val = validation[features]
y_val = validation["market_value"]

# Load preprocessor
preprocessor = joblib.load(
    "models/preprocessor.joblib"
)

# Transform input features
X_train = preprocessor.transform(X_train)
X_val = preprocessor.transform(X_val)

print("Preprocessing completed!")

# Initialise Linear Regression
model = LinearRegression()

# Train the model
print("\nTraining Linear Regression...")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print("Training completed!")

# Make predictions
predictions = model.predict(X_val)

# Prevent negative market values
predictions = np.maximum(predictions, 0)

# Evaluate model
mae = mean_absolute_error(y_val, predictions)

rmse = np.sqrt(
    mean_squared_error(y_val, predictions)
)

r2 = r2_score(y_val, predictions)

print("\nLINEAR REGRESSION RESULTS")
print(f"MAE: EUR {mae:,.2f}")
print(f"RMSE: EUR {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"Training time: {training_time:.2f} seconds")

# Compare actual and predicted values
comparison = pd.DataFrame({
    "Player": validation["name"],
    "Actual": y_val,
    "Predicted": predictions
})

comparison["Error"] = abs(
    comparison["Actual"] - comparison["Predicted"]
)

print("\nSAMPLE PREDICTIONS")
print(comparison.head(15).to_string(index=False))

# Save trained model
joblib.dump(
    model,
    "models/linear_regression.joblib"
)

# Save predictions
comparison.to_csv(
    "data/processed/linear_regression_predictions.csv",
    index=False
)

print("\nModel saved successfully!")
print("Linear Regression completed!")
