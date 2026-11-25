
import pandas as pd
import numpy as np
import joblib
import time

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
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

# Create polynomial age feature
train["age_squared"] = train["age"] ** 2

validation["age_squared"] = validation["age"] ** 2

print("Age squared feature created!")

# Numerical features
numerical_features = [
    "age",
    "age_squared",
    "goals",
    "assists",
    "minutes",
    "appearances",
    "goals_per_90",
    "assists_per_90",
    "goal_contributions",
    "contributions_per_90",
    "minutes_per_appearance"
]

# Categorical features
categorical_features = [
    "position",
    "last_club_id"
]

features = numerical_features + categorical_features

# Separate inputs and target
X_train = train[features]
y_train = train["market_value"]

X_val = validation[features]
y_val = validation["market_value"]

# Create updated preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# Fit using training data only
X_train = preprocessor.fit_transform(X_train)

X_val = preprocessor.transform(X_val)

# Transform target
y_train_log = np.log1p(y_train)

# Train Linear Regression
model = LinearRegression()

print("\nTraining Polynomial Log Linear Regression...")

start_time = time.time()

model.fit(X_train, y_train_log)

training_time = time.time() - start_time

# Generate predictions
predictions_log = model.predict(X_val)

predictions = np.expm1(predictions_log)

predictions = np.maximum(predictions, 0)

# Evaluation
mae = mean_absolute_error(y_val, predictions)

rmse = np.sqrt(
    mean_squared_error(y_val, predictions)
)

r2 = r2_score(y_val, predictions)

print("\nPOLYNOMIAL REGRESSION RESULTS")

print(f"MAE: EUR {mae:,.2f}")
print(f"RMSE: EUR {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")
print(f"Training time: {training_time:.2f} seconds")
print("Zero predictions:", (predictions == 0).sum())

# Compare all three experiments
results = pd.read_csv(
    "data/processed/regression_comparison.csv"
)

# Prevent duplicate results when running again
results = results[
    results["Model"] != "Polynomial Log Linear Regression"
]

new_result = pd.DataFrame([{
    "Model": "Polynomial Log Linear Regression",
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

# Sample predictions
comparison = pd.DataFrame({
    "Player": validation["name"],
    "Actual": y_val,
    "Predicted": predictions
})

comparison["Error"] = abs(
    comparison["Actual"] - comparison["Predicted"]
)

print("\nSAMPLE PREDICTIONS")
print(comparison.head(15).round(2).to_string(index=False))

# Save the new model
joblib.dump(
    model,
    "models/polynomial_log_regression.joblib"
)

# Save the new preprocessor separately
joblib.dump(
    preprocessor,
    "models/polynomial_preprocessor.joblib"
)

# Save results
results.to_csv(
    "data/processed/regression_comparison.csv",
    index=False
)

comparison.to_csv(
    "data/processed/polynomial_predictions.csv",
    index=False
)

print("\nModel and preprocessor saved successfully!")
print("Polynomial Regression completed!")
