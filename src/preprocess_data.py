
import pandas as pd
import joblib
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load final dataset
df = pd.read_csv(
    "data/processed/footmarket_final.csv",
    dtype={"last_club_id": str},
    parse_dates=["valuation_date"]
)

# Sort records chronologically
df = df.sort_values("valuation_date")

print("Dataset loaded:", df.shape)

# Numerical features
numerical_features = [
    "age",
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
target = "market_value"

# Chronological splitting
train = df[df["valuation_date"] < "2023-01-01"]

validation = df[
    (df["valuation_date"] >= "2023-01-01") &
    (df["valuation_date"] < "2025-01-01")
]

test = df[df["valuation_date"] >= "2025-01-01"]

# Separate inputs and target
X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]

X_test = test[features]
y_test = test[target]

# Display split information
print("\nDATASET SPLIT")

for name, dataset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test)
]:
    print(f"\n{name}")
    print("Records:", len(dataset))
    print("Percentage:", round(len(dataset) / len(df) * 100, 2))
    print("Start date:", dataset["valuation_date"].min())
    print("End date:", dataset["valuation_date"].max())

# Build preprocessing transformer
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

# Learn preprocessing parameters from training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Apply the same preprocessing to validation and test
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print("\nPREPROCESSED DATA")

print("Training shape:", X_train_processed.shape)
print("Validation shape:", X_val_processed.shape)
print("Testing shape:", X_test_processed.shape)

# Validate dataset splitting
assert len(train) + len(validation) + len(test) == len(df)

assert train["valuation_date"].max() < validation["valuation_date"].min()

assert validation["valuation_date"].max() < test["valuation_date"].min()

print("\nChronological split validation passed!")

# Create output directories
Path("models").mkdir(exist_ok=True)
Path("data/processed/splits").mkdir(
    parents=True,
    exist_ok=True
)

# Save preprocessing transformer
joblib.dump(
    preprocessor,
    "models/preprocessor.joblib"
)

# Save dataset splits
train.to_csv(
    "data/processed/splits/train.csv",
    index=False
)

validation.to_csv(
    "data/processed/splits/validation.csv",
    index=False
)

test.to_csv(
    "data/processed/splits/test.csv",
    index=False
)

print("\nPreprocessor saved successfully!")
print("Dataset splits saved successfully!")
print("Preprocessing completed!")
