import pandas as pd

# Load our generated dataset
df = pd.read_csv(
    "data/processed/footmarket_training_data.csv"
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Inspect numerical features
columns = [
    "age",
    "goals",
    "assists",
    "minutes",
    "appearances",
    "market_value"
]

print("\nNUMERICAL STATISTICS")
print(df[columns].describe().to_string())

# Check for invalid values
print("\nINVALID VALUES")

print("Negative goals:", (df["goals"] < 0).sum())
print("Negative assists:", (df["assists"] < 0).sum())
print("Negative minutes:", (df["minutes"] < 0).sum())
print("Zero minutes:", (df["minutes"] == 0).sum())
print("Invalid market values:", (df["market_value"] <= 0).sum())

print("\nPOSITION DISTRIBUTION")
print(df["position"].value_counts())

print("\nUNIQUE PLAYERS")
print(df["player_id"].nunique())

print("\nUNIQUE CLUBS")
print(df["last_club_id"].nunique())

print("\nVALUATION DATE RANGE")
print("Earliest:", df["valuation_date"].min())
print("Latest:", df["valuation_date"].max())