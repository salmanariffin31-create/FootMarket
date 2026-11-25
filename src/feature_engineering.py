import pandas as pd

# Load our dataset
df = pd.read_csv(
    "data/processed/footmarket_training_data.csv"
)

print("Dataset loaded successfully!")

# Create new performance features
df["goals_per_90"] = (
    df["goals"] / df["minutes"]
) * 90

df["assists_per_90"] = (
    df["assists"] / df["minutes"]
) * 90

df["goal_contributions"] = (
    df["goals"] + df["assists"]
)

df["contributions_per_90"] = (
    df["goal_contributions"] / df["minutes"]
) * 90

df["minutes_per_appearance"] = (
    df["minutes"] / df["appearances"]
)

print("\nNEW FEATURES")

columns = [
    "name",
    "goals_per_90",
    "assists_per_90",
    "goal_contributions",
    "contributions_per_90",
    "minutes_per_appearance"
]

print(df[columns].head(10).to_string(index=False))

print("\nFEATURE STATISTICS")
print(df[columns[1:]].describe().to_string())

# Save the enhanced dataset
df.to_csv(
    "data/processed/footmarket_features.csv",
    index=False
)

print("\nDataset shape:", df.shape)
print("Feature engineering completed!")


print("\nPLAYING TIME ANALYSIS")

print("Total records:", len(df))

print("Below 90 minutes:",
      (df["minutes"] < 90).sum())

print("Below 180 minutes:",
      (df["minutes"] < 180).sum())

print("Below 450 minutes:",
      (df["minutes"] < 450).sum())

print("Below 900 minutes:",
      (df["minutes"] < 900).sum())

print("\nEXTREME PERFORMANCE RECORDS")

print(
    df.nlargest(10, "goals_per_90")[
        [
            "name",
            "goals",
            "minutes",
            "appearances",
            "goals_per_90"
        ]
    ].to_string(index=False)
)

print("\nFINAL DATASET FILTERING")

original_count = len(df)

# Keep records with at least 90 minutes played
df_final = df[df["minutes"] >= 90].copy()

print("Original records:", original_count)
print("Remaining records:", len(df_final))
print("Removed records:", original_count - len(df_final))

# Validate the filtered dataset
print("\nFINAL VALIDATION")

print("Missing values:", df_final.isnull().sum().sum())

print("Duplicate records:", df_final.duplicated(
    subset=["player_id", "valuation_date"]
).sum())

print("Minimum minutes:", df_final["minutes"].min())
print("Maximum goals per 90:", df_final["goals_per_90"].max())
print("Maximum assists per 90:", df_final["assists_per_90"].max())
print("Unique players:", df_final["player_id"].nunique())

# Save final dataset
df_final.to_csv(
    "data/processed/footmarket_final.csv",
    index=False
)

print("\nFinal dataset shape:", df_final.shape)
print("Final dataset saved successfully!")