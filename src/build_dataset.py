
import duckdb
import pandas as pd
from pathlib import Path

# Connect to the original database
connection = duckdb.connect(
    "data/raw/transfermarkt-datasets.duckdb",
    read_only=True
)

# Create processed data folder
Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

print("FootMarket dataset builder ready!")
print("Extracting training dataset...")

# Build the training dataset
query = """
SELECT
    p.player_id,
    p.name,
    v.date AS valuation_date,
    p.position,

    DATE_DIFF('year', p.date_of_birth, v.date)
    - CASE
        WHEN STRFTIME(v.date, '%m-%d')
             < STRFTIME(p.date_of_birth, '%m-%d')
        THEN 1
        ELSE 0
      END AS age,

    ARG_MAX(a.player_club_id, a.date) AS last_club_id,

    SUM(a.goals) AS goals,
    SUM(a.assists) AS assists,
    SUM(a.minutes_played) AS minutes,
    COUNT(a.appearance_id) AS appearances,

    v.market_value_in_eur AS market_value

FROM player_valuations v

JOIN players p
    ON v.player_id = p.player_id

JOIN appearances a
    ON v.player_id = a.player_id
    AND a.date >= v.date - INTERVAL 365 DAY
    AND a.date < v.date

WHERE v.market_value_in_eur > 0
AND v.date >= '2013-07-03'
AND p.date_of_birth IS NOT NULL
AND p.position IN (
    'Attack',
    'Midfield',
    'Defender',
    'Goalkeeper'
)

GROUP BY
    p.player_id,
    p.name,
    p.position,
    p.date_of_birth,
    v.date,
    v.market_value_in_eur

ORDER BY v.date
"""

# Convert SQL results into a Pandas DataFrame
df = connection.execute(query).df()

print("\nExtraction completed!")
print("Dataset shape:", df.shape)

# Display first 10 records
print("\nFIRST 10 RECORDS")
print(df.head(10).to_string(index=False))

# Basic dataset validation
print("\nDATASET VALIDATION")

print("Total rows:", len(df))

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate player-date records:")
print(df.duplicated(
    subset=["player_id", "valuation_date"]
).sum())

print("\nMinimum age:", df["age"].min())
print("Maximum age:", df["age"].max())
print("Minimum appearances:", df["appearances"].min())

# Export dataset to CSV
df.to_csv(
    "data/processed/footmarket_training_data.csv",
    index=False
)

print("\nDataset saved successfully!")
print("Location: data/processed/footmarket_training_data.csv")

# Close database connection
connection.close()
