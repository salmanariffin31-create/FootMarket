
import duckdb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SOURCE = ROOT / "data/raw/transfermarkt-datasets.duckdb"
DESTINATION = ROOT / "data/deployment/footmarket.duckdb"

DESTINATION.parent.mkdir(parents=True, exist_ok=True)

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

source_path = str(SOURCE).replace("\\", "/").replace("'", "''")

with duckdb.connect(str(DESTINATION)) as conn:

    conn.execute(
        f"ATTACH '{source_path}' AS original (READ_ONLY)"
    )

    # Extract player information
    conn.execute("""
        CREATE OR REPLACE TABLE players AS
        SELECT
            player_id,
            name,
            image_url
        FROM original.players
    """)

    # Extract club information
    conn.execute("""
        CREATE OR REPLACE TABLE clubs AS
        SELECT
            club_id,
            name
        FROM original.clubs
    """)

    player_count = conn.execute("""
        SELECT COUNT(*) FROM players
    """).fetchone()[0]

    club_count = conn.execute("""
        SELECT COUNT(*) FROM clubs
    """).fetchone()[0]

    image_count = conn.execute("""
        SELECT COUNT(*) FROM players
        WHERE image_url IS NOT NULL
    """).fetchone()[0]

    conn.execute("DETACH original")
    conn.execute("CHECKPOINT")

original_size = SOURCE.stat().st_size / (1024 ** 2)
deployment_size = DESTINATION.stat().st_size / (1024 ** 2)

reduction = (
    1 - deployment_size / original_size
) * 100

print("\n=== FOOTMARKET DEPLOYMENT DATABASE ===")

print(f"Players: {player_count:,}")
print(f"Clubs: {club_count:,}")
print(f"Player image URLs: {image_count:,}")

print(f"\nOriginal database: {original_size:.2f} MB")
print(f"Deployment database: {deployment_size:.2f} MB")
print(f"Size reduction: {reduction:.2f}%")

print(f"\nSaved to: {DESTINATION}")
print("Deployment database created successfully!")
