"""Shared helpers for FootMarket: cached loaders, prediction ranges, formatting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

# Chronological split boundaries used when the model was trained
VALIDATION_START = pd.Timestamp("2023-01-01")
TEST_START = pd.Timestamp("2025-01-01")

# Highest value in the dataset; ranges are never drawn above it
MAX_RECORDED_VALUE = 200_000_000


# ---------------------------------------------------------------- data

@st.cache_data
def load_player_data():
    return pd.read_csv(
        ROOT / "data/processed/footmarket_final.csv",
        dtype={"last_club_id": str},
        parse_dates=["valuation_date"],
    )


@st.cache_data
def load_player_list():
    """One row per player, labelled for a searchable dropdown.

    Sorted by each player's highest recorded value, so well-known players
    appear first instead of an alphabetical list starting with "A.J.".
    Built once and cached (previously rebuilt with iterrows on every rerun).
    """
    df = load_player_data()

    players = (
        df.groupby("player_id", as_index=False)
        .agg(name=("name", "first"), peak_value=("market_value", "max"))
        .sort_values("peak_value", ascending=False)
    )
    players["label"] = (
        players["name"].astype(str)
        + " ("
        + players["player_id"].astype(str)
        + ")"
    )
    return players.reset_index(drop=True)


@st.cache_data
def load_club_options(club_ids, club_names):
    """Build {"Club name": club_id}. Appends the ID only for duplicate names."""
    clubs = pd.DataFrame({"club_id": list(club_ids), "name": list(club_names)})
    duplicated = clubs["name"].duplicated(keep=False)
    clubs["label"] = np.where(
        duplicated,
        clubs["name"] + " (" + clubs["club_id"] + ")",
        clubs["name"],
    )
    return dict(zip(clubs["label"], clubs["club_id"]))


@st.cache_data
def load_headline_metrics():
    """Metrics read from the evaluation files, so the UI never drifts."""
    evaluation = pd.read_csv(ROOT / "data/processed/final_evaluation.csv")
    test = evaluation[evaluation["Dataset"] == "Test"].iloc[0]

    dates = pd.read_csv(
        ROOT / "data/processed/footmarket_final.csv",
        usecols=["valuation_date"],
        parse_dates=["valuation_date"],
    )["valuation_date"]

    return {
        "train_records": int((dates < VALIDATION_START).sum()),
        "validation_records": int(
            ((dates >= VALIDATION_START) & (dates < TEST_START)).sum()
        ),
        "test_records": int((dates >= TEST_START).sum()),
        "total_records": int(len(dates)),
        "r2": float(test["R2"]),
        "mae": float(test["MAE"]),
        "rmse": float(test["RMSE"]),
    }


# ---------------------------------------------------------- predictions

def predict_value(model, preprocessor, row):
    """Predict a market value in euros from a one-row DataFrame."""
    data = row.copy()
    if "age_squared" not in data.columns:
        data["age_squared"] = data["age"] ** 2

    processed = preprocessor.transform(data[preprocessor.feature_names_in_])
    return float(max(np.expm1(model.predict(processed)[0]), 0))


@st.cache_data
def load_interval_table():
    path = ROOT / "data/processed/prediction_interval.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prediction_range(prediction):
    """Return (low, high) euros: a typical range around a point estimate.

    The ratios are the 10th and 90th percentiles of actual/predicted values
    on the validation set, calculated separately for each predicted-value
    tier (see src/build_prediction_interval.py). Wide ranges are expected:
    the model's median error is roughly 45% of the recorded value.
    """
    for tier in load_interval_table()["bins"]:
        if tier["upper"] is None or prediction < tier["upper"]:
            return (
                prediction * tier["low_ratio"],
                min(prediction * tier["high_ratio"], MAX_RECORDED_VALUE),
            )
    return prediction, prediction


# ------------------------------------------------------------ formatting

def euros(value):
    """Compact euro formatting: 58462953 -> '€58.5M', 750000 -> '€750K'."""
    value = float(value)
    if value >= 1_000_000:
        return f"€{value / 1_000_000:,.1f}M"
    if value >= 1_000:
        return f"€{value / 1_000:,.0f}K"
    return f"€{value:,.0f}"


def result_card_html(prediction, caption):
    """HTML for the main result card, including the typical range."""
    low, high = prediction_range(prediction)
    return f"""
<div class="fm-result">
    <div class="fm-result-label">ESTIMATED MARKET VALUE</div>
    <div class="fm-result-value">{euros(prediction)}</div>
    <div class="fm-result-caption">
        Typical range {euros(low)} &ndash; {euros(high)}
    </div>
    <div class="fm-result-caption">{caption}</div>
</div>
"""
