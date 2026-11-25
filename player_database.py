
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from html import escape

from football_media import (
    player_photo_html,
    club_badge_html
)

ROOT = Path(__file__).resolve().parent


@st.cache_data
def load_player_data():

    df = pd.read_csv(
        ROOT / "data/processed/footmarket_final.csv",
        dtype={"last_club_id": str},
        parse_dates=["valuation_date"]
    )

    return df


def show_player_database(model, preprocessor, clubs):

    df = load_player_data()

    st.header("Player Database")

    st.caption(
        "Explore historical football player statistics "
        "and machine learning valuations."
    )

    # Database statistics
    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Unique Players",
        f"{df['player_id'].nunique():,}"
    )

    c2.metric(
        "Valuation Records",
        f"{len(df):,}"
    )

    c3.metric(
        "Historical Coverage",
        "2013–2026"
    )

    st.divider()

    # Create searchable player list
    players = df[
        ["player_id", "name"]
    ].drop_duplicates(
        subset=["player_id"]
    ).sort_values("name")

    player_options = {
        f"{row['name']} ({row['player_id']})":
        row["player_id"]
        for _, row in players.iterrows()
    }

    selected_player = st.selectbox(
        "Search Football Player",
        options=list(player_options.keys()),
        index=None,
        placeholder="Type a player name..."
    )

    if selected_player is None:
        st.info("Search and select a football player.")
        return

    player_id = player_options[selected_player]

    # Retrieve selected player's historical records
    history = df[
        df["player_id"] == player_id
    ].sort_values("valuation_date")

    date_options = history[
        "valuation_date"
    ].dt.strftime("%Y-%m-%d").tolist()

    selected_date = st.selectbox(
        "Select Historical Valuation Date",
        options=date_options,
        index=len(date_options) - 1
    )

    player = history[
        history["valuation_date"]
        == pd.Timestamp(selected_date)
    ].iloc[0]

    # Convert club ID into club name
    club_mapping = dict(
        zip(clubs["club_id"], clubs["name"])
    )

    club_id = str(player["last_club_id"])

    club_name = club_mapping.get(
        club_id,
        f"Club ID: {club_id}"
    )

    
# Professional player profile
    st.divider()

    photo = player_photo_html(
        player["player_id"],
        player["name"]
    )

    badge = club_badge_html(
        player["last_club_id"],
        club_name
    )

    st.html(f"""
<div class="fm-database-profile">

    <div class="fm-database-photo">

        {photo}

    </div>

    <div class="fm-database-details">

        <div class="fm-eyebrow">
            HISTORICAL PLAYER PROFILE
        </div>

        <div class="fm-player-position"
             style="color: #B8F34A;">

            {escape(str(player["position"]).upper())}

        </div>

        <div class="fm-database-name">

            {escape(str(player["name"]))}

        </div>

        <div class="fm-club-identity">

            {badge}

            <span>{escape(str(club_name))}</span>

        </div>

        <div class="fm-player-age">

            AGE {int(player["age"])}
            | {escape(selected_date)}

        </div>

    </div>

</div>
""")

    st.caption(
        "Profile photographs may not correspond "
        "to the selected historical date."
    )


    # Performance statistics
    st.subheader("Player Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Goals", int(player["goals"]))
    c2.metric("Assists", int(player["assists"]))
    c3.metric("Appearances", int(player["appearances"]))

    c4, c5, c6 = st.columns(3)

    c4.metric("Minutes", f"{int(player['minutes']):,}")

    c5.metric(
        "Goals / 90",
        f"{player['goals_per_90']:.2f}"
    )

    c6.metric(
        "Assists / 90",
        f"{player['assists_per_90']:.2f}"
    )

    # Prepare model input
    player_input = pd.DataFrame([player.to_dict()])

    player_input["age_squared"] = (
        player_input["age"] ** 2
    )

    features = preprocessor.feature_names_in_

    # Historical prediction
    if st.button(
        "Predict Player Market Value",
        use_container_width=True
    ):

        processed = preprocessor.transform(
            player_input[features]
        )

        prediction_log = model.predict(processed)[0]

        prediction = max(
            np.expm1(prediction_log), 0
        )

        actual = player["market_value"]

        error = abs(actual - prediction)

        st.markdown(f"""
        <div class="fm-result">
            <div class="fm-result-label">
                ESTIMATED MARKET VALUE
            </div>
            <div class="fm-result-value">
                €{prediction:,.0f}
            </div>
            <div class="fm-result-caption">
                Historical Player Valuation
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Actual Recorded Value",
            f"€{actual:,.0f}"
        )

        c2.metric(
            "Predicted Value",
            f"€{prediction:,.0f}"
        )

        c3.metric(
            "Absolute Error",
            f"€{error:,.0f}"
        )

        if pd.Timestamp(selected_date) < pd.Timestamp("2023-01-01"):
            st.warning(
                "This record belongs to the training period."
            )
        elif pd.Timestamp(selected_date) < pd.Timestamp("2025-01-01"):
            st.info(
                "This record belongs to the validation period."
            )
        else:
            st.info(
                "This record belongs to the held-out test period."
            )

    # Historical valuation chart
    st.divider()

    st.subheader("Market Value History")

    chart_data = history[
        ["valuation_date", "market_value"]
    ].copy()

    chart_data["Market Value (€M)"] = (
        chart_data["market_value"] / 1000000
    )

    st.line_chart(
        chart_data,
        x="valuation_date",
        y="Market Value (€M)",
        color="#B8F34A"
    )

    # Historical statistics table
    with st.expander("View Historical Records"):

        st.dataframe(
            history[[
                "valuation_date",
                "age",
                "goals",
                "assists",
                "minutes",
                "appearances",
                "market_value"
            ]],
            hide_index=True,
            use_container_width=True
        )

    st.caption(
        "Historical data is not live. "
        "Predictions are experimental estimates."
    )
