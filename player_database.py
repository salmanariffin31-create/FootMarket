
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from html import escape

from football_media import (
    player_photo_html,
    club_badge_html
)
from fm_utils import (
    euros,
    load_player_data,
    load_player_list,
    predict_value,
    prediction_range,
    result_card_html,
)

ROOT = Path(__file__).resolve().parent


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

    # Searchable player list (cached; best-known players listed first)
    players = load_player_list()

    player_options = dict(
        zip(players["label"], players["player_id"])
    )

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
        width="stretch"
    ):

        prediction = predict_value(
            model,
            preprocessor,
            player_input
        )

        actual = player["market_value"]

        error = abs(actual - prediction)

        low, high = prediction_range(prediction)

        st.html(
            result_card_html(
                prediction,
                "Historical Player Valuation"
            )
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Actual Recorded Value",
            euros(actual)
        )

        c2.metric(
            "Predicted Value",
            euros(prediction)
        )

        c3.metric(
            "Absolute Error",
            euros(error)
        )

        if low <= actual <= high:
            st.success(
                f"The recorded value ({euros(actual)}) is inside "
                f"the typical range ({euros(low)} – {euros(high)})."
            )
        elif actual > high:
            st.warning(
                f"The recorded value ({euros(actual)}) is above "
                f"the typical range ({euros(low)} – {euros(high)}). "
                "The model tends to underestimate high-value players."
            )
        else:
            st.warning(
                f"The recorded value ({euros(actual)}) is below "
                f"the typical range ({euros(low)} – {euros(high)})."
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
            width="stretch"
        )

    st.caption(
        "Historical data is not live. "
        "Predictions are experimental estimates."
    )
