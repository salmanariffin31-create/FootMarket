
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from football_media import (
    get_player_photo,
    club_badge_html
)

from html import escape
from base64 import b64decode
from io import BytesIO
from PIL import Image, ImageDraw
from player_database import load_player_data


# Predict market value using V4 model
def predict_value(player, model, preprocessor):

    data = pd.DataFrame([player.to_dict()])

    data["age_squared"] = data["age"] ** 2

    features = preprocessor.feature_names_in_

    processed = preprocessor.transform(
        data[features]
    )

    prediction_log = model.predict(processed)[0]

    return max(np.expm1(prediction_log), 0)


# Select player and historical record
def select_player(df, players, side):

    options = dict(
        zip(players["label"], players["player_id"])
    )

    selected = st.selectbox(
        f"Select Player {side}",
        options=list(options.keys()),
        index=None,
        placeholder="Search player name...",
        key=f"player_{side}"
    )

    if selected is None:
        return None

    player_id = options[selected]

    history = df[
        df["player_id"] == player_id
    ].sort_values("valuation_date")

    dates = history[
        "valuation_date"
    ].dt.strftime("%Y-%m-%d").tolist()

    selected_date = st.selectbox(
        "Historical Valuation Date",
        options=dates,
        index=len(dates) - 1,
        key=f"date_{side}_{player_id}"
    )

    player = history[
        history["valuation_date"]
        == pd.Timestamp(selected_date)
    ].iloc[0]

    return player


# Display player profile card

# Display player profile with photograph and badge
def display_profile(player, club_mapping, color):

    club_id = str(player["last_club_id"])

    club = club_mapping.get(
        club_id,
        f"Club ID: {club_id}"
    )

    # Use Streamlit's native image renderer, not embedded HTML.
    photo = get_player_photo(player["player_id"])
    try:
        if not photo:
            raise ValueError("No photograph returned by football_media")
        image = Image.open(BytesIO(b64decode(photo.split(",", 1)[1])))
        image.load()
        available = True
    except Exception as error:
        print(f"[Photo] {player['name']}: {error}", flush=True)
        image = Image.new("RGB", (220, 250), "#142334")
        draw = ImageDraw.Draw(image)
        draw.ellipse((70, 24, 150, 104), fill="#526579")
        draw.rounded_rectangle((30, 125, 190, 249), radius=55, fill="#526579")
        available = False

    _, portrait, _ = st.columns([1, 2, 1])
    with portrait:
        st.image(image, width=220)
        if not available:
            st.caption("Player photograph unavailable")

    badge = club_badge_html(
        club_id,
        club
    )

    html = f"""
<div class="fm-player-card fm-media-card"
     style="border-top: 4px solid {color};">

    <div class="fm-profile-details">

        <div class="fm-player-position"
             style="color: {color};">
            {escape(str(player["position"]).upper())}
        </div>

        <div class="fm-player-name">
            {escape(str(player["name"]))}
        </div>

        <div class="fm-club-identity">
            {badge}
            <span>{escape(str(club))}</span>
        </div>

        <div class="fm-player-age">
            AGE {int(player["age"])}
            | {player["valuation_date"].strftime("%d %b %Y")}
        </div>

    </div>

</div>
"""

    st.html(html)


# Display comparison bars
def comparison_bar(label, value_a, value_b, decimals=0):

    maximum = max(value_a, value_b)

    width_a = value_a / maximum * 100 if maximum else 0

    width_b = value_b / maximum * 100 if maximum else 0

    html = f"""
<div class="fm-comparison-row">

    <div class="fm-comparison-labels">

        <span style="color: #B8F34A;">
            {value_a:,.{decimals}f}
        </span>

        <span class="fm-stat-label">
            {escape(label)}
        </span>

        <span style="color: #60A5FA;">
            {value_b:,.{decimals}f}
        </span>

    </div>

    <div class="fm-comparison-bars">

        <div class="fm-track fm-track-left">
            <div style="
                width: {width_a}%;
                background: #B8F34A;
                height: 100%;
                border-radius: 5px;">
            </div>
        </div>

        <div class="fm-track">
            <div style="
                width: {width_b}%;
                background: #60A5FA;
                height: 100%;
                border-radius: 5px;">
            </div>
        </div>

    </div>

</div>
"""

    st.html(html)


# Create radar chart
def show_radar(player_a, player_b, df):

    features = [
        "goals",
        "assists",
        "minutes",
        "appearances",
        "goals_per_90",
        "assists_per_90"
    ]

    labels = [
        "Goals",
        "Assists",
        "Minutes",
        "Appearances",
        "Goals / 90",
        "Assists / 90"
    ]

    # Use training data as reference
    reference = df[
        df["valuation_date"] < pd.Timestamp("2023-01-01")
    ]

    maximums = reference[features].quantile(0.95)

    maximums = maximums.replace(0, 1)

    # Normalise player statistics
    values_a = np.clip(
        player_a[features].astype(float).values
        / maximums.values,
        0, 1
    )

    values_b = np.clip(
        player_b[features].astype(float).values
        / maximums.values,
        0, 1
    )

    angles = np.linspace(
        0,
        2 * np.pi,
        len(features),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    values_a = values_a.tolist()
    values_b = values_b.tolist()

    values_a += values_a[:1]
    values_b += values_b[:1]

    # Create radar chart
    fig, ax = plt.subplots(
        figsize=(7, 6),
        subplot_kw={"polar": True}
    )

    fig.patch.set_facecolor("#111D2B")
    ax.set_facecolor("#111D2B")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Player A
    ax.plot(
        angles,
        values_a,
        color="#B8F34A",
        linewidth=2.5,
        label=player_a["name"]
    )

    ax.fill(
        angles,
        values_a,
        color="#B8F34A",
        alpha=0.15
    )

    # Player B
    ax.plot(
        angles,
        values_b,
        color="#60A5FA",
        linewidth=2.5,
        label=player_b["name"]
    )

    ax.fill(
        angles,
        values_b,
        color="#60A5FA",
        alpha=0.15
    )

    ax.set_xticks(angles[:-1])

    ax.set_xticklabels(
        labels,
        color="#F1F5F9",
        fontsize=10
    )

    ax.set_ylim(0, 1)

    ax.set_yticks([0.25, 0.5, 0.75, 1])

    ax.set_yticklabels(
        ["25%", "50%", "75%", "100%"],
        color="#94A3B8",
        fontsize=8
    )

    ax.grid(color="#304056")

    ax.spines["polar"].set_color("#304056")

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        facecolor="#111D2B"
    )

    for text in legend.get_texts():
        text.set_color("#F1F5F9")

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    st.caption(
        "Statistics are scaled against the 95th percentile "
        "of training records. Values above that threshold "
        "are capped at 100%. This is not an overall "
        "player ability rating."
    )


# Display historical market value chart
def show_market_history(df, player_a, player_b):

    # Only show records up to each selected valuation date.
    # This prevents later records from appearing in a historical comparison.
    history_a = df[
        (df["player_id"] == player_a["player_id"])
        & (df["valuation_date"] <= player_a["valuation_date"])
    ][["valuation_date", "market_value"]].sort_values(
        "valuation_date"
    )

    history_b = df[
        (df["player_id"] == player_b["player_id"])
        & (df["valuation_date"] <= player_b["valuation_date"])
    ][["valuation_date", "market_value"]].sort_values(
        "valuation_date"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    fig.patch.set_facecolor("#111D2B")
    ax.set_facecolor("#111D2B")

    ax.plot(
        history_a["valuation_date"],
        history_a["market_value"] / 1000000,
        color="#B8F34A",
        linewidth=2.5,
        label=player_a["name"]
    )

    ax.plot(
        history_b["valuation_date"],
        history_b["market_value"] / 1000000,
        color="#60A5FA",
        linewidth=2.5,
        label=player_b["name"]
    )

    ax.set_ylabel(
        "Recorded Market Value (EUR Millions)",
        color="#F1F5F9"
    )

    ax.set_xlabel(
        "Valuation Date",
        color="#F1F5F9"
    )

    ax.tick_params(colors="#94A3B8")

    ax.grid(
        alpha=0.15,
        color="#94A3B8"
    )

    for spine in ax.spines.values():
        spine.set_color("#304056")

    legend = ax.legend(facecolor="#111D2B")

    for text in legend.get_texts():
        text.set_color("#F1F5F9")

    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The chart shows recorded market-value history only up to "
        "each player's selected valuation date."
    )


# Identify dataset period
def get_dataset_period(date):

    if date < pd.Timestamp("2023-01-01"):
        return "Training record"

    elif date < pd.Timestamp("2025-01-01"):
        return "Validation record"

    else:
        return "Held-out test record"


# Main player comparison page
def show_player_comparison(model, preprocessor, clubs):

    df = load_player_data()

    # Page header
    st.html("""
<div class="fm-comparison-header">

    <div class="fm-eyebrow">
        SCOUTING INTELLIGENCE
    </div>

    <div class="fm-hero-title">
        HEAD <span>TO HEAD.</span>
    </div>

    <div class="fm-hero-description">
        Compare football player performance and machine
        learning market value estimates side by side.
    </div>

</div>
""")

    # Create searchable player list
    players = df[
        ["player_id", "name"]
    ].drop_duplicates(
        subset=["player_id"]
    ).sort_values("name").copy()

    players["label"] = (
        players["name"].astype(str)
        + " ("
        + players["player_id"].astype(str)
        + ")"
    )

    # Select two players
    left, right = st.columns(2)

    with left:

        player_a = select_player(
            df,
            players,
            "A"
        )

    with right:

        player_b = select_player(
            df,
            players,
            "B"
        )

    if player_a is None or player_b is None:

        st.info(
            "Select two players to begin the comparison."
        )

        return

    # Comparison warnings
    if player_a["player_id"] == player_b["player_id"]:

        st.warning(
            "Select two different players to compare their records."
        )
        return

    if player_a["position"] != player_b["position"]:

        st.info(
            "These players have different positions. "
            "Goals and assists may not be equally relevant."
        )

    if player_a["valuation_date"] != player_b["valuation_date"]:

        st.info(
            "The selected records have different historical "
            "dates and may represent different career stages."
        )

    # Convert club IDs into names
    club_mapping = dict(
        zip(
            clubs["club_id"].astype(str),
            clubs["name"]
        )
    )

    # Generate market value predictions
    prediction_a = predict_value(
        player_a,
        model,
        preprocessor
    )

    prediction_b = predict_value(
        player_b,
        model,
        preprocessor
    )

    actual_a = player_a["market_value"]

    actual_b = player_b["market_value"]

    error_a = abs(actual_a - prediction_a)
    error_b = abs(actual_b - prediction_b)

    error_pct_a = (
        error_a / actual_a * 100
        if actual_a and actual_a > 0
        else None
    )
    error_pct_b = (
        error_b / actual_b * 100
        if actual_b and actual_b > 0
        else None
    )

    st.divider()

    # Player profile cards
    left, right = st.columns(2)

    with left:

        display_profile(
            player_a,
            club_mapping,
            "#B8F34A"
        )

        st.caption(
            get_dataset_period(player_a["valuation_date"])
        )

        st.metric(
            "Model-Estimated Value",
            f"€{prediction_a:,.0f}"
        )

        st.metric(
            "Recorded Dataset Value",
            f"€{actual_a:,.0f}"
        )

        st.metric(
            "Absolute Prediction Error",
            f"€{error_a:,.0f}"
        )

        st.caption(
            f"Error relative to recorded value: "
            f"{error_pct_a:.1f}%"
            if error_pct_a is not None
            else "Percentage error unavailable because recorded value is zero."
        )

    with right:

        display_profile(
            player_b,
            club_mapping,
            "#60A5FA"
        )

        st.caption(
            get_dataset_period(player_b["valuation_date"])
        )

        st.metric(
            "Model-Estimated Value",
            f"€{prediction_b:,.0f}"
        )

        st.metric(
            "Recorded Dataset Value",
            f"€{actual_b:,.0f}"
        )

        st.metric(
            "Absolute Prediction Error",
            f"€{error_b:,.0f}"
        )

        st.caption(
            f"Error relative to recorded value: "
            f"{error_pct_b:.1f}%"
            if error_pct_b is not None
            else "Percentage error unavailable because recorded value is zero."
        )

    st.caption(
        "Recorded dataset value is the historical market value stored "
        "in the Transfermarkt dataset for the selected date. It is not "
        "a live transfer fee or a current official valuation."
    )

    st.divider()

    # Market value summary
    st.subheader("Valuation Comparison")

    difference = abs(prediction_a - prediction_b)

    if prediction_a > prediction_b:
        higher_player = player_a["name"]
        lower_player = player_b["name"]
        lower_value = prediction_b
    elif prediction_b > prediction_a:
        higher_player = player_b["name"]
        lower_player = player_a["name"]
        lower_value = prediction_a
    else:
        higher_player = None
        lower_player = None
        lower_value = None

    if higher_player is not None:
        if lower_value and lower_value > 0:
            percentage_difference = difference / lower_value * 100
            st.success(
                f"{higher_player} has a higher model-estimated value "
                f"by €{difference:,.0f} ({percentage_difference:.1f}% "
                f"higher than {lower_player})."
            )
        else:
            st.success(
                f"{higher_player} has a higher model-estimated value "
                f"by €{difference:,.0f}."
            )
    else:
        st.info(
            "Both players have equal model-estimated market values."
        )

    st.caption(
        "The percentage difference uses the lower estimate as its "
        "reference. A higher estimated market value does not necessarily "
        "mean a player has better football ability."
    )

    st.divider()

    # Performance comparison
    st.subheader("Performance Comparison")

    st.caption(
        "Recorded statistics from the 365 days "
        "before each selected valuation."
    )

    statistics = [
        ("Age", "age", 0),

        ("Goals", "goals", 0),

        ("Assists", "assists", 0),

        ("Appearances", "appearances", 0),

        ("Minutes Played", "minutes", 0),

        ("Goals per 90", "goals_per_90", 2),

        ("Assists per 90", "assists_per_90", 2),

        ("Goal Contributions", "goal_contributions", 0),

        (
            "Contributions per 90",
            "contributions_per_90",
            2
        ),

        (
            "Minutes per Appearance",
            "minutes_per_appearance",
            1
        )
    ]

    for label, column, decimals in statistics:

        comparison_bar(
            label,
            float(player_a[column]),
            float(player_b[column]),
            decimals
        )

    st.divider()

    # Radar chart
    st.subheader("Player Performance Radar")

    show_radar(
        player_a,
        player_b,
        df
    )

    st.divider()

    # Market value history
    st.subheader("Market Value History")

    show_market_history(
        df,
        player_a,
        player_b
    )

    st.divider()

    # Export comparison report
    comparison = pd.DataFrame([
        {
            "Player": player_a["name"],
            "Date": player_a["valuation_date"],
            "Position": player_a["position"],
            "Age": player_a["age"],
            "Goals": player_a["goals"],
            "Assists": player_a["assists"],
            "Minutes": player_a["minutes"],
            "Appearances": player_a["appearances"],
            "Goals per 90": player_a["goals_per_90"],
            "Assists per 90": player_a["assists_per_90"],
            "Actual Value": actual_a,
            "Predicted Value": prediction_a,
            "Absolute Error": error_a
        },
        {
            "Player": player_b["name"],
            "Date": player_b["valuation_date"],
            "Position": player_b["position"],
            "Age": player_b["age"],
            "Goals": player_b["goals"],
            "Assists": player_b["assists"],
            "Minutes": player_b["minutes"],
            "Appearances": player_b["appearances"],
            "Goals per 90": player_b["goals_per_90"],
            "Assists per 90": player_b["assists_per_90"],
            "Actual Value": actual_b,
            "Predicted Value": prediction_b,
            "Absolute Error": error_b
        }
    ])

    st.download_button(
        label="Download Player Comparison",
        data=comparison.to_csv(index=False),
        file_name="footmarket_player_comparison.csv",
        mime="text/csv"
    )

    # Page footer
    st.html("""
<div class="fm-footer">
    FOOTMARKET — SCOUTING INTELLIGENCE
    <br><br>
    Experimental machine learning predictions based
    on historical football data.
    <br>
    Not a professional transfer valuation service.
</div>
""")
