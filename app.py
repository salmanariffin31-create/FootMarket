
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import duckdb
from pathlib import Path
from player_database import show_player_database
from model_analytics import show_model_analytics
from player_comparison import show_player_comparison

# Website configuration
st.set_page_config(
    page_title="FootMarket",
    page_icon="⚽",
    layout="wide"
)

# Load custom CSS
def load_css():
    css_path = Path(__file__).parent / "assets" / "style.css"

    with open(css_path, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load(
        "models/weighted_log_regression.joblib"
    )

    preprocessor = joblib.load(
        "models/polynomial_preprocessor.joblib"
    )

    return model, preprocessor


# Load club information
@st.cache_data
def load_clubs():
    connection = duckdb.connect(
        "data/deployment/footmarket.duckdb",
        read_only=True
    )

    clubs = connection.execute("""
        SELECT DISTINCT club_id, name
        FROM clubs
        WHERE name IS NOT NULL
        AND club_id IS NOT NULL
        ORDER BY name
    """).df()

    connection.close()

    return clubs


model, preprocessor = load_model()
clubs = load_clubs()

# Get clubs recognised by the model
known_clubs = set(
    preprocessor.named_transformers_[
        "categorical"
    ].categories_[1]
)

clubs["club_id"] = clubs["club_id"].astype(str)

clubs = clubs[
    clubs["club_id"].isin(known_clubs)
]

club_options = {
    f"{row['name']} ({row['club_id']})": row["club_id"]
    for _, row in clubs.iterrows()
}

# Header
# Professional website header
st.markdown("""
<div class="fm-navbar">
    <div class="fm-logo">
        FOOT<span>MARKET</span>
    </div>
    <div class="fm-version">
        ML MODEL V4
    </div>
</div>

<div class="fm-hero">
    <div class="fm-eyebrow">
        FOOTBALL INTELLIGENCE
    </div>
    <div class="fm-hero-title">
        KNOW THE<br>
        <span>PLAYER'S VALUE.</span>
    </div>
    <div class="fm-hero-description">
        Discover data-driven football valuations
        powered by historical player performance
        and machine learning.
    </div>
</div>
""", unsafe_allow_html=True)


# Model performance
st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Training Records", "139,090")
col2.metric("Test R²", "0.6803")
col3.metric("Test MAE", "€3.135M")
col4.metric("Test RMSE", "€7.422M")

st.divider()
# Website navigation
# Website navigation

page = st.radio(
    "Navigation",
    [
        "Market Predictor",
        "Player Database",
        "Player Comparison",
        "Model Analytics"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()

if page == "Player Database":

    show_player_database(
        model,
        preprocessor,
        clubs
    )

    st.stop()

elif page == "Player Comparison":

    show_player_comparison(
        model,
        preprocessor,
        clubs
    )

    st.stop()

elif page == "Model Analytics":

    show_model_analytics()

    st.stop()
# Player input
st.header("Player Market Value Prediction")

st.caption(
    "Enter performance statistics from the previous 365 days."
)

with st.form("prediction_form"):

    left, right = st.columns(2)

    with left:

        age = st.number_input(
            "Player Age",
            min_value=15,
            max_value=45,
            value=25
        )

        position = st.selectbox(
            "Position",
            ["Attack", "Midfield", "Defender", "Goalkeeper"]
        )

        club_name = st.selectbox(
            "Last Recorded Club",
            options=list(club_options.keys())
        )

        appearances = st.number_input(
            "Appearances",
            min_value=1,
            max_value=68,
            value=30
        )

    with right:

        goals = st.number_input(
            "Goals",
            min_value=0,
            max_value=100,
            value=10
        )

        assists = st.number_input(
            "Assists",
            min_value=0,
            max_value=100,
            value=5
        )

        minutes = st.number_input(
            "Minutes Played",
            min_value=90,
            max_value=5790,
            value=2400
        )

    submitted = st.form_submit_button(
        "Estimate Market Value",
        use_container_width=True
    )

# Generate prediction
if submitted:

    if minutes > appearances * 120:

        st.error(
            "Minutes played cannot exceed "
            "120 minutes per appearance on average."
        )

    else:

        # Automatically calculate engineered features
        goals_per_90 = goals / minutes * 90

        assists_per_90 = assists / minutes * 90

        goal_contributions = goals + assists

        contributions_per_90 = (
            goal_contributions / minutes
        ) * 90

        minutes_per_appearance = (
            minutes / appearances
        )

        age_squared = age ** 2

        # Build model input
        player_data = pd.DataFrame([{
            "age": age,
            "age_squared": age_squared,
            "position": position,
            "last_club_id": club_options[club_name],
            "goals": goals,
            "assists": assists,
            "minutes": minutes,
            "appearances": appearances,
            "goals_per_90": goals_per_90,
            "assists_per_90": assists_per_90,
            "goal_contributions": goal_contributions,
            "contributions_per_90": contributions_per_90,
            "minutes_per_appearance": minutes_per_appearance
        }])

        # Apply saved preprocessing
        player_data = player_data[
            preprocessor.feature_names_in_
        ]

        processed_data = preprocessor.transform(
            player_data
        )

        # Predict log market value
        prediction_log = model.predict(
            processed_data
        )[0]

        # Convert prediction back into euros
        prediction = max(
            np.expm1(prediction_log), 0
        )

        st.divider()

        st.subheader("Prediction Result")

        st.markdown(f"""
<div class="fm-result">
    <div class="fm-result-label">
        ESTIMATED MARKET VALUE
    </div>

    <div class="fm-result-value">
        €{prediction:,.0f}
    </div>

    <div class="fm-result-caption">
        Weighted Linear Regression
        | Performance-Based Valuation
    </div>
</div>
""", unsafe_allow_html=True)

        st.success("Prediction generated successfully!")

        # Performance summary
        st.subheader("Player Performance Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Goals per 90",
            f"{goals_per_90:.2f}"
        )

        c2.metric(
            "Assists per 90",
            f"{assists_per_90:.2f}"
        )

        c3.metric(
            "Goal Contributions",
            goal_contributions
        )

        st.caption(
            "This is an experimental machine learning "
            "estimate, not a professional transfer valuation."
        )

# About the model
st.divider()

with st.expander("About the Machine Learning Model"):

    st.write("""
    **Algorithm:** Weighted Multiple Linear Regression

    **Dataset:** Transfermarkt historical football data

    **Total prepared records:** 200,363

    **Training records:** 139,090

    **Validation records:** 33,818

    **Testing records:** 27,455

    **Feature engineering:**
    - Age squared
    - Goals per 90 minutes
    - Assists per 90 minutes
    - Goal contributions
    - Contributions per 90 minutes
    - Minutes per appearance

    **Optimisation techniques:**
    - Logarithmic target transformation
    - Polynomial age feature
    - Sample weighting

    **Limitations:**
    - Underestimation of some high-value players
    - Limited defensive and goalkeeping statistics
    - No contract duration or injury information
    - Historical club and position limitations
    - Requires at least 90 recorded playing minutes
    """)
st.markdown("""
<div class="fm-footer">
    FOOTMARKET — Football Intelligence Platform
    <br><br>
    Powered by Python, DuckDB, Scikit-learn and Streamlit.
    <br>
    Experimental machine learning portfolio project.
    Not a professional transfer valuation service.
</div>
""", unsafe_allow_html=True)