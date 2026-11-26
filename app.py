import joblib
import duckdb
import pandas as pd
import streamlit as st
from pathlib import Path

from fm_utils import (
    euros,
    load_club_options,
    load_headline_metrics,
    predict_value,
    result_card_html,
)
from model_analytics import show_model_analytics
from player_comparison import show_player_comparison
from player_database import show_player_database

ROOT = Path(__file__).resolve().parent

# Website configuration
st.set_page_config(page_title="FootMarket", page_icon="⚽", layout="wide")


# Load custom CSS
def load_css():
    css_path = ROOT / "assets" / "style.css"
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# Load model and preprocessor
@st.cache_resource
def load_model():
    model = joblib.load(ROOT / "models/weighted_log_regression.joblib")
    preprocessor = joblib.load(ROOT / "models/polynomial_preprocessor.joblib")
    return model, preprocessor


# Load club information
@st.cache_data
def load_clubs(known_clubs):
    with duckdb.connect(
        str(ROOT / "data/deployment/footmarket.duckdb"), read_only=True
    ) as connection:
        clubs = connection.execute("""
            SELECT DISTINCT club_id, name
            FROM clubs
            WHERE name IS NOT NULL
            AND club_id IS NOT NULL
            ORDER BY name
        """).df()

    clubs["club_id"] = clubs["club_id"].astype(str)
    return clubs[clubs["club_id"].isin(known_clubs)].reset_index(drop=True)


model, preprocessor = load_model()

# Clubs recognised by the model
known_clubs = frozenset(
    preprocessor.named_transformers_["categorical"].categories_[1]
)
clubs = load_clubs(known_clubs)
club_options = load_club_options(tuple(clubs["club_id"]), tuple(clubs["name"]))

metrics = load_headline_metrics()

# Navbar (always visible, compact)
st.markdown("""
<div class="fm-navbar">
    <div class="fm-logo">FOOT<span>MARKET</span></div>
    <div class="fm-version">ML MODEL V4</div>
</div>
""", unsafe_allow_html=True)

# Website navigation
page = st.radio(
    "Navigation",
    ["Market Predictor", "Player Database", "Player Comparison", "Model Analytics"],
    horizontal=True,
    label_visibility="collapsed",
)

st.divider()

if page == "Player Database":
    show_player_database(model, preprocessor, clubs)
    st.stop()

elif page == "Player Comparison":
    show_player_comparison(model, preprocessor, clubs)
    st.stop()

elif page == "Model Analytics":
    show_model_analytics()
    st.stop()

# ---------------------------------------------------------- Market Predictor
# The hero banner and headline metrics only appear on the landing page, so the
# other pages start with their own content instead of repeating them.
st.markdown("""
<div class="fm-hero">
    <div class="fm-eyebrow">FOOTBALL INTELLIGENCE</div>
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

col1, col2, col3, col4 = st.columns(4)
col1.metric("Training Records", f"{metrics['train_records']:,}")
col2.metric("Test R²", f"{metrics['r2']:.4f}")
col3.metric("Test MAE", f"€{metrics['mae'] / 1_000_000:.3f}M")
col4.metric("Test RMSE", f"€{metrics['rmse'] / 1_000_000:.3f}M")

st.divider()

st.header("Player Market Value Prediction")
st.caption("Enter performance statistics from the previous 365 days.")

with st.form("prediction_form"):

    left, right = st.columns(2)

    with left:
        age = st.number_input("Player Age", min_value=15, max_value=45, value=25)

        position = st.selectbox(
            "Position", ["Attack", "Midfield", "Defender", "Goalkeeper"]
        )

        club_name = st.selectbox(
            "Last Recorded Club", options=list(club_options.keys())
        )

        appearances = st.number_input(
            "Appearances", min_value=1, max_value=68, value=30
        )

    with right:
        goals = st.number_input("Goals", min_value=0, max_value=100, value=10)
        assists = st.number_input("Assists", min_value=0, max_value=100, value=5)
        minutes = st.number_input(
            "Minutes Played", min_value=90, max_value=5790, value=2400
        )

    submitted = st.form_submit_button("Estimate Market Value", width="stretch")

if submitted:

    if minutes > appearances * 120:
        st.error(
            "Minutes played cannot exceed "
            "120 minutes per appearance on average."
        )

    else:
        # Engineered features
        goals_per_90 = goals / minutes * 90
        assists_per_90 = assists / minutes * 90
        goal_contributions = goals + assists
        contributions_per_90 = goal_contributions / minutes * 90
        minutes_per_appearance = minutes / appearances

        player_data = pd.DataFrame([{
            "age": age,
            "age_squared": age ** 2,
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
            "minutes_per_appearance": minutes_per_appearance,
        }])

        prediction = predict_value(model, preprocessor, player_data)

        st.divider()
        st.subheader("Prediction Result")

        st.html(result_card_html(
            prediction,
            "Weighted Log Linear Regression | Performance-Based Valuation",
        ))

        st.caption(
            "The range is where recorded values usually fell for players "
            "with similar estimates during validation (2023–2024). "
            "On the held-out test set, roughly 3 in 4 recorded values landed "
            "inside it. The model tends to underestimate stars."
        )

        st.subheader("Player Performance Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Goals per 90", f"{goals_per_90:.2f}")
        c2.metric("Assists per 90", f"{assists_per_90:.2f}")
        c3.metric("Goal Contributions", goal_contributions)

        st.caption(
            "This is an experimental machine learning "
            "estimate, not a professional transfer valuation."
        )

# About the model
st.divider()

with st.expander("About the Machine Learning Model"):
    st.write(f"""
    **Algorithm:** Weighted Multiple Linear Regression (log-transformed target)

    **Dataset:** Transfermarkt historical football data

    **Total prepared records:** {metrics['total_records']:,}

    **Training records:** {metrics['train_records']:,}

    **Validation records:** {metrics['validation_records']:,}

    **Testing records:** {metrics['test_records']:,}

    **Feature engineering:**
    - Age and age squared
    - Goals and assists per 90 minutes
    - Goal contributions and contributions per 90 minutes
    - Minutes per appearance

    **Optimisation techniques:**
    - Logarithmic target transformation
    - Age-squared feature
    - Sample weighting toward higher-value players

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
