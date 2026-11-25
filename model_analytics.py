
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parent


@st.cache_data
def load_analytics():

    comparison = pd.read_csv(
        ROOT / "data/processed/regression_comparison.csv"
    )

    evaluation = pd.read_csv(
        ROOT / "data/processed/final_evaluation.csv"
    )

    predictions = pd.read_csv(
        ROOT / "data/processed/final_test_predictions.csv"
    )

    return comparison, evaluation, predictions


def style_chart(fig, ax):

    fig.patch.set_facecolor("#111D2B")
    ax.set_facecolor("#111D2B")

    ax.tick_params(colors="#94A3B8")

    ax.xaxis.label.set_color("#F1F5F9")
    ax.yaxis.label.set_color("#F1F5F9")
    ax.title.set_color("#F1F5F9")

    for spine in ax.spines.values():
        spine.set_color("#304056")

    ax.grid(
        alpha=0.15,
        color="#94A3B8"
    )

    ax.set_axisbelow(True)


def show_model_analytics():

    comparison, evaluation, predictions = load_analytics()

    test = evaluation[
        evaluation["Dataset"] == "Test"
    ].iloc[0]

    validation = evaluation[
        evaluation["Dataset"] == "Validation"
    ].iloc[0]

    improvement = (
        (comparison.iloc[0]["MAE"] - validation["MAE"])
        / comparison.iloc[0]["MAE"]
    ) * 100

    st.header("Model Intelligence")

    st.caption(
        "Machine learning performance, evaluation "
        "and optimisation insights."
    )

    # Main performance metrics

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Test R²",
        f"{test['R2']:.4f}"
    )

    c2.metric(
        "Test MAE",
        f"€{test['MAE'] / 1000000:.3f}M"
    )

    c3.metric(
        "Validation MAE Reduction",
        f"{improvement:.2f}%"
    )

    c4.metric(
        "Test Records",
        f"{len(predictions):,}"
    )

    st.divider()

    # Model comparison

    st.subheader("Model Optimisation Journey")

    st.write(
        "Four Linear Regression experiments were "
        "conducted to improve prediction performance."
    )

    model_names = {
        "Original Linear Regression": "V1 Original",
        "Log Linear Regression": "V2 Log",
        "Polynomial Log Linear Regression": "V3 Polynomial",
        "Weighted Log Linear Regression": "V4 Weighted"
    }

    comparison["Version"] = comparison["Model"].map(
        model_names
    ).fillna(comparison["Model"])

    left, right = st.columns(2)

    with left:

        fig, ax = plt.subplots(figsize=(7, 4))

        colors = [
            "#B8F34A"
            if name == "V4 Weighted"
            else "#526D86"
            for name in comparison["Version"]
        ]

        bars = ax.bar(
            comparison["Version"],
            comparison["MAE"] / 1000000,
            color=colors
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            color="#F1F5F9",
            padding=5
        )

        ax.set_title("Mean Absolute Error")
        ax.set_ylabel("MAE (EUR Millions)")
        ax.set_ylim(
            0,
            comparison["MAE"].max() / 1000000 * 1.2
        )

        ax.tick_params(axis="x", rotation=15)

        style_chart(fig, ax)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    with right:

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.plot(
            comparison["Version"],
            comparison["R2"],
            color="#B8F34A",
            marker="o",
            linewidth=3,
            markersize=8
        )

        for i, value in enumerate(comparison["R2"]):

            ax.annotate(
                f"{value:.4f}",
                (i, value),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                color="#F1F5F9"
            )

        ax.set_title("R² Score Progression")
        ax.set_ylabel("R² Score")
        ax.set_ylim(0, 1)

        ax.tick_params(axis="x", rotation=15)

        style_chart(fig, ax)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    # Comparison table

    st.subheader("Regression Experiment Results")

    st.dataframe(
        comparison[["Model", "MAE", "RMSE", "R2"]],
        hide_index=True,
        use_container_width=True,
        column_config={
            "MAE": st.column_config.NumberColumn(
                format="€%.2f"
            ),
            "RMSE": st.column_config.NumberColumn(
                format="€%.2f"
            ),
            "R2": st.column_config.NumberColumn(
                format="%.4f"
            )
        }
    )

    st.divider()

    # Actual vs predicted chart

    st.subheader("Actual vs Predicted Market Values")

    fig, ax = plt.subplots(figsize=(10, 6))

    actual = predictions["Actual"] / 1000000
    predicted = predictions["Predicted"] / 1000000

    ax.scatter(
        actual,
        predicted,
        color="#B8F34A",
        alpha=0.2,
        s=12
    )

    maximum = max(actual.max(), predicted.max())

    ax.plot(
        [0, maximum],
        [0, maximum],
        color="#F1F5F9",
        linestyle="--",
        label="Perfect Prediction"
    )

    ax.set_xlabel("Actual Market Value (EUR Millions)")
    ax.set_ylabel("Predicted Market Value (EUR Millions)")
    ax.set_title("Final Test Dataset")

    legend = ax.legend(facecolor="#111D2B")

    for text in legend.get_texts():
        text.set_color("#F1F5F9")

    style_chart(fig, ax)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Points closer to the diagonal line represent "
        "more accurate predictions."
    )

    st.divider()

    # Test performance by position

    st.subheader("Performance by Player Position")

    position_summary = predictions.groupby(
        "Position"
    ).agg(
        Records=("Actual", "count"),
        Average_Value=("Actual", "mean"),
        MAE=("Absolute Error", "mean")
    ).reset_index()

    position_summary["Relative_MAE"] = (
        position_summary["MAE"]
        / position_summary["Average_Value"]
    ) * 100

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.barh(
        position_summary["Position"],
        position_summary["Relative_MAE"],
        color="#B8F34A"
    )

    ax.set_xlabel("MAE / Average Market Value (%)")
    ax.set_title("Relative Prediction Error by Position")

    style_chart(fig, ax)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        position_summary.round(2),
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # Validation versus test

    st.subheader("Validation vs Test Performance")

    st.dataframe(
        evaluation.round(4),
        hide_index=True,
        use_container_width=True
    )

    # Model limitations

    st.subheader("Model Limitations")

    st.warning(
        "FootMarket is an experimental valuation tool "
        "and should not be used for professional "
        "transfer decisions."
    )

    st.markdown("""
    - Limited defensive and goalkeeping statistics.
    - Underestimation of some high-value players.
    - No contract duration, injury or reputation data.
    - Historical data rather than live player statistics.
    - Same players may occur across chronological splits.
    - Minimum requirement of 90 recorded playing minutes.
    """)

    # Download predictions

    st.divider()

    st.download_button(
        label="Download Final Test Predictions",
        data=predictions.to_csv(index=False),
        file_name="footmarket_test_predictions.csv",
        mime="text/csv"
    )
