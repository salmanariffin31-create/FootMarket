# FootMarket — Football Market Value Prediction

A machine learning football analytics app that estimates player market values from historical performance statistics, using weighted log-linear regression.

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://footmarket-analytics.streamlit.app/)

**Live app:** https://footmarket-analytics.streamlit.app/

<!-- Add 2-3 screenshots here, e.g. docs/predictor.png, docs/comparison.png -->

## Features

- **Market Value Predictor** — estimate a value from age, position, club and last-365-day statistics, shown with a **typical range**, not just a single number.
- **Player Database** — browse 23,811 players and 200,363 historical valuations; compare the model against the recorded value for any date.
- **Player Comparison** — side-by-side estimates, statistics, radar chart and value history, with a note when the two ranges overlap.
- **Model Analytics** — model versions, error by value tier, and honest accuracy metrics (median % error, share of predictions within ±25% / ±50%).
- **Media** — player photographs and club badges, with placeholders when unavailable.
- **CSV export** — download comparison reports and test-set predictions.

## Dataset

Historical data from the [Transfermarkt Datasets](https://github.com/dcaribou/transfermarkt-datasets) project.

| Dataset information | Value |
|---|---:|
| Prepared records | 200,363 |
| Unique players | 23,811 |
| Training records (before 2023) | 139,090 |
| Validation records (2023–2024) | 33,818 |
| Test records (2025 onward) | 27,455 |
| Historical coverage | 2013–2026 |

A chronological split is used so the model is never trained on records from after the period it is evaluated on. The same *players* can still appear in several splits (at different dates).

## Methodology

Four regression models were compared on the validation set:

| Version | Model | Validation MAE | Validation R² |
|---|---|---:|---:|
| V1 | Multiple linear regression | €3.63M | 0.571 |
| V2 | Log-transformed target | €3.06M | 0.645 |
| V3 | Log target + age² | €2.98M | 0.642 |
| **V4** | **Log target + age² + sample weights** | **€2.78M** | **0.695** |

V4 was selected on validation performance and evaluated **once** on the untouched test set.

**Features:** age, age², position, last club, goals, assists, minutes, appearances, goals/90, assists/90, goal contributions, contributions/90, minutes per appearance — all computed from the 365 days before each valuation date.

## Results (held-out test set, 27,455 records)

| Metric | Result |
|---|---:|
| R² | 0.6803 |
| Mean absolute error | €3.135M |
| Root mean squared error | €7.422M |
| **Median error (% of recorded value)** | **≈ 45%** |
| Predictions within ±25% of recorded value | ≈ 27% |
| Predictions within ±50% of recorded value | ≈ 56% |

R² and MAE are measured in euros, so a small number of very valuable players dominate them. The percentage metrics describe a typical single prediction more fairly: **individual estimates are rough.**

### Known weakness: high-value players

The model under-predicts stars. Among the 417 test records with a recorded value above €50M, the average recorded value is about €80.9M while the average prediction is about €50.5M.

### Prediction ranges

Because single numbers imply false precision, the app shows a *typical range* next to every estimate. Ranges are built from the 10th–90th percentile of `actual / predicted` on the **validation** set, separately for four predicted-value tiers (under €1M, €1–5M, €5–20M, over €20M), and are capped at €200M (the highest value in the dataset). On the held-out test set the recorded value fell inside its range roughly 72–83% of the time depending on tier — useful, but not a guarantee.

Regenerate the ranges with:

```bash
python src/build_prediction_interval.py
```

## Technology

Python · scikit-learn · pandas · NumPy · DuckDB · Streamlit · Matplotlib · Pillow · joblib

## Project structure

```
app.py                  Streamlit entry point and Market Predictor page
fm_utils.py             Shared cached loaders, prediction ranges, formatting
player_database.py      Player Database page
player_comparison.py    Player Comparison page
model_analytics.py      Model Analytics page
football_media.py       Player photos and club badges
assets/style.css        Custom theme
models/                 Trained models and preprocessors (.joblib)
data/deployment/        Small DuckDB file used by the app (2 MB)
data/processed/         Final dataset, evaluation results, prediction ranges
src/                    Data preparation, training and evaluation scripts
```

## Running locally

```bash
git clone https://github.com/salmanariffin31-create/FootMarket.git
cd FootMarket
pip install -r requirements.txt
python -m streamlit run app.py
```

The repository includes the trained models and processed data needed to run the app.

## Reproducing the model

The raw database (~200 MB) is not stored in this repository. Download `transfermarkt-datasets.duckdb` from the [Transfermarkt Datasets project](https://github.com/dcaribou/transfermarkt-datasets) into `data/raw/`, then run from the project root, in this order:

```bash
python src/build_dataset.py
python src/feature_engineering.py
python src/preprocess_data.py              # creates train / validation / test splits
python src/train_linear_regression.py
python src/train_log_regression.py
python src/train_polynomial_regression.py
python src/train_weighted_regression.py    # final V4 model
python src/evaluate_final_model.py
python src/create_deployment_database.py
python src/build_prediction_interval.py
```

## Limitations

- Estimates are rough: the median error is about 45% of the recorded value.
- High-value players are systematically under-predicted.
- No contract length, injuries, transfer demand or potential.
- Limited defensive and goalkeeper statistics.
- Values are historical Transfermarkt reference estimates, not transfer fees, and are not live.

## Data and image rights

Football data comes from the Transfermarkt Datasets project. Player photographs and club badges are loaded from Transfermarkt at runtime and remain the property of their respective owners; they are not part of this repository's code.

## Disclaimer

FootMarket is an independent educational and machine learning portfolio project. It is not affiliated with Transfermarkt, football clubs or professional valuation services. Predictions are not financial or transfer advice.
