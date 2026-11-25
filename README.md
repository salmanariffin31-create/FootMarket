# FootMarket — Football Market Value Prediction

A machine learning-powered football analytics platform that estimates player market values using historical performance statistics and weighted linear regression.

## Project Overview

FootMarket explores the relationship between football player performance and market valuation through machine learning, historical data analysis and an interactive web application.

The platform allows users to estimate player market values, explore historical player records, compare footballers and evaluate model performance.

## Key Features

- **Market Value Predictor:** Estimate market values using player characteristics and performance statistics.
- **Player Database:** Explore historical player valuations and performance records.
- **Player Comparison:** Compare two footballers through valuation estimates, performance statistics, radar charts and historical trends.
- **Model Analytics:** Evaluate regression models and analyse prediction errors.
- **Player Media Integration:** Dynamic player photographs and club badges.
- **CSV Export:** Download player comparison reports.

## Dataset

Historical football data sourced from the Transfermarkt Datasets project.

| Dataset Information | Value |
|---|---:|
| Prepared records | 200,363 |
| Unique players | 23,811 |
| Training records | 139,090 |
| Validation records | 33,818 |
| Testing records | 27,455 |
| Historical coverage | 2013–2026 |

A chronological dataset split was implemented to reduce temporal data leakage.

## Machine Learning Methodology

Four regression approaches were developed and evaluated:

1. Baseline Multiple Linear Regression
2. Log-Transformed Linear Regression
3. Polynomial Age Regression
4. Weighted Log-Transformed Regression

The fourth model was selected based on its validation performance.

### Feature Engineering

- Player age and age squared
- Playing position
- Last recorded club
- Goals and assists
- Minutes played
- Appearances
- Goals per 90 minutes
- Assists per 90 minutes
- Goal contributions
- Contributions per 90 minutes
- Minutes per appearance

Performance features were calculated using records from the 365 days preceding each valuation date.

## Final Model Performance

| Evaluation Metric | Held-out Test Result |
|---|---:|
| R² Score | 0.6803 |
| Mean Absolute Error | €3.135 million |
| Root Mean Squared Error | €7.422 million |
| Test samples | 27,455 |

The model explains approximately 68% of the variance in recorded market valuations on the held-out test dataset.

## Technology Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- DuckDB
- Streamlit
- Matplotlib
- Pillow
- Joblib
- HTML and CSS
- Git and GitHub

## System Architecture

Historical Football Data → Data Cleaning → Feature Engineering → Chronological Dataset Splitting → Model Training → Model Evaluation → Streamlit Web Application

## Database Optimisation

A lightweight deployment database was created by extracting only the player and club information required by the application.

- Original database: 201.01 MB
- Deployment database: 2.01 MB
- Storage reduction: approximately 99%

The original database is retained locally for data preparation and model reproducibility.

## Running Locally

Clone the repository and navigate to the project directory.

```bash
pip install -r requirements.txt
```

Ensure the required deployment database, processed datasets and trained model files are available.

Launch the application:

```bash
python -m streamlit run app.py
```

## Model Limitations

- The model may underestimate high-value players.
- Contract duration, injuries and transfer demand are not directly considered.
- Defensive and goalkeeping statistics are limited.
- Future player potential is not explicitly measured.
- Historical valuations are based on available dataset records rather than live market information.

Transfermarkt market values are reference estimates, not confirmed transfer fees.

## Data Source

Transfermarkt Datasets by dcaribou:

https://github.com/dcaribou/transfermarkt-datasets

Third-party football data, photographs and club logos remain subject to their respective ownership and usage rights.

## Future Improvements

- Experimentation with ensemble learning algorithms.
- Additional defensive and advanced performance statistics.
- Improved valuation modelling for young players.
- Expanded league and competition features.
- Automated dataset updates.

## Disclaimer

FootMarket is an independent educational and machine learning portfolio project. It is not affiliated with Transfermarkt, football clubs or professional football valuation services.

The predictions should not be interpreted as professional financial or transfer advice.