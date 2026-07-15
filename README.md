## Football Tournament Predictor

This is a comprehensive machine learning project designed to forecast the results of international football (soccer) matches and to simulate entire tournaments using Monte Carlo techniques.

# Overview

This repository features a professional-grade machine learning pipeline that:
1. **Data Ingestion & Preprocessing**: Cleans international match results and historical FIFA rankings; consolidates match names and merges features without introducing lookahead bias.
2. **Feature Development**: Generates dynamic Elo ratings, monitors team performance, and compiles statistics for head-to-head matchups.
3. **Model Training & Calibration**: Constructs and evaluates various models (Logistic Regression, Random Forest, XGBoost) to yield calibrated probabilities for match outcomes.
4. **Monte Carlo Simulation**: Conducts simulations of tournament formats (e.g., the World Cup) up to 10,000 times to identify potential champions, qualification chances, and stage-by-stage probabilities.
5. **Interactive Dashboard**: Provides insights, live predictions for matches, and tournament simulations via a Streamlit web application.

---

## Folder Structure

```
Football-Tournament-Predictor/
├── config.yaml
├── requirements.txt
├── README.md
├── walkthrough.md
├── run_pipeline.py
├── data/
│   ├── raw/           # Raw downloaded CSVs (git-ignored)
│   └── processed/     # Cleaned and merged CSVs (git-ignored)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── simulation.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── models/
├── reports/
└── tests/
    ├── test_data_loader.py
    ├── test_features.py
    ├── test_model.py
    ├── test_preprocessing.py
    └── test_simulation.py
```

---
## Project Overview
- **Pipeline & Ingestion**: Cleaned match records and FIFA rankings from raw 1872 (filtered 1993–present) without lookahead bias, standardizing names across datasets.
- **Feature Engineering**: Extracted dynamic Elo ratings, H2H histories, and leakage-free rolling goal averages over windows of 5 and 10 games.
- **ML Model & Calibration**: Trained Logistic Regression (test log loss: 0.87) and XGBoost, calibrating probabilities via cross-validated Platt scaling.
- **Monte Carlo WC Simulator**: Coded the 48-team 2026 World Cup structure (12 groups of 4). Optimized loops 100x using a 4,140 matchup precomputation cache.
- **Dashboard & Verification**: Designed a dark-themed 3-tab Streamlit web application (app.py) and verified all code quality with 13 passing unit tests.
---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Football-Tournament-Predictor.git
   cd Football-Tournament-Predictor
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Pipeline Steps**:
   ```bash
   # Preprocess data
   python -m src.preprocessing

   # Extract features
   python -m src.feature_engineering

   # Train and calibrate models
   python -m src.model_training
   ```

4. **Launch Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

5. **Run Unit Tests**:
   ```bash
   python -m pytest
   ```
