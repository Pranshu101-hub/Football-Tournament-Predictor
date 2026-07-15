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
Built a full football analytics and World Cup simulation project from scratch, covering data collection, cleaning, feature engineering, modeling, calibration, simulation, and deployment. I combined 3 major data sources, standardized team names, avoided leakage with time-aware merges, and created rolling form, Elo, head-to-head, rest-day, and confederation features. I then trained and compared Logistic Regression and XGBoost using a chronological split, calibrated the final model, and saved it for reuse. On top of that, I built a fast Monte Carlo simulator for the 2026 World Cup’s 48-team format with 12 groups and a 32-team knockout stage, plus a Streamlit dashboard for tournament odds, custom match predictions, and standings. The project is fully tested, logged, documented, and ready for end-to-end use, with 13 passing unit tests and a precomputed matchup cache that speeds simulations by over 100x.


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
