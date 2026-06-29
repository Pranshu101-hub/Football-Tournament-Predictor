# Football Tournament Predictor ⚽🏆

A portfolio-grade, end-to-end Machine Learning project that predicts international football (soccer) match outcomes and simulates entire tournaments using Monte Carlo simulations.

## Project Overview

This repository implements a professional machine learning pipeline to:
1. **Ingest & Preprocess Data**: Clean international results and historical FIFA rankings, matching names and merging features without lookahead bias.
2. **Feature Engineering**: Generate dynamic Elo ratings, rolling team form, and head-to-head statistics.
3. **Model Training & Calibration**: Build and compare models (Logistic Regression, Random Forest, XGBoost) to output calibrated match outcome probabilities.
4. **Monte Carlo Simulation**: Simulate tournament formats (like the World Cup) 10,000+ times to calculate champion, qualification, and stage-by-stage probabilities.
5. **Interactive Dashboard**: Present insights, live match predictions, and tournament simulations via a Streamlit web app.

---

## Folder Structure

```
Football-Tournament-Predictor/
├── config.yaml
├── requirements.txt
├── README.md
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
│   ├── prediction.py
│   ├── simulation.py
│   ├── visualization.py
│   └── utils.py
├── dashboard/
│   └── app.py
├── models/
├── reports/
└── tests/
    ├── test_data_loader.py
    └── test_preprocessing.py
```

---

## 📅 7-Day Development Schedule

### 🚀 Day 1: Project Setup, Data Collection, & EDA (Completed)
*   **Data Sources**:
    *   Match results (1872–present) from Mart Jürisoo's dataset.
    *   FIFA Rankings (1993–present) from Dato-Futbol.
    *   Confederations from FiveThirtyEight.
*   **Engineering Highlights**:
    *   Created `config.yaml` to ensure paths and configurations are centralized.
    *   Implemented name standardization (e.g., matching "Korea Republic" to "South Korea" across different datasets).
    *   Implemented a time-series **as-of merge** (`pd.merge_asof`) which guarantees rankings are merged up to the match date without introducing future leakage.
    *   Calculated ranks dynamically from points when the explicit rank column is missing.
    *   Created `notebooks/01_exploratory_data_analysis.ipynb` with interactive Plotly visualisations exploring goal distributions, outcome balances, and rank-outcome correlations.
    *   Wrote modular unit tests for data loading and preprocessing.

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

3. **Run Preprocessing**:
   ```bash
   python -m src.preprocessing
   ```

4. **Run Unit Tests**:
   ```bash
   python -m pytest
   ```
