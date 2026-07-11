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

## 📅 Development

### 🚀 Phase 1: Project Setup, Data Collection, & EDA (Completed)
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

### 📈 Phase 2: Feature Engineering & Data Validation (Completed)
*   **Engineering Highlights**:
    *   Developed a dynamic **chess-style Elo rating** simulator tracking team strength sequentially match-by-match.
    *   Incorporated **official FIFA weighting components** (K-factor based on match importance) and a **Goal Margin Index** to scale Elo updates based on victory dominance.
    *   Implemented vectorized, chronological **rolling averages** (windows of 5 and 10 games) representing team form (goals scored, goals conceded, match points) shifted by 1 row to prevent data leakage.
    *   Added dynamic **Head-to-Head (H2H)** history tracking (average goal difference, win-draw ratios) for specific team matchups prior to kickoff.
    *   Added contextual rest indicators (days since last match) and confederation delta flags.
    *   Wrote comprehensive unit tests verifying calculations and ensuring no data leakage.

### 🧠 Phase 3: Model Training, Evaluation, & Probability Calibration (Completed)
*   **Engineering Highlights**:
    *   Set up a temporal train/test split (split date `2022-01-01`) to validate models chronologically and prevent future-leakage evaluation.
    *   Built a preprocessing pipeline scaling numeric features and one-hot encoding categorical variables (confederations).
    *   Trained and evaluated a baseline **Regularized Logistic Regression** model and an advanced **XGBoost Classifier**.
    *   Compared classifiers using Log Loss, Accuracy, and ROC-AUC metrics (XGBoost selected as best predictor).
    *   Implemented **Probability Calibration** using `CalibratedClassifierCV` with Sigmoid scaling to ensure predictions closely match real-world relative frequencies.
    *   Serialized the final calibrated model to `models/football_model.pkl` using pickle.
    *   Wrote unit tests for label mapping and temporal splits.

### 🎲 Phase 4: Monte Carlo Tournament Simulator (Completed)
*   **Engineering Highlights**:
    *   Hardcoded the full **2026 FIFA World Cup structure** (48 teams in 12 groups of 4, qualifying top 2 + 8 best third-placed teams into a 32-team knockout bracket).
    *   Implemented a high-performance **probability precomputation cache** (resolves all 4,140 possible home/away/neutral team matchups in 0.1s in a single batch prediction), speeding up the Monte Carlo engine by over **100x**.
    *   Developed a non-deterministic **match simulator** sampling outcomes (Win, Draw, Loss) from multinomial calibrated probabilities and generating realistic goal counts using Poisson distributions.
    *   Coded a robust **Group Stage engine** tracking points (3 for W, 1 for D), Goal Difference, Goals Scored, and ranking tables accordingly.
    *   Built a **Knockout Stage engine** simulating single-elimination games with extra time/penalty shootouts decision resolution.
    *   Created `tests/test_simulation.py` verifying probability sum conservation and bracket outcomes.

### 💻 Phase 5: Interactive Web Dashboard (Completed)
*   **Engineering Highlights**:
    *   Developed a premium-grade **Streamlit web application** (`dashboard/app.py`) built with vanilla CSS styling overrides for dark mode visualization.
    *   Integrated resource caching (`@st.cache_resource`) to hold the `TournamentSimulator` instance, avoiding matchup precomputation overhead on re-renders.
    *   Designed three interactive tabs:
        *   **Tournament Simulations**: Displays champion winning probabilities in a clean horizontal Plotly bar chart, including metrics (Favorite, unique winners, dark horse) and a tabular overview of all 48 teams' odds.
        *   **Custom Match Predictor**: Allows users to configure any matchup (including neutral venue toggles), view W-D-L distribution graphs, and simulate live scorelines (complete with penalty shootout resolutions).
        *   **Groups & Standings**: Renders the 12 groups of the 2026 World Cup in a structured card layout.

### 🛠️ Phase 6: Refactoring, Testing & Logging (Completed)
*   **Engineering Highlights**:
    *   Cleaned and standardized active comments to a minimalistic, single-line format across all modules.
    *   Added commented-out debug print statements at key ingestion, feature merging, model validation, and simulation checkpoints.
    *   Inserted commented-out alternative implementation code blocks (e.g. EWMA Head-to-Head weights, isotonic probability calibration, and Elo-weighted shootout models) as clean references.
    *   Created `run_pipeline.py` to automate the execution of preprocessing, feature engineering, model training, and verification simulation steps sequentially with detailed runtime execution logging.
    *   Verified full code quality with isolated, green, and passing unit tests (13 tests passed).

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
