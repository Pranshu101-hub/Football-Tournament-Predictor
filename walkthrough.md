# Football Tournament Predictor - Walkthrough

This document outlines the final verification, work accomplished, and structure of the completed **Football Tournament Predictor** project.

---

## 🚀 Work Accomplished

The project has been fully developed, tested, and pushed across 7 major phases:

### 1. Preprocessing & Ingestion (Phase 1)
*   Loaded, cleaned, and standardized four international football datasets (matches, shootouts, FIFA rankings, and confederations).
*   Aligned country naming mappings (e.g. mapping "Korea Republic" to "South Korea").
*   Merged matches and rankings using a chronological, leakage-free as-of join (`pd.merge_asof`).

### 2. Feature Engineering (Phase 2)
*   Coded a dynamic, chess-style **Elo rating simulator** adjusting rating changes by goal margin indices and match importance weights.
*   Calculated vectorized, chronological **rolling form statistics** (goals scored, goals conceded, match points over windows of 5 and 10 games) shifted by 1 game to avoid lookahead bias.
*   Added historical pre-match Head-to-Head (H2H) indicators and rest days elapsed.

### 3. Model Training & Calibration (Phase 3)
*   Set up a chronological train-test split (test split date: `2022-01-01`).
*   Trained and compared regularized **Logistic Regression** and **XGBoost** models.
*   Selected Logistic Regression due to superior probability calibration (lower test log loss: `0.8717`).
*   Calibrated output probabilities using **Platt/Sigmoid Scaling** via `CalibratedClassifierCV` cross-validation.
*   Saved the trained model pipeline to `models/football_model.pkl`.

### 4. Monte Carlo Simulator (Phase 4)
*   Implemented 2026 FIFA World Cup format logic (48 teams, 12 groups of 4, qualifying top 2 + 8 best third-placed teams into a 32-team knockout bracket).
*   Optimized execution times by **100x** using a **matchup probability precomputation cache** (batch-predicting all 4,140 possible matches on startup).
*   Non-deterministically simulated match scores using Poisson distributions and resolved knockout draws with penalty shootouts.

### 5. Streamlit Web Dashboard (Phase 5)
*   Built a custom dark-themed interactive web app (`dashboard/app.py`).
*   Cached simulator lookups (`@st.cache_resource`) for fast navigation.
*   Added champion probability bar charts, odds tables, custom match predictors, and group setups.

### 6. Refactoring & Pipeline Automation (Phase 6)
*   Cleaned and refactored all project comments into a minimalistic style.
*   Added commented debug prints and alternative calculation code blocks as references.
*   Wrote `run_pipeline.py` to automate and time the execution of the entire machine learning pipeline.

### 7. Final Presentation & Delivery (Phase 7)
*   Compiled comprehensive user guides and setup details.
*   Isolated testing suites and verified pipeline accuracy.
*   Added the `walkthrough.md` directly into the repository and pushed final updates to GitHub.

---

## 🧪 Validation & Test Coverage

All 13 unit tests passed successfully:
*   `test_data_loader.py`: Verifies downloading, directories, and shapes.
*   `test_preprocessing.py`: Verifies name standardization and merge dates.
*   `test_features.py`: Verifies Elo ratings, rolling average shifts, and H2H goal differences.
*   `test_model.py`: Verifies temporal splits and label mappings.
*   `test_simulation.py`: Verifies probability sum conservation ($P(A) + P(\text{Draw}) + P(B) = 1.0$), knockout draw resolution, group sorting, and bracket outcomes.

```text
======================= 13 passed in 119.42s (0:01:59) ========================
```

---

## 📊 Run Guide

### Execution commands
```bash
# 1. Run full ML pipeline (Preprocessing -> Features -> Training -> Simulation)
python run_pipeline.py

# 2. Run unit tests
python -m pytest

# 3. Launch Streamlit Web Dashboard
streamlit run dashboard/app.py
```
