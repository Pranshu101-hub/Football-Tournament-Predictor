import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix, roc_auc_score
from src.utils import setup_logger, load_config, get_absolute_path

logger = setup_logger("model_training")

class ModelTrainer:
    """Trains, compares, calibrates, and saves predictive ML models for match outcomes."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.processed_dir = get_absolute_path(self.config["paths"]["processed_dir"])
        self.models_dir = get_absolute_path(self.config["paths"]["models_dir"])
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.features_path = os.path.join(self.processed_dir, "matches_features.csv")
        self.model_save_path = os.path.join(self.models_dir, "football_model.pkl")
        
        self.test_split_date = pd.to_datetime(self.config["preprocessing"]["test_split_date"])
        self.target_col = self.config["model"]["target"]
        
        # Outcomes mapping: L (0) = Away Win, D (1) = Draw, W (2) = Home Win
        self.label_map = {"L": 0, "D": 1, "W": 2}
        self.inverse_label_map = {0: "L", 1: "D", 2: "W"}
        
        # Define features list
        self.num_features = [
            "home_rank", "away_rank", "home_points", "away_points", "rank_diff", "points_diff",
            "home_elo", "away_elo", "elo_diff",
            "home_rolling_goals_scored_w5", "away_rolling_goals_scored_w5",
            "home_rolling_goals_conceded_w5", "away_rolling_goals_conceded_w5",
            "home_rolling_match_points_w5", "away_rolling_match_points_w5",
            "home_rolling_goals_scored_w10", "away_rolling_goals_scored_w10",
            "home_rolling_goals_conceded_w10", "away_rolling_goals_conceded_w10",
            "home_rolling_match_points_w10", "away_rolling_match_points_w10",
            "h2h_win_rate_home", "h2h_goal_diff_home",
            "home_days_since_last_match", "away_days_since_last_match"
        ]
        
        self.cat_features = ["home_confederation", "away_confederation"]
        self.bin_features = ["neutral", "confederation_same"]

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Loads features dataset, performs chronological split, and encodes target labels."""
        logger.info(f"Loading feature dataset from {self.features_path}...")
        df = pd.read_csv(self.features_path)
        df["date"] = pd.to_datetime(df["date"])
        
        # Map target labels
        df["target"] = df[self.target_col].map(self.label_map)
        
        # Verify no missing targets
        df = df.dropna(subset=["target"]).reset_index(drop=True)
        
        # Set features lists
        all_features = self.num_features + self.cat_features + self.bin_features
        
        # Split data temporally to prevent lookahead validation leakage
        train_mask = df["date"] < self.test_split_date
        test_mask = df["date"] >= self.test_split_date
        
        X_train = df.loc[train_mask, all_features]
        y_train = df.loc[train_mask, "target"]
        
        X_test = df.loc[test_mask, all_features]
        y_test = df.loc[test_mask, "target"]
        
        logger.info(
            f"Data split completed. "
            f"Train set: {X_train.shape[0]} matches (before {self.test_split_date.date()}). "
            f"Test set: {X_test.shape[0]} matches (since {self.test_split_date.date()})."
        )
        return X_train, y_train, X_test, y_test

    def get_preprocessing_pipeline(self) -> ColumnTransformer:
        """Returns standard preprocessor scaling numbers and one-hot encoding categories."""
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features),
                ("bin", "passthrough", self.bin_features)
            ]
        )
        return preprocessor

    def evaluate_model(self, model: Pipeline, X: pd.DataFrame, y: pd.Series, label: str) -> Dict[str, float]:
        """Evaluates model performance and prints standard classification metrics."""
        probs = model.predict_proba(X)
        preds = model.predict(X)
        
        acc = accuracy_score(y, preds)
        loss = log_loss(y, probs)
        roc_auc = roc_auc_score(y, probs, multi_class="ovr", average="macro")
        
        logger.info(f"=== Evaluation Metrics: {label} ===")
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"Log Loss: {loss:.4f}")
        logger.info(f"ROC-AUC (OVR): {roc_auc:.4f}")
        
        # Detailed report
        report = classification_report(y, preds, target_names=["Away Win", "Draw", "Home Win"])
        logger.info(f"\nClassification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y, preds)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        return {"accuracy": acc, "log_loss": loss, "roc_auc": roc_auc}

    def train_and_compare(self) -> Pipeline:
        """Trains baseline (LogReg) vs. advanced models (XGBoost), selects best, and calibrates it."""
        X_train, y_train, X_test, y_test = self.prepare_data()
        preprocessor = self.get_preprocessing_pipeline()
        
        # 1. Baseline: Logistic Regression
        lr_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        logger.info("Training Baseline: Logistic Regression...")
        lr_pipeline.fit(X_train, y_train)
        lr_metrics = self.evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression (Baseline)")
        
        # 2. Advanced: XGBoost
        xgb_params = self.config["model"]["xgb_params"]
        xgb_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**xgb_params))
        ])
        
        logger.info("Training Advanced: XGBoost...")
        xgb_pipeline.fit(X_train, y_train)
        xgb_metrics = self.evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost Classifier")
        
        # 3. Model Selection
        # Choose the model with the lowest Log Loss (most important for probability simulations)
        if xgb_metrics["log_loss"] < lr_metrics["log_loss"]:
            logger.info("Selected XGBoost as the primary model due to lower Log Loss.")
            best_pipeline = xgb_pipeline
        else:
            logger.info("Selected Logistic Regression as the primary model due to lower Log Loss.")
            best_pipeline = lr_pipeline
            
        # 4. Probability Calibration (Sigmoid scaling on best model)
        # Calibration ensures predicted probabilities match real world relative frequencies
        logger.info("Calibrating probability outputs using CalibratedClassifierCV...")
        
        # Get raw classifier and preprocessor to calibrate
        raw_classifier = best_pipeline.named_steps["classifier"]
        
        # Fit preprocessor on X_train first
        X_train_transformed = preprocessor.fit_transform(X_train)
        
        # Calibrate classifier using time-series split cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        calibrated_clf = CalibratedClassifierCV(estimator=raw_classifier, method="sigmoid", cv=tscv)
        calibrated_clf.fit(X_train_transformed, y_train)
        
        # Rebuild full pipeline with calibrated classifier
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", calibrated_clf)
        ])
        
        logger.info("Evaluating Calibrated Pipeline...")
        self.evaluate_model(final_pipeline, X_test, y_test, "Final Calibrated Pipeline")
        
        # Save model
        with open(self.model_save_path, "wb") as f:
            pickle.dump(final_pipeline, f)
        logger.info(f"Model saved successfully to {self.model_save_path}")
        
        return final_pipeline

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_compare()
