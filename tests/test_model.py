import os
import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from src.model_training import ModelTrainer

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        self.trainer = ModelTrainer("config.yaml")
        
        # Create a mock matches feature dataset for testing
        self.mock_features = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2021-01-01", "2022-06-01", "2023-01-01"]),
            "home_team": ["France", "Brazil", "Argentina", "England", "Germany"],
            "away_team": ["Germany", "Italy", "Spain", "Germany", "France"],
            "outcome": ["W", "D", "L", "W", "D"], # W=2, D=1, L=0
            
            # Numeric features
            "home_rank": [2, 1, 3, 5, 12],
            "away_rank": [15, 7, 8, 15, 2],
            "home_points": [1700.0, 1800.0, 1750.0, 1680.0, 1620.0],
            "away_points": [1600.0, 1650.0, 1660.0, 1600.0, 1700.0],
            "rank_diff": [-13, -6, -5, -10, 10],
            "points_diff": [100.0, 150.0, 90.0, 80.0, -80.0],
            "home_elo": [1600.0, 1650.0, 1620.0, 1580.0, 1550.0],
            "away_elo": [1550.0, 1580.0, 1590.0, 1550.0, 1600.0],
            "elo_diff": [50.0, 70.0, 30.0, 30.0, -50.0],
            
            # Rolling features (set w5 and w10 to match lists in trainer)
            "home_rolling_goals_scored_w5": [1.5]*5,
            "away_rolling_goals_scored_w5": [1.2]*5,
            "home_rolling_goals_conceded_w5": [1.0]*5,
            "away_rolling_goals_conceded_w5": [1.3]*5,
            "home_rolling_match_points_w5": [1.8]*5,
            "away_rolling_match_points_w5": [1.4]*5,
            "home_rolling_goals_scored_w10": [1.5]*5,
            "away_rolling_goals_scored_w10": [1.2]*5,
            "home_rolling_goals_conceded_w10": [1.0]*5,
            "away_rolling_goals_conceded_w10": [1.3]*5,
            "home_rolling_match_points_w10": [1.8]*5,
            "away_rolling_match_points_w10": [1.4]*5,
            
            "h2h_win_rate_home": [0.5]*5,
            "h2h_goal_diff_home": [0.0]*5,
            "home_days_since_last_match": [10]*5,
            "away_days_since_last_match": [12]*5,
            
            # Categorical features
            "home_confederation": ["UEFA", "CONMEBOL", "CONMEBOL", "UEFA", "UEFA"],
            "away_confederation": ["UEFA", "UEFA", "UEFA", "UEFA", "UEFA"],
            
            # Binary features
            "neutral": [0, 0, 1, 0, 0],
            "confederation_same": [1, 0, 0, 1, 1]
        })

    def test_label_mappings(self):
        self.assertEqual(self.trainer.label_map["W"], 2)
        self.assertEqual(self.trainer.label_map["D"], 1)
        self.assertEqual(self.trainer.label_map["L"], 0)

    def test_prepare_data_temporal_split(self):
        # Save mock features temporarily to location trainer expects
        temp_features_path = self.trainer.features_path
        # Create directories if missing
        os.makedirs(os.path.dirname(temp_features_path), exist_ok=True)
        self.mock_features.to_csv(temp_features_path, index=False)

        try:
            X_train, y_train, X_test, y_test = self.trainer.prepare_data()
            
            # Since split date is 2022-01-01:
            # Train should contain matches from: 2020-01-01, 2020-06-01, 2021-01-01 (3 rows)
            # Test should contain matches from: 2022-06-01, 2023-01-01 (2 rows)
            self.assertEqual(X_train.shape[0], 3)
            self.assertEqual(X_test.shape[0], 2)
            
            # Verify target labels map correctly
            # Row 0: W -> 2, Row 1: D -> 1, Row 2: L -> 0
            self.assertEqual(y_train.iloc[0], 2)
            self.assertEqual(y_train.iloc[1], 1)
            self.assertEqual(y_train.iloc[2], 0)
        finally:
            # Clean up
            if os.path.exists(temp_features_path):
                os.remove(temp_features_path)

if __name__ == '__main__':
    unittest.main()
