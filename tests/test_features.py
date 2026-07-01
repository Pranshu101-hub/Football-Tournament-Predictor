import unittest
import pandas as pd
import numpy as np
from src.feature_engineering import FeatureExtractor

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor("config.yaml")

    def test_get_match_importance(self):
        # Test K-factors matching FIFA official values
        self.assertEqual(self.extractor.get_match_importance("FIFA World Cup"), 60.0)
        self.assertEqual(self.extractor.get_match_importance("Friendly"), 10.0)
        self.assertEqual(self.extractor.get_match_importance("UEFA Euro qualifying"), 35.0)

    def test_calculate_elo_progression(self):
        # Create a mock series of matches where Team A consistently wins
        matches = pd.DataFrame({
            "date": pd.to_datetime(["2021-01-01", "2021-01-05", "2021-01-10"]),
            "home_team": ["Team A", "Team A", "Team B"],
            "away_team": ["Team B", "Team C", "Team C"],
            "home_score": [3, 2, 1],
            "away_score": [1, 0, 1],
            "neutral": [False, False, True],
            "tournament": ["Friendly", "Friendly", "Friendly"]
        })

        df_elo, tracker = self.extractor.calculate_elo(matches)

        # Assertions
        # 1. First match: Pre-match Elo ratings must be initial ratings (1500)
        self.assertEqual(df_elo.iloc[0]["home_elo"], 1500.0)
        self.assertEqual(df_elo.iloc[0]["away_elo"], 1500.0)
        
        # 2. Team A wins game 1. Their rating in tracker must increase, Team B's must decrease
        self.assertTrue(tracker["Team A"] > 1500.0)
        self.assertTrue(tracker["Team B"] < 1500.0)
        
        # 3. Second match: Team A pre-match Elo must be updated Elo from game 1 (greater than 1500)
        self.assertTrue(df_elo.iloc[1]["home_elo"] > 1500.0)
        # 4. Third match: Team B pre-match Elo must match their rating after game 1 (less than 1500)
        # Since Team B did not play in match 2, its rating after match 1 is unchanged.
        self.assertTrue(df_elo.iloc[2]["home_elo"] < 1500.0)

    def test_rolling_stats_no_leakage(self):
        # Create mock matches with neutral column
        matches = pd.DataFrame({
            "date": pd.to_datetime(["2021-01-01", "2021-01-05", "2021-01-10"]),
            "home_team": ["Team A", "Team B", "Team A"],
            "away_team": ["Team B", "Team A", "Team C"],
            "home_score": [4, 1, 0],
            "away_score": [0, 2, 3],  # In Match 2: Team A is away and scores 2 goals
            "outcome": ["W", "L", "L"], # W, Away Win, Away Win
            "neutral": [False, False, False],
            "tournament": ["Friendly", "Friendly", "Friendly"]
        })

        df_rolling = self.extractor.compute_rolling_stats(matches)

        # Match 1: No history, should have default values
        m1 = df_rolling.iloc[0]
        # Match 2: Team A has played 1 match (scored 4 goals).
        # Since we shift by 1 to prevent leakage, their pre-match rolling scored for Match 2 should be exactly 4.0
        m2 = df_rolling.iloc[1]
        self.assertEqual(m2["away_rolling_goals_scored_w5"], 4.0)

        # Match 3: Team A has played 2 matches.
        # Match 1: Scored 4. Match 2: Scored 2.
        # Pre-match rolling goals scored for Match 3 should be (4 + 2)/2 = 3.0.
        # It must NOT include the 0 goals Team A scores in Match 3 (which would be leakage).
        m3 = df_rolling.iloc[2]
        self.assertEqual(m3["home_rolling_goals_scored_w5"], 3.0)

if __name__ == '__main__':
    unittest.main()
