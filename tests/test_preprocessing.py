import unittest
import pandas as pd
from src.preprocessing import FootballPreprocessor, clean_team_name

class TestFootballPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = FootballPreprocessor("config.yaml")

    def test_clean_team_name(self):
        # Test mapped and unmapped names
        self.assertEqual(clean_team_name("Korea Republic"), "South Korea")
        self.assertEqual(clean_team_name("Germany"), "Germany")
        self.assertEqual(clean_team_name("USA"), "United States")

    def test_merge_rankings_no_lookahead_bias(self):
        # Create a mock matches dataset
        matches = pd.DataFrame({
            "date": pd.to_datetime(["2020-05-15", "2020-06-15"]),
            "home_team": ["Argentina", "Brazil"],
            "away_team": ["Brazil", "Argentina"],
            "home_score": [1, 2],
            "away_score": [0, 1],
            "tournament": ["Friendly", "Friendly"],
            "neutral": [False, False]
        })

        # Create a mock rankings dataset
        rankings = pd.DataFrame({
            "rank_date": pd.to_datetime(["2020-05-01", "2020-05-01", "2020-06-01", "2020-06-01"]),
            "country_full": ["Argentina", "Brazil", "Argentina", "Brazil"],
            "rank": [2, 1, 3, 2],  # Argentina goes from 2 to 3, Brazil goes from 1 to 2
            "points": [1600.0, 1700.0, 1590.0, 1680.0],
            "confederation": ["CONMEBOL", "CONMEBOL", "CONMEBOL", "CONMEBOL"]
        })

        # Preprocess rankings
        rankings_cleaned = self.preprocessor.clean_rankings(rankings)
        
        # Merge
        merged = self.preprocessor.merge_rankings(matches, rankings_cleaned)

        # Assert match 1 (2020-05-15):
        # Home: Argentina. As-of ranking should be 2020-05-01 -> Rank 2, Points 1600.0
        # Away: Brazil. As-of ranking should be 2020-05-01 -> Rank 1, Points 1700.0
        m1 = merged.iloc[0]
        self.assertEqual(m1["home_team"], "Argentina")
        self.assertEqual(m1["home_rank"], 2)
        self.assertEqual(m1["away_rank"], 1)

        # Assert match 2 (2020-06-15):
        # Home: Brazil. As-of ranking should be 2020-06-01 -> Rank 2, Points 1680.0
        # Away: Argentina. As-of ranking should be 2020-06-01 -> Rank 3, Points 1590.0
        m2 = merged.iloc[1]
        self.assertEqual(m2["home_team"], "Brazil")
        self.assertEqual(m2["home_rank"], 2)
        self.assertEqual(m2["away_rank"], 3)

if __name__ == '__main__':
    unittest.main()
