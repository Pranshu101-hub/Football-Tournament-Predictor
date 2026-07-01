import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from src.utils import setup_logger, load_config, get_absolute_path

logger = setup_logger("feature_engineering")

class FeatureExtractor:
    """Extracts features from preprocessed matches and rankings, focusing on Elo, rolling stats, and H2H."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.processed_dir = get_absolute_path(self.config["paths"]["processed_dir"])
        
        # Load configuration params
        self.initial_elo = self.config["features"]["elo"]["initial_rating"]
        self.k_factor_base = self.config["features"]["elo"]["k_factor_base"]
        self.home_elo_advantage = self.config["features"]["elo"]["home_advantage"]
        self.rolling_windows = self.config["features"]["rolling_windows"]
        
        self.processed_matches_path = os.path.join(self.processed_dir, "matches_cleaned.csv")
        self.features_output_path = os.path.join(self.processed_dir, "matches_features.csv")

    def get_match_importance(self, tournament: str) -> float:
        """Determines Elo K-factor weight based on match importance.
        
        Matches FIFA's official weighting system:
        - Friendlies: 10
        - Nations League / minor tournaments: 25
        - Confederations Qualifiers / World Cup Qualifiers: 35
        - Confederations Final Tournaments: 40
        - World Cup Final Tournaments: 60
        """
        t = str(tournament).lower()
        if "world cup" in t:
            if "qualif" in t:
                return 35.0
            return 60.0
        elif "euro" in t or "copa am" in t or "african cup" in t or "asian cup" in t or "gold cup" in t:
            if "qualif" in t:
                return 35.0
            return 40.0
        elif "nations league" in t:
            return 25.0
        elif "friendly" in t:
            return 10.0
        return 20.0  # default for other tournaments

    def calculate_elo(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Calculates dynamic pre-match Elo ratings chronologically.
        
        Applies K-factor scaling based on match importance and goal margins.
        """
        logger.info("Starting chronological Elo calculations...")
        df_sorted = df.sort_values(by="date").copy()
        
        # Elo ratings tracker: team name -> current Elo rating
        elo_tracker = {}
        
        home_elos = []
        away_elos = []
        
        for idx, row in df_sorted.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_score = row["home_score"]
            away_score = row["away_score"]
            neutral = row["neutral"]
            tournament = row["tournament"]
            
            # Initialize ratings if not seen before
            r_home = elo_tracker.get(home, self.initial_elo)
            r_away = elo_tracker.get(away, self.initial_elo)
            
            # Save pre-match Elos
            home_elos.append(r_home)
            away_elos.append(r_away)
            
            # Calculate expected outcome
            # Adjust home rating if venue is not neutral
            r_home_adjusted = r_home + (0.0 if neutral else self.home_elo_advantage)
            
            dr_home = r_home_adjusted - r_away
            dr_away = r_away - r_home_adjusted
            
            expected_home = 1.0 / (1.0 + 10.0 ** (-dr_home / 400.0))
            expected_away = 1.0 / (1.0 + 10.0 ** (-dr_away / 400.0))
            
            # Actual outcome (1 = Win, 0.5 = Draw, 0 = Loss)
            if home_score > away_score:
                actual_home, actual_away = 1.0, 0.0
            elif home_score < away_score:
                actual_home, actual_away = 0.0, 1.0
            else:
                actual_home, actual_away = 0.5, 0.5
                
            # Compute K-factor and adjust for goal margin index (FIFA formula)
            k = self.get_match_importance(tournament)
            goal_diff = abs(home_score - away_score)
            
            # Goal margin multiplier
            if goal_diff <= 1:
                margin_multiplier = 1.0
            elif goal_diff == 2:
                margin_multiplier = 1.5
            elif goal_diff == 3:
                margin_multiplier = 1.75
            else:  # >= 4 goals
                margin_multiplier = 1.75 + (goal_diff - 3) / 8.0
                
            # Update ratings
            new_r_home = r_home + k * margin_multiplier * (actual_home - expected_home)
            new_r_away = r_away + k * margin_multiplier * (actual_away - expected_away)
            
            # Save updated ratings back to tracker
            elo_tracker[home] = new_r_home
            elo_tracker[away] = new_r_away
            
        df_sorted["home_elo"] = home_elos
        df_sorted["away_elo"] = away_elos
        df_sorted["elo_diff"] = df_sorted["home_elo"] - df_sorted["away_elo"]
        
        logger.info(f"Elo calculations completed. Tracked {len(elo_tracker)} teams.")
        return df_sorted, elo_tracker

    def compute_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates rolling average statistics from team history to avoid data leakage."""
        logger.info("Computing rolling form features (without lookahead)...")
        
        # 1. Pivot matches to 'long' format (one row per team per match)
        home_cols = {
            "date": "date",
            "home_team": "team",
            "away_team": "opponent",
            "home_score": "goals_scored",
            "away_score": "goals_conceded",
            "outcome": "match_outcome",
            "neutral": "neutral"
        }
        
        away_cols = {
            "date": "date",
            "away_team": "team",
            "home_team": "opponent",
            "away_score": "goals_scored",
            "home_score": "goals_conceded",
            "outcome": "match_outcome",
            "neutral": "neutral"
        }
        
        df_home = df[list(home_cols.keys())].rename(columns=home_cols)
        df_home["is_home"] = 1
        df_home["match_points"] = df_home["match_outcome"].map({"W": 3, "D": 1, "L": 0})
        
        df_away = df[list(away_cols.keys())].rename(columns=away_cols)
        df_away["is_home"] = 0
        df_away["match_points"] = df_away["match_outcome"].map({"W": 0, "D": 1, "L": 3})  # invert for away perspective
        
        # Combine home & away matches into a single chronological timeline
        long_df = pd.concat([df_home, df_away], axis=0).sort_values(by=["team", "date"]).reset_index(drop=True)
        
        # Shift rows by 1 within each team group to compute PRE-match features
        # If we didn't shift, the rolling statistics for a match would include that match's goals (leakage!)
        long_df["shifted_goals_scored"] = long_df.groupby("team")["goals_scored"].shift(1)
        long_df["shifted_goals_conceded"] = long_df.groupby("team")["goals_conceded"].shift(1)
        long_df["shifted_match_points"] = long_df.groupby("team")["match_points"].shift(1)
        long_df["shifted_match"] = long_df.groupby("team")["date"].shift(1)
        
        # Days since last match (rest feature)
        long_df["days_since_last_match"] = (long_df["date"] - long_df["shifted_match"]).dt.days
        long_df["days_since_last_match"] = long_df["days_since_last_match"].fillna(30)  # default for first game
        
        # Calculate rolling metrics for configured windows (e.g. 5 and 10)
        for w in self.rolling_windows:
            long_df[f"rolling_goals_scored_w{w}"] = (
                long_df.groupby("team")["shifted_goals_scored"]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            long_df[f"rolling_goals_conceded_w{w}"] = (
                long_df.groupby("team")["shifted_goals_conceded"]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            # Win/Draw/Loss rate approximation based on rolling points average
            # 3 = 100% win, 0 = 100% loss, 1 = 100% draw.
            long_df[f"rolling_match_points_w{w}"] = (
                long_df.groupby("team")["shifted_match_points"]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
        # Fill first match NaNs with global averages
        for w in self.rolling_windows:
            long_df[f"rolling_goals_scored_w{w}"] = long_df[f"rolling_goals_scored_w{w}"].fillna(1.2)
            long_df[f"rolling_goals_conceded_w{w}"] = long_df[f"rolling_goals_conceded_w{w}"].fillna(1.2)
            long_df[f"rolling_match_points_w{w}"] = long_df[f"rolling_match_points_w{w}"].fillna(1.3)  # roughly 1.3 points/game average
            
        # Split back to home and away components
        home_features = long_df[long_df["is_home"] == 1].copy()
        away_features = long_df[long_df["is_home"] == 0].copy()
        
        # Rename columns to merge back into df
        home_rename = {
            col: f"home_{col}" for col in home_features.columns 
            if col not in ["date", "team", "opponent", "is_home"]
        }
        away_rename = {
            col: f"away_{col}" for col in away_features.columns 
            if col not in ["date", "team", "opponent", "is_home"]
        }
        
        home_features = home_features.rename(columns=home_rename)
        away_features = away_features.rename(columns=away_rename)
        
        # Merge back to original matches df
        df_merged = pd.merge(
            df,
            home_features[["date", "team"] + list(home_rename.values())],
            left_on=["date", "home_team"],
            right_on=["date", "team"],
            how="left"
        ).drop(columns=["team"])
        
        df_merged = pd.merge(
            df_merged,
            away_features[["date", "team"] + list(away_rename.values())],
            left_on=["date", "away_team"],
            right_on=["date", "team"],
            how="left"
        ).drop(columns=["team"])
        
        return df_merged

    def compute_h2h_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates historical Head-to-Head (H2H) statistics prior to each match."""
        logger.info("Computing pre-match Head-to-Head statistics...")
        df_sorted = df.sort_values(by="date").copy()
        
        # Store for calculated features
        h2h_win_rate_home = []
        h2h_goal_diff_home = []
        
        # Track past matches: (TeamA, TeamB) -> list of past match goal differences (from TeamA perspective)
        # Note: sorted keys ensures both directions share tracking or we track both directions explicitly
        h2h_history = {}
        
        for idx, row in df_sorted.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_score = row["home_score"]
            away_score = row["away_score"]
            
            # Fetch past history
            pair = (home, away)
            reverse_pair = (away, home)
            
            past_results = h2h_history.get(pair, [])
            past_results_rev = h2h_history.get(reverse_pair, [])
            
            # Calculate current pre-match H2H features
            # Combine direct and reversed history (inverting values for reverse)
            all_past_home_diffs = []
            for d in past_results:
                all_past_home_diffs.append(d)
            for d in past_results_rev:
                all_past_home_diffs.append(-d)  # inverted if home team was away team in past
                
            if len(all_past_home_diffs) == 0:
                # No past encounters, default to neutral stats
                h2h_win_rate_home.append(0.5)
                h2h_goal_diff_home.append(0.0)
            else:
                avg_diff = np.mean(all_past_home_diffs)
                # fraction of wins/draws: +ve diff is win/draw
                wins_or_draws = sum(1 for d in all_past_home_diffs if d >= 0)
                win_rate = wins_or_draws / len(all_past_home_diffs)
                
                h2h_win_rate_home.append(win_rate)
                h2h_goal_diff_home.append(avg_diff)
                
            # Add this match to history log
            current_diff = home_score - away_score
            if pair not in h2h_history:
                h2h_history[pair] = []
            h2h_history[pair].append(current_diff)
            
        df_sorted["h2h_win_rate_home"] = h2h_win_rate_home
        df_sorted["h2h_goal_diff_home"] = h2h_goal_diff_home
        
        return df_sorted

    def extract_all_features(self) -> pd.DataFrame:
        """Main pipeline orchestration: loads, adds features, and saves dataset."""
        if not os.path.exists(self.processed_matches_path):
            raise FileNotFoundError(f"Cleaned matches dataset missing at {self.processed_matches_path}")
            
        df_matches = pd.read_csv(self.processed_matches_path)
        df_matches["date"] = pd.to_datetime(df_matches["date"])
        
        # 1. Dynamic Elo Ratings
        df_features, elo_tracker = self.calculate_elo(df_matches)
        
        # 2. Rolling statistics (form)
        df_features = self.compute_rolling_stats(df_features)
        
        # 3. Head to Head statistics
        df_features = self.compute_h2h_stats(df_features)
        
        # 4. Contextual and Delta features
        df_features["rank_diff"] = df_features["home_rank"] - df_features["away_rank"]
        df_features["points_diff"] = df_features["home_points"] - df_features["away_points"]
        df_features["confederation_same"] = (df_features["home_confederation"] == df_features["away_confederation"]).astype(int)
        
        # Drop columns not needed for modeling (e.g. shift residuals)
        drop_cols = [c for c in df_features.columns if "shifted" in c]
        df_features = df_features.drop(columns=drop_cols)
        
        # Save output
        df_features.to_csv(self.features_output_path, index=False)
        logger.info(f"Saved feature dataset with shape {df_features.shape} to {self.features_output_path}")
        
        return df_features

if __name__ == "__main__":
    extractor = FeatureExtractor()
    df_feat = extractor.extract_all_features()
    print("Features extracted successfully!")
    print("Columns in feature dataset:")
    print(df_feat.columns.tolist()[:20])
