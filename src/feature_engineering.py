import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from src.utils import setup_logger, load_config, get_absolute_path

logger = setup_logger("feature_engineering")

class FeatureExtractor:
    # Extracts Elo, rolling stats, and H2H features
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.processed_dir = get_absolute_path(self.config["paths"]["processed_dir"])
        self.initial_elo = self.config["features"]["elo"]["initial_rating"]
        self.k_factor_base = self.config["features"]["elo"]["k_factor_base"]
        self.home_elo_advantage = self.config["features"]["elo"]["home_advantage"]
        self.rolling_windows = self.config["features"]["rolling_windows"]
        self.processed_matches_path = os.path.join(self.processed_dir, "matches_cleaned.csv")
        self.features_output_path = os.path.join(self.processed_dir, "matches_features.csv")

    def get_match_importance(self, tournament: str) -> float:
        # Determine K-factor weight
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
        return 20.0

    def calculate_elo(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        # Calculate pre-match Elo ratings chronologically
        logger.info("Starting chronological Elo calculations...")
        df_sorted = df.sort_values(by="date").copy()
        
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
            
            r_home = elo_tracker.get(home, self.initial_elo)
            r_away = elo_tracker.get(away, self.initial_elo)
            
            home_elos.append(r_home)
            away_elos.append(r_away)
            
            r_home_adjusted = r_home + (0.0 if neutral else self.home_elo_advantage)
            dr_home = r_home_adjusted - r_away
            dr_away = r_away - r_home_adjusted
            
            expected_home = 1.0 / (1.0 + 10.0 ** (-dr_home / 400.0))
            expected_away = 1.0 / (1.0 + 10.0 ** (-dr_away / 400.0))
            
            if home_score > away_score:
                actual_home, actual_away = 1.0, 0.0
            elif home_score < away_score:
                actual_home, actual_away = 0.0, 1.0
            else:
                actual_home, actual_away = 0.5, 0.5
                
            k = self.get_match_importance(tournament)
            goal_diff = abs(home_score - away_score)
            
            if goal_diff <= 1:
                margin_multiplier = 1.0
            elif goal_diff == 2:
                margin_multiplier = 1.5
            elif goal_diff == 3:
                margin_multiplier = 1.75
            else:
                margin_multiplier = 1.75 + (goal_diff - 3) / 8.0
                
            new_r_home = r_home + k * margin_multiplier * (actual_home - expected_home)
            new_r_away = r_away + k * margin_multiplier * (actual_away - expected_away)
            
            # [DEBUG print Elo updates for a specific team]
            # if home == "Argentina" or away == "Argentina":
            #     print(f"[DEBUG ELO] date={row['date']} {home} vs {away} | r_home={r_home:.1f}->{new_r_home:.1f} | r_away={r_away:.1f}->{new_r_away:.1f}")
            
            elo_tracker[home] = new_r_home
            elo_tracker[away] = new_r_away
            
        df_sorted["home_elo"] = home_elos
        df_sorted["away_elo"] = away_elos
        df_sorted["elo_diff"] = df_sorted["home_elo"] - df_sorted["away_elo"]
        
        logger.info(f"Elo calculations completed. Tracked {len(elo_tracker)} teams.")
        return df_sorted, elo_tracker

    def compute_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate rolling team averages shifted to prevent leakage
        logger.info("Computing rolling form features...")
        
        home_cols = {
            "date": "date", "home_team": "team", "away_team": "opponent",
            "home_score": "goals_scored", "away_score": "goals_conceded",
            "outcome": "match_outcome", "neutral": "neutral"
        }
        
        away_cols = {
            "date": "date", "away_team": "team", "home_team": "opponent",
            "away_score": "goals_scored", "home_score": "goals_conceded",
            "outcome": "match_outcome", "neutral": "neutral"
        }
        
        df_home = df[list(home_cols.keys())].rename(columns=home_cols)
        df_home["is_home"] = 1
        df_home["match_points"] = df_home["match_outcome"].map({"W": 3, "D": 1, "L": 0})
        
        df_away = df[list(away_cols.keys())].rename(columns=away_cols)
        df_away["is_home"] = 0
        df_away["match_points"] = df_away["match_outcome"].map({"W": 0, "D": 1, "L": 3})
        
        long_df = pd.concat([df_home, df_away], axis=0).sort_values(by=["team", "date"]).reset_index(drop=True)
        
        # Shift to get pre-match attributes
        long_df["shifted_goals_scored"] = long_df.groupby("team")["goals_scored"].shift(1)
        long_df["shifted_goals_conceded"] = long_df.groupby("team")["goals_conceded"].shift(1)
        long_df["shifted_match_points"] = long_df.groupby("team")["match_points"].shift(1)
        long_df["shifted_match"] = long_df.groupby("team")["date"].shift(1)
        
        long_df["days_since_last_match"] = (long_df["date"] - long_df["shifted_match"]).dt.days
        long_df["days_since_last_match"] = long_df["days_since_last_match"].fillna(30)
        
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
            long_df[f"rolling_match_points_w{w}"] = (
                long_df.groupby("team")["shifted_match_points"]
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
        for w in self.rolling_windows:
            long_df[f"rolling_goals_scored_w{w}"] = long_df[f"rolling_goals_scored_w{w}"].fillna(1.2)
            long_df[f"rolling_goals_conceded_w{w}"] = long_df[f"rolling_goals_conceded_w{w}"].fillna(1.2)
            long_df[f"rolling_match_points_w{w}"] = long_df[f"rolling_match_points_w{w}"].fillna(1.3)
            
        home_features = long_df[long_df["is_home"] == 1].copy()
        away_features = long_df[long_df["is_home"] == 0].copy()
        
        home_rename = {col: f"home_{col}" for col in home_features.columns if col not in ["date", "team", "opponent", "is_home"]}
        away_rename = {col: f"away_{col}" for col in away_features.columns if col not in ["date", "team", "opponent", "is_home"]}
        
        home_features = home_features.rename(columns=home_rename)
        away_features = away_features.rename(columns=away_rename)
        
        df_merged = pd.merge(
            df,
            home_features[["date", "team"] + list(home_rename.values())],
            left_on=["date", "home_team"], right_on=["date", "team"], how="left"
        ).drop(columns=["team"])
        
        df_merged = pd.merge(
            df_merged,
            away_features[["date", "team"] + list(away_rename.values())],
            left_on=["date", "away_team"], right_on=["date", "team"], how="left"
        ).drop(columns=["team"])
        
        # [DEBUG verify columns after rolling merges]
        # print("[DEBUG compute_rolling_stats] merged cols:", [c for c in df_merged.columns if "rolling" in c][:4])
        
        return df_merged

    def compute_h2h_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        # Calculate pre-match head-to-head stats
        logger.info("Computing pre-match H2H stats...")
        df_sorted = df.sort_values(by="date").copy()
        
        h2h_win_rate_home = []
        h2h_goal_diff_home = []
        h2h_history = {}
        
        for idx, row in df_sorted.iterrows():
            home = row["home_team"]
            away = row["away_team"]
            home_score = row["home_score"]
            away_score = row["away_score"]
            
            pair = (home, away)
            reverse_pair = (away, home)
            
            past_results = h2h_history.get(pair, [])
            past_results_rev = h2h_history.get(reverse_pair, [])
            
            all_past_home_diffs = []
            for d in past_results:
                all_past_home_diffs.append(d)
            for d in past_results_rev:
                all_past_home_diffs.append(-d)
                
            if len(all_past_home_diffs) == 0:
                h2h_win_rate_home.append(0.5)
                h2h_goal_diff_home.append(0.0)
            else:
                avg_diff = np.mean(all_past_home_diffs)
                wins_or_draws = sum(1 for d in all_past_home_diffs if d >= 0)
                win_rate = wins_or_draws / len(all_past_home_diffs)
                
                h2h_win_rate_home.append(win_rate)
                h2h_goal_diff_home.append(avg_diff)
                
            current_diff = home_score - away_score
            if pair not in h2h_history:
                h2h_history[pair] = []
            h2h_history[pair].append(current_diff)
            
        df_sorted["h2h_win_rate_home"] = h2h_win_rate_home
        df_sorted["h2h_goal_diff_home"] = h2h_goal_diff_home
        
        # [UNUSED ALTERNATIVE EXPONENTIAL WEIGHTED H2H BLOCK - for reference]
        # def compute_ewma_h2h(diffs, alpha=0.1):
        #     # prioritizes recent matches in H2H history
        #     weighted = 0.0
        #     weight_sum = 0.0
        #     for i, d in enumerate(diffs):
        #         w = (1 - alpha) ** (len(diffs) - 1 - i)
        #         weighted += d * w
        #         weight_sum += w
        #     return weighted / weight_sum if weight_sum > 0 else 0.0
            
        return df_sorted

    def extract_all_features(self) -> pd.DataFrame:
        # Load preprocessed matches, extract and save features
        if not os.path.exists(self.processed_matches_path):
            raise FileNotFoundError(f"Cleaned matches missing at {self.processed_matches_path}")
            
        df_matches = pd.read_csv(self.processed_matches_path)
        df_matches["date"] = pd.to_datetime(df_matches["date"])
        
        df_features, elo_tracker = self.calculate_elo(df_matches)
        df_features = self.compute_rolling_stats(df_features)
        df_features = self.compute_h2h_stats(df_features)
        
        df_features["rank_diff"] = df_features["home_rank"] - df_features["away_rank"]
        df_features["points_diff"] = df_features["home_points"] - df_features["away_points"]
        df_features["confederation_same"] = (df_features["home_confederation"] == df_features["away_confederation"]).astype(int)
        
        drop_cols = [c for c in df_features.columns if "shifted" in c]
        df_features = df_features.drop(columns=drop_cols)
        
        df_features.to_csv(self.features_output_path, index=False)
        logger.info(f"Saved feature dataset with shape {df_features.shape} to {self.features_output_path}")
        return df_features

if __name__ == "__main__":
    extractor = FeatureExtractor()
    df_feat = extractor.extract_all_features()
