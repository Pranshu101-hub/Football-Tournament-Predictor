import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from src.utils import setup_logger, load_config, get_absolute_path

logger = setup_logger("simulation")

class TournamentSimulator:
    """Simulates a 48-team international football tournament using Monte Carlo simulations."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.processed_dir = get_absolute_path(self.config["paths"]["processed_dir"])
        self.models_dir = get_absolute_path(self.config["paths"]["models_dir"])
        
        self.features_path = os.path.join(self.processed_dir, "matches_features.csv")
        self.model_path = os.path.join(self.models_dir, "football_model.pkl")
        
        self.n_simulations = self.config["simulation"]["n_simulations"]
        
        # Load trained model pipeline
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model missing at {self.model_path}. Run model training first.")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
            
        # Load features to extract the latest states of all teams
        self.df_features = pd.read_csv(self.features_path)
        self.team_states = self._extract_latest_team_states()
        
        # Default 2026 World Cup 48-team group configuration (12 groups of 4)
        # Groups are populated with a representative mix of top international teams
        self.groups_2026 = {
            "A": ["United States", "Jamaica", "Cameroon", "New Zealand"],
            "B": ["Mexico", "Ecuador", "Saudi Arabia", "Honduras"],
            "C": ["Canada", "Uruguay", "Egypt", "Slovakia"],
            "D": ["Argentina", "Poland", "Morocco", "Costa Rica"],
            "E": ["France", "Colombia", "Australia", "Mali"],
            "F": ["England", "Peru", "South Korea", "Tunisia"],
            "G": ["Spain", "Japan", "Nigeria", "Canada"], # Note: Standardized team names
            "H": ["Portugal", "Chile", "Algeria", "Panama"],
            "I": ["Belgium", "Switzerland", "Senegal", "United Arab Emirates"],
            "J": ["Netherlands", "Croatia", "Iran", "South Africa"],
            "K": ["Italy", "Denmark", "Sweden", "Iraq"],
            "L": ["Germany", "Austria", "Japan", "Uzbekistan"]
        }
        
        # Ensure team names match team states (fix potential name duplicates in manual groups)
        # We will dynamically map names using our standardization logic
        self._standardize_group_names()
        self._precompute_match_probabilities()

    def _standardize_group_names(self) -> None:
        """Cleans up names in groups to ensure we have a valid state for every team."""
        from src.preprocessing import clean_team_name
        standardized_groups = {}
        for g, teams in self.groups_2026.items():
            cleaned_teams = []
            for t in teams:
                ct = clean_team_name(t)
                # If team has no historical state, default to a close alternative or mock one
                if ct not in self.team_states:
                    logger.warning(f"Team {ct} has no historical data. Creating a default state...")
                    self.team_states[ct] = self._create_default_team_state(ct)
                cleaned_teams.append(ct)
            standardized_groups[g] = cleaned_teams
        self.groups_2026 = standardized_groups

    def _precompute_match_probabilities(self) -> None:
        """Pre-computes and caches match probabilities for all possible team pairings to speed up Monte Carlo loops."""
        logger.info("Pre-computing probability cache for all team pairings...")
        all_teams = []
        for teams in self.groups_2026.values():
            all_teams.extend(teams)
        all_teams = list(set(all_teams)) # 48 unique teams
        
        matchups_list = []
        keys = []
        
        for t1 in all_teams:
            for t2 in all_teams:
                if t1 == t2:
                    continue
                # Pre-compute both neutral=True and neutral=False
                for neutral in [True, False]:
                    feat_df = self._build_match_features(t1, t2, neutral)
                    matchups_list.append(feat_df.iloc[0])
                    keys.append((t1, t2, neutral))
                    
        # Batch predict in one go to bypass pandas/sklearn loop overhead
        df_batch = pd.DataFrame(matchups_list)
        probs_all = self.model.predict_proba(df_batch)
        
        # Populate cache
        self.prob_cache = {}
        for idx, key in enumerate(keys):
            probs = probs_all[idx]
            # 0: B win (L), 1: Draw (D), 2: A win (W)
            self.prob_cache[key] = (probs[2], probs[1], probs[0])
            
        logger.info(f"Pre-computed probabilities for {len(self.prob_cache)} matchups.")

    def _extract_latest_team_states(self) -> Dict[str, Dict[str, Any]]:
        """Extracts the latest feature values for each team at the end of the historical dataset."""
        logger.info("Extracting latest team states for simulation features...")
        df = self.df_features.sort_values(by="date").copy()
        
        team_states = {}
        # Iterate matches and record the latest stats for each team as home/away
        for idx, row in df.iterrows():
            date = row["date"]
            h = row["home_team"]
            a = row["away_team"]
            
            # Record state for home team
            h_state = {
                "rank": row["home_rank"],
                "points": row["home_points"],
                "elo": row["home_elo"],
                "confederation": row["home_confederation"],
                "rolling_goals_scored_w5": row["home_rolling_goals_scored_w5"],
                "rolling_goals_conceded_w5": row["home_rolling_goals_conceded_w5"],
                "rolling_match_points_w5": row["home_rolling_match_points_w5"],
                "rolling_goals_scored_w10": row["home_rolling_goals_scored_w10"],
                "rolling_goals_conceded_w10": row["home_rolling_goals_conceded_w10"],
                "rolling_match_points_w10": row["home_rolling_match_points_w10"],
                "days_since_last_match": 10.0 # Default rest
            }
            
            # Record state for away team
            a_state = {
                "rank": row["away_rank"],
                "points": row["away_points"],
                "elo": row["away_elo"],
                "confederation": row["away_confederation"],
                "rolling_goals_scored_w5": row["away_rolling_goals_scored_w5"],
                "rolling_goals_conceded_w5": row["away_rolling_goals_conceded_w5"],
                "rolling_match_points_w5": row["away_rolling_match_points_w5"],
                "rolling_goals_scored_w10": row["away_rolling_goals_scored_w10"],
                "rolling_goals_conceded_w10": row["away_rolling_goals_conceded_w10"],
                "rolling_match_points_w10": row["away_rolling_match_points_w10"],
                "days_since_last_match": 10.0 # Default rest
            }
            
            team_states[h] = h_state
            team_states[a] = a_state
            
        logger.info(f"Extracted states for {len(team_states)} national teams.")
        return team_states

    def _create_default_team_state(self, name: str) -> Dict[str, Any]:
        """Creates a mock/average state for a country missing in our historical logs."""
        return {
            "rank": 80.0,
            "points": 1200.0,
            "elo": 1400.0,
            "confederation": "CONCACAF",
            "rolling_goals_scored_w5": 1.0,
            "rolling_goals_conceded_w5": 1.5,
            "rolling_match_points_w5": 1.0,
            "rolling_goals_scored_w10": 1.0,
            "rolling_goals_conceded_w10": 1.5,
            "rolling_match_points_w10": 1.0,
            "days_since_last_match": 10.0
        }

    def _build_match_features(self, team_a: str, team_b: str, neutral: bool = True) -> pd.DataFrame:
        """Dynamically constructs the input feature DataFrame for a match between A and B."""
        state_a = self.team_states[team_a]
        state_b = self.team_states[team_b]
        
        # Build features from home (A) vs away (B) perspective
        features = {
            # Rankings & Elos
            "home_rank": state_a["rank"],
            "away_rank": state_b["rank"],
            "home_points": state_a["points"],
            "away_points": state_b["points"],
            "rank_diff": state_a["rank"] - state_b["rank"],
            "points_diff": state_a["points"] - state_b["points"],
            "home_elo": state_a["elo"],
            "away_elo": state_b["elo"],
            "elo_diff": state_a["elo"] - state_b["elo"],
            
            # Rolling window 5
            "home_rolling_goals_scored_w5": state_a["rolling_goals_scored_w5"],
            "away_rolling_goals_scored_w5": state_b["rolling_goals_scored_w5"],
            "home_rolling_goals_conceded_w5": state_a["rolling_goals_conceded_w5"],
            "away_rolling_goals_conceded_w5": state_b["rolling_goals_conceded_w5"],
            "home_rolling_match_points_w5": state_a["rolling_match_points_w5"],
            "away_rolling_match_points_w5": state_b["rolling_match_points_w5"],
            
            # Rolling window 10
            "home_rolling_goals_scored_w10": state_a["rolling_goals_scored_w10"],
            "away_rolling_goals_scored_w10": state_b["rolling_goals_scored_w10"],
            "home_rolling_goals_conceded_w10": state_a["rolling_goals_conceded_w10"],
            "away_rolling_goals_conceded_w10": state_b["rolling_goals_conceded_w10"],
            "home_rolling_match_points_w10": state_a["rolling_match_points_w10"],
            "away_rolling_match_points_w10": state_b["rolling_match_points_w10"],
            
            # H2H (default neutral for tournament simulations)
            "h2h_win_rate_home": 0.5,
            "h2h_goal_diff_home": 0.0,
            
            "home_days_since_last_match": 4.0, # Tournament standard rest
            "away_days_since_last_match": 4.0,
            
            # Categorical & Binary
            "home_confederation": state_a["confederation"],
            "away_confederation": state_b["confederation"],
            "neutral": 1 if neutral else 0,
            "confederation_same": 1 if state_a["confederation"] == state_b["confederation"] else 0
        }
        
        return pd.DataFrame([features])

    def predict_match_probabilities(self, team_a: str, team_b: str, neutral: bool = True) -> Tuple[float, float, float]:
        """Calculates probabilities for Team A winning, Drawing, or Team B winning.
        
        Uses pre-computed cache for fast lookups.
        """
        key = (team_a, team_b, neutral)
        if hasattr(self, "prob_cache") and key in self.prob_cache:
            return self.prob_cache[key]
            
        # Fallback if key is not found or cache is not initialized
        X = self._build_match_features(team_a, team_b, neutral)
        probs = self.model.predict_proba(X)[0]
        return probs[2], probs[1], probs[0]

    def simulate_match_outcome(
        self, 
        team_a: str, 
        team_b: str, 
        is_knockout: bool = False,
        neutral: bool = True
    ) -> Tuple[str, int, int]:
        """Samples match outcome using predicted probabilities (non-deterministic).
        
        If it's a knockout match, resolves draws via a penalty shootout.
        
        Returns:
            Tuple[str, int, int]: (Winner name, Team A Goals, Team B Goals)
        """
        prob_a, prob_draw, prob_b = self.predict_match_probabilities(team_a, team_b, neutral)
        
        # Non-deterministic Monte Carlo sampling: choose outcome based on random draw
        r = np.random.rand()
        
        if r < prob_b:
            outcome = "B" # Team B Wins
        elif r < (prob_b + prob_draw):
            outcome = "D" # Draw
        else:
            outcome = "A" # Team A Wins
            
        # Sample realistic scores using Poisson distributions centered around average scores
        # We enforce that the goals scored align with the sampled outcome.
        # Lambda parameters for goal scoring (higher for the winning team)
        if outcome == "A":
            goals_a = np.random.poisson(1.8) + 1  # guaranteed at least 1
            goals_b = np.random.poisson(0.8)
            if goals_b >= goals_a:
                goals_b = goals_a - 1
            winner = team_a
        elif outcome == "B":
            goals_b = np.random.poisson(1.8) + 1
            goals_a = np.random.poisson(0.8)
            if goals_a >= goals_b:
                goals_a = goals_b - 1
            winner = team_b
        else: # Draw
            goals_a = np.random.poisson(1.1)
            goals_b = goals_a
            winner = "Draw"
            
        # Resolve Draw in knockout stage via penalty shootout (50/50 probability)
        if is_knockout and outcome == "D":
            shootout_winner = team_a if np.random.rand() < 0.5 else team_b
            winner = shootout_winner
            
        return winner, goals_a, goals_b

    def simulate_group_stage(self) -> Dict[str, List[Tuple[str, int, int, int]]]:
        """Simulates all matches in the 12 groups.
        
        Returns:
            Dict[str, List[Tuple[str, int, int, int]]]: Group tables containing:
            (Team, Points, Goal Difference, Goals Scored) sorted by rules.
        """
        group_results = {}
        
        for g, teams in self.groups_2026.items():
            # Track stats: team -> [points, goal_diff, goals_scored]
            stats = {t: [0, 0, 0] for t in teams}
            
            # Round-robin: 6 matches per group
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    t1, t2 = teams[i], teams[j]
                    winner, g1, g2 = self.simulate_match_outcome(t1, t2, is_knockout=False)
                    
                    # Update goals scored and conceded
                    stats[t1][2] += g1
                    stats[t2][2] += g2
                    
                    stats[t1][1] += (g1 - g2)
                    stats[t2][1] += (g2 - g1)
                    
                    if winner == t1:
                        stats[t1][0] += 3
                    elif winner == t2:
                        stats[t2][0] += 3
                    else: # Draw
                        stats[t1][0] += 1
                        stats[t2][0] += 1
                        
            # Convert to list and sort by: Points -> Goal Difference -> Goals Scored
            table = [(team, s[0], s[1], s[2]) for team, s in stats.items()]
            table.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
            group_results[g] = table
            
        return group_results

    def simulate_tournament(self) -> str:
        """Simulates a single full tournament from group stage through final.
        
        Returns:
            str: Name of the Champion team.
        """
        # 1. Group Stage
        group_tables = self.simulate_group_stage()
        
        # 2. Extract Qualifiers (Top 2 from each of the 12 groups + 8 best third-placed teams)
        advancing_teams = []
        third_placed_teams = []
        
        for g, table in group_tables.items():
            advancing_teams.append(table[0][0]) # Winner
            advancing_teams.append(table[1][0]) # Runner-up
            third_placed_teams.append((table[2][0], table[2][1], table[2][2], table[2][3])) # Third place
            
        # Rank and pick 8 best third-placed teams
        third_placed_teams.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
        for i in range(8):
            advancing_teams.append(third_placed_teams[i][0])
            
        # Verify we have exactly 32 teams advancing
        assert len(advancing_teams) == 32
        
        # Shuffle randomly to simulate cup bracket pairings (mocking actual bracket slots)
        np.random.shuffle(advancing_teams)
        
        # 3. Knockout Rounds
        # Round of 32
        r32_winners = []
        for i in range(0, 32, 2):
            w, _, _ = self.simulate_match_outcome(advancing_teams[i], advancing_teams[i+1], is_knockout=True)
            r32_winners.append(w)
            
        # Round of 16
        r16_winners = []
        for i in range(0, 16, 2):
            w, _, _ = self.simulate_match_outcome(r32_winners[i], r32_winners[i+1], is_knockout=True)
            r16_winners.append(w)
            
        # Quarter-finals
        qf_winners = []
        for i in range(0, 8, 2):
            w, _, _ = self.simulate_match_outcome(r16_winners[i], r16_winners[i+1], is_knockout=True)
            qf_winners.append(w)
            
        # Semi-finals
        sf_winners = []
        for i in range(0, 4, 2):
            w, _, _ = self.simulate_match_outcome(qf_winners[i], qf_winners[i+1], is_knockout=True)
            sf_winners.append(w)
            
        # Final
        champion, _, _ = self.simulate_match_outcome(sf_winners[0], sf_winners[1], is_knockout=True)
        
        return champion

    def run_monte_carlo(self) -> Dict[str, Any]:
        """Runs the Monte Carlo engine n_simulations times and aggregates probabilities."""
        logger.info(f"Starting Monte Carlo simulation engine with {self.n_simulations} iterations...")
        
        champion_counts = {}
        
        for sim in range(self.n_simulations):
            if (sim + 1) % 2000 == 0:
                logger.info(f"Simulated {sim + 1}/{self.n_simulations} tournaments...")
            champ = self.simulate_tournament()
            champion_counts[champ] = champion_counts.get(champ, 0) + 1
            
        # Convert counts to probabilities
        champ_probs = {
            team: count / self.n_simulations for team, count in champion_counts.items()
        }
        # Sort by probability descending
        sorted_probs = sorted(champ_probs.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("Monte Carlo simulation completed successfully.")
        return {
            "champions_probs": sorted_probs,
            "raw_counts": champion_counts
        }

if __name__ == "__main__":
    simulator = TournamentSimulator()
    
    # Run a single test match prediction
    team_a, team_b = "Argentina", "France"
    p_win_a, p_draw, p_win_b = simulator.predict_match_probabilities(team_a, team_b)
    print(f"\nMatch Prediction ({team_a} vs {team_b}):")
    print(f"{team_a} Win: {p_win_a*100:.2f}% | Draw: {p_draw*100:.2f}% | {team_b} Win: {p_win_b*100:.2f}%")
    
    # Run Monte Carlo (subset 1,000 for quick testing in CLI run)
    simulator.n_simulations = 1000
    results = simulator.run_monte_carlo()
    print("\nTop 5 Simulated World Cup Champions:")
    for team, prob in results["champions_probs"][:5]:
        print(f"{team}: {prob*100:.2f}%")
