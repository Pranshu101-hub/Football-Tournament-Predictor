import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from src.utils import setup_logger, load_config, get_absolute_path

logger = setup_logger("preprocessing")

# Standard team name mapping to align different datasets
TEAM_NAME_MAP = {
    "USA": "United States",
    "US Virgin Islands": "U.S. Virgin Islands",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Korea DPR": "North Korea",
    "Congo DR": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "North Macedonia": "Macedonia",
    "Timor-Leste": "East Timor",
    "St. Vincent / Grenadines": "Saint Vincent and the Grenadines",
    "St. Kitts & Nevis": "Saint Kitts and Nevis",
    "St. Lucia": "Saint Lucia",
    "Brunei Darussalam": "Brunei",
    "Kyrgyz Republic": "Kyrgyzstan",
    "Viet Nam": "Vietnam",
    "Curacao": "Curaçao",
    "São Tomé and Príncipe": "Sao Tome and Principe",
    "Türkiye": "Turkey"
}

def clean_team_name(name: str):
    #standardizes team names based on the defined mapping.
    if not isinstance(name, str):
        return str(name)
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)

class FootballPreprocessor:
    #Preprocesses raw football datasets and merges them without lookahead bias.
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.start_year = self.config["preprocessing"]["start_year"]
        self.processed_dir = get_absolute_path(self.config["paths"]["processed_dir"])
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.processed_matches_path = os.path.join(self.processed_dir, "matches_cleaned.csv")
        self.processed_rankings_path = os.path.join(self.processed_dir, "rankings_cleaned.csv")

    def clean_rankings(self, df_rankings: pd.DataFrame) -> pd.DataFrame:
        """Cleans and standardizes the FIFA rankings dataset."""
        logger.info("Cleaning FIFA rankings dataset...")
        df = df_rankings.copy()
        
        # Standardize columns based on Dato-Futbol layout
        # Date column is typically rank_date or date
        date_col = "rank_date" if "rank_date" in df.columns else ("date" if "date" in df.columns else None)
        if not date_col:
            raise KeyError("Could not find a valid date column in rankings dataset.")
            
        df["rank_date"] = pd.to_datetime(df[date_col])
        
        # Country column is typically country_full, country, or team
        country_col = None
        for col in ["country_full", "country", "team"]:
            if col in df.columns:
                country_col = col
                break
        if not country_col:
            raise KeyError("Could not find a valid country name column in rankings dataset.")
            
        df["team"] = df[country_col].apply(clean_team_name)
        
        # In some datasets, points might be total_points
        if "total_points" in df.columns:
            df["points"] = df["total_points"]
        elif "points" not in df.columns:
            df["points"] = 0.0
            
        # Calculate rank dynamically if missing
        if "rank" not in df.columns:
            logger.info("Rank column not found in raw rankings data. Calculating rank dynamically from points...")
            # Rank teams on each date descending by points
            df["rank"] = df.groupby("rank_date")["points"].rank(ascending=False, method="min")
            
        # Keep relevant columns
        keep_cols = ["rank_date", "team", "rank", "points", "confederation"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]
        
        # Sort for merge_asof
        df = df.sort_values(by="rank_date").reset_index(drop=True)
        
        logger.info(f"Cleaned rankings shape: {df.shape}")
        return df

    def clean_matches(self, df_results: pd.DataFrame, df_shootouts: pd.DataFrame) -> pd.DataFrame:
        """Cleans and standardizes the match results dataset."""
        logger.info("Cleaning match results dataset...")
        df = df_results.copy()
        
        # Datetime conversion
        df["date"] = pd.to_datetime(df["date"])
        
        # Filter post start year (FIFA ranking era)
        df = df[df["date"].dt.year >= self.start_year].reset_index(drop=True)
        
        # Standardize team names
        df["home_team"] = df["home_team"].apply(clean_team_name)
        df["away_team"] = df["away_team"].apply(clean_team_name)
        
        # Merge shootout info
        # Shootout columns: date, home_team, away_team, winner, first_shooter
        df_shootouts_clean = df_shootouts.copy()
        df_shootouts_clean["date"] = pd.to_datetime(df_shootouts_clean["date"])
        df_shootouts_clean["home_team"] = df_shootouts_clean["home_team"].apply(clean_team_name)
        df_shootouts_clean["away_team"] = df_shootouts_clean["away_team"].apply(clean_team_name)
        
        df = pd.merge(
            df,
            df_shootouts_clean[["date", "home_team", "away_team", "winner"]],
            on=["date", "home_team", "away_team"],
            how="left"
        )
        
        # Create target outcome: W (Home Win), D (Draw), L (Away Win) from home perspective
        def determine_outcome(row):
            if row["home_score"] > row["away_score"]:
                return "W"
            elif row["home_score"] < row["away_score"]:
                return "L"
            else:
                return "D"
                
        df["outcome"] = df.apply(determine_outcome, axis=1)
        
        # Goal difference
        df["goal_difference"] = df["home_score"] - df["away_score"]
        
        # Sort by date for merge_asof
        df = df.sort_values(by="date").reset_index(drop=True)
        
        logger.info(f"Cleaned matches shape: {df.shape}")
        return df

    def merge_rankings(self, df_matches: pd.DataFrame, df_rankings: pd.DataFrame) -> pd.DataFrame:
        """Merges FIFA rankings into matches dataset using merge_asof to prevent leakage."""
        logger.info("Merging rankings into match results using merge_asof...")
        
        # Ensure sorted by date
        df_matches = df_matches.sort_values(by="date").reset_index(drop=True)
        df_rankings = df_rankings.sort_values(by="rank_date").reset_index(drop=True)
        
        # Temporary team column for matching home team
        df_rankings_home = df_rankings.rename(columns={
            "team": "home_team",
            "rank": "home_rank",
            "points": "home_points",
            "confederation": "home_confederation",
            "rank_date": "home_rank_date"
        })
        
        # Merge for home team
        # We find the latest ranking date <= match date
        merged = pd.merge_asof(
            df_matches,
            df_rankings_home,
            left_on="date",
            right_on="home_rank_date",
            by="home_team",
            direction="backward"
        )
        
        # Temporary team column for matching away team
        df_rankings_away = df_rankings.rename(columns={
            "team": "away_team",
            "rank": "away_rank",
            "points": "away_points",
            "confederation": "away_confederation",
            "rank_date": "away_rank_date"
        })
        
        # Merge for away team
        merged = pd.merge_asof(
            merged,
            df_rankings_away,
            left_on="date",
            right_on="away_rank_date",
            by="away_team",
            direction="backward"
        )
        
        # Fill missing rankings with a default high rank (e.g., 200) and 0 points
        merged["home_rank"] = merged["home_rank"].fillna(200.0)
        merged["away_rank"] = merged["away_rank"].fillna(200.0)
        merged["home_points"] = merged["home_points"].fillna(0.0)
        merged["away_points"] = merged["away_points"].fillna(0.0)
        
        logger.info(f"Merged matches shape: {merged.shape}")
        return merged

    def add_confederations(self, df: pd.DataFrame, df_conf: pd.DataFrame) -> pd.DataFrame:
        """Adds home_confederation and away_confederation to matches based on confederations database."""
        logger.info("Adding confederation information to matches...")
        
        # Standardize 538 confederations dataset
        df_conf_clean = df_conf.copy()
        df_conf_clean["team"] = df_conf_clean["country"].apply(clean_team_name)
        conf_map = dict(zip(df_conf_clean["team"], df_conf_clean["confederation"]))
        
        # Comprehensive fallback dictionary for countries not in 538 dataset
        fallback_conf = {
            # UEFA (Europe)
            "England": "UEFA", "Scotland": "UEFA", "Wales": "UEFA", "Northern Ireland": "UEFA",
            "Republic of Ireland": "UEFA", "Gibraltar": "UEFA", "Kosovo": "UEFA",
            "Bosnia and Herzegovina": "UEFA", "North Macedonia": "UEFA", "Macedonia": "UEFA",
            "Montenegro": "UEFA", "Serbia": "UEFA", "Czech Republic": "UEFA", "Slovakia": "UEFA",
            "Slovenia": "UEFA", "Croatia": "UEFA", "Ukraine": "UEFA", "Belarus": "UEFA",
            "Moldova": "UEFA", "Lithuania": "UEFA", "Latvia": "UEFA", "Estonia": "UEFA",
            "Georgia": "UEFA", "Armenia": "UEFA", "Azerbaijan": "UEFA", "Kazakhstan": "UEFA",
            "Cyprus": "UEFA", "Malta": "UEFA", "Faroe Islands": "UEFA", "Andorra": "UEFA",
            "San Marino": "UEFA", "Liechtenstein": "UEFA", "Luxembourg": "UEFA",
            
            # CONMEBOL (South America)
            "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
            "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Peru": "CONMEBOL",
            "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Venezuela": "CONMEBOL",
            "Bolivia": "CONMEBOL",
            
            # CONCACAF (North / Central America)
            "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
            "Costa Rica": "CONCACAF", "Panama": "CONCACAF", "Honduras": "CONCACAF",
            "El Salvador": "CONCACAF", "Jamaica": "CONCACAF", "Haiti": "CONCACAF",
            "Trinidad and Tobago": "CONCACAF", "Guatemala": "CONCACAF", "Curaçao": "CONCACAF",
            "Suriname": "CONCACAF", "Martinique": "CONCACAF", "Guadeloupe": "CONCACAF",
            "Bermuda": "CONCACAF", "Cuba": "CONCACAF", "Nicaragua": "CONCACAF",
            
            # CAF (Africa)
            "Senegal": "CAF", "Morocco": "CAF", "Tunisia": "CAF", "Algeria": "CAF",
            "Egypt": "CAF", "Nigeria": "CAF", "Cameroon": "CAF", "Ghana": "CAF",
            "Ivory Coast": "CAF", "Mali": "CAF", "Burkina Faso": "CAF", "South Africa": "CAF",
            "DR Congo": "CAF", "Cape Verde": "CAF", "Guinea": "CAF", "Equatorial Guinea": "CAF",
            "Zambia": "CAF", "Uganda": "CAF", "Gabon": "CAF", "Angola": "CAF", "Mauritania": "CAF",
            
            # AFC (Asia)
            "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Australia": "AFC",
            "Saudi Arabia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "United Arab Emirates": "AFC",
            "Uzbekistan": "AFC", "Oman": "AFC", "China": "AFC", "Jordan": "AFC", "Bahrain": "AFC",
            "Syria": "AFC", "Vietnam": "AFC", "Thailand": "AFC", "India": "AFC", "Palestine": "AFC",
            
            # OFC (Oceania)
            "New Zealand": "OFC", "Fiji": "OFC", "Solomon Islands": "OFC", "Tahiti": "OFC",
            "Vanuatu": "OFC", "New Caledonia": "OFC", "Papua New Guinea": "OFC", "Samoa": "OFC"
        }
        
        # Merge maps
        full_map = {**fallback_conf, **conf_map}
        
        # Function to resolve confederation with fallback using tournament context
        def get_conf(team: str, tournament: str = "") -> str:
            if team in full_map:
                return full_map[team]
            # Try to guess based on tournament name keywords
            t = str(tournament).lower()
            if "uefa" in t or "euro" in t:
                return "UEFA"
            elif "copa am" in t:
                return "CONMEBOL"
            elif "concacaf" in t or "gold cup" in t:
                return "CONCACAF"
            elif "african" in t or "caf" in t:
                return "CAF"
            elif "afc" in t or "asian cup" in t:
                return "AFC"
            elif "oceania" in t or "ofc" in t:
                return "OFC"
            return "Other"
            
        df = df.copy()
        df["home_confederation"] = df.apply(lambda r: get_conf(r["home_team"], r["tournament"]), axis=1)
        df["away_confederation"] = df.apply(lambda r: get_conf(r["away_team"], r["tournament"]), axis=1)
        
        return df

    def process_and_save(
        self, 
        df_results: pd.DataFrame, 
        df_shootouts: pd.DataFrame, 
        df_rankings: pd.DataFrame,
        df_confederations: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs the entire preprocessing pipeline and saves cleaned datasets."""
        cleaned_rankings = self.clean_rankings(df_rankings)
        cleaned_matches_raw = self.clean_matches(df_results, df_shootouts)
        
        merged_matches = self.merge_rankings(cleaned_matches_raw, cleaned_rankings)
        merged_matches = self.add_confederations(merged_matches, df_confederations)
        
        # Save to csv
        merged_matches.to_csv(self.processed_matches_path, index=False)
        cleaned_rankings.to_csv(self.processed_rankings_path, index=False)
        logger.info(f"Saved processed data to {self.processed_dir}")
        
        return merged_matches, cleaned_rankings

if __name__ == "__main__":
    from src.data_loader import FootballDataLoader
    loader = FootballDataLoader()
    r, s, rk, c = loader.load_raw_data()
    preprocessor = FootballPreprocessor()
    m_clean, r_clean = preprocessor.process_and_save(r, s, rk, c)
    print("Preprocessed matches head:")
    print(m_clean[["date", "home_team", "away_team", "home_rank", "away_rank", "outcome", "home_confederation"]].head())
