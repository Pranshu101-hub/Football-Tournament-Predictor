import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from src.utils import setup_logger, load_config, get_absolute_path

logger = setup_logger("preprocessing")

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

def clean_team_name(name: str) -> str:
    if not isinstance(name, str):
        return str(name)
    name = name.strip()
    return TEAM_NAME_MAP.get(name, name)

class FootballPreprocessor:
    # Cleans and merges raw matches and rankings
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.start_year = self.config["preprocessing"]["start_year"]
        self.processed_dir = get_absolute_path(self.config["paths"]["processed_dir"])
        os.makedirs(self.processed_dir, exist_ok=True)
        self.processed_matches_path = os.path.join(self.processed_dir, "matches_cleaned.csv")
        self.processed_rankings_path = os.path.join(self.processed_dir, "rankings_cleaned.csv")

    def clean_rankings(self, df_rankings: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning FIFA rankings...")
        df = df_rankings.copy()
        
        date_col = "rank_date" if "rank_date" in df.columns else ("date" if "date" in df.columns else None)
        if not date_col:
            raise KeyError("Could not find date column in rankings.")
            
        df["rank_date"] = pd.to_datetime(df[date_col])
        
        country_col = None
        for col in ["country_full", "country", "team"]:
            if col in df.columns:
                country_col = col
                break
        if not country_col:
            raise KeyError("Could not find country column in rankings.")
            
        df["team"] = df[country_col].apply(clean_team_name)
        
        if "total_points" in df.columns:
            df["points"] = df["total_points"]
        elif "points" not in df.columns:
            df["points"] = 0.0
            
        if "rank" not in df.columns:
            logger.info("Rank column missing, calculating dynamically...")
            df["rank"] = df.groupby("rank_date")["points"].rank(ascending=False, method="min")
            
        keep_cols = ["rank_date", "team", "rank", "points", "confederation"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]
        df = df.sort_values(by="rank_date").reset_index(drop=True)
        
        # [DEBUG print sample rankings]
        # print("[DEBUG clean_rankings] head:\n", df.head(3))
        
        logger.info(f"Cleaned rankings: {df.shape}")
        return df

    def clean_matches(self, df_results: pd.DataFrame, df_shootouts: pd.DataFrame) -> pd.DataFrame:
        logger.info("Cleaning matches...")
        df = df_results.copy()
        # Drop matches with missing team names or score values
        df = df.dropna(subset=["home_team", "away_team", "home_score", "away_score"]).reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"].dt.year >= self.start_year].reset_index(drop=True)
        df["home_team"] = df["home_team"].apply(clean_team_name)
        df["away_team"] = df["away_team"].apply(clean_team_name)
        
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
        
        def determine_outcome(row):
            if row["home_score"] > row["away_score"]:
                return "W"
            elif row["home_score"] < row["away_score"]:
                return "L"
            else:
                return "D"
                
        df["outcome"] = df.apply(determine_outcome, axis=1)
        df["goal_difference"] = df["home_score"] - df["away_score"]
        df = df.sort_values(by="date").reset_index(drop=True)
        
        # [DEBUG match outcome mapping validation]
        # print(f"[DEBUG outcome counts] {df['outcome'].value_counts().to_dict()}")
        
        logger.info(f"Cleaned matches: {df.shape}")
        return df

    def merge_rankings(self, df_matches: pd.DataFrame, df_rankings: pd.DataFrame) -> pd.DataFrame:
        logger.info("Merging rankings (merge_asof)...")
        df_matches = df_matches.sort_values(by="date").reset_index(drop=True)
        df_rankings = df_rankings.sort_values(by="rank_date").reset_index(drop=True)
        
        df_rankings_home = df_rankings.rename(columns={
            "team": "home_team",
            "rank": "home_rank",
            "points": "home_points",
            "confederation": "home_confederation",
            "rank_date": "home_rank_date"
        })
        
        merged = pd.merge_asof(
            df_matches,
            df_rankings_home,
            left_on="date",
            right_on="home_rank_date",
            by="home_team",
            direction="backward"
        )
        
        df_rankings_away = df_rankings.rename(columns={
            "team": "away_team",
            "rank": "away_rank",
            "points": "away_points",
            "confederation": "away_confederation",
            "rank_date": "away_rank_date"
        })
        
        merged = pd.merge_asof(
            merged,
            df_rankings_away,
            left_on="date",
            right_on="away_rank_date",
            by="away_team",
            direction="backward"
        )
        
        merged["home_rank"] = merged["home_rank"].fillna(200.0)
        merged["away_rank"] = merged["away_rank"].fillna(200.0)
        merged["home_points"] = merged["home_points"].fillna(0.0)
        merged["away_points"] = merged["away_points"].fillna(0.0)
        
        # [DEBUG merge rankings check]
        # print("[DEBUG merge_rankings] null counts:\n", merged[["home_rank", "away_rank"]].isnull().sum())
        
        logger.info(f"Merged matches: {merged.shape}")
        return merged

    def add_confederations(self, df: pd.DataFrame, df_conf: pd.DataFrame) -> pd.DataFrame:
        logger.info("Adding confederation metadata...")
        df_conf_clean = df_conf.copy()
        df_conf_clean["team"] = df_conf_clean["country"].apply(clean_team_name)
        conf_map = dict(zip(df_conf_clean["team"], df_conf_clean["confederation"]))
        
        fallback_conf = {
            "England": "UEFA", "Scotland": "UEFA", "Wales": "UEFA", "Northern Ireland": "UEFA",
            "Republic of Ireland": "UEFA", "Gibraltar": "UEFA", "Kosovo": "UEFA",
            "Bosnia and Herzegovina": "UEFA", "North Macedonia": "UEFA", "Macedonia": "UEFA",
            "Montenegro": "UEFA", "Serbia": "UEFA", "Czech Republic": "UEFA", "Slovakia": "UEFA",
            "Slovenia": "UEFA", "Croatia": "UEFA", "Ukraine": "UEFA", "Belarus": "UEFA",
            "Moldova": "UEFA", "Lithuania": "UEFA", "Latvia": "UEFA", "Estonia": "UEFA",
            "Georgia": "UEFA", "Armenia": "UEFA", "Azerbaijan": "UEFA", "Kazakhstan": "UEFA",
            "Cyprus": "UEFA", "Malta": "UEFA", "Faroe Islands": "UEFA", "Andorra": "UEFA",
            "San Marino": "UEFA", "Liechtenstein": "UEFA", "Luxembourg": "UEFA",
            
            "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
            "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Peru": "CONMEBOL",
            "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Venezuela": "CONMEBOL",
            "Bolivia": "CONMEBOL",
            
            "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
            "Costa Rica": "CONCACAF", "Panama": "CONCACAF", "Honduras": "CONCACAF",
            "El Salvador": "CONCACAF", "Jamaica": "CONCACAF", "Haiti": "CONCACAF",
            "Trinidad and Tobago": "CONCACAF", "Guatemala": "CONCACAF", "Curaçao": "CONCACAF",
            "Suriname": "CONCACAF", "Martinique": "CONCACAF", "Guadeloupe": "CONCACAF",
            "Bermuda": "CONCACAF", "Cuba": "CONCACAF", "Nicaragua": "CONCACAF",
            
            "Senegal": "CAF", "Morocco": "CAF", "Tunisia": "CAF", "Algeria": "CAF",
            "Egypt": "CAF", "Nigeria": "CAF", "Cameroon": "CAF", "Ghana": "CAF",
            "Ivory Coast": "CAF", "Mali": "CAF", "Burkina Faso": "CAF", "South Africa": "CAF",
            "DR Congo": "CAF", "Cape Verde": "CAF", "Guinea": "CAF", "Equatorial Guinea": "CAF",
            "Zambia": "CAF", "Uganda": "CAF", "Gabon": "CAF", "Angola": "CAF", "Mauritania": "CAF",
            
            "Japan": "AFC", "South Korea": "AFC", "Iran": "AFC", "Australia": "AFC",
            "Saudi Arabia": "AFC", "Qatar": "AFC", "Iraq": "AFC", "United Arab Emirates": "AFC",
            "Uzbekistan": "AFC", "Oman": "AFC", "China": "AFC", "Jordan": "AFC", "Bahrain": "AFC",
            "Syria": "AFC", "Vietnam": "AFC", "Thailand": "AFC", "India": "AFC", "Palestine": "AFC",
            
            "New Zealand": "OFC", "Fiji": "OFC", "Solomon Islands": "OFC", "Tahiti": "OFC",
            "Vanuatu": "OFC", "New Caledonia": "OFC", "Papua New Guinea": "OFC", "Samoa": "OFC"
        }
        
        full_map = {**fallback_conf, **conf_map}
        
        def get_conf(team: str, tournament: str = "") -> str:
            if team in full_map:
                return full_map[team]
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
        # Run preprocessing and save outputs
        cleaned_rankings = self.clean_rankings(df_rankings)
        cleaned_matches_raw = self.clean_matches(df_results, df_shootouts)
        
        merged_matches = self.merge_rankings(cleaned_matches_raw, cleaned_rankings)
        merged_matches = self.add_confederations(merged_matches, df_confederations)
        
        merged_matches.to_csv(self.processed_matches_path, index=False)
        cleaned_rankings.to_csv(self.processed_rankings_path, index=False)
        
        # [UNUSED MOCK ANALYSIS CODE BLOCK - for context reference]
        # def log_confederation_distribution(df):
        #     dist = df['home_confederation'].value_counts()
        #     print(f"[DEBUG conf_dist]\n{dist}")
            
        logger.info(f"Saved preprocessed files to {self.processed_dir}")
        return merged_matches, cleaned_rankings

if __name__ == "__main__":
    from src.data_loader import FootballDataLoader
    loader = FootballDataLoader()
    r, s, rk, c = loader.load_raw_data()
    preprocessor = FootballPreprocessor()
    m_clean, r_clean = preprocessor.process_and_save(r, s, rk, c)
