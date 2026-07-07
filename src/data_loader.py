import os
import requests
import pandas as pd
from typing import Tuple
from src.utils import setup_logger, load_config, get_absolute_path

logger = setup_logger("data_loader")

class FootballDataLoader:
    # Handles fetching and loading raw datasets
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.raw_dir = get_absolute_path(self.config["paths"]["raw_dir"])
        os.makedirs(self.raw_dir, exist_ok=True)
        self.results_url = self.config["paths"]["raw_results_url"]
        self.shootouts_url = self.config["paths"]["raw_shootouts_url"]
        self.rankings_url = self.config["paths"]["raw_rankings_url"]
        self.confederations_url = self.config["paths"]["raw_confederations_url"]
        self.results_path = os.path.join(self.raw_dir, "results.csv")
        self.shootouts_path = os.path.join(self.raw_dir, "shootouts.csv")
        self.rankings_path = os.path.join(self.raw_dir, "rankings.csv")
        self.confederations_path = os.path.join(self.raw_dir, "confederations.csv")

    def _download_file(self, url: str, dest_path: str) -> None:
        # Download helper with local caching
        if os.path.exists(dest_path):
            logger.info(f"File exists: {dest_path}. Skipping.")
            return

        logger.info(f"Downloading from {url} to {dest_path}...")
        # print(f"[DEBUG _download_file] starting request url={url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
            logger.info("Download complete.")
            # print("[DEBUG _download_file] success")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise e

    def download_all(self) -> None:
        # Download all files
        self._download_file(self.results_url, self.results_path)
        self._download_file(self.shootouts_url, self.shootouts_path)
        self._download_file(self.rankings_url, self.rankings_path)
        self._download_file(self.confederations_url, self.confederations_path)

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Load datasets to pandas dataframes
        self.download_all()
        
        logger.info("Loading results...")
        df_results = pd.read_csv(self.results_path)
        
        logger.info("Loading shootouts...")
        df_shootouts = pd.read_csv(self.shootouts_path)
        
        logger.info("Loading rankings...")
        df_rankings = pd.read_csv(self.rankings_path)

        logger.info("Loading confederations...")
        df_confederations = pd.read_csv(self.confederations_path)
        
        # [DEBUG print shapes]
        # print(f"Raw results shape: {df_results.shape}")
        # print(f"Raw rankings shape: {df_rankings.shape}")
        
        # [UNSUITED MOCK PRE-CLEANUP BLOCK - for reference]
        # def verify_columns(df, name):
        #     missing = [col for col in ['date', 'home_team', 'away_team'] if col not in df.columns]
        #     if missing:
        #         print(f"[DEBUG WARNING] {name} is missing core columns {missing}")
        #     return len(missing) == 0

        logger.info("Raw data loaded successfully.")
        return df_results, df_shootouts, df_rankings, df_confederations

if __name__ == "__main__":
    loader = FootballDataLoader()
    r, s, rk, c = loader.load_raw_data()
    # print(r.head(2))
