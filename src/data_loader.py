import os
import requests
import pandas as pd
from typing import Tuple, Optional
from src.utils import setup_logger, load_config, get_absolute_path

logger = setup_logger("data_loader")

class FootballDataLoader:
    """Class responsible for fetching, caching, and loading raw football datasets."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.raw_dir = get_absolute_path(self.config["paths"]["raw_dir"])
        os.makedirs(self.raw_dir, exist_ok=True)
        
        self.results_url = self.config["paths"]["raw_results_url"]
        self.shootouts_url = self.config["paths"]["raw_shootouts_url"]
        self.rankings_url = self.config["paths"]["raw_rankings_url"]
        
        self.results_path = os.path.join(self.raw_dir, "results.csv")
        self.shootouts_path = os.path.join(self.raw_dir, "shootouts.csv")
        self.rankings_path = os.path.join(self.raw_dir, "rankings.csv")
        self.confederations_url = self.config["paths"]["raw_confederations_url"]
        self.confederations_path = os.path.join(self.raw_dir, "confederations.csv")

    def _download_file(self, url: str, dest_path: str) -> None:
        """Downloads a file from a URL to a local destination if it doesn't exist."""
        if os.path.exists(dest_path):
            logger.info(f"File already exists: {dest_path}. Skipping download.")
            return

        logger.info(f"Downloading data from {url} to {dest_path}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
            logger.info("Download completed successfully.")
        except Exception as e:
            logger.error(f"Failed to download from {url}: {e}")
            raise e

    def download_all(self) -> None:
        """Downloads all configured datasets."""
        self._download_file(self.results_url, self.results_path)
        self._download_file(self.shootouts_url, self.shootouts_path)
        self._download_file(self.rankings_url, self.rankings_path)
        self._download_file(self.confederations_url, self.confederations_path)

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads and returns the raw datasets as Pandas DataFrames.
        
        Downloads them first if they are not already cached.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames of results, shootouts, rankings, confederations.
        """
        self.download_all()
        
        logger.info("Loading results dataset...")
        df_results = pd.read_csv(self.results_path)
        
        logger.info("Loading shootouts dataset...")
        df_shootouts = pd.read_csv(self.shootouts_path)
        
        logger.info("Loading rankings dataset...")
        df_rankings = pd.read_csv(self.rankings_path)

        logger.info("Loading confederations dataset...")
        df_confederations = pd.read_csv(self.confederations_path)
        
        logger.info(
            f"Loaded results: {df_results.shape}, shootouts: {df_shootouts.shape}, "
            f"rankings: {df_rankings.shape}, confederations: {df_confederations.shape}"
        )
        return df_results, df_shootouts, df_rankings, df_confederations

if __name__ == "__main__":
    # Test script running directly
    loader = FootballDataLoader()
    r, s, rk, c = loader.load_raw_data()
    print("Results head:")
    print(r.head(2))
    print("\nRankings head:")
    print(rk.head(2))
    print("\nConfederations head:")
    print(c.head(2))
