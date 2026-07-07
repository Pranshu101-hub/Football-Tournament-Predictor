import os
import logging
import yaml
from typing import Any, Dict

def setup_logger(name: str = "football_predictor", log_file: str = "project.log", level: int = logging.INFO) -> logging.Logger:
    # Setup logger with console and file handlers
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    c_handler = logging.StreamHandler()
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File handler
    try:
        f_handler = logging.FileHandler(log_file, encoding="utf-8")
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
    except Exception as e:
        # print(f"[DEBUG LOG SETUP FAILED] name={name} err={e}")
        print(f"Warning: Could not create log file {log_file} due to: {e}")

    # [OPTIONAL alternative log levels]
    # if level == logging.DEBUG:
    #     logger.setLevel(logging.DEBUG)

    return logger

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    # Load project YAML config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # print(f"[DEBUG load_config] loaded config paths: {config.get('paths')}")
    return config

def get_absolute_path(relative_path: str) -> str:
    # Get absolute path from project root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root_dir, relative_path)
