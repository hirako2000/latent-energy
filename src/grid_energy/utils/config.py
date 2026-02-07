import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Project Root (calculated relative to this file)
    ROOT_DIR: Path = Path(__file__).resolve().parents[3]
    
    # HF Settings
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")
    HF_DATASET_ID: str = os.getenv("HF_DATASET_ID", "VGRP-Bench/VGRP-Bench")
    
    # Storage Paths
    BRONZE_DIR: Path = ROOT_DIR / os.getenv("BRONZE_PATH", "data/bronze")
    SILVER_DIR: Path = ROOT_DIR / os.getenv("SILVER_PATH", "data/silver")
    GOLD_DIR: Path = ROOT_DIR / os.getenv("GOLD_PATH", "data/gold")

settings = Settings()