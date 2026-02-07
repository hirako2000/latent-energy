import shutil
from huggingface_hub import hf_hub_download
from grid_energy.utils.config import settings

NONOGRAM_SUBSETS = [
    "nonogram_5x5",
    "nonogram_8x8",
    "nonogram_12x12"
]

def fetch_bronze_data():
    """
    Downloads raw parquet files from Hugging Face as the Bronze layer.
    """
    token = settings.HF_TOKEN if settings.HF_TOKEN.strip() else None
    
    if not token:
        print("Notice: No HF_TOKEN found. Proceeding with public access.")

    settings.BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    for subset in NONOGRAM_SUBSETS:
        filename = f"{subset}/test-00000-of-00001.parquet"
        
        print(f"Fetching {subset}...")
        try:
            local_cached_path = hf_hub_download(
                repo_id=settings.HF_DATASET_ID,
                filename=filename,
                repo_type="dataset",
                token=token
            )
            
            target_dir = settings.BRONZE_DIR / subset
            target_dir.mkdir(exist_ok=True)
            
            target_path = target_dir / "test.parquet"
            shutil.copy(local_cached_path, target_path)
            
            print(f"  ✓ Stored: {target_path}")
        except Exception as e:
            print(f"  ✗ Failed to download {subset}: {e}")

if __name__ == "__main__":
    fetch_bronze_data()