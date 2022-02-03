from pathlib import Path

class Config:
    RANDOM_SEED = 57
    ASSETS_PATH = Path("./assets")
    DATASET_PATH = Path("./data")
    ORIGINAL_DATASET_FILE_PATH = DATASET_PATH / "dataset.csv"
    FEATURES_PATH = ASSETS_PATH / "features"
    MODELS_PATH = ASSETS_PATH / "models"
    METRICS_PATH = ASSETS_PATH / "metrics"
    METRICS_FILE_PATH = METRICS_PATH / "metrics.json"
