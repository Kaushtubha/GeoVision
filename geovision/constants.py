"""Constants and enumerated types across GeoVision."""

from enum import Enum
from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"
WEIGHTS_DIR = REPO_ROOT / "weights"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
MLRUNS_DIR = REPO_ROOT / "mlruns"


class TaskType(str, Enum):
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CHANGE_DETECTION = "change_detection"
    RETRIEVAL = "retrieval"
    VLM_GROUNDING = "vlm_grounding"


# LandCover.ai 5 class mapping (including Background)
LANDCOVER_AI_CLASSES = {
    0: "Background",
    1: "Building",
    2: "Woodland",
    3: "Water",
    4: "Road",
}

# NWPU VHR-10 10 detection classes
NWPU_VHR10_CLASSES = [
    "airplane",
    "ship",
    "storage_tank",
    "baseball_diamond",
    "tennis_court",
    "basketball_court",
    "ground_track_field",
    "harbor",
    "bridge",
    "vehicle",
]

# EuroSAT 10 land-use classes
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]
