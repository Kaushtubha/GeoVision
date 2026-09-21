"""Data loading, datasets, downloaders, and transform utilities for GeoVision."""

from geovision.data.datasets import (
    EuroSATDataset,
    LandCoverAIDataset,
    LEVIRCDDataset,
    NWPUVHR10Dataset,
)
from geovision.data.downloaders import (
    download_dataset,
    download_eurosat,
    download_landcover_ai,
    download_levir_cd,
    download_nwpu_vhr10,
    verify_dataset_integrity,
)
from geovision.data.statistics import calculate_dataset_stats
from geovision.data.transforms import (
    get_change_detection_transforms,
    get_classification_transforms,
    get_detection_transforms,
    get_segmentation_transforms,
)

__all__ = [
    "EuroSATDataset",
    "NWPUVHR10Dataset",
    "LandCoverAIDataset",
    "LEVIRCDDataset",
    "download_dataset",
    "download_eurosat",
    "download_nwpu_vhr10",
    "download_landcover_ai",
    "download_levir_cd",
    "verify_dataset_integrity",
    "calculate_dataset_stats",
    "get_classification_transforms",
    "get_detection_transforms",
    "get_segmentation_transforms",
    "get_change_detection_transforms",
]
