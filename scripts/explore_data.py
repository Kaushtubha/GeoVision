"""CLI script to inspect datasets and compute statistical distributions."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.constants import RAW_DATA_DIR
from geovision.data.statistics import calculate_dataset_stats
from geovision.logger import get_logger

logger = get_logger("geovision.scripts.explore")


def main():
    parser = argparse.ArgumentParser(description="Explore dataset statistics in GeoVision.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Path to raw datasets directory",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    dataset_map = {
        "EuroSAT": ("EuroSAT", "classification"),
        "NWPU_VHR10": ("NWPU_VHR10", "detection"),
        "LandCoverAI": ("LandCoverAI", "segmentation"),
        "LEVIR_CD": ("LEVIR_CD", "change"),
    }

    logger.info("=" * 60)
    logger.info("GeoVision Dataset Statistical Summary")
    logger.info("=" * 60)

    found = False
    for label, (folder_name, task) in dataset_map.items():
        ds_path = data_dir / folder_name
        if ds_path.exists():
            found = True
            stats = calculate_dataset_stats(ds_path, task)
            logger.info(f"\n[Dataset: {label}] (Task: {task})")
            logger.info(f"  • Total Samples: {stats['total_samples']}")
            logger.info(f"  • Disk Footprint: {stats['total_disk_mb']} MB")
            if stats.get("class_counts"):
                logger.info(f"  • Class Distribution: {stats['class_counts']}")
            if stats.get("unique_classes_found"):
                logger.info(f"  • Unique Mask Classes: {stats['unique_classes_found']}")
            if stats.get("image_shapes"):
                logger.info(f"  • Sample Dimensions: {stats['image_shapes']}")

    if not found:
        logger.warning(
            f"No processed datasets found in {data_dir}. Run `python scripts/download_datasets.py --smoke` first!"
        )


if __name__ == "__main__":
    main()
