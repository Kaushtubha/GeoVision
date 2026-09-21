"""CLI script to download datasets for GeoVision."""

import argparse
import sys
from pathlib import Path

# Ensure geovision is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.constants import RAW_DATA_DIR
from geovision.data.downloaders import download_dataset, verify_dataset_integrity
from geovision.logger import get_logger

logger = get_logger("geovision.scripts.download")


def parse_args():
    parser = argparse.ArgumentParser(description="Download datasets for GeoVision.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "eurosat", "nwpu_vhr10", "landcover_ai", "levir_cd"],
        help="Dataset to download (default: all)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Generate lightweight smoke-test dataset for fast CPU verification (< 5MB)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Output directory for raw data",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dest = Path(args.output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    targets = (
        ["eurosat", "nwpu_vhr10", "landcover_ai", "levir_cd"]
        if args.dataset == "all"
        else [args.dataset]
    )

    logger.info(f"Starting download pipeline (Smoke Mode: {args.smoke}) for: {targets}")

    for dataset_name in targets:
        logger.info(f"--- Processing {dataset_name} ---")
        dataset_path = download_dataset(dataset_name, target_dir=dest, smoke=args.smoke)
        report = verify_dataset_integrity(dataset_name, dataset_path)
        logger.info(f"Verification: {report['status']} ({report['total_files']} files)")

    logger.info("All requested datasets ready!")


if __name__ == "__main__":
    main()
