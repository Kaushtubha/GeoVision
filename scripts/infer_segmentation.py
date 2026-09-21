"""CLI script to run Land-Cover Semantic Segmentation inference and area reporting."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.constants import WEIGHTS_DIR
from geovision.data.downloaders import download_dataset
from geovision.logger import get_logger
from geovision.segmentation.infer import segment_image

logger = get_logger("geovision.scripts.infer_segmentation")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Land-Cover Semantic Segmentation inference on an aerial/satellite image.")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input aerial image or GeoTIFF file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(WEIGHTS_DIR / "segmentation" / "best_segmenter.pth"),
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-mask",
        type=str,
        default="experiments/runs/segmentation/predicted_mask.png",
        help="Path to save output integer class mask PNG/GeoTIFF.",
    )
    parser.add_argument(
        "--output-overlay",
        type=str,
        default="experiments/runs/segmentation/overlay.png",
        help="Path to save colorized visual overlay image.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="experiments/runs/segmentation/distribution.json",
        help="Path to save land-cover class area distribution JSON report.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Sliding window tile size in pixels.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Overlap in pixels between adjacent sliding windows.",
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=0.5,
        help="Ground Sampling Distance in meters/pixel (default: 0.5m for LandCover.ai).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run inference in smoke mode on a sample tile.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = args.image
    if image_path is None or not Path(image_path).exists():
        if args.smoke or image_path is None:
            logger.info("No input image specified or file not found. Fetching smoke LandCover.ai sample...")
            dataset_dir = download_dataset("landcover_ai", smoke=True)
            img_candidates = list((dataset_dir / "images").glob("*.png")) + list((dataset_dir / "images").glob("*.jpg"))
            if img_candidates:
                image_path = img_candidates[0]
                logger.info(f"Using smoke test sample: {image_path}")
            else:
                # Generate synthetic test image
                image_path = Path("experiments/runs/segmentation/sample_test.png")
                image_path.parent.mkdir(parents=True, exist_ok=True)
                sample_arr = np.random.randint(40, 220, size=(512, 512, 3), dtype=np.uint8)
                Image.fromarray(sample_arr).save(image_path)
                logger.info(f"Generated synthetic test image: {image_path}")
        else:
            raise FileNotFoundError(f"Input image does not exist: {image_path}")

    ckpt = args.checkpoint if Path(args.checkpoint).exists() else None

    logger.info(f"Running Land-Cover Segmentation Inference on: {image_path}")
    result = segment_image(
        image_input=image_path,
        checkpoint_path=ckpt,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device,
        gsd_meters=args.gsd,
        output_mask_path=args.output_mask,
        output_overlay_path=args.output_overlay,
    )

    logger.info("=== Land-Cover Area Distribution Summary ===")
    dist = result["distribution"]
    logger.info(f"Total Area: {dist.get('total_area_sq_km', 'N/A')} km² ({dist['total_pixels']:,} pixels)")
    for cls_name, pct in dist["percentages"].items():
        area_ha = dist.get("area_hectares", {}).get(cls_name, 0.0)
        logger.info(f"  - {cls_name:12s}: {pct:6.2f}% ({area_ha:8.4f} ha | {dist['pixel_counts'][cls_name]:,} px)")

    if args.output_json:
        out_json_path = Path(args.output_json)
        out_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(dist, f, indent=2)
        logger.info(f"Saved distribution report to {out_json_path}")

    logger.info(f"Saved output mask to: {result['saved_files'].get('mask_path')}")
    logger.info(f"Saved visual overlay to: {result['saved_files'].get('overlay_path')}")


if __name__ == "__main__":
    main()
