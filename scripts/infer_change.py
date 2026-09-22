"""CLI script to run bi-temporal change detection inference and generate visual reports."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.change.infer import detect_changes
from geovision.constants import WEIGHTS_DIR
from geovision.data.downloaders import download_dataset
from geovision.logger import get_logger

logger = get_logger("geovision.scripts.infer_change")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Bi-Temporal Change Detection inference on image pairs.")
    parser.add_argument(
        "--image-a",
        type=str,
        default=None,
        help="Path to Pre-event (T1) image or GeoTIFF.",
    )
    parser.add_argument(
        "--image-b",
        type=str,
        default=None,
        help="Path to Post-event (T2) image or GeoTIFF.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(WEIGHTS_DIR / "change" / "best_change_detector.pth"),
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-mask",
        type=str,
        default="experiments/runs/change/change_mask.png",
        help="Path to save binary change mask PNG/GeoTIFF.",
    )
    parser.add_argument(
        "--output-overlay",
        type=str,
        default="experiments/runs/change/change_overlay.png",
        help="Path to save change highlight overlay on T2 image.",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="experiments/runs/change/tri_panel_report.png",
        help="Path to save 3-panel comparison image (T1, T2, Change Overlay).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="experiments/runs/change/change_stats.json",
        help="Path to save quantitative change shift JSON report.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=256,
        help="Sliding window tile size.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=32,
        help="Overlap in pixels between adjacent tiles.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Change probability threshold.",
    )
    parser.add_argument(
        "--gsd",
        type=float,
        default=0.5,
        help="Ground Sampling Distance in meters/pixel (default: 0.5m for LEVIR-CD).",
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
        help="Run inference in smoke mode on sample bi-temporal pair.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    img_a_path = args.image_a
    img_b_path = args.image_b

    if img_a_path is None or img_b_path is None or not Path(img_a_path).exists() or not Path(img_b_path).exists():
        if args.smoke or img_a_path is None:
            logger.info("No input pair specified or files not found. Fetching smoke LEVIR-CD sample...")
            dataset_dir = download_dataset("levir_cd", smoke=True)
            a_candidates = list((dataset_dir / "A").glob("*.png")) + list((dataset_dir / "A").glob("*.jpg"))
            if a_candidates:
                img_a_path = a_candidates[0]
                img_b_path = dataset_dir / "B" / img_a_path.name
                logger.info(f"Using smoke test pair: {img_a_path.name}")
            else:
                out_dir = Path("experiments/runs/change")
                out_dir.mkdir(parents=True, exist_ok=True)
                img_a_path = out_dir / "sample_t1.png"
                img_b_path = out_dir / "sample_t2.png"
                sample_a = np.random.randint(50, 180, size=(256, 256, 3), dtype=np.uint8)
                sample_b = sample_a.copy()
                # Add simulated building change
                sample_b[50:120, 50:120] = np.random.randint(190, 255, size=(70, 70, 3), dtype=np.uint8)
                Image.fromarray(sample_a).save(img_a_path)
                Image.fromarray(sample_b).save(img_b_path)
                logger.info(f"Generated synthetic test pair at {out_dir}")
        else:
            raise FileNotFoundError(f"One or both input images not found: {img_a_path}, {img_b_path}")

    ckpt = args.checkpoint if Path(args.checkpoint).exists() else None

    logger.info(f"Running Bi-Temporal Change Detection Inference on: {img_a_path} and {img_b_path}")
    result = detect_changes(
        image_a=img_a_path,
        image_b=img_b_path,
        checkpoint_path=ckpt,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
        device=args.device,
        gsd_meters=args.gsd,
        output_mask_path=args.output_mask,
        output_overlay_path=args.output_overlay,
        output_report_path=args.output_report,
    )

    logger.info("=== Quantitative Change Shift Statistics ===")
    stats = result["statistics"]
    logger.info(f"Total Area: {stats.get('total_area_sq_km', 'N/A')} km² ({stats['total_pixels']:,} pixels)")
    logger.info(
        f"Altered / Changed Area: {stats.get('changed_area_hectares', 'N/A')} ha "
        f"({stats.get('changed_area_sq_km', 'N/A')} km² | {stats['changed_pixels']:,} px | {stats['change_percentage']:.2f}%)"
    )

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved change shift statistics to {out_json}")

    logger.info(f"Saved binary change mask to: {result['saved_files'].get('mask_path')}")
    logger.info(f"Saved change overlay to: {result['saved_files'].get('overlay_path')}")
    logger.info(f"Saved 3-panel report to: {result['saved_files'].get('report_path')}")


if __name__ == "__main__":
    main()
