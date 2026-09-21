"""CLI script to train/fine-tune YOLO object detection models."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.config import load_config
from geovision.detection.train import train_yolo_detector
from geovision.logger import get_logger

logger = get_logger("geovision.scripts.train_detection")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO Object Detector on satellite imagery.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--smoke", action="store_true", help="Run 1-epoch smoke test on CPU (< 3 mins)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu, cuda, 0, etc.)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, smoke=args.smoke)

    if args.epochs is not None:
        cfg.detection.epochs = args.epochs
    if args.batch_size is not None:
        cfg.detection.batch_size = args.batch_size
    if args.device is not None:
        cfg.detection.device = args.device

    logger.info("=" * 65)
    logger.info(f"GeoVision Detection Training Pipeline (Smoke: {cfg.smoke})")
    logger.info("=" * 65)

    results = train_yolo_detector(cfg=cfg, smoke=cfg.smoke)
    logger.info("=" * 65)
    logger.info("Training Run Summary:")
    for k, v in results.get("metrics", {}).items():
        logger.info(f"  • {k}: {v}")
    logger.info(f"  • Weights Saved: {results.get('weights_path')}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
