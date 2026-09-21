"""CLI script to evaluate YOLO object detector on test/validation set."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.detection.evaluate import evaluate_yolo_detector
from geovision.logger import get_logger

logger = get_logger("geovision.scripts.eval_detection")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Object Detector.")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Path to weights file")
    parser.add_argument("--data-yaml", type=str, default=None, help="Path to data.yaml dataset config")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--smoke", action="store_true", help="Evaluate on smoke dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 65)
    logger.info("GeoVision Object Detection Evaluation Benchmark")
    logger.info("=" * 65)

    results = evaluate_yolo_detector(
        weights_path=args.weights,
        data_yaml=args.data_yaml,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        smoke=args.smoke,
    )

    logger.info("\n" + "=" * 65)
    logger.info("Macro Metrics:")
    logger.info(f"  • mAP@50:     {results['mAP@50']:.4f}")
    logger.info(f"  • mAP@75:     {results['mAP@75']:.4f}")
    logger.info(f"  • mAP@50-95:  {results['mAP@50-95']:.4f}")
    logger.info("-" * 65)
    logger.info("Per-Class AP@50 Breakdown:")
    for cls_name, ap_val in results.get("per_class_AP@50", {}).items():
        logger.info(f"  • {cls_name:<20}: {ap_val:.4f}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
