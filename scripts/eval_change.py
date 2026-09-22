"""CLI script to evaluate bi-temporal change detection models."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from geovision.change.evaluate import evaluate_change_detection
from geovision.change.losses import ChangeLoss
from geovision.change.model import SiameseChangeDetector
from geovision.constants import WEIGHTS_DIR
from geovision.data.datasets import LEVIRCDDataset
from geovision.data.downloaders import download_dataset
from geovision.logger import get_logger

logger = get_logger("geovision.scripts.eval_change")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoVision Bi-Temporal Change Detection model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(WEIGHTS_DIR / "change" / "best_change_detector.pth"),
        help="Path to trained model checkpoint .pth file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate on lightweight smoke dataset.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary change classification.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation on ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation metrics as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    if ckpt_path.exists():
        logger.info(f"Loading checkpoint from {ckpt_path}")
        model = SiameseChangeDetector.load(ckpt_path, device=args.device)
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found. Initializing baseline model for evaluation.")
        model = SiameseChangeDetector(device=args.device)

    dataset_path = download_dataset("levir_cd", smoke=args.smoke)
    dataset = LEVIRCDDataset(root_dir=dataset_path)
    dataloader = DataLoader(dataset, batch_size=2 if args.smoke else 4, shuffle=False)

    criterion = ChangeLoss()
    results = evaluate_change_detection(
        model=model,
        dataloader=dataloader,
        device=args.device,
        criterion=criterion,
        threshold=args.threshold,
    )

    logger.info("=== Bi-Temporal Change Detection Evaluation Summary ===")
    logger.info(f"Change-IoU: {results['change_iou']:.4f}")
    logger.info(f"F1-Score: {results['f1_score']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall: {results['recall']:.4f}")
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation report to {out_p}")


if __name__ == "__main__":
    main()
