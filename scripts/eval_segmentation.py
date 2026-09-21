"""CLI script to evaluate Land-Cover Semantic Segmentation models."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from geovision.constants import LANDCOVER_AI_CLASSES, WEIGHTS_DIR
from geovision.data.datasets import LandCoverAIDataset
from geovision.data.downloaders import download_dataset
from geovision.logger import get_logger
from geovision.segmentation.evaluate import evaluate_segmentation
from geovision.segmentation.losses import DiceCELoss
from geovision.segmentation.model import LandCoverSegmenter

logger = get_logger("geovision.scripts.eval_segmentation")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GeoVision Land-Cover Semantic Segmentation model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(WEIGHTS_DIR / "segmentation" / "best_segmenter.pth"),
        help="Path to trained model checkpoint .pth file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate on lightweight smoke dataset.",
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
        help="Optional path to save evaluation results as JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)

    if ckpt_path.exists():
        logger.info(f"Loading checkpoint from {ckpt_path}")
        model = LandCoverSegmenter.load(ckpt_path, device=args.device)
    else:
        logger.warning(f"Checkpoint {ckpt_path} not found. Initializing base model for evaluation.")
        model = LandCoverSegmenter(device=args.device)

    dataset_path = download_dataset("landcover_ai", smoke=args.smoke)
    dataset = LandCoverAIDataset(root_dir=dataset_path)
    dataloader = DataLoader(dataset, batch_size=2 if args.smoke else 4, shuffle=False)

    criterion = DiceCELoss()
    results = evaluate_segmentation(
        model=model,
        dataloader=dataloader,
        device=args.device,
        criterion=criterion,
        num_classes=5,
        class_names=LANDCOVER_AI_CLASSES,
    )

    logger.info("--- Evaluation Summary ---")
    logger.info(f"Mean IoU: {results['mIoU']:.4f}")
    logger.info(f"Mean Dice: {results['mean_dice']:.4f}")
    logger.info(f"Overall Accuracy: {results['overall_accuracy']:.4f}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation metrics to {out_p}")


if __name__ == "__main__":
    main()
