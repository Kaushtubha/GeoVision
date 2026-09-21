"""CLI script to train Land-Cover Semantic Segmentation models."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.config import SegmentationConfig
from geovision.logger import get_logger
from geovision.segmentation.train import train_segmentation

logger = get_logger("geovision.scripts.train_segmentation")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoVision Land-Cover Semantic Segmentation model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/segmentation.yaml",
        help="Path to segmentation config YAML file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast 1-epoch CPU smoke-test training on small synthetic/cached data.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default=None,
        help="Override model architecture (e.g. Unet, FPN, DeepLabV3Plus).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Override encoder backbone (e.g. mobilenet_v3_small, resnet18).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)

    cfg = SegmentationConfig()
    if cfg_path.exists():
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            cfg = SegmentationConfig(**raw)

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.architecture is not None:
        cfg.architecture = args.architecture
    if args.encoder is not None:
        cfg.encoder_name = args.encoder
    if args.device is not None:
        cfg.device = args.device

    logger.info("Initializing GeoVision Segmentation training pipeline...")
    results = train_segmentation(
        config=cfg,
        smoke=args.smoke,
    )
    logger.info(f"Training completed successfully! Best checkpoint: {results['best_checkpoint']}")


if __name__ == "__main__":
    main()
