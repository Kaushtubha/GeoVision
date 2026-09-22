"""CLI script to train bi-temporal satellite change detection models."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.change.train import train_change_detection
from geovision.config import ChangeConfig
from geovision.logger import get_logger

logger = get_logger("geovision.scripts.train_change")


def parse_args():
    parser = argparse.ArgumentParser(description="Train GeoVision Bi-Temporal Change Detection model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/change.yaml",
        help="Path to change detection config YAML file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast 1-epoch CPU smoke-test training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Override encoder backbone (e.g. resnet18, cnn).",
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

    cfg = ChangeConfig()
    if cfg_path.exists():
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            cfg = ChangeConfig(**raw)

    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.encoder is not None:
        cfg.encoder_name = args.encoder
    if args.device is not None:
        cfg.device = args.device

    logger.info("Initializing GeoVision Bi-Temporal Change Detection training pipeline...")
    results = train_change_detection(
        config=cfg,
        smoke=args.smoke,
    )
    logger.info(f"Training completed successfully! Best checkpoint: {results['best_checkpoint']}")


if __name__ == "__main__":
    main()
