"""CLI script to train/fine-tune multimodal satellite scene embedders."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.config import RetrievalConfig
from geovision.logger import get_logger
from geovision.retrieval.train import train_retrieval_embedder

logger = get_logger("geovision.scripts.train_retrieval")


def parse_args():
    parser = argparse.ArgumentParser(description="Train / fine-tune GeoVision satellite scene embedder.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval.yaml",
        help="Path to retrieval config YAML file.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run fast 1-epoch CPU smoke-test training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for AdamW.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu' or 'cuda').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config)

    cfg = RetrievalConfig()
    if cfg_path.exists():
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            cfg = RetrievalConfig(**raw)

    cfg.device = args.device

    logger.info("Starting Satellite Scene Retrieval training...")
    results = train_retrieval_embedder(
        config=cfg,
        smoke=args.smoke,
        epochs=args.epochs,
        learning_rate=args.lr,
    )
    logger.info(f"Training completed successfully! Best checkpoint: {results['best_checkpoint']}")


if __name__ == "__main__":
    main()
