"""CLI script to benchmark satellite scene retrieval and zero-shot classification."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from torch.utils.data import DataLoader

from geovision.config import RetrievalConfig
from geovision.constants import EUROSAT_CLASSES
from geovision.data.datasets import EuroSATDataset
from geovision.data.downloaders import download_dataset
from geovision.logger import get_logger
from geovision.retrieval.embedder import SatelliteSceneEmbedder
from geovision.retrieval.evaluate import evaluate_retrieval

logger = get_logger("geovision.scripts.eval_retrieval")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate satellite scene retrieval metrics on EuroSAT.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval.yaml",
        help="Path to retrieval config YAML file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pth).",
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
        help="Computation device ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="experiments/runs/retrieval/eval_metrics.json",
        help="Path to save evaluation metrics as JSON.",
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

    if args.checkpoint and Path(args.checkpoint).exists():
        logger.info(f"Loading embedder from checkpoint: {args.checkpoint}")
        embedder = SatelliteSceneEmbedder.load(args.checkpoint, device=args.device)
    else:
        embedder = SatelliteSceneEmbedder(
            model_name=cfg.model_name,
            pretrained=cfg.pretrained if not args.smoke else None,
            embedding_dim=cfg.embedding_dim,
            device=args.device,
        )

    dataset_path = download_dataset("eurosat", smoke=args.smoke)
    dataset = EuroSATDataset(root_dir=dataset_path, split="all")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    results = evaluate_retrieval(
        embedder=embedder,
        dataloader=loader,
        class_names=EUROSAT_CLASSES,
        k_values=(1, 5, 10),
    )

    logger.info("=== Satellite Scene Retrieval Benchmark Results ===")
    logger.info(f"Recall@1 : {results.get('recall@1', 0.0):.4f}")
    logger.info(f"Recall@5 : {results.get('recall@5', 0.0):.4f}")
    logger.info(f"Recall@10: {results.get('recall@10', 0.0):.4f}")
    logger.info(f"MRR      : {results.get('mrr', 0.0):.4f}")
    logger.info(f"mAP      : {results.get('map', 0.0):.4f}")
    logger.info(f"Zero-Shot Top-1 Accuracy: {results.get('zeroshot_top1_accuracy', 0.0):.4f}")
    logger.info(f"Zero-Shot Top-5 Accuracy: {results.get('zeroshot_top5_accuracy', 0.0):.4f}")

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation metrics to {out_p}")


if __name__ == "__main__":
    main()
