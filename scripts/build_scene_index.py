"""CLI script to embed and index EuroSAT satellite imagery into the Qdrant vector database."""

import argparse
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
from geovision.retrieval.evaluate import extract_embeddings_and_labels
from geovision.retrieval.index import SceneVectorIndex

logger = get_logger("geovision.scripts.build_scene_index")


def parse_args():
    parser = argparse.ArgumentParser(description="Build satellite scene vector index in Qdrant.")
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
        help="Optional path to fine-tuned embedder checkpoint (.pth).",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=None,
        help="Path to local Qdrant vector database storage directory.",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Run vector index purely in memory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Index lightweight smoke dataset subset.",
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

    qdrant_path = args.qdrant_path or cfg.qdrant_path

    # Initialize embedder
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

    # Initialize vector index
    index = SceneVectorIndex(
        collection_name=cfg.collection_name,
        vector_size=cfg.embedding_dim,
        db_path=qdrant_path if not args.in_memory else None,
        in_memory=args.in_memory,
    )
    index.reset()

    # Load dataset
    dataset_path = download_dataset("eurosat", smoke=args.smoke)
    dataset = EuroSATDataset(root_dir=dataset_path, split="all")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    logger.info(f"Extracting embeddings for {len(dataset)} scenes...")
    embeddings, labels, paths = extract_embeddings_and_labels(embedder, loader)

    payloads = [
        {
            "image_path": paths[i],
            "class_label": int(labels[i]),
            "class_name": EUROSAT_CLASSES[labels[i]] if labels[i] < len(EUROSAT_CLASSES) else "Unknown",
        }
        for i in range(len(embeddings))
    ]

    indexed_count = index.upsert_scenes(embeddings, payloads)
    logger.info(f"Successfully indexed {indexed_count} satellite scenes into Qdrant collection '{cfg.collection_name}'!")


if __name__ == "__main__":
    main()
