"""CLI script to search satellite scenes using natural language text prompts or image queries."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from geovision.config import RetrievalConfig
from geovision.logger import get_logger
from geovision.retrieval.embedder import SatelliteSceneEmbedder
from geovision.retrieval.index import SceneVectorIndex

logger = get_logger("geovision.scripts.query_scene")


def create_result_collage(
    results: list[dict],
    output_path: Path,
    thumb_size: tuple[int, int] = (160, 160),
) -> None:
    """Create a visual collage of top-k retrieved satellite images with scores."""
    valid_results = [r for r in results if Path(r["payload"].get("image_path", "")).exists()]
    if not valid_results:
        return

    n = len(valid_results)
    collage = Image.new("RGB", (n * thumb_size[0], thumb_size[1] + 40), color=(20, 20, 20))
    draw = ImageDraw.Draw(collage)

    for i, res in enumerate(valid_results):
        img_p = Path(res["payload"]["image_path"])
        with Image.open(img_p) as im:
            thumb = im.convert("RGB").resize(thumb_size)

        x_offset = i * thumb_size[0]
        collage.paste(thumb, (x_offset, 0))

        score_text = f"#{i+1} {res['payload'].get('class_name', '')[:10]}\nSim: {res['score']:.3f}"
        draw.text((x_offset + 5, thumb_size[1] + 5), score_text, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(output_path)
    logger.info(f"Saved visual retrieval collage to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Query satellite scene database using text or image.")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Natural language text query (e.g. 'river near agricultural fields').",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to query image.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest scene matches to return.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/retrieval.yaml",
        help="Path to retrieval config YAML file.",
    )
    parser.add_argument(
        "--qdrant-path",
        type=str,
        default=None,
        help="Path to Qdrant vector database storage.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="experiments/runs/retrieval/query_results.json",
        help="Path to save ranked search results as JSON.",
    )
    parser.add_argument(
        "--output-collage",
        type=str,
        default="experiments/runs/retrieval/retrieval_collage.png",
        help="Path to save top-K image collage.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run query in smoke mode with automatic sample fallback.",
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
        db_path=qdrant_path,
        in_memory=False,
    )

    # If index is empty or in smoke mode, populate with sample scenes if needed
    if index.count() == 0 or args.smoke:
        from geovision.data.datasets import EuroSATDataset
        from geovision.data.downloaders import download_dataset
        logger.info("Vector index has no entries or smoke mode active. Indexing sample scenes...")
        dataset_path = download_dataset("eurosat", smoke=True)
        ds = EuroSATDataset(root_dir=dataset_path)
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
        from geovision.retrieval.evaluate import extract_embeddings_and_labels
        embs, lbls, paths = extract_embeddings_and_labels(embedder, loader)
        from geovision.constants import EUROSAT_CLASSES
        payloads = [
            {"image_path": paths[i], "class_name": EUROSAT_CLASSES[lbls[i]]}
            for i in range(len(embs))
        ]
        index.upsert_scenes(embs, payloads)

    query_text = args.text or "Forest area with trees and vegetation"
    if args.image and Path(args.image).exists():
        logger.info(f"Searching by image query: {args.image}")
        hits = index.search_by_image(args.image, embedder, top_k=args.top_k)
    else:
        logger.info(f"Searching by text query: '{query_text}'")
        hits = index.search_by_text(query_text, embedder, top_k=args.top_k)

    logger.info(f"--- Top {len(hits)} Retrieval Results ---")
    for rank, hit in enumerate(hits, 1):
        c_name = hit["payload"].get("class_name", "Unknown")
        p = hit["payload"].get("image_path", "")
        logger.info(f"  #{rank}: Class='{c_name}' | Cosine Sim={hit['score']:.4f} | Path={p}")

    if args.output_json:
        out_j = Path(args.output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        with open(out_j, "w", encoding="utf-8") as f:
            json.dump(hits, f, indent=2)
        logger.info(f"Saved results to {out_j}")

    if args.output_collage:
        create_result_collage(hits, Path(args.output_collage))


if __name__ == "__main__":
    main()
