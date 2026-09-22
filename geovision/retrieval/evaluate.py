"""Evaluation routine for satellite scene retrieval and zero-shot metric learning."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from geovision.constants import EUROSAT_CLASSES
from geovision.logger import get_logger
from geovision.metrics.retrieval import (
    evaluate_retrieval_metrics,
    evaluate_zeroshot_accuracy,
)
from geovision.retrieval.embedder import SatelliteSceneEmbedder

logger = get_logger("geovision.retrieval.evaluate")


def extract_embeddings_and_labels(
    embedder: SatelliteSceneEmbedder,
    dataloader: DataLoader,
    progress_bar: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract normalized feature embeddings, labels, and file paths across a dataloader."""
    embedder.eval()
    all_embeddings = []
    all_labels = []
    all_paths = []

    pbar = tqdm(dataloader, desc="Extracting Scene Embeddings", disable=not progress_bar)
    with torch.no_grad():
        for batch in pbar:
            images = batch["image"]
            labels = batch["label"]
            paths = batch.get("path", [])

            emb = embedder.encode_image(images)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.append(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
            all_paths.extend(paths if isinstance(paths, list) else list(paths))

    embeddings_arr = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, embedder.embedding_dim))
    labels_arr = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.int64)

    return embeddings_arr, labels_arr, all_paths


def evaluate_retrieval(
    embedder: SatelliteSceneEmbedder,
    dataloader: DataLoader,
    class_names: list[str] | None = None,
    k_values: tuple[int, ...] = (1, 5, 10),
    progress_bar: bool = True,
) -> dict[str, Any]:
    """Benchmark image-to-image retrieval and zero-shot classification on a dataset loader.

    Args:
        embedder: SatelliteSceneEmbedder instance.
        dataloader: PyTorch DataLoader for EuroSAT dataset.
        class_names: List of class strings.
        k_values: Top-K values for Recall evaluation.
        progress_bar: Whether to show progress bars.

    Returns:
        Dictionary of Recall@K, MRR, mAP, and zero-shot accuracy metrics.
    """
    classes = class_names or EUROSAT_CLASSES

    # 1. Extract visual embeddings
    embeddings, labels, paths = extract_embeddings_and_labels(
        embedder=embedder,
        dataloader=dataloader,
        progress_bar=progress_bar,
    )

    if len(embeddings) == 0:
        logger.warning("No samples found for retrieval evaluation.")
        return {"recall@1": 0.0, "recall@5": 0.0, "mrr": 0.0, "map": 0.0}

    # 2. Compute image-to-image scene retrieval metrics
    retrieval_res = evaluate_retrieval_metrics(
        query_embeddings=embeddings,
        query_labels=labels,
        gallery_embeddings=embeddings,
        gallery_labels=labels,
        k_values=k_values,
        exclude_self=True,
    )

    # 3. Compute zero-shot classification accuracy with text prompts
    text_embeddings = embedder.build_class_text_embeddings(class_names=classes)
    zeroshot_res = evaluate_zeroshot_accuracy(
        image_embeddings=embeddings,
        labels=labels,
        text_embeddings=text_embeddings,
        top_k=(1, 5),
    )

    combined_results = {
        **retrieval_res,
        **zeroshot_res,
        "num_evaluated_scenes": len(embeddings),
    }

    logger.info(
        f"Retrieval Benchmark -> Recall@1: {combined_results.get('recall@1', 0.0):.4f} | "
        f"Recall@5: {combined_results.get('recall@5', 0.0):.4f} | "
        f"MRR: {combined_results.get('mrr', 0.0):.4f} | "
        f"Zero-Shot Top-1: {combined_results.get('zeroshot_top1_accuracy', 0.0):.4f}"
    )

    return combined_results
