"""Evaluation metrics for satellite scene retrieval and metric learning (Recall@K, MRR, mAP, Zero-Shot Accuracy)."""

from typing import Any

import numpy as np

from geovision.logger import get_logger

logger = get_logger("geovision.metrics.retrieval")


def evaluate_retrieval_metrics(
    query_embeddings: np.ndarray | Any,
    query_labels: np.ndarray | Any,
    gallery_embeddings: np.ndarray | Any,
    gallery_labels: np.ndarray | Any,
    k_values: tuple[int, ...] = (1, 5, 10),
    exclude_self: bool = False,
) -> dict[str, float]:
    """Compute Recall@K, Mean Reciprocal Rank (MRR), and Mean Average Precision (mAP) for vector retrieval.

    Args:
        query_embeddings: Normalized query vectors of shape (N_q, D).
        query_labels: Class labels of queries of shape (N_q,).
        gallery_embeddings: Normalized gallery vectors of shape (N_g, D).
        gallery_labels: Class labels of gallery items of shape (N_g,).
        k_values: K thresholds to compute recall at.
        exclude_self: If True, excludes the top-1 self match (useful when query == gallery).

    Returns:
        Dictionary containing Recall@1, Recall@5, Recall@10, MRR, and mAP.
    """
    if hasattr(query_embeddings, "detach"):
        query_embeddings = query_embeddings.detach().cpu().numpy()
    if hasattr(query_labels, "detach"):
        query_labels = query_labels.detach().cpu().numpy()
    if hasattr(gallery_embeddings, "detach"):
        gallery_embeddings = gallery_embeddings.detach().cpu().numpy()
    if hasattr(gallery_labels, "detach"):
        gallery_labels = gallery_labels.detach().cpu().numpy()

    q_emb = np.asarray(query_embeddings, dtype=np.float32)
    q_lbl = np.asarray(query_labels, dtype=np.int64)
    g_emb = np.asarray(gallery_embeddings, dtype=np.float32)
    g_lbl = np.asarray(gallery_labels, dtype=np.int64)

    # Normalize vectors if not already unit length
    q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
    g_norm = np.linalg.norm(g_emb, axis=1, keepdims=True)
    q_emb = q_emb / np.maximum(q_norm, 1e-8)
    g_emb = g_emb / np.maximum(g_norm, 1e-8)

    # Compute cosine similarity matrix: (N_q, N_g)
    similarity_matrix = np.dot(q_emb, g_emb.T)
    n_queries = len(q_lbl)

    recall_counts = {k: 0 for k in k_values}
    reciprocal_ranks = []
    avg_precisions = []

    for i in range(n_queries):
        target_label = q_lbl[i]
        sims = similarity_matrix[i]

        # Sort indices in descending order of similarity
        sorted_indices = np.argsort(-sims)

        if exclude_self:
            # Exclude self if query is in gallery
            sorted_indices = sorted_indices[sorted_indices != i]

        retrieved_labels = g_lbl[sorted_indices]
        matches = (retrieved_labels == target_label)

        # Compute Recall@K
        for k in k_values:
            top_k_matches = matches[:k]
            if np.any(top_k_matches):
                recall_counts[k] += 1

        # Compute MRR (first matching rank)
        match_positions = np.where(matches)[0]
        if len(match_positions) > 0:
            first_rank = match_positions[0] + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / first_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Compute Average Precision (AP)
        if len(match_positions) > 0:
            cum_matches = np.cumsum(matches)
            ranks = np.arange(1, len(matches) + 1)
            precisions = cum_matches / ranks
            ap = np.sum(precisions * matches) / len(match_positions)
            avg_precisions.append(ap)
        else:
            avg_precisions.append(0.0)

    results = {}
    for k in k_values:
        results[f"recall@{k}"] = float(recall_counts[k] / n_queries) if n_queries > 0 else 0.0

    results["mrr"] = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    results["map"] = float(np.mean(avg_precisions)) if avg_precisions else 0.0

    return results


def evaluate_zeroshot_accuracy(
    image_embeddings: np.ndarray | Any,
    labels: np.ndarray | Any,
    text_embeddings: np.ndarray | Any,
    top_k: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Evaluate zero-shot classification accuracy by comparing image embeddings to text class embeddings.

    Args:
        image_embeddings: (N, D) normalized image feature vectors.
        labels: (N,) true integer class indices.
        text_embeddings: (C, D) normalized text prompt vectors for all C classes.
        top_k: Top-K accuracy thresholds (e.g., top-1, top-5).

    Returns:
        Dictionary of zero-shot accuracy metrics.
    """
    if hasattr(image_embeddings, "detach"):
        image_embeddings = image_embeddings.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    if hasattr(text_embeddings, "detach"):
        text_embeddings = text_embeddings.detach().cpu().numpy()

    img_emb = np.asarray(image_embeddings, dtype=np.float32)
    lbl = np.asarray(labels, dtype=np.int64)
    txt_emb = np.asarray(text_embeddings, dtype=np.float32)

    # Cosine similarity between images and class text descriptions: (N, C)
    similarity = np.dot(img_emb, txt_emb.T)
    n_samples = len(lbl)

    results = {}
    for k in top_k:
        k_val = min(k, txt_emb.shape[0])
        # Get top-k predicted class indices
        top_k_preds = np.argsort(-similarity, axis=1)[:, :k_val]
        correct = np.any(top_k_preds == lbl[:, np.newaxis], axis=1)
        results[f"zeroshot_top{k}_accuracy"] = float(np.mean(correct)) if n_samples > 0 else 0.0

    return results
