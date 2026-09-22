"""Comprehensive unit tests for Metric Learning & Satellite Scene Retrieval."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from geovision.constants import EUROSAT_CLASSES
from geovision.metrics.retrieval import (
    evaluate_retrieval_metrics,
    evaluate_zeroshot_accuracy,
)
from geovision.retrieval.embedder import SatelliteSceneEmbedder
from geovision.retrieval.index import SceneVectorIndex
from geovision.retrieval.train import train_retrieval_embedder


def test_satellite_scene_embedder_shapes_and_norm():
    """Test image and text embedding output shapes and L2 normalization."""
    embedder = SatelliteSceneEmbedder(
        model_name="ViT-B-32",
        pretrained=None,
        embedding_dim=512,
        device="cpu",
    )

    # Test dummy images tensor
    dummy_imgs = torch.randn(3, 3, 224, 224)
    img_embs = embedder.encode_image(dummy_imgs)
    assert img_embs.shape == (3, 512)
    # Check L2 unit length
    norms = torch.norm(img_embs, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    # Test text prompts
    prompts = ["a satellite photo of a forest", "river and water body", "urban industrial area"]
    txt_embs = embedder.encode_text(prompts)
    assert txt_embs.shape == (3, 512)
    txt_norms = torch.norm(txt_embs, p=2, dim=-1)
    assert torch.allclose(txt_norms, torch.ones_like(txt_norms), atol=1e-4)

    # Test class prompt ensemble
    class_embs = embedder.build_class_text_embeddings(class_names=EUROSAT_CLASSES)
    assert class_embs.shape == (len(EUROSAT_CLASSES), 512)


def test_retrieval_metrics_calculator_known_cases():
    """Test evaluate_retrieval_metrics on synthetic orthogonal and identical vectors."""
    # 4 classes, 4 queries, 4 gallery items
    # Diagonal matches exactly
    query_embs = np.eye(4, dtype=np.float32)
    query_lbls = np.array([0, 1, 2, 3])
    gallery_embs = np.eye(4, dtype=np.float32)
    gallery_lbls = np.array([0, 1, 2, 3])

    res = evaluate_retrieval_metrics(
        query_embeddings=query_embs,
        query_labels=query_lbls,
        gallery_embeddings=gallery_embs,
        gallery_labels=gallery_lbls,
        k_values=(1, 2),
        exclude_self=False,
    )

    assert pytest.approx(res["recall@1"], 0.001) == 1.0
    assert pytest.approx(res["recall@2"], 0.001) == 1.0
    assert pytest.approx(res["mrr"], 0.001) == 1.0
    assert pytest.approx(res["map"], 0.001) == 1.0

    # Test zero-shot evaluation
    txt_embs = np.eye(4, dtype=np.float32)
    zs_res = evaluate_zeroshot_accuracy(
        image_embeddings=query_embs,
        labels=query_lbls,
        text_embeddings=txt_embs,
        top_k=(1, 2),
    )
    assert pytest.approx(zs_res["zeroshot_top1_accuracy"], 0.001) == 1.0


def test_scene_vector_index_ops():
    """Test SceneVectorIndex collection creation, upsert, and search."""
    temp_dir = Path(tempfile.mkdtemp())
    try:
        index = SceneVectorIndex(
            collection_name="test_scenes",
            vector_size=64,
            db_path=temp_dir / "qdrant_test",
            in_memory=True,
        )

        n_samples = 10
        vectors = np.random.randn(n_samples, 64).astype(np.float32)
        payloads = [{"scene_id": i, "class_name": f"Class_{i % 3}"} for i in range(n_samples)]

        count = index.upsert_scenes(vectors, payloads)
        assert count == n_samples
        assert index.count() >= n_samples

        # Test vector search
        query_vec = vectors[0]
        hits = index.search_by_vector(query_vec, top_k=3)
        assert len(hits) == 3
        # Top hit should have score close to 1.0
        assert pytest.approx(hits[0]["score"], 0.01) == 1.0
        assert hits[0]["payload"]["scene_id"] == 0
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_smoke_train_retrieval():
    """Test end-to-end contrastive fine-tuning in smoke mode."""
    results = train_retrieval_embedder(
        smoke=True,
        epochs=1,
        experiment_name="Test-Retrieval-Smoke",
    )

    assert "best_checkpoint" in results
    assert Path(results["best_checkpoint"]).exists()
    assert results["epochs_completed"] == 1
    assert "best_recall@1" in results
