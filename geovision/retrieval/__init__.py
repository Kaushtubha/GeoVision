"""Satellite Scene Retrieval and Metric Learning Module."""

from geovision.retrieval.embedder import SatelliteSceneEmbedder
from geovision.retrieval.evaluate import evaluate_retrieval, extract_embeddings_and_labels
from geovision.retrieval.index import SceneVectorIndex
from geovision.retrieval.train import train_retrieval_embedder

__all__ = [
    "SatelliteSceneEmbedder",
    "SceneVectorIndex",
    "train_retrieval_embedder",
    "evaluate_retrieval",
    "extract_embeddings_and_labels",
]
