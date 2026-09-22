"""Embedded Qdrant vector index for satellite scene search and multimodal retrieval."""

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from geovision.logger import get_logger
from geovision.retrieval.embedder import SatelliteSceneEmbedder

logger = get_logger("geovision.retrieval.index")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False


class InMemoryVectorStore:
    """Zero-dependency fallback in-memory vector database."""

    def __init__(self, vector_size: int = 512):
        self.vector_size = vector_size
        self.vectors: list[np.ndarray] = []
        self.payloads: list[dict[str, Any]] = []
        self.ids: list[str] = []

    def upsert(self, vectors: np.ndarray, payloads: list[dict[str, Any]], ids: list[str] | None = None) -> None:
        n = len(vectors)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]

        for i in range(n):
            vec = vectors[i] / np.maximum(np.linalg.norm(vectors[i]), 1e-8)
            self.vectors.append(vec)
            self.payloads.append(payloads[i])
            self.ids.append(ids[i])

    def search(self, query_vector: np.ndarray, limit: int = 5) -> list[dict[str, Any]]:
        if not self.vectors:
            return []

        q_norm = query_vector / np.maximum(np.linalg.norm(query_vector), 1e-8)
        matrix = np.stack(self.vectors, axis=0)  # (N, D)
        sims = np.dot(matrix, q_norm)  # (N,)

        top_indices = np.argsort(-sims)[:limit]
        results = []
        for idx in top_indices:
            results.append({
                "id": self.ids[idx],
                "score": float(sims[idx]),
                "payload": self.payloads[idx],
            })
        return results

    def count(self) -> int:
        return len(self.vectors)


class SceneVectorIndex:
    """Satellite scene vector database interface backed by Qdrant or fallback memory store."""

    def __init__(
        self,
        collection_name: str = "geovision_scenes",
        vector_size: int = 512,
        db_path: str | Path | None = None,
        in_memory: bool = False,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.db_path = str(db_path) if db_path and not in_memory else None
        self.in_memory = in_memory or (self.db_path is None)

        self.client = None
        self.memory_store = None

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize Qdrant client connection or in-memory fallback."""
        if _HAS_QDRANT:
            try:
                if self.in_memory:
                    self.client = QdrantClient(":memory:")
                else:
                    Path(self.db_path).mkdir(parents=True, exist_ok=True)
                    self.client = QdrantClient(path=self.db_path)

                self._ensure_collection_exists()
                logger.info(f"Initialized Qdrant vector index (collection='{self.collection_name}', path={self.db_path or ':memory:'})")
                return
            except Exception as e:
                logger.warning(f"Failed to initialize Qdrant ({e}). Falling back to InMemoryVectorStore.")

        self.memory_store = InMemoryVectorStore(vector_size=self.vector_size)
        logger.info("Using InMemoryVectorStore for scene indexing.")

    def _ensure_collection_exists(self) -> None:
        """Create Qdrant collection if not already present."""
        if self.client is None:
            return

        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection '{self.collection_name}' with Cosine distance.")

    def reset(self) -> None:
        """Re-create / clear the collection."""
        if self.client is not None:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._ensure_collection_exists()
        else:
            self.memory_store = InMemoryVectorStore(vector_size=self.vector_size)

    def upsert_scenes(
        self,
        vectors: np.ndarray | torch.Tensor,
        payloads: list[dict[str, Any]],
        ids: list[int | str] | None = None,
    ) -> int:
        """Insert or update scene vectors with associated metadata.

        Args:
            vectors: (N, D) float array of normalized feature embeddings.
            payloads: List of N metadata dictionaries.
            ids: Optional list of unique IDs.

        Returns:
            Number of points indexed.
        """
        if hasattr(vectors, "detach"):
            vectors = vectors.detach().cpu().numpy()
        vectors_np = np.asarray(vectors, dtype=np.float32)

        n = len(vectors_np)
        if len(payloads) != n:
            raise ValueError(f"Length mismatch: {n} vectors vs {len(payloads)} payloads")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(n)]
        else:
            ids = [str(i) for i in ids]

        if self.client is not None:
            points = [
                qmodels.PointStruct(
                    id=ids[i],
                    vector=vectors_np[i].tolist(),
                    payload=payloads[i],
                )
                for i in range(n)
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        else:
            self.memory_store.upsert(vectors_np, payloads, ids)

        logger.info(f"Indexed {n} scene vectors into '{self.collection_name}'")
        return n

    def search_by_vector(
        self,
        query_vector: np.ndarray | torch.Tensor | list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search nearest scenes using a query vector."""
        if hasattr(query_vector, "detach"):
            query_vector = query_vector.detach().cpu().numpy()
        q_vec = np.asarray(query_vector, dtype=np.float32).flatten()

        # Normalize
        q_vec = q_vec / np.maximum(np.linalg.norm(q_vec), 1e-8)

        if self.client is not None:
            # Query Qdrant
            try:
                hits = self.client.query_points(
                    collection_name=self.collection_name,
                    query=q_vec.tolist(),
                    limit=top_k,
                ).points
            except Exception:
                hits = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=q_vec.tolist(),
                    limit=top_k,
                )

            results = []
            for hit in hits:
                results.append({
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "payload": hit.payload or {},
                })
            return results
        else:
            return self.memory_store.search(q_vec, limit=top_k)

    def search_by_text(
        self,
        text_query: str,
        embedder: SatelliteSceneEmbedder,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search scene database using a natural language text query."""
        text_emb = embedder.encode_text(text_query)[0]
        return self.search_by_vector(text_emb, top_k=top_k)

    def search_by_image(
        self,
        image_input: str | Path | np.ndarray | Image.Image,
        embedder: SatelliteSceneEmbedder,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search scene database using a query image."""
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as im:
                img_arr = np.array(im.convert("RGB"))
        elif isinstance(image_input, Image.Image):
            img_arr = np.array(image_input.convert("RGB"))
        else:
            img_arr = image_input

        img_emb = embedder.encode_image(img_arr)[0]
        return self.search_by_vector(img_emb, top_k=top_k)

    def count(self) -> int:
        """Return total indexed points."""
        if self.client is not None:
            try:
                return self.client.get_collection(self.collection_name).points_count or 0
            except Exception:
                return 0
        return self.memory_store.count()
