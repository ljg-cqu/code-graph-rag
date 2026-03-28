"""Qdrant vector storage backend wrapper.

Wraps the existing Qdrant client functionality to conform to VectorBackend protocol.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from loguru import logger

from . import logs as ls
from .config import settings
from .constants import PAYLOAD_NODE_ID, PAYLOAD_QUALIFIED_NAME
from .utils.dependencies import has_qdrant_client
from .vector_backend import VectorBackend

_RETRIEVE_BATCH_SIZE = 1000


class QdrantBackend(VectorBackend):
    """Qdrant vector storage backend.

    Uses Qdrant's HNSW index for efficient similarity search.
    Supports both local file storage and remote Qdrant server.

    Advantages:
    - Proven scale (billions of vectors)
    - Remote/cloud deployment via QDRANT_URI
    - Quantization support (scalar, product, binary)
    - Specialized vector optimization
    """

    def __init__(self) -> None:
        self._client = None
        self._initialized = False

    def _check_dependencies(self) -> bool:
        """Check if Qdrant client is available."""
        if not has_qdrant_client():
            logger.warning("Qdrant client not installed. Install with: pip install qdrant-client")
            return False
        return True

    def _get_client(self):
        """Get or create Qdrant client."""
        if not self._check_dependencies():
            raise RuntimeError("Qdrant client not available")

        if self._client is None:
            from qdrant_client import QdrantClient

            if settings.QDRANT_URI:
                self._client = QdrantClient(url=settings.QDRANT_URI)
            else:
                self._client = QdrantClient(path=settings.QDRANT_DB_PATH)
        return self._client

    def initialize(self) -> None:
        """Create Qdrant collection if it doesn't exist."""
        if not self._check_dependencies():
            logger.warning("Skipping Qdrant initialization - client not available")
            return

        if self._initialized:
            return

        from qdrant_client.models import Distance, VectorParams

        client = self._get_client()
        collection_name = settings.QDRANT_COLLECTION_NAME

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=settings.QDRANT_VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                ls.QDRANT_COLLECTION_CREATED.format(
                    collection=collection_name,
                    dim=settings.QDRANT_VECTOR_DIM,
                )
            )

        self._initialized = True
        logger.info(ls.QDRANT_BACKEND_READY.format(collection=collection_name))

    def _upsert_with_retry(self, points: list) -> None:
        """Upsert points with retry logic."""
        client = self._get_client()
        max_attempts = settings.QDRANT_UPSERT_RETRIES
        base_delay = settings.QDRANT_RETRY_BASE_DELAY

        for attempt in range(1, max_attempts + 1):
            try:
                client.upsert(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    points=points,
                )
                return
            except Exception as e:
                if attempt == max_attempts:
                    raise
                delay = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    ls.EMBEDDING_STORE_RETRY.format(
                        attempt=attempt,
                        max_attempts=max_attempts,
                        delay=delay,
                        error=e,
                    )
                )
                time.sleep(delay)

    def store_batch(
        self, points: Sequence[tuple[int, list[float], str]]
    ) -> int:
        """Store embeddings in Qdrant collection."""
        if not points or not self._check_dependencies():
            return 0

        from qdrant_client.models import PointStruct

        point_structs = [
            PointStruct(
                id=node_id,
                vector=embedding,
                payload={
                    PAYLOAD_NODE_ID: node_id,
                    PAYLOAD_QUALIFIED_NAME: qualified_name,
                },
            )
            for node_id, embedding, qualified_name in points
        ]

        try:
            self._upsert_with_retry(point_structs)
            logger.debug(ls.EMBEDDING_BATCH_STORED.format(count=len(point_structs)))
            return len(point_structs)
        except Exception as e:
            logger.warning(ls.EMBEDDING_BATCH_FAILED.format(error=e))
            return 0

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[int, float]]:
        """Search for similar embeddings using Qdrant query."""
        if not self._check_dependencies():
            return []

        effective_top_k = top_k if top_k > 0 else settings.VECTOR_SEARCH_TOP_K
        client = self._get_client()

        try:
            # Note: Qdrant filters would require Filter model construction
            # For now, we rely on post-filtering in application code
            result = client.query_points(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                query=query_embedding,
                limit=effective_top_k,
            )

            return [
                (
                    hit.payload.get(PAYLOAD_NODE_ID, hit.id) if hit.payload else hit.id,
                    hit.score,
                )
                for hit in result.points
            ]
        except Exception as e:
            logger.warning(ls.EMBEDDING_SEARCH_FAILED.format(error=e))
            return []

    def delete_batch(self, node_ids: Sequence[int]) -> int:
        """Delete points from Qdrant collection."""
        if not node_ids or not self._check_dependencies():
            return 0

        client = self._get_client()
        try:
            client.delete(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points_selector=list(node_ids),
            )
            return len(node_ids)
        except Exception as e:
            logger.warning(ls.VECTOR_DELETE_FAILED.format(error=e))
            return 0

    def verify_ids(self, expected_ids: set[int]) -> set[int]:
        """Verify which IDs are stored in Qdrant."""
        if not expected_ids or not self._check_dependencies():
            return set()

        client = self._get_client()
        found_ids: set[int] = set()
        ids_list = list(expected_ids)

        for i in range(0, len(ids_list), _RETRIEVE_BATCH_SIZE):
            chunk = ids_list[i : i + _RETRIEVE_BATCH_SIZE]
            points = client.retrieve(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                ids=chunk,
                with_payload=False,
                with_vectors=False,
            )
            found_ids.update(p.id for p in points if isinstance(p.id, int))

        return found_ids

    def get_stats(self) -> dict:
        """Return Qdrant collection statistics."""
        if not self._check_dependencies():
            return {"backend": "qdrant", "available": False}

        client = self._get_client()
        try:
            info = client.get_collection(settings.QDRANT_COLLECTION_NAME)
            return {
                "backend": "qdrant",
                "available": True,
                "total_embeddings": info.points_count or 0,
                "vector_dim": info.config.params.vectors.size,
                "status": info.status,
            }
        except Exception:
            return {"backend": "qdrant", "available": True, "total_embeddings": 0}

    def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        if not self._check_dependencies():
            return False
        try:
            client = self._get_client()
            client.get_collections()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close Qdrant client."""
        if self._client is not None:
            try:
                self._client.close()
                logger.info(ls.QDRANT_CLIENT_CLOSED)
            except Exception:
                pass
            self._client = None