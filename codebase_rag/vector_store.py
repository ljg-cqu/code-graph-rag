"""
Vector store module - backward compatible wrapper for dual backend support.

This module provides backward-compatible functions that work with either
Qdrant or Memgraph native vector storage, controlled by VECTOR_STORE_BACKEND setting.

For direct backend access, use vector_backend.py module.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from . import logs as ls
from .config import settings
from .constants import PAYLOAD_NODE_ID, PAYLOAD_QUALIFIED_NAME
from .utils.dependencies import has_qdrant_client
from .vector_backend import VectorBackend, get_shared_backend, close_shared_backend

_RETRIEVE_BATCH_SIZE = 1000

# Global backend instance for backward compatibility
_BACKEND: VectorBackend | None = None


def _get_backend() -> VectorBackend:
    """Get the configured vector backend."""
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = get_shared_backend()
    return _BACKEND


def close_vector_backend() -> None:
    """Close the vector backend connection."""
    global _BACKEND
    close_shared_backend()
    _BACKEND = None


# Backward compatibility: expose Qdrant-specific functions when backend is qdrant
if has_qdrant_client():
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    _CLIENT: QdrantClient | None = None

    def close_qdrant_client() -> None:
        """Close Qdrant client (backward compatibility)."""
        global _CLIENT
        if _CLIENT is not None:
            _CLIENT.close()
            _CLIENT = None
        # Also close backend
        close_vector_backend()

    def get_qdrant_client() -> QdrantClient:
        """Get Qdrant client directly (backward compatibility).

        Note: For new code, use vector_backend.get_vector_backend() instead.
        """
        global _CLIENT
        if _CLIENT is None:
            if settings.QDRANT_URI:
                _CLIENT = QdrantClient(url=settings.QDRANT_URI)
            else:
                _CLIENT = QdrantClient(path=settings.QDRANT_DB_PATH)
            if not _CLIENT.collection_exists(settings.QDRANT_COLLECTION_NAME):
                _CLIENT.create_collection(
                    collection_name=settings.QDRANT_COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=settings.QDRANT_VECTOR_DIM, distance=Distance.COSINE
                    ),
                )
        return _CLIENT

else:

    def close_qdrant_client() -> None:
        """Close Qdrant client (no-op when qdrant not installed)."""
        close_vector_backend()

    def get_qdrant_client():
        """Get Qdrant client (not available)."""
        raise RuntimeError("Qdrant client not installed. Install with: pip install qdrant-client")


def store_embedding(
    node_id: int, embedding: list[float], qualified_name: str
) -> None:
    """Store a single embedding (backward compatibility).

    Uses configured backend (Memgraph or Qdrant).
    """
    store_embedding_batch([(node_id, embedding, qualified_name)])


def store_embedding_batch(
    points: Sequence[tuple[int, list[float], str]],
) -> int:
    """Store embeddings in batch using configured backend.

    Args:
        points: Sequence of (node_id, embedding, qualified_name) tuples.

    Returns:
        Number of successfully stored embeddings.
    """
    if not points:
        return 0

    backend = _get_backend()
    return backend.store_batch(points)


def delete_project_embeddings(project_name: str, node_ids: Sequence[int]) -> None:
    """Delete embeddings for a project.

    Args:
        project_name: Project name for logging.
        node_ids: Sequence of node IDs to delete.
    """
    if not node_ids:
        return

    backend = _get_backend()

    try:
        logger.info(
            ls.QDRANT_DELETE_PROJECT.format(
                count=len(node_ids), project=project_name
            )
        )
        deleted_count = backend.delete_batch(node_ids)
        logger.info(ls.QDRANT_DELETE_PROJECT_DONE.format(project=project_name))
        if deleted_count < len(node_ids):
            logger.warning(
                f"Only deleted {deleted_count} of {len(node_ids)} embeddings"
            )
    except Exception as e:
        logger.warning(
            ls.QDRANT_DELETE_PROJECT_FAILED.format(project=project_name, error=e)
        )


def verify_stored_ids(expected_ids: set[int]) -> set[int]:
    """Verify which IDs have embeddings stored.

    Args:
        expected_ids: Set of node IDs to check.

    Returns:
        Set of IDs that exist in the backend.
    """
    if not expected_ids:
        return set()

    backend = _get_backend()
    return backend.verify_ids(expected_ids)


def search_embeddings(
    query_embedding: list[float], top_k: int | None = None
) -> list[tuple[int, float]]:
    """Search for similar embeddings.

    Args:
        query_embedding: Query vector (768-dim).
        top_k: Number of results (default: from settings).

    Returns:
        List of (node_id, score) tuples.
    """
    backend = _get_backend()
    effective_top_k = top_k if top_k is not None else settings.VECTOR_SEARCH_TOP_K

    try:
        return backend.search(query_embedding, effective_top_k)
    except Exception as e:
        logger.warning(ls.EMBEDDING_SEARCH_FAILED.format(error=e))
        return []
