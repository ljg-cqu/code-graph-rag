"""Backward-compatible vector store helpers for Memgraph native vector storage."""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from . import logs as ls
from .config import settings
from .vector_backend import VectorBackend, close_shared_backend, get_shared_backend

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


def store_embedding(node_id: int, embedding: list[float], qualified_name: str) -> None:
    """Store a single embedding (backward compatibility).

    Uses the configured Memgraph backend.
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
            ls.VECTOR_DELETE_PROJECT.format(count=len(node_ids), project=project_name)
        )
        deleted_count = backend.delete_batch(node_ids)
        logger.info(ls.VECTOR_DELETE_PROJECT_DONE.format(project=project_name))
        if deleted_count < len(node_ids):
            logger.warning(
                f"Only deleted {deleted_count} of {len(node_ids)} embeddings"
            )
    except Exception as e:
        logger.warning(
            ls.VECTOR_DELETE_PROJECT_FAILED.format(project=project_name, error=e)
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
        List of (node_id, similarity) tuples.
    """
    backend = _get_backend()
    effective_top_k = top_k if top_k is not None else settings.VECTOR_SEARCH_TOP_K

    try:
        return backend.search(query_embedding, effective_top_k)
    except Exception as e:
        logger.warning(ls.EMBEDDING_SEARCH_FAILED.format(error=e))
        return []
