"""Vector backend protocol and factory for dual backend support."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from loguru import logger


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector storage backends.

    Supports both Qdrant and Memgraph native vector storage.
    """

    def initialize(self) -> None:
        """Initialize the backend (create indexes, collections).

        Should be called before any store/search operations.
        """
        ...

    def store_batch(
        self, points: Sequence[tuple[int, list[float], str]]
    ) -> int:
        """Store embeddings in batch.

        Args:
            points: Sequence of (node_id, embedding, qualified_name)
                - node_id: Memgraph internal vertex ID
                - embedding: 768-dimensional vector (UniXcoder)
                - qualified_name: Fully qualified name for reference

        Returns:
            Number of successfully stored points
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[tuple[int, float]]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query vector (768-dim)
            top_k: Number of results to return
            filters: Optional filters (e.g., {"project_prefix": "myproject."})

        Returns:
            List of (node_id, similarity) tuples sorted by relevance
        """
        ...

    def delete_batch(self, node_ids: Sequence[int]) -> int:
        """Delete embeddings by node IDs.

        Args:
            node_ids: Sequence of Memgraph node IDs to delete

        Returns:
            Number of successfully deleted points
        """
        ...

    def verify_ids(self, expected_ids: set[int]) -> set[int]:
        """Verify which IDs are stored.

        Args:
            expected_ids: Set of IDs to check

        Returns:
            Set of IDs that exist in the backend
        """
        ...

    def close(self) -> None:
        """Cleanup resources.

        Called when shutting down the application.
        """
        ...

    def get_stats(self) -> dict:
        """Return backend statistics.

        Returns:
            Dict with stats like total_embeddings, backend_type, etc.
        """
        ...

    def health_check(self) -> bool:
        """Check if backend is healthy and operational.

        Returns:
            True if backend is healthy, False otherwise
        """
        ...


def get_vector_backend() -> VectorBackend:
    """Factory function to get configured vector backend.

    Reads VECTOR_STORE_BACKEND from settings and returns appropriate backend.
    Default is 'memgraph' for native vector storage.

    Returns:
        VectorBackend instance (MemgraphBackend or QdrantBackend)

    Raises:
        ValueError: If backend name is unknown
    """
    from .config import settings

    backend_name = settings.VECTOR_STORE_BACKEND.lower()

    if backend_name == "qdrant":
        from .vector_store_qdrant import QdrantBackend
        logger.info("Using Qdrant vector backend")
        return QdrantBackend()
    elif backend_name == "memgraph":
        from .vector_store_memgraph import MemgraphBackend
        logger.info("Using Memgraph native vector backend")
        return MemgraphBackend()
    else:
        raise ValueError(
            f"Unknown vector backend: {backend_name}. "
            "Supported: 'memgraph' (default), 'qdrant'"
        )


# Global backend instance (lazy initialization)
_BACKEND_INSTANCE: VectorBackend | None = None


def get_shared_backend() -> VectorBackend:
    """Get shared backend instance (singleton pattern).

    Creates backend on first call, reuses on subsequent calls.
    Call close_shared_backend() to cleanup.

    Returns:
        Shared VectorBackend instance
    """
    global _BACKEND_INSTANCE
    if _BACKEND_INSTANCE is None:
        _BACKEND_INSTANCE = get_vector_backend()
        _BACKEND_INSTANCE.initialize()
    return _BACKEND_INSTANCE


def close_shared_backend() -> None:
    """Close and cleanup the shared backend instance."""
    global _BACKEND_INSTANCE
    if _BACKEND_INSTANCE is not None:
        _BACKEND_INSTANCE.close()
        _BACKEND_INSTANCE = None