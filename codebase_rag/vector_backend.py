"""Vector backend protocol and factory for Memgraph vector storage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, overload, runtime_checkable

from loguru import logger

from .config import settings
from .types_defs import ResultRow


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for vector storage backends.

    The production backend is Memgraph native vector storage.
    """

    def initialize(self) -> None:
        """Initialize the backend (create indexes, collections).

        Should be called before any store/search operations.
        """
        ...

    def store_batch(self, points: Sequence[tuple[int, list[float], str]]) -> int:
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

    @overload
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        include_context: Literal[False] = False,
        max_context_depth: int = 2,
    ) -> list[tuple[int, float]]: ...

    @overload
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        include_context: Literal[True] = True,
        max_context_depth: int = 2,
    ) -> list[ResultRow]: ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        include_context: bool = False,
        max_context_depth: int = 2,
    ) -> list[tuple[int, float]] | list[ResultRow]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query vector (768-dim)
            top_k: Number of results to return
            filters: Optional filters (e.g., {"project_prefix": "myproject."})
            include_context: Whether to include expanded graph context rows.
            max_context_depth: BFS depth limit when include_context is enabled.

        Returns:
            Vector match tuples by default, or full result rows when context is requested.
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


def get_vector_backend(is_document: bool = False) -> VectorBackend:
    """Factory function to get configured vector backend.

    Memgraph native vector storage is the only supported backend.

    Args:
        is_document: If True, use document graph vector backend, else use code graph backend.

    Returns:
        VectorBackend instance (MemgraphBackend)
    """
    from .vector_store_memgraph import MemgraphBackend

    configured_backend = (
        settings.DOC_VECTOR_STORE_BACKEND if is_document else settings.VECTOR_STORE_BACKEND
    ).strip().lower()
    if configured_backend != "memgraph":
        backend_scope = "DOC_VECTOR_STORE_BACKEND" if is_document else "VECTOR_STORE_BACKEND"
        raise ValueError(
            f"{backend_scope}={configured_backend!r} is not supported. Only 'memgraph' is available."
        )

    logger.info(
        f"Using Memgraph native vector backend for {'document' if is_document else 'code'}"
    )
    return MemgraphBackend(is_document=is_document)


# Global backend instances (lazy initialization)
_BACKEND_INSTANCE: VectorBackend | None = None
_DOC_BACKEND_INSTANCE: VectorBackend | None = None


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


def get_shared_backend_for_documents() -> VectorBackend:
    """Get shared document backend instance.

    Creates document backend on first call, reuses on subsequent calls.

    Returns:
        Shared VectorBackend instance for documents
    """
    global _DOC_BACKEND_INSTANCE
    if _DOC_BACKEND_INSTANCE is None:
        _DOC_BACKEND_INSTANCE = get_vector_backend(is_document=True)
        _DOC_BACKEND_INSTANCE.initialize()
    return _DOC_BACKEND_INSTANCE


def close_shared_backend() -> None:
    """Close and cleanup the shared backend instance."""
    global _BACKEND_INSTANCE, _DOC_BACKEND_INSTANCE
    if _BACKEND_INSTANCE is not None:
        _BACKEND_INSTANCE.close()
        _BACKEND_INSTANCE = None
    if _DOC_BACKEND_INSTANCE is not None:
        _DOC_BACKEND_INSTANCE.close()
        _DOC_BACKEND_INSTANCE = None
