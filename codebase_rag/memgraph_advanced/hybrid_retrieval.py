"""Hybrid retrieval combining vector, text, and graph signals."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..config import HybridRetrievalConfig
    from ..embeddings.protocols import EmbeddingProviderProtocol
    from ..services import QueryProtocol
    from ..vector_backend import VectorBackend

# Module-level shared dependencies (NOT the retriever itself)
_SHARED_EMBEDDING_PROVIDER: EmbeddingProviderProtocol | None = None
_PROVIDER_LOCK = threading.Lock()


def get_shared_embedding_provider() -> EmbeddingProviderProtocol:
    """Get or create shared embedding provider instance.

    This avoids redundant provider initialization (API key validation,
    model loading, etc.) across multiple HybridRetriever instances.
    """
    global _SHARED_EMBEDDING_PROVIDER

    if _SHARED_EMBEDDING_PROVIDER is None:
        with _PROVIDER_LOCK:
            if _SHARED_EMBEDDING_PROVIDER is None:
                from ..config import settings
                from ..embeddings import get_embedding_provider

                config = settings.active_embedding_config
                _SHARED_EMBEDDING_PROVIDER = get_embedding_provider(
                    provider=config.provider,
                    model_id=config.model_id,
                )

    return _SHARED_EMBEDDING_PROVIDER


def reset_shared_embedding_provider() -> None:
    """Reset shared provider (e.g., when configuration changes)."""
    global _SHARED_EMBEDDING_PROVIDER
    with _PROVIDER_LOCK:
        _SHARED_EMBEDDING_PROVIDER = None


def create_hybrid_retriever(graph_ingestor: QueryProtocol) -> HybridRetriever:
    """Factory function that creates HybridRetriever with shared dependencies.

    This is the recommended way to create HybridRetriever instances.
    Shares vector_backend and embedding_provider across all instances,
    avoiding redundant initialization while allowing proper context manager
    usage for graph_ingestor.

    Args:
        graph_ingestor: MemgraphIngestor instance (use as context manager)

    Returns:
        HybridRetriever configured with shared dependencies

    Example:
        with MemgraphIngestor(...) as ingestor:
            retriever = create_hybrid_retriever(ingestor)
            results = retriever.search("query")
    """
    from ..config import settings
    from ..vector_backend import get_shared_backend

    return HybridRetriever(
        graph_ingestor=graph_ingestor,
        vector_backend=get_shared_backend(),
        embedding_provider=get_shared_embedding_provider(),
        config=settings.hybrid_retrieval_config,
    )


def _coerce_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _coerce_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def _coerce_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return default


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""

    node_id: int
    name: str
    qualified_name: str
    node_type: str
    file_path: str
    start_line: int
    end_line: int
    vector_score: float
    text_score: float
    pagerank_score: float
    community_score: float
    graph_score: float
    combined_score: float


class HybridRetriever:
    """Multi-modal retrieval combining vector, text, and graph signals."""

    def __init__(
        self,
        graph_ingestor: QueryProtocol | None = None,
        vector_backend: VectorBackend | None = None,
        embedding_provider: EmbeddingProviderProtocol | None = None,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        # Validate required dependencies
        if graph_ingestor is None:
            raise ValueError("graph_ingestor is required for HybridRetriever")
        if vector_backend is None:
            raise ValueError("vector_backend is required for HybridRetriever")
        if embedding_provider is None:
            raise ValueError("embedding_provider is required for HybridRetriever")

        # Validate vector backend health
        if not vector_backend.health_check():
            raise RuntimeError("Vector backend health check failed - connection not ready")

        # Validate embedding provider
        try:
            test_embedding = embedding_provider.embed("test")
            if not isinstance(test_embedding, list) or len(test_embedding) == 0:
                raise ValueError("Embedding provider returned invalid embedding")
        except Exception as e:
            raise ValueError(f"Embedding provider validation failed: {e}")

        self.graph_ingestor = graph_ingestor
        self.vector_backend = vector_backend
        self.embedding_provider = embedding_provider
        self.config = config

    def search(self, query: str, top_k: int = 10) -> list[HybridSearchResult]:
        from ..config import HybridRetrievalConfig

        cfg = self.config or HybridRetrievalConfig()
        query_embedding = self.embedding_provider.embed(query)

        # Fetch more results than needed for filtering
        vector_pairs: list[tuple[int, float]] = self.vector_backend.search(
            query_embedding, top_k=top_k * 3
        )

        if not vector_pairs:
            return []

        # Early filter based on minimum similarity threshold
        min_similarity = cfg.min_similarity_threshold
        filtered_pairs = [
            pair for pair in vector_pairs
            if pair[1] >= min_similarity
        ][:top_k * 2]  # Limit after filtering

        if not filtered_pairs:
            logger.debug(
                f"All vector results below min_similarity_threshold={min_similarity}"
            )
            return []

        node_ids = [pair[0] for pair in filtered_pairs]
        similarity_map = {pair[0]: pair[1] for pair in filtered_pairs}

        metadata_cypher = """
        MATCH (n)
        WHERE id(n) IN $node_ids
        RETURN id(n) AS node_id,
               n.name AS name,
               n.qualified_name AS qualified_name,
               labels(n)[0] AS node_type,
               n.path AS file_path,
               n.start_line AS start_line,
               n.end_line AS end_line,
               COALESCE(n.pagerank_score, 0.1) AS pagerank_score,
               COALESCE(n.community_importance, 0.0) AS community_score
        """

        records = self.graph_ingestor.fetch_all(metadata_cypher, {"node_ids": node_ids})
        results: list[HybridSearchResult] = []
        graph_weight = cfg.pagerank_weight + cfg.community_weight

        for record in records:
            node_id = _coerce_int(record.get("node_id"), 0)
            vector_score = similarity_map.get(node_id, 0.0)
            pagerank_score = _coerce_float(record.get("pagerank_score"), 0.1)
            community_score = _coerce_float(record.get("community_score"), 0.0)
            graph_score = (
                (
                    pagerank_score * cfg.pagerank_weight
                    + community_score * cfg.community_weight
                )
                / graph_weight
                if graph_weight > 0
                else pagerank_score
            )
            combined_score = (
                vector_score * cfg.vector_weight + graph_score * graph_weight
            )
            results.append(
                HybridSearchResult(
                    node_id=node_id,
                    name=_coerce_str(record.get("name")),
                    qualified_name=_coerce_str(record.get("qualified_name")),
                    node_type=_coerce_str(record.get("node_type")),
                    file_path=_coerce_str(record.get("file_path")),
                    start_line=_coerce_int(record.get("start_line"), 0),
                    end_line=_coerce_int(record.get("end_line"), 0),
                    vector_score=vector_score,
                    text_score=0.0,
                    pagerank_score=pagerank_score,
                    community_score=community_score,
                    graph_score=graph_score,
                    combined_score=combined_score,
                )
            )

        results.sort(key=lambda r: r.combined_score, reverse=True)
        logger.debug(
            f"Hybrid search returned {len(results)} results for query: {query[:50]!r}"
        )
        return results[:top_k]
