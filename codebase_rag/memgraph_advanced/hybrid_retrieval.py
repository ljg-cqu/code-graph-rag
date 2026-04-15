"""Hybrid retrieval combining vector, text, and graph signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..config import HybridRetrievalConfig
    from ..embeddings.protocols import EmbeddingProviderProtocol
    from ..services.graph_service import MemgraphIngestor
    from ..vector_backend import VectorBackend


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
        graph_ingestor: MemgraphIngestor | None = None,
        vector_backend: VectorBackend | None = None,
        embedding_provider: EmbeddingProviderProtocol | None = None,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self.graph_ingestor = graph_ingestor
        self.vector_backend = vector_backend
        self.embedding_provider = embedding_provider
        self.config = config

    def search(self, query: str, top_k: int = 10) -> list[HybridSearchResult]:
        if not self.vector_backend or not self.embedding_provider or not self.graph_ingestor:
            return []

        from ..config import HybridRetrievalConfig

        cfg = self.config or HybridRetrievalConfig()
        query_embedding = self.embedding_provider.embed(query)
        vector_pairs: list[tuple[int, float]] = self.vector_backend.search(
            query_embedding, top_k=top_k * 2
        )

        if not vector_pairs:
            return []

        node_ids = [pair[0] for pair in vector_pairs]
        similarity_map = {pair[0]: pair[1] for pair in vector_pairs}

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
            node_id = int(record["node_id"])
            vector_score = similarity_map.get(node_id, 0.0)
            pagerank_score = float(record.get("pagerank_score") or 0.1)
            community_score = float(record.get("community_score") or 0.0)
            graph_score = (
                (pagerank_score * cfg.pagerank_weight + community_score * cfg.community_weight)
                / graph_weight
                if graph_weight > 0
                else pagerank_score
            )
            combined_score = vector_score * cfg.vector_weight + graph_score * graph_weight
            results.append(
                HybridSearchResult(
                    node_id=node_id,
                    name=str(record.get("name") or ""),
                    qualified_name=str(record.get("qualified_name") or ""),
                    node_type=str(record.get("node_type") or ""),
                    file_path=str(record.get("file_path") or ""),
                    start_line=int(record.get("start_line") or 0),
                    end_line=int(record.get("end_line") or 0),
                    vector_score=vector_score,
                    text_score=0.0,
                    pagerank_score=pagerank_score,
                    community_score=community_score,
                    graph_score=graph_score,
                    combined_score=combined_score,
                )
            )

        results.sort(key=lambda r: r.combined_score, reverse=True)
        logger.debug(f"Hybrid search returned {len(results)} results for query: {query[:50]!r}")
        return results[:top_k]
