from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from codebase_rag.config import HybridRetrievalConfig
from codebase_rag.memgraph_advanced import HybridRetriever, HybridSearchResult


@pytest.mark.integration
class TestHybridRetrievalIntegration:
    def test_hybrid_retriever_complete_flow(
        self,
        mock_memgraph_backend: MagicMock,
        mock_graph_ingestor: MagicMock,
        mock_embedding_provider: MagicMock,
    ) -> None:
        config = HybridRetrievalConfig(
            vector_weight=0.6,
            pagerank_weight=0.3,
            community_weight=0.1,
            top_k=5,
        )

        retriever = HybridRetriever(
            graph_ingestor=mock_graph_ingestor,
            vector_backend=mock_memgraph_backend,
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        results = retriever.search("test query", top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, HybridSearchResult) for r in results)
        assert all(r.combined_score >= 0 for r in results)

    def test_hybrid_retriever_with_low_similarity_filtering(
        self,
        mock_memgraph_backend: MagicMock,
        mock_graph_ingestor: MagicMock,
        mock_embedding_provider: MagicMock,
    ) -> None:
        config = HybridRetrievalConfig(min_similarity_threshold=0.8)
        retriever = HybridRetriever(
            graph_ingestor=mock_graph_ingestor,
            vector_backend=mock_memgraph_backend,
            embedding_provider=mock_embedding_provider,
            config=config,
        )

        mock_memgraph_backend.search.return_value = [
            (1, 0.9),
            (2, 0.85),
            (3, 0.7),
            (4, 0.6),
        ]

        results = retriever.search("test query", top_k=10)
        assert len(results) == 2
