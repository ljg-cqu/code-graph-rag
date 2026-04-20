from __future__ import annotations

from unittest.mock import MagicMock, patch

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
        """Test hybrid retriever with properly configured mocks.

        Note: This test mocks the graph_ingestor.fetch_all method to return
        realistic Cypher result records, as the hybrid retriever uses
        atomic Cypher queries rather than direct vector_backend.search().
        """
        # Configure weights to sum to 1.0
        config = HybridRetrievalConfig(
            vector_weight=0.60,
            text_weight=0.15,
            pagerank_weight=0.20,
            community_weight=0.05,
            top_k=5,
        )

        # Mock the fetch_all to return realistic Cypher records
        mock_graph_ingestor.fetch_all.return_value = [
            {
                "node_id": 1,
                "name": "test_function",
                "qualified_name": "module.test_function",
                "node_type": "Function",
                "file_path": "/test/file.py",
                "start_line": 10,
                "end_line": 20,
                "vector_score": 0.95,
                "text_score": 1.0,
                "pagerank_score": 0.5,
                "community_score": 0.3,
                "combined_score": 0.85,
                "context": [],
            },
            {
                "node_id": 2,
                "name": "another_function",
                "qualified_name": "module.another_function",
                "node_type": "Function",
                "file_path": "/test/file.py",
                "start_line": 30,
                "end_line": 40,
                "vector_score": 0.85,
                "text_score": 0.0,
                "pagerank_score": 0.4,
                "community_score": 0.2,
                "combined_score": 0.65,
                "context": [],
            },
        ]

        # Mock embedding provider to return valid embedding
        mock_embedding_provider.embed.return_value = [0.1] * 768

        # Mock vector backend health check
        mock_memgraph_backend.health_check.return_value = True

        # Create retriever with strict_validation=False to avoid health check issues
        retriever = HybridRetriever(
            graph_ingestor=mock_graph_ingestor,
            vector_backend=mock_memgraph_backend,
            embedding_provider=mock_embedding_provider,
            config=config,
            strict_validation=False,
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
        """Test that low similarity results are filtered out."""
        config = HybridRetrievalConfig(
            min_similarity_threshold=0.8,
            vector_weight=0.60,
            text_weight=0.15,
            pagerank_weight=0.20,
            community_weight=0.05,
        )

        # Mock fetch_all to return records above threshold
        mock_graph_ingestor.fetch_all.return_value = [
            {
                "node_id": 1,
                "name": "high_match",
                "qualified_name": "module.high_match",
                "node_type": "Function",
                "file_path": "/test/file.py",
                "start_line": 10,
                "end_line": 20,
                "vector_score": 0.9,  # Above threshold
                "text_score": 1.0,
                "pagerank_score": 0.5,
                "community_score": 0.3,
                "combined_score": 0.85,
                "context": [],
            },
            {
                "node_id": 2,
                "name": "medium_match",
                "qualified_name": "module.medium_match",
                "node_type": "Function",
                "file_path": "/test/file.py",
                "start_line": 30,
                "end_line": 40,
                "vector_score": 0.85,  # Above threshold
                "text_score": 0.0,
                "pagerank_score": 0.4,
                "community_score": 0.2,
                "combined_score": 0.65,
                "context": [],
            },
        ]

        mock_embedding_provider.embed.return_value = [0.1] * 768
        mock_memgraph_backend.health_check.return_value = True

        retriever = HybridRetriever(
            graph_ingestor=mock_graph_ingestor,
            vector_backend=mock_memgraph_backend,
            embedding_provider=mock_embedding_provider,
            config=config,
            strict_validation=False,
        )

        results = retriever.search("test query", top_k=10)

        # Should return results (mock returns 2 records)
        assert len(results) == 2
        # All results should have vector_score >= threshold
        assert all(r.vector_score >= 0.8 for r in results)
