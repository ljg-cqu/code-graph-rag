"""Tests for QueryMethodOrchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.retrieval.query_orchestrator import (
    CombinedQueryResult,
    QueryIntent,
    QueryMethod,
    QueryMethodOrchestrator,
    QueryMethodResult,
)


class TestQueryIntentClassification:
    """Test query intent classification."""

    def test_classifies_structural_intent(self) -> None:
        assert QueryMethodOrchestrator.classify_intent(
            "What functions call authenticate?"
        ) == QueryIntent.STRUCTURAL

    def test_classifies_functional_intent(self) -> None:
        assert QueryMethodOrchestrator.classify_intent(
            "How does authentication work?"
        ) == QueryIntent.FUNCTIONAL

    def test_classifies_semantic_intent(self) -> None:
        assert QueryMethodOrchestrator.classify_intent(
            "Find functions similar to login"
        ) == QueryIntent.SEMANTIC

    def test_classifies_validation_intent(self) -> None:
        assert QueryMethodOrchestrator.classify_intent(
            "Is this code valid and correct?"
        ) == QueryIntent.VALIDATION

    def test_defaults_to_exploratory(self) -> None:
        assert (
            QueryMethodOrchestrator.classify_intent("Tell me about auth")
            == QueryIntent.EXPLORATORY
        )


class TestQueryMethodSelection:
    """Test method selection based on intent."""

    @pytest.fixture
    def orchestrator(self) -> QueryMethodOrchestrator:
        mock_graph = MagicMock()
        return QueryMethodOrchestrator(code_graph=mock_graph)

    def test_functional_selects_semantic_and_graph(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        methods = orchestrator.select_methods(QueryIntent.FUNCTIONAL)
        assert QueryMethod.SEMANTIC_SEARCH in methods
        assert QueryMethod.GRAPH_TRAVERSAL in methods

    def test_structural_selects_graph_methods(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        methods = orchestrator.select_methods(QueryIntent.STRUCTURAL)
        assert QueryMethod.GRAPH_TRAVERSAL in methods
        assert QueryMethod.GRAPH_NAVIGATION in methods

    def test_exploratory_selects_multiple(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        methods = orchestrator.select_methods(QueryIntent.EXPLORATORY)
        assert len(methods) >= 2


class TestQueryMethodOrchestrator:
    """Test QueryMethodOrchestrator execution."""

    @pytest.fixture
    def mock_graph(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def orchestrator(self, mock_graph: MagicMock) -> QueryMethodOrchestrator:
        return QueryMethodOrchestrator(code_graph=mock_graph)

    def test_execute_returns_combined_result(
        self, orchestrator: QueryMethodOrchestrator, mock_graph: MagicMock
    ) -> None:
        mock_graph.fetch_all.return_value = [
            {
                "node_id": 1,
                "qualified_name": "auth.login",
                "name": "login",
                "type": "Function",
            }
        ]

        with patch(
            "codebase_rag.retrieval.query_orchestrator.QueryMethodOrchestrator._get_hybrid_retriever"
        ) as mock_retriever:
            mock_retriever.return_value.search.return_value = []

            result = orchestrator.execute("How does authentication work?")

            assert isinstance(result, CombinedQueryResult)
            assert result.intent == QueryIntent.FUNCTIONAL

    def test_merge_and_rank_combines_scores(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        results = [
            QueryMethodResult(
                method=QueryMethod.SEMANTIC_SEARCH,
                items=[
                    {"qualified_name": "auth.login", "similarity": 0.8},
                ],
                execution_time_ms=10.0,
            ),
            QueryMethodResult(
                method=QueryMethod.KEYWORD_SEARCH,
                items=[
                    {"qualified_name": "auth.login", "similarity": 0.6},
                ],
                execution_time_ms=5.0,
            ),
        ]

        merged = orchestrator._merge_and_rank(results, top_k=5)

        assert len(merged) == 1
        assert merged[0]["qualified_name"] == "auth.login"
        assert merged[0]["method_count"] == 2

    def test_merge_and_rank_boosts_multi_method(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        results = [
            QueryMethodResult(
                method=QueryMethod.SEMANTIC_SEARCH,
                items=[
                    {"qualified_name": "fn1", "similarity": 0.5},
                    {"qualified_name": "fn2", "similarity": 0.6},
                ],
                execution_time_ms=10.0,
            ),
            QueryMethodResult(
                method=QueryMethod.KEYWORD_SEARCH,
                items=[
                    {"qualified_name": "fn1", "similarity": 0.5},
                ],
                execution_time_ms=5.0,
            ),
        ]

        merged = orchestrator._merge_and_rank(results, top_k=5)

        assert merged[0]["qualified_name"] == "fn1"
        assert merged[0]["method_count"] == 2


class TestQueryMethodResult:
    """Test QueryMethodResult dataclass."""

    def test_creates_result(self) -> None:
        result = QueryMethodResult(
            method=QueryMethod.SEMANTIC_SEARCH,
            items=[{"name": "test"}],
            execution_time_ms=100.0,
        )
        assert result.method == QueryMethod.SEMANTIC_SEARCH
        assert len(result.items) == 1
        assert result.error is None

    def test_creates_error_result(self) -> None:
        result = QueryMethodResult(
            method=QueryMethod.GRAPH_TRAVERSAL,
            items=[],
            execution_time_ms=50.0,
            error="Connection failed",
        )
        assert result.error == "Connection failed"
