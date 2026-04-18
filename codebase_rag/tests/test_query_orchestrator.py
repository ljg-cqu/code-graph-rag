"""Tests for QueryMethodOrchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.retrieval.query_orchestrator import (
    CombinedQueryResult,
    IntegrityWarning,
    QueryIntent,
    QueryMethod,
    QueryMethodOrchestrator,
    QueryMethodResult,
    verify_graph_result_integrity,
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


class TestQueryIntentWithConfidence:
    """Test query intent classification with confidence scoring."""

    def test_returns_tuple(self) -> None:
        intent, confidence = QueryMethodOrchestrator.classify_intent_with_confidence(
            "What functions call authenticate?"
        )
        assert isinstance(intent, QueryIntent)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_high_confidence_for_clear_intent(self) -> None:
        intent, confidence = QueryMethodOrchestrator.classify_intent_with_confidence(
            "call hierarchy for authenticate"
        )
        assert intent == QueryIntent.STRUCTURAL
        assert confidence >= 0.5

    def test_low_confidence_for_ambiguous_query(self) -> None:
        intent, confidence = QueryMethodOrchestrator.classify_intent_with_confidence(
            "auth"
        )
        assert confidence < 0.5


class TestQueryMethodSelection:
    """Test method selection based on intent."""

    @pytest.fixture
    def orchestrator(self) -> QueryMethodOrchestrator:
        mock_graph = MagicMock()
        return QueryMethodOrchestrator(code_graph=mock_graph)

    def test_functional_selects_primary_semantic_and_graph(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        primary, secondary = orchestrator.select_methods(QueryIntent.FUNCTIONAL)
        assert QueryMethod.SEMANTIC_SEARCH in primary
        assert QueryMethod.GRAPH_TRAVERSAL in primary

    def test_structural_selects_graph_methods(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        primary, secondary = orchestrator.select_methods(QueryIntent.STRUCTURAL)
        assert QueryMethod.GRAPH_TRAVERSAL in primary
        assert QueryMethod.GRAPH_NAVIGATION in primary

    def test_exploratory_selects_multiple_primary(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        primary, secondary = orchestrator.select_methods(QueryIntent.EXPLORATORY)
        assert len(primary) >= 2


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

    def test_execute_async_returns_combined_result(
        self, orchestrator: QueryMethodOrchestrator, mock_graph: MagicMock
    ) -> None:
        import asyncio

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

            result = asyncio.run(orchestrator.execute_async("How does authentication work?"))

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


class TestCircuitBreaker:
    """Test circuit breaker logic."""

    @pytest.fixture
    def orchestrator(self) -> QueryMethodOrchestrator:
        mock_graph = MagicMock()
        return QueryMethodOrchestrator(code_graph=mock_graph)

    def test_circuit_breaker_resets_on_success(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        result = QueryMethodResult(
            method=QueryMethod.SEMANTIC_SEARCH,
            items=[{"name": "test"}],
            execution_time_ms=100.0,
        )
        orchestrator._circuit_breaker[QueryMethod.SEMANTIC_SEARCH] = 2
        orchestrator._update_circuit_breaker(QueryMethod.SEMANTIC_SEARCH, result)
        assert orchestrator._circuit_breaker[QueryMethod.SEMANTIC_SEARCH] == 0

    def test_circuit_breaker_increments_on_failure(
        self, orchestrator: QueryMethodOrchestrator
    ) -> None:
        result = QueryMethodResult(
            method=QueryMethod.GRAPH_TRAVERSAL,
            items=[],
            execution_time_ms=50.0,
            error="Connection failed",
        )
        orchestrator._update_circuit_breaker(QueryMethod.GRAPH_TRAVERSAL, result)
        assert orchestrator._circuit_breaker[QueryMethod.GRAPH_TRAVERSAL] == 1


class TestIntegrityVerification:
    """Test graph data integrity verification."""

    def test_detects_missing_file(self, tmp_path) -> None:
        items = [
            {
                "qualified_name": "auth.login",
                "file_path": "nonexistent.py",
                "start_line": 10,
                "end_line": 20,
            }
        ]
        warnings = verify_graph_result_integrity(items, tmp_path, max_checks=1)
        assert len(warnings) == 1
        assert warnings[0].severity == "hard"
        assert "does not exist" in warnings[0].issue

    def test_detects_out_of_bounds_lines(self, tmp_path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\n")

        items = [
            {
                "qualified_name": "auth.login",
                "file_path": "test.py",
                "start_line": 10,
                "end_line": 20,
            }
        ]
        warnings = verify_graph_result_integrity(items, tmp_path, max_checks=1)
        assert len(warnings) == 1
        assert warnings[0].severity == "soft"
        assert "exceeds file length" in warnings[0].issue

    def test_passes_valid_results(self, tmp_path) -> None:
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5\n")

        items = [
            {
                "qualified_name": "auth.login",
                "file_path": "test.py",
                "start_line": 1,
                "end_line": 3,
            }
        ]
        warnings = verify_graph_result_integrity(items, tmp_path, max_checks=1)
        assert len(warnings) == 0


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


class TestCombinedQueryResult:
    """Test CombinedQueryResult dataclass with new integrity fields."""

    def test_default_integrity_fields(self) -> None:
        result = CombinedQueryResult(
            query="test query",
            intent=QueryIntent.EXPLORATORY,
            methods_used=[QueryMethod.SEMANTIC_SEARCH],
            items=[],
            execution_time_ms=100.0,
        )
        assert result.integrity_warnings == []
        assert result.integrity_check_count == 0
        assert result.integrity_pass_count == 0
        assert result.intent_confidence == 0.0

    def test_integrity_warnings_populated(self) -> None:
        warning = IntegrityWarning(
            severity="hard",
            item="auth.login",
            issue="File not found",
            action="Re-index",
        )
        result = CombinedQueryResult(
            query="test query",
            intent=QueryIntent.EXPLORATORY,
            methods_used=[QueryMethod.SEMANTIC_SEARCH],
            items=[],
            execution_time_ms=100.0,
            integrity_warnings=[warning],
            integrity_check_count=1,
            integrity_pass_count=0,
        )
        assert len(result.integrity_warnings) == 1
        assert result.integrity_warnings[0].severity == "hard"
