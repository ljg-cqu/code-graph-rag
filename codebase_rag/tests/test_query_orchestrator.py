"""Tests for QueryMethodOrchestrator.

NOTE: This module now uses LLM-driven orchestration per the LLM-First Orchestration spec.
The previous rule-based methods (classify_intent, select_methods) have been replaced
with LLMQueryPlanner for intent classification and method selection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.orchestrator.llm_query_planner import (
    GraphAlgorithm,
    LLMQueryPlanner,
    QueryIntent,
    QueryMethod,
    QueryPlan,
)
from codebase_rag.retrieval.query_orchestrator import (
    CombinedQueryResult,
    IntegrityWarning,
    QueryMethodOrchestrator,
    QueryMethodResult,
    verify_graph_result_integrity,
)


class TestLLMQueryPlanner:
    """Test LLM-driven query planning (replaces rule-based classification)."""

    @pytest.fixture
    def planner(self) -> LLMQueryPlanner:
        return LLMQueryPlanner()

    def test_query_plan_structure(self) -> None:
        """Test that QueryPlan has all required fields."""
        plan = QueryPlan(
            methods=[QueryMethod.SEMANTIC_SEARCH],
            reasoning="Test reasoning",
            fallback_methods=[QueryMethod.KEYWORD_SEARCH],
            expected_entities=["auth", "login"],
            requires_file_read=False,
            intent=QueryIntent.FUNCTIONAL,
            algorithm=GraphAlgorithm.NONE,
        )
        assert plan.methods == [QueryMethod.SEMANTIC_SEARCH]
        assert plan.intent == QueryIntent.FUNCTIONAL
        assert plan.expected_entities == ["auth", "login"]

    def test_query_plan_defaults(self) -> None:
        """Test QueryPlan default values."""
        plan = QueryPlan(methods=[QueryMethod.GRAPH_TRAVERSAL])
        assert plan.fallback_methods == []
        assert plan.expected_entities == []
        assert plan.requires_file_read is False
        assert plan.intent is None
        assert plan.algorithm == GraphAlgorithm.NONE

    def test_planner_caching(self, planner: LLMQueryPlanner) -> None:
        """Test that planner has caching infrastructure."""
        # Cache key should be deterministic for same query
        key1 = planner._get_cache_key("test query")
        key2 = planner._get_cache_key("test query")
        assert key1 == key2
        assert len(key1) == 16  # SHA256 hex digest truncated

    def test_planner_cache_operations(self, planner: LLMQueryPlanner) -> None:
        """Test cache store and retrieve."""
        plan = QueryPlan(
            methods=[QueryMethod.SEMANTIC_SEARCH],
            reasoning="Cached plan",
        )
        key = "test_key"

        # Store in cache
        planner._cache_plan(key, plan)

        # Retrieve from cache
        cached = planner._get_cached_plan(key)
        assert cached is not None
        assert cached.reasoning == "Cached plan"

        # Non-existent key returns None
        assert planner._get_cached_plan("nonexistent") is None


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
