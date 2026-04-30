"""Tests for QueryRouter and CONCEPT_ONLY mode."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from codebase_rag.shared.query_router import (
    QueryMode,
    QueryRequest,
    QueryRouter,
    Source,
)


class TestQueryMode:
    """Test QueryMode enum values."""

    def test_concept_only_mode_exists(self):
        """CONCEPT_ONLY enum value must exist."""
        assert QueryMode.CONCEPT_ONLY == "concept_only"
        assert QueryMode.CONCEPT_ONLY.value == "concept_only"

    def test_all_modes_are_strings(self):
        """All QueryMode values must be strings."""
        for mode in QueryMode:
            assert isinstance(mode.value, str)


class TestConceptOnlyRouting:
    """Test CONCEPT_ONLY query routing."""

    @pytest.fixture
    def mock_concept_graph(self):
        """Return a mocked concept graph connection."""
        conn = MagicMock()
        return_value = [
            {
                "name": "Critical Thinking",
                "entity_category": "EVENT_PROCESS",
                "entity_subtype": "Cognitive Process",
                "entity_emoji": "⏱️",
                "qualified_name": "default:Critical Thinking",
                "definition": "The objective analysis and evaluation of an issue.",
                "related_concepts": ["Self-Corrective Thinking", "Metacognition"],
                "references": ["doc1.md", "doc2.md"],
            }
        ]
        conn.fetch_all.return_value = return_value

        async def async_fetch_all(*args, **kwargs):
            return return_value

        conn.fetch_all_async = async_fetch_all
        return conn

    def test_concept_only_routes_to_concept_graph(self, mock_concept_graph):
        """CONCEPT_ONLY mode queries the concept graph."""
        router = QueryRouter(concept_graph=mock_concept_graph)
        request = QueryRequest(
            question="Critical Thinking",
            mode=QueryMode.CONCEPT_ONLY,
        )
        response = router.query(request)

        assert response.mode == QueryMode.CONCEPT_ONLY
        assert "Critical Thinking" in response.answer
        mock_concept_graph.fetch_all.assert_called_once()

    def test_concept_only_returns_error_when_no_concept_graph(self):
        """CONCEPT_ONLY returns helpful error when concept graph is unavailable."""
        router = QueryRouter()
        request = QueryRequest(
            question="test",
            mode=QueryMode.CONCEPT_ONLY,
        )
        response = router.query(request)

        assert response.mode == QueryMode.CONCEPT_ONLY
        assert "not available" in response.answer.lower()
        assert any("not configured" in w for w in response.warnings)

    def test_concept_only_disabled_in_code_only_mode(self):
        """Concept queries are disabled when router is in CODE_ONLY mode."""
        router = QueryRouter(concept_graph=MagicMock())
        router.current_mode = QueryMode.CODE_ONLY
        request = QueryRequest(
            question="test",
            mode=QueryMode.CONCEPT_ONLY,
        )
        response = router._query_concept_only(request)

        assert "disabled" in response.answer.lower()
        assert any("disabled" in w for w in response.warnings)

    def test_concept_only_forced_in_code_only_mode(self):
        """Concept queries work in CODE_ONLY mode when forced."""
        mock_graph = MagicMock()
        mock_graph.fetch_all.return_value = [
            {
                "name": "Test Concept",
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": "",
                "entity_emoji": "💡",
                "qualified_name": "default:Test Concept",
                "definition": "",
                "references": [],
            }
        ]
        router = QueryRouter(concept_graph=mock_graph)
        router.current_mode = QueryMode.CODE_ONLY
        request = QueryRequest(
            question="test",
            mode=QueryMode.CONCEPT_ONLY,
            forced=True,
        )
        response = router._query_concept_only(request)

        assert response.mode == QueryMode.CONCEPT_ONLY
        assert "disabled" not in response.answer.lower()
        mock_graph.fetch_all.assert_called_once()

    def test_concept_results_include_entity_properties(self, mock_concept_graph):
        """Concept query results include entity_category, entity_subtype, entity_emoji."""
        router = QueryRouter(concept_graph=mock_concept_graph)
        request = QueryRequest(
            question="Critical Thinking",
            mode=QueryMode.CONCEPT_ONLY,
        )
        response = router.query(request)

        assert len(response.sources) == 1
        source = response.sources[0]
        assert source.type == "concept"
        assert source.node_type == "EVENT_PROCESS"
        assert source.qualified_name == "default:Critical Thinking"
        assert "⏱️" in response.answer

    def test_concept_keyword_search_lowercasing(self, mock_concept_graph):
        """Keywords are lowercased before Cypher comparison."""
        router = QueryRouter(concept_graph=mock_concept_graph)
        request = QueryRequest(
            question="Critical Thinking",
            mode=QueryMode.CONCEPT_ONLY,
        )
        router.query(request)

        call_args = mock_concept_graph.fetch_all.call_args
        params = call_args[0][1]
        assert all(kw.islower() for kw in params["keywords"])

    def test_concept_empty_results(self):
        """Empty concept results produce appropriate message."""
        mock_graph = MagicMock()
        mock_graph.fetch_all.return_value = []
        router = QueryRouter(concept_graph=mock_graph)
        request = QueryRequest(
            question="nonexistent concept",
            mode=QueryMode.CONCEPT_ONLY,
        )
        response = router.query(request)

        assert "No concepts found" in response.answer
        assert response.sources == []

    def test_concept_async_routing(self, mock_concept_graph):
        """Async query routes CONCEPT_ONLY correctly."""
        router = QueryRouter(concept_graph=mock_concept_graph)
        request = QueryRequest(
            question="Critical Thinking",
            mode=QueryMode.CONCEPT_ONLY,
        )

        import asyncio

        response = asyncio.run(router.query_async(request))
        assert response.mode == QueryMode.CONCEPT_ONLY
        assert "Critical Thinking" in response.answer

    def test_concept_only_async_method_exists(self, mock_concept_graph):
        """_query_concept_only_async method exists and is callable."""
        router = QueryRouter(concept_graph=mock_concept_graph)
        assert hasattr(router, "_query_concept_only_async")
        import inspect
        assert inspect.iscoroutinefunction(router._query_concept_only_async)

    def test_concept_related_concepts_in_response(self, mock_concept_graph):
        """Concept response includes related concepts from graph traversal."""
        router = QueryRouter(concept_graph=mock_concept_graph)
        request = QueryRequest(
            question="Critical Thinking",
            mode=QueryMode.CONCEPT_ONLY,
        )
        response = router.query(request)

        assert "Self-Corrective Thinking" in response.answer
        assert "Metacognition" in response.answer

    def test_concept_cypher_includes_relationship_traversal(self, mock_concept_graph):
        """Keyword Cypher includes concept-to-concept relationship traversal."""
        router = QueryRouter(concept_graph=mock_concept_graph)
        request = QueryRequest(
            question="Critical Thinking",
            mode=QueryMode.CONCEPT_ONLY,
        )
        router.query(request)

        cypher = mock_concept_graph.fetch_all.call_args[0][0]
        assert "OPTIONAL MATCH (c)-[rel:" in cypher
        assert "COMPOSITIONAL" in cypher
        assert "CAUSAL" in cypher
        assert "->(related:Concept)" in cypher
        assert "related_concepts" in cypher


class TestCodeLegacyKeywordCaseSensitivity:
    """Test _query_code_legacy fallback handles mixed-case keywords."""

    def test_code_fallback_lowercases_keywords(self):
        """expected_entities keywords are lowercased before Cypher execution."""
        mock_code_graph = MagicMock()
        mock_code_graph.fetch_all.return_value = []
        router = QueryRouter(code_graph=mock_code_graph)
        from codebase_rag.orchestrator.llm_query_planner import QueryPlan

        request = QueryRequest(
            question="How does ResolveEntityCategory work?",
            mode=QueryMode.CODE_ONLY,
            plan=QueryPlan(
                methods=["keyword_search"],
                reasoning="test",
                expected_entities=["ResolveEntityCategory"],
            ),
        )
        router.query(request)

        call_args = mock_code_graph.fetch_all.call_args
        params = call_args[0][1]
        assert params["keywords"] == ["resolveentitycategory"]

    def test_code_fallback_cypher_uses_tolower(self):
        """Cypher query uses toLower() on name and qualified_name."""
        mock_code_graph = MagicMock()
        mock_code_graph.fetch_all.return_value = []
        router = QueryRouter(code_graph=mock_code_graph)
        from codebase_rag.orchestrator.llm_query_planner import QueryPlan

        request = QueryRequest(
            question="How does ResolveEntityCategory work?",
            mode=QueryMode.CODE_ONLY,
            plan=QueryPlan(
                methods=["keyword_search"],
                reasoning="test",
                expected_entities=["ResolveEntityCategory"],
            ),
        )
        router.query(request)

        cypher = mock_code_graph.fetch_all.call_args[0][0]
        assert "toLower(n.name) CONTAINS kw" in cypher
        assert "toLower(n.qualified_name) CONTAINS kw" in cypher


class TestSourceType:
    """Test Source dataclass supports concept type."""

    def test_source_accepts_concept_type(self):
        """Source.type must accept 'concept' value."""
        source = Source(
            type="concept",
            path="concept_graph",
            node_type="ABSTRACT_CONCEPT",
            qualified_name="default:Test",
        )
        assert source.type == "concept"
        assert source.to_dict()["type"] == "concept"
