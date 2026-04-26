"""Tests for JSON graph query engine."""
from __future__ import annotations

from typing import Any

from codebase_rag.json_queries import (
    JsonEntityResult,
    JsonGraphQueryEngine,
)


class MockExecutor:
    """Mock MemgraphIngestor for testing."""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = records or []
        self.queries: list[tuple[str, dict[str, Any]]] = []

    def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((query, params or {}))
        return self.records

    def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        self.queries.append((query, params or {}))


class TestJsonEntityResult:
    def test_to_dict_truncates_long_description(self) -> None:
        result = JsonEntityResult(
            unique_id="test::1",
            name="Test",
            entity_type="Role",
            entity_category="AGENT_ROLE",
            entity_subtype="Role",
            entity_emoji="🎭",
            description="x" * 300,
        )
        d = result.to_dict()
        assert d["description"].endswith("...")
        assert len(d["description"]) == 203


class TestKeywordSearch:
    def test_basic_keyword_search(self) -> None:
        records = [
            {
                "unique_id": "ds::1",
                "name": "Strategic Thinking",
                "entity_type": "Mindset",
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": "Mindset",
                "entity_emoji": "💡",
                "description": "A cognitive capability",
                "pagerank_score": 0.5,
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.keyword_search("strategic thinking", top_k=5)
        assert len(results) == 1
        assert results[0].name == "Strategic Thinking"
        assert results[0].text_score == 1.0

    def test_empty_keywords_returns_empty(self) -> None:
        engine = JsonGraphQueryEngine(executor=MockExecutor())
        results = engine.keyword_search("a b c", top_k=5)
        assert results == []

    def test_category_filter(self) -> None:
        records = [
            {
                "unique_id": "ds::1",
                "name": "CTO Role",
                "entity_type": "Role",
                "entity_category": "AGENT_ROLE",
                "entity_subtype": "Role",
                "entity_emoji": "🎭",
                "description": "Chief Technology Officer",
                "pagerank_score": 0.8,
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.keyword_search("officer", top_k=5, entity_category="AGENT_ROLE")
        assert len(results) == 1


class TestSemanticSearchFallback:
    def test_no_embedding_provider_falls_back(self) -> None:
        records = [
            {
                "unique_id": "ds::1",
                "name": "Strategic Thinking",
                "entity_type": "Mindset",
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": "Mindset",
                "entity_emoji": "💡",
                "description": "A cognitive capability",
                "pagerank_score": 0.5,
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.semantic_search("strategic thinking", top_k=5)
        assert len(results) == 1
        assert results[0].name == "Strategic Thinking"


class TestFindRelatedEntities:
    def test_related_entities(self) -> None:
        records = [
            {
                "unique_id": "ds::2",
                "name": "Technical Architecture",
                "entity_type": "Competency",
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": "Competency",
                "entity_emoji": "💡",
                "description": "Designing systems",
                "pagerank_score": 0.6,
                "depth": 1,
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.find_related_entities("Strategic Thinking", top_k=5)
        assert len(results) == 1
        assert results[0].name == "Technical Architecture"


class TestFindRelationships:
    def test_find_relationships(self) -> None:
        records = [
            {
                "relationship_type": "CAUSAL",
                "relationship_category": "CAUSAL",
                "relationship_emoji": "⚡",
                "verb": "enables",
                "strength": 0.95,
                "from_name": "Strategic Thinking",
                "from_type": "Mindset",
                "to_name": "Technical Architecture",
                "to_type": "Competency",
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.find_relationships("Strategic Thinking")
        assert len(results) == 1
        assert results[0].verb == "enables"
        assert results[0].relationship_category == "CAUSAL"
        assert results[0].relationship_emoji == "⚡"
        assert results[0].strength == 0.95

    def test_filter_by_category(self) -> None:
        records = [
            {
                "relationship_type": "CAUSAL",
                "relationship_category": "CAUSAL",
                "relationship_emoji": "⚡",
                "verb": "enables",
                "strength": 0.9,
                "from_name": "A",
                "from_type": "Mindset",
                "to_name": "B",
                "to_type": "Competency",
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.find_relationships("A", relationship_category="CAUSAL")
        assert len(results) == 1

    def test_filter_by_verb(self) -> None:
        records = [
            {
                "relationship_type": "CAUSAL",
                "relationship_category": "CAUSAL",
                "relationship_emoji": "⚡",
                "verb": "enables",
                "strength": 0.9,
                "from_name": "A",
                "from_type": "Mindset",
                "to_name": "B",
                "to_type": "Competency",
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.find_relationships("A", verb="enables")
        assert len(results) == 1

    def test_relationship_defaults(self) -> None:
        """Test that missing properties use defaults."""
        records = [
            {
                "relationship_type": "RELATED_TO",
                "verb": "related-to",
                "from_name": "A",
                "to_name": "B",
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.find_relationships("A")
        assert len(results) == 1
        assert results[0].relationship_category == "RELATED_TO"
        assert results[0].relationship_emoji == "🔗"
        assert results[0].verb == "related-to"

    def test_backward_compatible_category_filter(self) -> None:
        """Test that category filter works with COALESCE (property or type)."""
        # Simulates old data without relationship_category property
        records = [
            {
                "relationship_type": "CAUSAL",
                # Note: relationship_category is missing (old data)
                "verb": "enables",
                "from_name": "A",
                "to_name": "B",
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        # Query should still work - uses defaults
        results = engine.find_relationships("A")
        assert len(results) == 1
        assert results[0].relationship_category == "RELATED_TO"  # Default
        assert results[0].relationship_type == "CAUSAL"  # From type(r)


class TestGetEntitiesByCategory:
    def test_filter_by_category(self) -> None:
        records = [
            {
                "unique_id": "ds::1",
                "name": "CTO",
                "entity_type": "Role",
                "entity_category": "AGENT_ROLE",
                "entity_subtype": "Role",
                "entity_emoji": "🎭",
                "description": "Chief Technology Officer",
                "pagerank_score": 0.8,
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.get_entities_by_category("AGENT_ROLE", top_k=5)
        assert len(results) == 1
        assert results[0].entity_category == "AGENT_ROLE"


class TestGetImportantEntities:
    def test_ordered_by_pagerank(self) -> None:
        records = [
            {
                "unique_id": "ds::1",
                "name": "CTO",
                "entity_type": "Role",
                "entity_category": "AGENT_ROLE",
                "entity_subtype": "Role",
                "entity_emoji": "🎭",
                "description": "Chief Technology Officer",
                "pagerank_score": 0.9,
            },
            {
                "unique_id": "ds::2",
                "name": "Engineer",
                "entity_type": "Role",
                "entity_category": "AGENT_ROLE",
                "entity_subtype": "Role",
                "entity_emoji": "🎭",
                "description": "Software Engineer",
                "pagerank_score": 0.5,
            },
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        results = engine.get_important_entities(top_k=5)
        assert len(results) == 2
        assert results[0].pagerank_score == 0.9


class TestGetEntityTaxonomy:
    def test_taxonomy_with_root(self) -> None:
        records = [
            {
                "root_name": "CTO Framework",
                "descendant_name": "Foundations Layer",
                "descendant_id": "ds::2",
                "descendant_type": "Layer",
                "relationship_type": "COMPOSITIONAL",
                "depth": 1,
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        taxonomy = engine.get_entity_taxonomy(root_entity="CTO Framework")
        assert "CTO Framework" in taxonomy
        assert len(taxonomy["CTO Framework"]["children"]) == 1
        assert taxonomy["CTO Framework"]["children"][0]["name"] == "Foundations Layer"

    def test_full_taxonomy_without_root(self) -> None:
        records = [
            {
                "root_name": "CTO Framework",
                "descendant_name": "Foundations Layer",
                "descendant_id": "ds::2",
                "descendant_type": "Layer",
                "relationship_type": "COMPOSITIONAL",
                "depth": 1,
            }
        ]
        engine = JsonGraphQueryEngine(executor=MockExecutor(records))
        taxonomy = engine.get_entity_taxonomy()
        assert "CTO Framework" in taxonomy
