"""Unit tests for ConceptConsolidator.

Covers confidence-weighted plurality voting, alias merging, definition/context
resolution, and cross-document deduplication.
"""

from __future__ import annotations

import pytest

from codebase_rag.document.concept_consolidator import (
    ConceptConsolidator,
    ConsolidatedConcept,
)


class TestConceptConsolidator:
    """Test ConceptConsolidator conflict resolution."""

    def test_plurality_vote_wins(self):
        """Two chunks extract same concept with different categories → plurality vote wins."""
        consolidator = ConceptConsolidator()
        consolidator.add(
            {
                "qualified_name": "test:Empathetic Leadership",
                "workspace": "test",
                "name": "Empathetic Leadership",
                "aliases": ["EL"],
                "type": "EVENT_PROCESS",
                "definition": "def1",
                "confidence": 0.9,
                "entity_category": "EVENT_PROCESS",
                "entity_subtype": None,
                "entity_emoji": "⏱️",
                "context": "context1",
            },
            "chunk1",
        )
        consolidator.add(
            {
                "qualified_name": "test:Empathetic Leadership",
                "workspace": "test",
                "name": "Empathetic Leadership",
                "aliases": ["Empathy"],
                "type": "PROPERTY_ATTRIBUTE",
                "definition": "def2",
                "confidence": 0.8,
                "entity_category": "PROPERTY_ATTRIBUTE",
                "entity_subtype": None,
                "entity_emoji": "📏",
                "context": "context2",
            },
            "chunk2",
        )

        result = consolidator.consolidate()
        assert len(result) == 1
        node = result[0]
        # EVENT_PROCESS wins: 0.9 > 0.8 (single vote each, highest confidence wins tie)
        assert node["entity_category"] == "EVENT_PROCESS"
        assert node["type"] == "EVENT_PROCESS"
        # Aliases union
        assert sorted(node["aliases"]) == ["EL", "Empathy"]
        # Definition from winning source
        assert node["definition"] == "def1"
        # Context from winning source
        assert node["context"] == "context1"
        # Max confidence
        assert node["confidence"] == 0.9
        # Source chunks
        assert sorted(node["source_chunks"]) == ["chunk1", "chunk2"]

    def test_plurality_vote_multiple_votes(self):
        """Three chunks: EVENT_PROCESS (0.9), PROPERTY_ATTRIBUTE (0.8), PROPERTY_ATTRIBUTE (0.7)
        → PROPERTY_ATTRIBUTE wins (2 votes vs 1)."""
        consolidator = ConceptConsolidator()
        for category, conf, chunk in [
            ("EVENT_PROCESS", 0.9, "chunk1"),
            ("PROPERTY_ATTRIBUTE", 0.8, "chunk2"),
            ("PROPERTY_ATTRIBUTE", 0.7, "chunk3"),
        ]:
            consolidator.add(
                {
                    "qualified_name": "test:Concept",
                    "workspace": "test",
                    "name": "Concept",
                    "aliases": [],
                    "type": category,
                    "definition": f"def_{chunk}",
                    "confidence": conf,
                    "entity_category": category,
                    "entity_subtype": None,
                    "entity_emoji": "💡",
                    "context": f"ctx_{chunk}",
                },
                chunk,
            )

        result = consolidator.consolidate()
        assert len(result) == 1
        node = result[0]
        # PROPERTY_ATTRIBUTE: 0.8 + 0.7 = 1.5 > EVENT_PROCESS: 0.9
        assert node["entity_category"] == "PROPERTY_ATTRIBUTE"
        # Winning source is chunk2 (highest confidence for PROPERTY_ATTRIBUTE)
        assert node["definition"] == "def_chunk2"
        assert node["context"] == "ctx_chunk2"

    def test_definition_tie_breaker_longest(self):
        """When two sources have same confidence for winning category,
        longest definition wins."""
        consolidator = ConceptConsolidator()
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": [],
                "type": "ABSTRACT_CONCEPT",
                "definition": "short",
                "confidence": 0.8,
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": None,
                "entity_emoji": "💡",
                "context": "ctx1",
            },
            "chunk1",
        )
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": [],
                "type": "ABSTRACT_CONCEPT",
                "definition": "this is a much longer definition",
                "confidence": 0.8,
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": None,
                "entity_emoji": "💡",
                "context": "ctx2",
            },
            "chunk2",
        )

        result = consolidator.consolidate()
        assert result[0]["definition"] == "this is a much longer definition"

    def test_aliases_merge_and_dedup(self):
        """Aliases from multiple sources are unioned and deduplicated."""
        consolidator = ConceptConsolidator()
        for aliases, chunk in [
            (["A", "B"], "chunk1"),
            (["B", "C"], "chunk2"),
            (["C", "D"], "chunk3"),
        ]:
            consolidator.add(
                {
                    "qualified_name": "test:Concept",
                    "workspace": "test",
                    "name": "Concept",
                    "aliases": aliases,
                    "type": "ABSTRACT_CONCEPT",
                    "definition": "def",
                    "confidence": 0.9,
                    "entity_category": "ABSTRACT_CONCEPT",
                    "entity_subtype": None,
                    "entity_emoji": "💡",
                    "context": "ctx",
                },
                chunk,
            )

        result = consolidator.consolidate()
        assert result[0]["aliases"] == ["A", "B", "C", "D"]

    def test_workspace_conflict_raises(self):
        """Conflicting workspaces for same QN should raise ValueError."""
        consolidator = ConceptConsolidator()
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "ws1",
                "name": "Concept",
                "aliases": [],
                "type": "ABSTRACT_CONCEPT",
                "definition": "def",
                "confidence": 0.9,
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": None,
                "entity_emoji": "💡",
                "context": "ctx",
            },
            "chunk1",
        )
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "ws2",
                "name": "Concept",
                "aliases": [],
                "type": "ABSTRACT_CONCEPT",
                "definition": "def",
                "confidence": 0.8,
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": None,
                "entity_emoji": "💡",
                "context": "ctx",
            },
            "chunk2",
        )

        with pytest.raises(ValueError, match="conflicting workspaces"):
            consolidator.consolidate()

    def test_name_conflict_raises(self):
        """Conflicting names for same QN should raise ValueError."""
        consolidator = ConceptConsolidator()
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "ConceptA",
                "aliases": [],
                "type": "ABSTRACT_CONCEPT",
                "definition": "def",
                "confidence": 0.9,
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": None,
                "entity_emoji": "💡",
                "context": "ctx",
            },
            "chunk1",
        )
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "ConceptB",
                "aliases": [],
                "type": "ABSTRACT_CONCEPT",
                "definition": "def",
                "confidence": 0.8,
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": None,
                "entity_emoji": "💡",
                "context": "ctx",
            },
            "chunk2",
        )

        with pytest.raises(ValueError, match="conflicting names"):
            consolidator.consolidate()

    def test_legacy_node_missing_confidence_defaults_to_zero(self):
        """Legacy node with missing confidence should still participate in voting."""
        consolidator = ConceptConsolidator()
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": [],
                "type": "EVENT_PROCESS",
                "definition": "def1",
                "confidence": 0.9,
                "entity_category": "EVENT_PROCESS",
                "entity_subtype": None,
                "entity_emoji": "⏱️",
                "context": "ctx1",
            },
            "chunk1",
        )
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": [],
                "type": "PROPERTY_ATTRIBUTE",
                "definition": "def2",
                # Missing confidence → defaults to 0.0
                "entity_category": "PROPERTY_ATTRIBUTE",
                "entity_subtype": None,
                "entity_emoji": "📏",
                "context": "ctx2",
            },
            "chunk2",
        )

        result = consolidator.consolidate()
        # EVENT_PROCESS wins: 0.9 > 0.0
        assert result[0]["entity_category"] == "EVENT_PROCESS"

    def test_entity_subtype_from_winning_source(self):
        """entity_subtype should come from the same source that won entity_category."""
        consolidator = ConceptConsolidator()
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": [],
                "type": "CONCRETE_ENTITY",
                "definition": "def1",
                "confidence": 0.9,
                "entity_category": "CONCRETE_ENTITY",
                "entity_subtype": "Container Image",
                "entity_emoji": "🧱",
                "context": "ctx1",
            },
            "chunk1",
        )
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": [],
                "type": "ABSTRACT_CONCEPT",
                "definition": "def2",
                "confidence": 0.5,
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": "Design Pattern",
                "entity_emoji": "💡",
                "context": "ctx2",
            },
            "chunk2",
        )

        result = consolidator.consolidate()
        assert result[0]["entity_category"] == "CONCRETE_ENTITY"
        assert result[0]["entity_subtype"] == "Container Image"

    def test_emoji_server_resolved_from_winning_category(self):
        """entity_emoji should be server-resolved from winning category, ignoring LLM-provided."""
        consolidator = ConceptConsolidator()
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": [],
                "type": "CONCRETE_ENTITY",
                "definition": "def",
                "confidence": 0.9,
                "entity_category": "CONCRETE_ENTITY",
                "entity_subtype": None,
                "entity_emoji": "wrong_emoji",
                "context": "ctx",
            },
            "chunk1",
        )

        result = consolidator.consolidate()
        # Should be resolved from ENTITY_CATEGORY_EMOJI_MAP, not "wrong_emoji"
        assert result[0]["entity_emoji"] == "🧱"

    def test_multiple_concepts_sorted_by_qn(self):
        """Multiple concepts should be returned sorted by qualified_name."""
        consolidator = ConceptConsolidator()
        for qn in ["test:Zebra", "test:Apple", "test:Mango"]:
            consolidator.add(
                {
                    "qualified_name": qn,
                    "workspace": "test",
                    "name": qn.split(":")[1],
                    "aliases": [],
                    "type": "ABSTRACT_CONCEPT",
                    "definition": "def",
                    "confidence": 0.9,
                    "entity_category": "ABSTRACT_CONCEPT",
                    "entity_subtype": None,
                    "entity_emoji": "💡",
                    "context": "ctx",
                },
                "chunk1",
            )

        result = consolidator.consolidate()
        qns = [r["qualified_name"] for r in result]
        assert qns == ["test:Apple", "test:Mango", "test:Zebra"]

    def test_max_confidence_across_all_sources(self):
        """Confidence should be max across all sources, not just winning category."""
        consolidator = ConceptConsolidator()
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": [],
                "type": "EVENT_PROCESS",
                "definition": "def1",
                "confidence": 0.6,
                "entity_category": "EVENT_PROCESS",
                "entity_subtype": None,
                "entity_emoji": "⏱️",
                "context": "ctx1",
            },
            "chunk1",
        )
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": [],
                "type": "PROPERTY_ATTRIBUTE",
                "definition": "def2",
                "confidence": 0.95,
                "entity_category": "PROPERTY_ATTRIBUTE",
                "entity_subtype": None,
                "entity_emoji": "📏",
                "context": "ctx2",
            },
            "chunk2",
        )

        result = consolidator.consolidate()
        # PROPERTY_ATTRIBUTE wins (0.95 > 0.6)
        assert result[0]["entity_category"] == "PROPERTY_ATTRIBUTE"
        # But confidence is max across ALL sources
        assert result[0]["confidence"] == 0.95

    def test_cross_document_merge_with_existing_concepts(self):
        """Simulate DocumentGraphUpdater pattern: existing concepts loaded with penalty."""
        consolidator = ConceptConsolidator()
        # New extraction
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": ["NewAlias"],
                "type": "EVENT_PROCESS",
                "definition": "new def",
                "confidence": 0.9,
                "entity_category": "EVENT_PROCESS",
                "entity_subtype": None,
                "entity_emoji": "⏱️",
                "context": "new ctx",
            },
            "chunk1",
        )
        # Existing from graph (confidence penalized by 0.9)
        consolidator.add(
            {
                "qualified_name": "test:Concept",
                "workspace": "test",
                "name": "Concept",
                "aliases": ["OldAlias"],
                "type": "PROPERTY_ATTRIBUTE",
                "definition": "old def",
                "confidence": 0.5 * 0.9,  # 0.45
                "entity_category": "PROPERTY_ATTRIBUTE",
                "entity_subtype": None,
                "entity_emoji": "📏",
                "context": "old ctx",
            },
            "__existing__",
        )

        result = consolidator.consolidate()
        # EVENT_PROCESS wins: 0.9 > 0.45
        assert result[0]["entity_category"] == "EVENT_PROCESS"
        # Aliases include both
        assert sorted(result[0]["aliases"]) == ["NewAlias", "OldAlias"]
        # Source chunks include both
        assert sorted(result[0]["source_chunks"]) == ["__existing__", "chunk1"]


class TestConsolidatedConcept:
    """Test ConsolidatedConcept dataclass."""

    def test_dataclass_creation(self):
        concept = ConsolidatedConcept(
            qualified_name="test:Concept",
            workspace="test",
            name="Concept",
            aliases=["A", "B"],
            type="ABSTRACT_CONCEPT",
            definition="A concept",
            confidence=0.9,
            entity_category="ABSTRACT_CONCEPT",
            entity_subtype=None,
            entity_emoji="💡",
            context="some context",
            source_chunks=["chunk1", "chunk2"],
        )
        assert concept.qualified_name == "test:Concept"
        assert concept.source_chunks == ["chunk1", "chunk2"]
