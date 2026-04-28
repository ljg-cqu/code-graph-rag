"""Tests for relationship category diversity enforcement.

Covers _check_relationship_diversity, _apply_category_caps, and
rebalancing behavior.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock, patch

import pytest

from codebase_rag.document.concept_extraction import (
    ConceptRelationship,
    ExtractedConcept,
    LLMConceptExtractor,
    _apply_category_caps,
    _check_relationship_diversity,
    _rebalance_relationships,
)


class TestCheckRelationshipDiversity:
    """Tests for _check_relationship_diversity."""

    def test_no_skew_when_diverse(self):
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="is-a",
                category="HIERARCHICAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="C",
                verb="contains",
                category="COMPOSITIONAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="D",
                verb="causes",
                category="CAUSAL",
                strength=0.8,
            ),
        ]
        is_skewed = _check_relationship_diversity(rels, "chunk:1")
        assert is_skewed is False

    def test_skew_when_single_category_dominant(self):
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="C",
                verb="enables",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="D",
                verb="triggers",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="E",
                verb="produces",
                category="CAUSAL",
                strength=0.8,
            ),
        ]
        with patch("loguru.logger") as mock_logger:
            is_skewed = _check_relationship_diversity(rels, "chunk:test")
            assert is_skewed is True
            mock_logger.warning.assert_called_once()
            assert "skew detected" in mock_logger.warning.call_args[0][0]
            assert "CAUSAL" in mock_logger.warning.call_args[0][0]

    def test_empty_relationships(self):
        is_skewed = _check_relationship_diversity([], "chunk:empty")
        assert is_skewed is False

    def test_threshold_just_above_boundary(self):
        """Just above 60% should trigger skew."""
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="C",
                verb="enables",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="D",
                verb="triggers",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="E",
                verb="is-a",
                category="HIERARCHICAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="F",
                verb="contains",
                category="COMPOSITIONAL",
                strength=0.8,
            ),
        ]
        # 3/5 = 60% exactly → should NOT trigger (must be > 60%)
        is_skewed = _check_relationship_diversity(rels, "chunk:test")
        assert is_skewed is False

    def test_threshold_above_60_percent(self):
        """4 CAUSAL out of 5 = 80% > 60% → should trigger."""
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="C",
                verb="enables",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="D",
                verb="triggers",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="E",
                verb="produces",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="F",
                verb="is-a",
                category="HIERARCHICAL",
                strength=0.8,
            ),
        ]
        is_skewed = _check_relationship_diversity(rels, "chunk:test")
        assert is_skewed is True


class TestApplyCategoryCaps:
    """Tests for _apply_category_caps."""

    def test_no_cap_when_under_ratio(self):
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="C",
                verb="is-a",
                category="HIERARCHICAL",
                strength=0.9,
            ),
        ]
        result = _apply_category_caps(rels, max_ratio=0.50)
        # 1 CAUSAL out of 2 = 50%, max_allowed = int(2 * 0.5) = 1, 1 <= 1 so no cap
        assert len(result) == 2

    def test_drops_excess_by_strength(self):
        """When category exceeds 50%, drop weakest relationships."""
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.9,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="C",
                verb="enables",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="D",
                verb="triggers",
                category="CAUSAL",
                strength=0.7,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="E",
                verb="produces",
                category="CAUSAL",
                strength=0.6,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="F",
                verb="is-a",
                category="HIERARCHICAL",
                strength=0.5,
            ),
        ]
        # 4 CAUSAL out of 5 = 80% > 50% → max_allowed = int(5 * 0.5) = 2
        result = _apply_category_caps(rels, max_ratio=0.50)
        causal_rels = [r for r in result if r.category == "CAUSAL"]
        assert len(causal_rels) == 2
        # Highest strength ones kept
        assert causal_rels[0].strength == 0.9
        assert causal_rels[1].strength == 0.8

    def test_empty_relationships(self):
        result = _apply_category_caps([], max_ratio=0.50)
        assert result == []

    def test_deterministic_ordering(self):
        """Result should be sorted deterministically."""
        rels = [
            ConceptRelationship(
                from_concept="Z",
                to_concept="A",
                verb="causes",
                category="CAUSAL",
                strength=0.5,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.9,
            ),
            ConceptRelationship(
                from_concept="M",
                to_concept="N",
                verb="causes",
                category="CAUSAL",
                strength=0.7,
            ),
        ]
        result = _apply_category_caps(rels, max_ratio=0.50)
        # With 3 total, max_allowed = int(3*0.5) = 1
        assert len(result) == 1
        assert result[0].strength == 0.9


class TestRebalanceRelationships:
    """Tests for _rebalance_relationships with mocked Agent."""

    @pytest.mark.asyncio
    async def test_rebalance_uses_agent(self):
        # Use an unknown verb so resolve_category doesn't override the rebalance result
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="brand-new-verb",
                category="CAUSAL",
                strength=0.8,
            ),
        ]

        mock_rebalanced = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="brand-new-verb",
                category="ANALOGICAL",
                strength=0.8,
            ),
        ]

        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value=Mock(output=mock_rebalanced))

        with (
            patch("codebase_rag.compat.pydantic_ai.Agent", return_value=mock_agent),
            patch("codebase_rag.services.llm._create_chat_model", return_value=Mock()),
        ):
            result = await _rebalance_relationships(rels, "some content", "chunk:1")

        assert len(result) == 1
        assert result[0].category == "ANALOGICAL"
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_rebalance_rejects_hallucinated_changes(self):
        """If LLM changes from_concept/to_concept/verb, keep original."""
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.8,
            ),
        ]

        # LLM hallucinated a change to from_concept
        mock_rebalanced = [
            ConceptRelationship(
                from_concept="X",
                to_concept="B",
                verb="causes",
                category="ANALOGICAL",
                strength=0.8,
            ),
        ]

        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value=Mock(output=mock_rebalanced))

        with (
            patch("codebase_rag.compat.pydantic_ai.Agent", return_value=mock_agent),
            patch("codebase_rag.services.llm._create_chat_model", return_value=Mock()),
        ):
            result = await _rebalance_relationships(rels, "some content", "chunk:1")

        # Should keep original since from_concept changed
        assert result[0].from_concept == "A"
        assert result[0].category == "CAUSAL"

    @pytest.mark.asyncio
    async def test_rebalance_preserves_category_for_known_verbs(self):
        """Registry verbs are excluded from rebalancing to preserve vetted categories."""
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.8,
            ),
        ]

        mock_agent = Mock()
        mock_agent.run = AsyncMock()

        with (
            patch("codebase_rag.compat.pydantic_ai.Agent", return_value=mock_agent),
            patch("codebase_rag.services.llm._create_chat_model", return_value=Mock()),
        ):
            result = await _rebalance_relationships(rels, "some content", "chunk:1")

        assert result[0].category == "CAUSAL"
        mock_agent.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_rebalance_only_non_registry_verbs(self):
        """Only non-registry verbs are passed to the rebalancing agent."""
        from codebase_rag.document.concept_extraction import VERB_REGISTRY

        registry_verb = "causes"
        non_registry_verb = "brand-new-verb"
        assert registry_verb in VERB_REGISTRY
        assert non_registry_verb not in VERB_REGISTRY

        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb=registry_verb,
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="C",
                verb=non_registry_verb,
                category="CAUSAL",
                strength=0.8,
            ),
        ]

        mock_rebalanced = [
            ConceptRelationship(
                from_concept="A",
                to_concept="C",
                verb=non_registry_verb,
                category="ANALOGICAL",
                strength=0.8,
            ),
        ]

        mock_agent = Mock()
        mock_agent.run = AsyncMock(return_value=Mock(output=mock_rebalanced))

        with (
            patch("codebase_rag.compat.pydantic_ai.Agent", return_value=mock_agent),
            patch("codebase_rag.services.llm._create_chat_model", return_value=Mock()),
        ):
            result = await _rebalance_relationships(rels, "some content", "chunk:1")

        registry_result = next(r for r in result if r.verb == registry_verb)
        non_registry_result = next(r for r in result if r.verb == non_registry_verb)

        assert registry_result.category == "CAUSAL"
        assert non_registry_result.category == "ANALOGICAL"
        mock_agent.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_sets_was_rebalanced_on_skew(self):
        """ExtractionResult.was_rebalanced is True when diversity check triggers."""
        extractor = LLMConceptExtractor()
        mock_agent = Mock()
        mock_result = Mock()
        mock_result.output = Mock()
        mock_result.output.concepts = [
            ExtractedConcept(
                name="X",
                definition="def",
                confidence=0.9,
                entity_category="ABSTRACT_CONCEPT",
            )
        ]
        mock_result.output.relationships = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="C",
                verb="enables",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="D",
                verb="triggers",
                category="CAUSAL",
                strength=0.8,
            ),
            ConceptRelationship(
                from_concept="A",
                to_concept="E",
                verb="produces",
                category="CAUSAL",
                strength=0.8,
            ),
        ]
        mock_result.output.was_rebalanced = False
        mock_agent.run = AsyncMock(return_value=mock_result)
        extractor.agent = mock_agent
        extractor._initialization_failed = False

        with patch(
            "codebase_rag.document.concept_extraction._rebalance_relationships",
            new=AsyncMock(return_value=mock_result.output.relationships),
        ):
            result = await extractor.extract("content", "chunk:1")

        assert result.was_rebalanced is True

    @pytest.mark.asyncio
    async def test_rebalance_failure_returns_original(self):
        """If agent fails, return original relationships."""
        rels = [
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="causes",
                category="CAUSAL",
                strength=0.8,
            ),
        ]

        with (
            patch(
                "codebase_rag.compat.pydantic_ai.Agent",
                side_effect=Exception("agent failed"),
            ),
            patch("codebase_rag.services.llm._create_chat_model", return_value=Mock()),
        ):
            result = await _rebalance_relationships(rels, "some content", "chunk:1")

        assert len(result) == 1
        assert result[0].category == "CAUSAL"
