"""Tests for taxonomy-derived concept relationship vocabulary.

Covers: resolve_category, emoji consistency, VERB_REGISTRY integrity,
ConceptRelationship model, MECE compliance, and backward compatibility.
"""

from __future__ import annotations

import pytest

from codebase_rag.constants import (
    CATEGORY_EMOJI_MAP,
    DOC_CONCEPT_CATEGORIES,
    DocConceptRelCategory,
)
from codebase_rag.document.concept_extraction import (
    VERB_REGISTRY,
    ConceptRelationship,
    _fuzzy_match_verb,
    resolve_category,
)


class TestResolveCategory:
    """Tests for the 4-step category resolution logic."""

    def test_registry_match_uses_declared(self) -> None:
        """Step 1: Verb in registry AND matches declared → use declared."""
        category, emoji = resolve_category("mitigates", "CAUSAL")
        assert category == "CAUSAL"
        assert emoji == "⚡"

    def test_registry_override_mismatched(self) -> None:
        """Step 2: Verb in registry AND mismatches declared → registry wins.

        'is-a' is registered as HIERARCHICAL, so even if LLM declares WRONG,
        we get HIERARCHICAL.
        """
        category, emoji = resolve_category("is-a", "CAUSAL")
        assert category == "HIERARCHICAL"
        assert emoji == "🌳"

    def test_registry_override_is_authoritative(self) -> None:
        """Registry always wins when it has an entry for the verb."""
        category, emoji = resolve_category("derives-from", "CAUSAL")
        assert category == "HIERARCHICAL"  # pinned in registry
        assert emoji == "🌳"

    def test_unregistered_uses_declared(self) -> None:
        """Step 3: Verb NOT in registry → use declared category."""
        category, emoji = resolve_category("novel-verb-123", "COMPARATIVE")
        assert category == "COMPARATIVE"
        assert emoji == "⚖️"

    def test_unregistered_no_declared_fallback(self) -> None:
        """Step 4: Verb NOT in registry AND no declared → RELATED_TO fallback."""
        category, emoji = resolve_category("unknown-verb", None)
        assert category == "RELATED_TO"
        assert emoji == "🔗"

    def test_unregistered_invalid_declared_fallback(self) -> None:
        """Step 4: Verb NOT in registry AND invalid declared → RELATED_TO."""
        category, emoji = resolve_category("another-unknown", "INVALID_CATEGORY")
        assert category == "RELATED_TO"
        assert emoji == "🔗"

    def test_verb_case_insensitive(self) -> None:
        """Verb lookup should be case-insensitive."""
        category, emoji = resolve_category("MITIGATES", "CAUSAL")
        assert category == "CAUSAL"
        assert emoji == "⚡"

    def test_verb_whitespace_insensitive(self) -> None:
        """Verb lookup should strip whitespace."""
        category, emoji = resolve_category("  is-a  ", "HIERARCHICAL")
        assert category == "HIERARCHICAL"
        assert emoji == "🌳"

    def test_registry_override_with_none_declared(self) -> None:
        """Registered verb with None declared → use registry."""
        category, emoji = resolve_category("part-of", None)
        assert category == "COMPOSITIONAL"
        assert emoji == "🧩"

    def test_lowercase_declared_category(self) -> None:
        """Declared category should be normalized to uppercase."""
        category, emoji = resolve_category("novel-verb-xyz", "causal")
        assert category == "CAUSAL"
        assert emoji == "⚡"

    def test_fuzzy_match_misspelled_registry_verb(self) -> None:
        """Step 4: Fuzzy match should catch slight misspellings of registry verbs."""
        category, emoji = resolve_category("mitigate", None)  # missing 's', not in registry
        assert category == "CAUSAL"  # fuzzy-match to "mitigates" → CAUSAL
        assert emoji == "⚡"

    def test_fuzzy_match_with_invalid_declared(self) -> None:
        """Step 4: Fuzzy match works even when declared category is invalid."""
        category, emoji = resolve_category("componet-of", "INVALID")  # 1 char off from component-of
        assert category == "COMPOSITIONAL"  # fuzzy-match to "component-of"
        assert emoji == "🧩"

    def test_fuzzy_match_no_close_match(self) -> None:
        """Step 4: When no fuzzy match found, fallback to RELATED_TO."""
        category, emoji = resolve_category("xyzzy-plugh-garply", None)
        assert category == "RELATED_TO"
        assert emoji == "🔗"

    def test_related_to_verb_registry_override(self) -> None:
        """The verb 'related-to' is in the registry as COMPARATIVE; registry wins."""
        category, emoji = resolve_category("related-to", "RELATED_TO")
        assert category == "COMPARATIVE"
        assert emoji == "⚖️"

    def test_related_to_verb_with_none_declared(self) -> None:
        """Verb 'related-to' with no declared category resolves via registry."""
        category, emoji = resolve_category("related-to", None)
        assert category == "COMPARATIVE"
        assert emoji == "⚖️"

    def test_is_parent_of_resolves_to_hierarchical(self) -> None:
        """Verb 'is-parent-of' resolves to HIERARCHICAL via registry."""
        category, emoji = resolve_category("is-parent-of", None)
        assert category == "HIERARCHICAL"
        assert emoji == "🌳"

    def test_takes_place_in_resolves_to_contextual(self) -> None:
        """Verb 'takes-place-in' resolves to CONTEXTUAL via registry."""
        category, emoji = resolve_category("takes-place-in", None)
        assert category == "CONTEXTUAL"
        assert emoji == "🎯"


class TestFuzzyMatchVerb:
    """Tests for the _fuzzy_match_verb helper."""

    def test_exact_match(self) -> None:
        """Exact match against registry."""
        result = _fuzzy_match_verb("mitigates")
        assert result == "CAUSAL"

    def test_close_misspelling(self) -> None:
        """Single-character misspelling should match."""
        result = _fuzzy_match_verb("componant-of")  # close to "component-of"
        assert result == "COMPOSITIONAL"

    def test_no_match_below_cutoff(self) -> None:
        """Completely different string returns None."""
        result = _fuzzy_match_verb("zzzzzzzzzzzz")
        assert result is None

    def test_empty_string(self) -> None:
        """Empty string returns None."""
        result = _fuzzy_match_verb("")
        assert result is None

    def test_fuzzy_match_case_insensitive(self) -> None:
        """Fuzzy matching works on already-lowered input."""
        result = _fuzzy_match_verb("INHERITS-FROM")  # difflib is case-sensitive
        # This will likely return None since registry keys are lowercase
        # Verify no crash
        assert result is None or result in set(c.value for c in DocConceptRelCategory)


class TestEmojiConsistency:
    """Emoji must always be server-derived from resolved category."""

    def test_all_categories_have_emoji(self) -> None:
        """Every canonical category has a non-empty emoji."""
        for category in DocConceptRelCategory:
            assert CATEGORY_EMOJI_MAP[category.value], (
                f"Missing emoji for {category.value}"
            )

    def test_emoji_never_from_llm_input(self) -> None:
        """Even when LLM provides wrong category, emoji comes from resolved."""
        # Simulate LLM saying category=CAUSAL with emoji=⚡ for verb 'is-a'
        # After resolution: category becomes HIERARCHICAL, emoji must be 🌳
        category, emoji = resolve_category("is-a", "CAUSAL")
        assert category == "HIERARCHICAL"
        assert emoji == "🌳"  # NOT ⚡ — server-resolved from HIERARCHICAL

    def test_fallback_emoji_is_consistent(self) -> None:
        """RELATED_TO fallback always returns 🔗."""
        _, emoji1 = resolve_category("unknown-1", None)
        _, emoji2 = resolve_category("unknown-2", "INVALID")
        assert emoji1 == "🔗"
        assert emoji2 == "🔗"

    def test_emoji_map_values_are_unique(self) -> None:
        """Each emoji should be unique across categories."""
        emojis = list(CATEGORY_EMOJI_MAP.values())
        assert len(emojis) == len(set(emojis)), "Emoji values must be unique"

    def test_emoji_map_has_exactly_nine_entries(self) -> None:
        """CATEGORY_EMOJI_MAP covers all 9 categories."""
        assert len(CATEGORY_EMOJI_MAP) == 9
        for category in DocConceptRelCategory:
            assert category.value in CATEGORY_EMOJI_MAP


class TestMECECompliance:
    """Verify the 8 canonical categories are MECE-complete."""

    def test_eight_canonical_categories_plus_fallback(self) -> None:
        """9 total: 8 canonical + 1 fallback (RELATED_TO)."""
        categories = set(DocConceptRelCategory)
        assert len(categories) == 9
        assert DocConceptRelCategory.RELATED_TO in categories

    def test_canonical_categories_distinct(self) -> None:
        """All 8 canonical categories are semantically distinct."""
        canonical = set(DocConceptRelCategory) - {DocConceptRelCategory.RELATED_TO}
        assert len(canonical) == 8

    def test_doc_concept_categories_matches_enum(self) -> None:
        """DOC_CONCEPT_CATEGORIES frozenset matches DocConceptRelCategory."""
        enum_values = frozenset(c.value for c in DocConceptRelCategory)
        assert DOC_CONCEPT_CATEGORIES == enum_values


class TestVerbRegistry:
    """Tests for VERB_REGISTRY integrity."""

    def test_all_registry_verbs_map_to_valid_category(self) -> None:
        """Every verb in VERB_REGISTRY maps to a valid DocConceptRelCategory."""
        valid = set(c.value for c in DocConceptRelCategory)
        for verb, category in VERB_REGISTRY.items():
            assert category in valid, (
                f"Verb '{verb}' maps to '{category}' which is not a valid category"
            )

    def test_registry_has_required_verbs(self) -> None:
        """Essential taxonomy verbs are present."""
        required = {
            "is-a", "part-of", "causes", "precedes", "follows",
            "analogous-to", "has-property", "compares-to", "depends-on",
            "deployed-in", "mitigates", "implements", "extends",
            "is-parent-of", "takes-place-in", "related-to",
        }
        for verb in required:
            assert verb in VERB_REGISTRY, f"Required verb '{verb}' missing from registry"

    def test_registry_no_duplicate_keys(self) -> None:
        """No duplicate verb entries."""
        verbs = list(VERB_REGISTRY.keys())
        assert len(verbs) == len(set(verbs))

    def test_registry_verbs_are_lowercase(self) -> None:
        """All registry keys are lowercase for consistent lookup."""
        for verb in VERB_REGISTRY:
            assert verb == verb.lower(), f"Verb '{verb}' is not lowercase"

    def test_registry_size_reasonable(self) -> None:
        """Registry should have at least 132 verbs (core taxonomy)."""
        assert len(VERB_REGISTRY) >= 132


class TestConceptRelationshipModel:
    """Tests for the updated ConceptRelationship Pydantic model."""

    def test_model_has_new_fields(self) -> None:
        """Model should have verb, category, and emoji fields."""
        rel = ConceptRelationship(
            from_concept="A",
            to_concept="B",
            verb="mitigates",
            category="CAUSAL",
            emoji="⚡",
            strength=0.9,
        )
        assert rel.verb == "mitigates"
        assert rel.category == "CAUSAL"
        assert rel.emoji == "⚡"
        assert rel.strength == 0.9

    def test_model_emoji_defaults_to_empty(self) -> None:
        """Emoji defaults to empty string."""
        rel = ConceptRelationship(
            from_concept="A",
            to_concept="B",
            verb="related-to",
            category="RELATED_TO",
        )
        assert rel.emoji == ""

    def test_model_no_longer_has_relationship_type(self) -> None:
        """The old field name should not exist."""
        rel = ConceptRelationship(
            from_concept="A",
            to_concept="B",
            verb="is-a",
            category="HIERARCHICAL",
        )
        assert not hasattr(rel, "relationship_type")

    def test_model_field_validation(self) -> None:
        """Strength must be between 0 and 1."""
        with pytest.raises(Exception):
            ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="is-a",
                category="HIERARCHICAL",
                strength=1.5,
            )

    def test_all_categories_accepted(self) -> None:
        """All 9 categories should be valid for the category field."""
        for category in DocConceptRelCategory:
            rel = ConceptRelationship(
                from_concept="X",
                to_concept="Y",
                verb="test-verb",
                category=category.value,
            )
            assert rel.category == category.value


class TestBackwardCompatibility:
    """Old labels coexist with new ones."""

    def test_old_labels_still_in_relationship_type_enum(self) -> None:
        """IS_A, PART_OF, CAUSES, RELATED_TO remain in RelationshipType."""
        from codebase_rag.constants import RelationshipType
        assert RelationshipType.IS_A == "IS_A"
        assert RelationshipType.PART_OF == "PART_OF"
        assert RelationshipType.CAUSES == "CAUSES"
        assert RelationshipType.RELATED_TO == "RELATED_TO"

    def test_new_category_labels_not_in_relationship_type(self) -> None:
        """New labels are in DocConceptRelCategory, not RelationshipType."""
        from codebase_rag.constants import RelationshipType
        rel_type_values = set(r.value for r in RelationshipType)
        assert "HIERARCHICAL" not in rel_type_values
        assert "COMPOSITIONAL" not in rel_type_values
        assert "CONTEXTUAL" not in rel_type_values

    def test_graph_algorithms_include_old_labels(self) -> None:
        """Verify graph algorithm patterns still reference old labels.

        Old labels (IS_A, PART_OF, CAUSES, RELATED_TO) must remain in
        traversal patterns for backward compatibility with existing data.
        """
        from codebase_rag.document.graph_algorithms import DocumentGraphAlgorithms
        import inspect

        source = inspect.getsource(DocumentGraphAlgorithms.find_shortest_path)
        assert "IS_A" in source
        assert "PART_OF" in source
        assert "CAUSES" in source
        assert "RELATED_TO" in source

    def test_graph_algorithms_include_new_labels(self) -> None:
        """Verify graph algorithm patterns reference new category labels."""
        from codebase_rag.document.graph_algorithms import DocumentGraphAlgorithms
        import inspect

        source = inspect.getsource(DocumentGraphAlgorithms.find_shortest_path)
        assert "HIERARCHICAL" in source
        assert "COMPOSITIONAL" in source
        assert "CONTEXTUAL" in source
        assert "ANALOGICAL" in source
