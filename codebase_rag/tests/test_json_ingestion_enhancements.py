"""Tests for JSON ingestion enhancements (entity taxonomy, verb inference, emoji)."""
from __future__ import annotations

from codebase_rag.json_ingestion import (
    _entity_properties,
    _extract_emoji_prefix,
    _infer_relationship_verb,
    _normalize_relationship_category,
    _relationship_properties,
)


class TestExtractEmojiPrefix:
    def test_extracts_single_emoji(self) -> None:
        clean, emoji = _extract_emoji_prefix("👤 CTO Role")
        assert clean == "CTO Role"
        assert emoji == "👤"

    def test_extracts_emoji_with_variation_selector(self) -> None:
        clean, emoji = _extract_emoji_prefix("⚙️ Technical Architecture")
        assert clean == "Technical Architecture"
        assert "⚙" in emoji

    def test_no_emoji_returns_unchanged(self) -> None:
        clean, emoji = _extract_emoji_prefix("CTO Role")
        assert clean == "CTO Role"
        assert emoji is None

    def test_empty_string(self) -> None:
        clean, emoji = _extract_emoji_prefix("")
        assert clean == ""
        assert emoji is None

    def test_extracts_multiple_emojis(self) -> None:
        clean, emoji = _extract_emoji_prefix("🧠💡 Strategic Thinking")
        assert clean == "Strategic Thinking"
        assert emoji == "🧠💡"


class TestNormalizeRelationshipCategory:
    def test_attribute_maps_to_attributive(self) -> None:
        assert _normalize_relationship_category("ATTRIBUTE") == "ATTRIBUTIVE"

    def test_inferred_maps_to_related_to(self) -> None:
        assert _normalize_relationship_category("INFERRED") == "RELATED_TO"

    def test_causal_passthrough(self) -> None:
        assert _normalize_relationship_category("CAUSAL") == "CAUSAL"

    def test_compositional_passthrough(self) -> None:
        assert _normalize_relationship_category("COMPOSITIONAL") == "COMPOSITIONAL"

    def test_empty_string_defaults_to_related_to(self) -> None:
        assert _normalize_relationship_category("") == "RELATED_TO"

    def test_whitespace_defaults_to_related_to(self) -> None:
        assert _normalize_relationship_category("   ") == "RELATED_TO"

    def test_unknown_category_defaults_to_related_to(self) -> None:
        assert _normalize_relationship_category("UNKNOWN_CATEGORY") == "RELATED_TO"

    def test_case_insensitive(self) -> None:
        assert _normalize_relationship_category("causal") == "CAUSAL"
        assert _normalize_relationship_category("CaUsAl") == "CAUSAL"


class TestInferRelationshipVerb:
    def test_mindset_to_competency_causal(self) -> None:
        assert _infer_relationship_verb("CAUSAL", "Mindset", "Competency") == "enables"

    def test_framework_to_competency_causal(self) -> None:
        assert _infer_relationship_verb("CAUSAL", "Framework", "Competency") == "develops"

    def test_framework_to_framework_compositional(self) -> None:
        assert (
            _infer_relationship_verb("COMPOSITIONAL", "Framework", "Framework")
            == "contains"
        )

    def test_competency_to_competency_hierarchical(self) -> None:
        assert (
            _infer_relationship_verb("HIERARCHICAL", "Competency", "Competency")
            == "subtype-of"
        )

    def test_unknown_type_pair_uses_default(self) -> None:
        assert _infer_relationship_verb("CAUSAL", "Unknown", "Unknown") == "influences"

    def test_unknown_category_uses_fallback(self) -> None:
        assert _infer_relationship_verb("UNKNOWN", "A", "B") == "related-to"

    def test_attribute_normalization(self) -> None:
        assert (
            _infer_relationship_verb("ATTRIBUTE", "Framework", "Property")
            == "characterizes"
        )


class TestEntityProperties:
    def test_entity_category_applied(self) -> None:
        entity = {
            "id": "test-1",
            "name": "Strategic Thinking",
            "type": "Mindset",
            "properties": {"description": "Test description"},
        }
        props = _entity_properties("test-dataset", entity, {})
        assert props["entity_category"] == "ABSTRACT_CONCEPT"
        assert props["entity_subtype"] == "Mindset"
        assert props["entity_emoji"] == "💡"

    def test_emoji_extracted_from_name(self) -> None:
        entity = {
            "id": "test-1",
            "name": "👤 CTO Role",
            "type": "Role",
            "properties": {},
        }
        props = _entity_properties("test-dataset", entity, {})
        assert props["name"] == "CTO Role"
        assert props["emoji"] == "👤"
        assert props["display_name"] == "👤 CTO Role"

    def test_qualified_name_alias(self) -> None:
        entity = {
            "id": "test-1",
            "name": "Entity",
            "type": "Thing",
            "properties": {},
        }
        props = _entity_properties("test-dataset", entity, {})
        assert props["qualified_name"] == "test-dataset::test-1"
        assert props["unique_id"] == "test-dataset::test-1"

    def test_default_graph_scores(self) -> None:
        entity = {
            "id": "test-1",
            "name": "Entity",
            "type": "Thing",
            "properties": {},
        }
        props = _entity_properties("test-dataset", entity, {})
        assert props["pagerank_score"] == 0.1
        assert props["community_id"] == -1
        assert props["community_importance"] == 0.0


class TestRelationshipProperties:
    def test_verb_inferred_when_missing(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
        }
        entities_by_id = {
            "src-1": {"type": "Mindset"},
            "tgt-1": {"type": "Competency"},
        }
        props = _relationship_properties(
            "test-dataset", relationship, {}, entities_by_id=entities_by_id
        )
        assert props["verb"] == "enables"

    def test_existing_verb_preserved(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
            "properties": {"verb": "custom-verb"},
        }
        entities_by_id = {
            "src-1": {"type": "Mindset"},
            "tgt-1": {"type": "Competency"},
        }
        props = _relationship_properties(
            "test-dataset", relationship, {}, entities_by_id=entities_by_id
        )
        assert props["verb"] == "custom-verb"

    def test_no_entities_by_id_skips_inference(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert "verb" not in props

    def test_relationship_category_stored(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["relationship_category"] == "CAUSAL"

    def test_relationship_category_normalized(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "ATTRIBUTE",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["relationship_category"] == "ATTRIBUTIVE"

    def test_relationship_emoji_stored(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["relationship_emoji"] == "⚡"

    def test_relationship_emoji_for_hierarchical(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "HIERARCHICAL",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["relationship_emoji"] == "🌳"

    def test_relationship_emoji_fallback(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "UNKNOWN_CATEGORY",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["relationship_emoji"] == "🔗"  # default
