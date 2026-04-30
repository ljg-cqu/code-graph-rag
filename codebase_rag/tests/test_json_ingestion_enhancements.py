"""Tests for JSON ingestion enhancements (entity taxonomy, verb inference, emoji)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from codebase_rag.json_ingestion import (
    DatasetReferences,
    _entity_labels,
    _entity_properties,
    _extract_emoji_prefix,
    _infer_relationship_verb,
    _normalize_relationship_category,
    _relationship_properties,
    ingest_relationships,
    sanitize_edge_label,
)
from codebase_rag.schemas import JSONRelationship


class TestExtractEmojiPrefix:
    def test_extracts_single_emoji(self) -> None:
        clean, emoji = _extract_emoji_prefix("👤 CTO Role")
        assert clean == "CTO Role"
        assert emoji == "👤"

    def test_extracts_emoji_with_variation_selector(self) -> None:
        clean, emoji = _extract_emoji_prefix("⚙️ Technical Architecture")
        assert clean == "Technical Architecture"
        assert emoji is not None
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

    def test_extracts_zwj_sequence(self) -> None:
        clean, emoji = _extract_emoji_prefix("👨‍🦲 Bald Man")
        assert clean == "Bald Man"
        assert emoji == "👨‍🦲"

    def test_extracts_family_emoji(self) -> None:
        clean, emoji = _extract_emoji_prefix("👩‍👩‍👧‍👦 Family")
        assert clean == "Family"
        assert emoji == "👩‍👩‍👧‍👦"

    def test_extracts_emoji_with_skin_tone(self) -> None:
        clean, emoji = _extract_emoji_prefix("👋🏿 Wave")
        assert clean == "Wave"
        assert emoji == "👋🏿"

    def test_extracts_flag_emoji(self) -> None:
        clean, emoji = _extract_emoji_prefix("🇺🇸 USA")
        assert clean == "USA"
        assert emoji == "🇺🇸"


class TestNormalizeRelationshipCategory:
    def test_attribute_maps_to_attributive(self) -> None:
        assert _normalize_relationship_category("ATTRIBUTE") == "ATTRIBUTIVE"

    def test_attributes_maps_to_attributive(self) -> None:
        assert _normalize_relationship_category("ATTRIBUTES") == "ATTRIBUTIVE"

    def test_inferred_maps_to_related_to(self) -> None:
        assert _normalize_relationship_category("INFERRED") == "RELATED_TO"

    def test_inference_maps_to_related_to(self) -> None:
        assert _normalize_relationship_category("INFERENCE") == "RELATED_TO"

    def test_causes_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("CAUSES") == "CAUSAL"

    def test_part_of_maps_to_compositional(self) -> None:
        assert _normalize_relationship_category("PART_OF") == "COMPOSITIONAL"

    def test_is_a_maps_to_hierarchical(self) -> None:
        assert _normalize_relationship_category("IS_A") == "HIERARCHICAL"

    def test_causal_passthrough(self) -> None:
        assert _normalize_relationship_category("CAUSAL") == "CAUSAL"

    def test_compositional_passthrough(self) -> None:
        assert _normalize_relationship_category("COMPOSITIONAL") == "COMPOSITIONAL"

    def test_empty_string_defaults_to_related_to(self) -> None:
        assert _normalize_relationship_category("") == "RELATED_TO"

    def test_whitespace_defaults_to_related_to(self) -> None:
        assert _normalize_relationship_category("   ") == "RELATED_TO"

    def test_unknown_category_defaults_to_related_to(self) -> None:
        with patch("codebase_rag.json_ingestion.logger") as mock_logger:
            assert _normalize_relationship_category("UNKNOWN_CATEGORY") == "RELATED_TO"
        mock_logger.warning.assert_called_once()
        assert "Unknown relationship category" in mock_logger.warning.call_args[0][0]

    def test_case_insensitive(self) -> None:
        assert _normalize_relationship_category("causal") == "CAUSAL"
        assert _normalize_relationship_category("CaUsAl") == "CAUSAL"

    def test_comprises_maps_to_compositional(self) -> None:
        assert _normalize_relationship_category("comprises") == "COMPOSITIONAL"

    def test_contains_maps_to_compositional(self) -> None:
        assert _normalize_relationship_category("contains") == "COMPOSITIONAL"

    def test_similar_to_maps_to_comparative(self) -> None:
        assert _normalize_relationship_category("similar-to") == "COMPARATIVE"

    def test_contrasts_with_maps_to_comparative(self) -> None:
        assert _normalize_relationship_category("contrasts-with") == "COMPARATIVE"

    def test_relates_to_maps_to_comparative(self) -> None:
        assert _normalize_relationship_category("relates-to") == "COMPARATIVE"

    def test_precedes_maps_to_sequential(self) -> None:
        assert _normalize_relationship_category("precedes") == "SEQUENTIAL"

    def test_follows_maps_to_sequential(self) -> None:
        assert _normalize_relationship_category("follows") == "SEQUENTIAL"

    def test_preceded_by_maps_to_sequential(self) -> None:
        assert _normalize_relationship_category("preceded-by") == "SEQUENTIAL"

    def test_followed_by_maps_to_sequential(self) -> None:
        assert _normalize_relationship_category("followed-by") == "SEQUENTIAL"

    def test_triggers_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("triggers") == "CAUSAL"

    def test_influences_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("influences") == "CAUSAL"

    def test_results_in_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("results-in") == "CAUSAL"

    def test_leads_to_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("leads-to") == "CAUSAL"

    def test_correlates_with_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("correlates-with") == "CAUSAL"

    def test_mitigates_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("mitigates") == "CAUSAL"

    def test_reduces_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("reduces") == "CAUSAL"

    def test_depends_on_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("depends-on") == "CAUSAL"

    def test_blocks_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("blocks") == "CAUSAL"

    def test_drives_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("drives") == "CAUSAL"

    def test_facilitates_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("facilitates") == "CAUSAL"

    def test_promotes_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("promotes") == "CAUSAL"

    def test_partof_maps_to_compositional(self) -> None:
        assert _normalize_relationship_category("partof") == "COMPOSITIONAL"

    def test_instanceof_maps_to_hierarchical(self) -> None:
        assert _normalize_relationship_category("instanceof") == "HIERARCHICAL"

    def test_type_of_maps_to_hierarchical(self) -> None:
        assert _normalize_relationship_category("type_of") == "HIERARCHICAL"

    def test_kind_of_maps_to_hierarchical(self) -> None:
        assert _normalize_relationship_category("kind_of") == "HIERARCHICAL"

    def test_prevents_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("prevents") == "CAUSAL"

    def test_enables_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("enables") == "CAUSAL"

    def test_inhibits_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("inhibits") == "CAUSAL"

    def test_produces_maps_to_causal(self) -> None:
        assert _normalize_relationship_category("produces") == "CAUSAL"


class TestSanitizeEdgeLabel:
    def test_replaces_hyphens_with_underscores(self) -> None:
        assert sanitize_edge_label("has-property") == "has_property"

    def test_replaces_dots_with_underscores(self) -> None:
        assert sanitize_edge_label("part.of") == "part_of"

    def test_leading_digit_gets_prefix(self) -> None:
        assert sanitize_edge_label("123_relation") == "R_123_relation"

    def test_passthrough_for_valid_label(self) -> None:
        assert sanitize_edge_label("normal") == "normal"

    def test_replaces_spaces_with_underscores(self) -> None:
        assert sanitize_edge_label("has property with spaces") == "has_property_with_spaces"

    def test_collapses_multiple_special_chars(self) -> None:
        assert sanitize_edge_label("a---b___c") == "a_b_c"

    def test_preserves_case(self) -> None:
        assert sanitize_edge_label("iPhone_hasFeature") == "iPhone_hasFeature"

    def test_empty_label_raises(self) -> None:
        with pytest.raises(ValueError):
            sanitize_edge_label("")

    def test_only_special_chars_raises(self) -> None:
        with pytest.raises(ValueError):
            sanitize_edge_label("---")


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

    def test_explicit_entity_category_override(self) -> None:
        entity = {
            "id": "test-1",
            "name": "Custom Entity",
            "type": "UnknownType",
            "entity_category": "EVENT_PROCESS",
            "properties": {},
        }
        props = _entity_properties("test-dataset", entity, {})
        assert props["entity_category"] == "EVENT_PROCESS"
        assert props["entity_emoji"] == "⏱️"

    def test_explicit_entity_category_case_insensitive(self) -> None:
        entity = {
            "id": "test-1",
            "name": "Custom Entity",
            "type": "UnknownType",
            "entity_category": "concrete_entity",
            "properties": {},
        }
        props = _entity_properties("test-dataset", entity, {})
        assert props["entity_category"] == "CONCRETE_ENTITY"
        assert props["entity_emoji"] == "🧱"

    def test_source_emoji_preserved(self) -> None:
        entity = {
            "id": "test-1",
            "name": "Critical Thinking",
            "type": "Cognitive Process",
            "emoji": "🧠",
            "properties": {},
        }
        props = _entity_properties("test-dataset", entity, {})
        assert props["entity_emoji_source"] == "🧠"
        assert props["entity_emoji"] == "⏱️"

    def test_source_emoji_omitted_when_absent(self) -> None:
        entity = {
            "id": "test-1",
            "name": "Critical Thinking",
            "type": "Cognitive Process",
            "properties": {},
        }
        props = _entity_properties("test-dataset", entity, {})
        assert "entity_emoji_source" not in props


class TestRelationshipProperties:
    def test_verb_inferred_when_missing(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "category": "CAUSAL",
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

    def test_verb_stored_from_original_relationship(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["verb"] == "CAUSAL"

    def test_category_stored(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["category"] == "CAUSAL"

    def test_category_normalized(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "ATTRIBUTE",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["category"] == "ATTRIBUTIVE"

    def test_emoji_stored(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["emoji"] == "⚡"

    def test_emoji_for_hierarchical(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "HIERARCHICAL",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["emoji"] == "🌳"

    def test_emoji_fallback(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "UNKNOWN_CATEGORY",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["emoji"] == "🔗"  # default

    def test_missing_category_logs_warning(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "originates_from",
        }
        with patch("codebase_rag.json_ingestion.logger") as mock_logger:
            props = _relationship_properties("test-dataset", relationship, {})
        assert props["category"] == "RELATED_TO"
        warning_messages = [
            str(call.args[0]) for call in mock_logger.warning.call_args_list
        ]
        assert any("missing 'category' field" in msg for msg in warning_messages)

    def test_symmetric_stored_as_is_symmetric(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
            "symmetric": True,
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["is_symmetric"] is True

    def test_symmetric_false_stored(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
            "symmetric": False,
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["is_symmetric"] is False

    def test_inferred_stored_as_snake_case(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
            "inferred": True,
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["is_inferred"] is True
        assert "isInferred" not in props

    def test_legacy_isInferred_normalized(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "CAUSAL",
            "isInferred": True,
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["is_inferred"] is True
        assert "isInferred" not in props

    def test_original_verb_stored(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "causes",
            "category": "CAUSAL",
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["original_verb"] == "causes"
        assert props["verb"] == "causes"

    def test_original_verb_stored_when_verb_overridden(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "causes",
            "category": "CAUSAL",
            "properties": {"verb": "custom"},
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["original_verb"] == "causes"
        assert props["verb"] == "custom"

    def test_deep_causal_analysis_flattened(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "causes",
            "category": "CAUSAL",
            "properties": {
                "deep_causal_analysis": {
                    "root_cause_chain": [
                        {"factor": "Evolutionary Pressure", "depth": 3, "confidence": 0.70},
                        {"factor": "Survival Optimization", "depth": 2, "confidence": 0.80},
                        {"factor": "Energy Efficiency", "depth": 1, "confidence": 0.85},
                    ],
                    "ultimate_effects": [
                        {"effect": "Decision Errors", "depth": 1, "confidence": 0.85},
                    ],
                    "termination_reason": "max_depth_reached",
                },
            },
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert "deep_causal_analysis" not in props
        assert props["causal_depth"] == 3
        assert props["causal_termination"] == "max_depth_reached"
        assert "root_cause_chain" in props
        assert "ultimate_effects" in props

    def test_trust_score_for_explicit_relationship(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "causes",
            "inferred": False,
            "properties": {"confidence": 0.85},
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["is_inferred"] is False
        assert props["trust_score"] == 0.85

    def test_trust_score_for_inferred_relationship(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "causes",
            "inferred": True,
            "properties": {"confidence": 0.72},
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["is_inferred"] is True
        assert props["trust_score"] == pytest.approx(0.576, abs=0.001)

    def test_trust_score_defaults_to_one_when_no_confidence(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "causes",
            "inferred": False,
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["trust_score"] == 1.0

    def test_trust_score_for_inferred_without_confidence(self) -> None:
        relationship = {
            "source": "src-1",
            "target": "tgt-1",
            "relationship": "causes",
            "inferred": True,
        }
        props = _relationship_properties("test-dataset", relationship, {})
        assert props["trust_score"] == 0.8


class TestSymmetricBidirectionalEdges:
    def test_symmetric_relationship_creates_reverse_edge(self) -> None:
        class _MockGraphConn:
            def __init__(self):
                self.calls: list[dict] = []

            def fetch_all(self, query: str, params: dict):
                self.calls.append({"query": query, "params": params})
                if "MATCH (n:JsonEntity" in query and "LIMIT 2" in query:
                    return [{"unique_id": f"test-dataset::{params['reference']}"}]
                if "deleted" in query:
                    return [{"deleted": 0}]
                return [{"relationship_id": 1}]

        graph_conn = _MockGraphConn()
        refs = DatasetReferences(
            ids={"System 1": "test-dataset::sys1", "System 2": "test-dataset::sys2"}
        )
        relationships = [
            {
                "source": "System 1",
                "target": "System 2",
                "relationship": "contrasts_with",
                "category": "COMPARATIVE",
                "symmetric": True,
                "properties": {"confidence": 0.9},
            }
        ]
        summary = ingest_relationships(
            "test-dataset",
            relationships,
            refs,
            {},
            graph_connection=graph_conn,
        )
        assert summary.ingested == 2
        # Two MERGE calls: one forward, one reverse
        merge_calls = [
            c for c in graph_conn.calls if "MERGE (a)-[r:" in c["query"]
        ]
        assert len(merge_calls) == 2
        # First call is forward
        assert merge_calls[0]["params"]["source_id"] == "test-dataset::sys1"
        assert merge_calls[0]["params"]["target_id"] == "test-dataset::sys2"
        # Second call is reverse (params retain original names; Cypher swaps them)
        assert merge_calls[1]["params"]["source_id"] == "test-dataset::sys1"
        assert merge_calls[1]["params"]["target_id"] == "test-dataset::sys2"

    def test_non_symmetric_relationship_creates_single_edge(self) -> None:
        class _MockGraphConn:
            def __init__(self):
                self.calls: list[dict] = []

            def fetch_all(self, query: str, params: dict):
                self.calls.append({"query": query, "params": params})
                if "MATCH (n:JsonEntity" in query and "LIMIT 2" in query:
                    return [{"unique_id": f"test-dataset::{params['reference']}"}]
                return [{"relationship_id": 1}]

        graph_conn = _MockGraphConn()
        refs = DatasetReferences(
            ids={"A": "test-dataset::a", "B": "test-dataset::b"}
        )
        relationships = [
            {
                "source": "A",
                "target": "B",
                "relationship": "causes",
                "category": "CAUSAL",
                "symmetric": False,
            }
        ]
        summary = ingest_relationships(
            "test-dataset",
            relationships,
            refs,
            {},
            graph_connection=graph_conn,
        )
        assert summary.ingested == 1
        merge_calls = [
            c for c in graph_conn.calls if "MERGE (a)-[r:" in c["query"]
        ]
        assert len(merge_calls) == 1


class TestEntityLabels:
    def test_basic_labels(self) -> None:
        entity = {"type": "Person", "labels": ["Employee", "Manager"]}
        labels = _entity_labels(entity)
        assert labels == ["JsonEntity", "Person", "Employee", "Manager"]

    def test_case_insensitive_deduplication(self) -> None:
        entity = {
            "type": "Cognitive Process",
            "labels": ["COGNITIVE_PROCESS", "Cognitive Process", "cognitive process"],
        }
        labels = _entity_labels(entity)
        lower_labels = [label.lower() for label in labels]
        assert len(labels) == len(set(lower_labels))
        assert "JsonEntity" in labels
        assert "CognitiveProcess" in labels

    def test_deduplicates_type_against_labels(self) -> None:
        entity = {"type": "Person", "labels": ["person", "Person", "PERSON"]}
        labels = _entity_labels(entity)
        assert labels.count("Person") == 1
        assert labels.count("person") == 0
        assert labels.count("PERSON") == 0

    def test_empty_labels_ignored(self) -> None:
        entity = {"type": "Thing", "labels": ["", "  ", "Valid"]}
        labels = _entity_labels(entity)
        assert "Valid" in labels
        assert "" not in labels
        assert "  " not in labels


class TestJSONRelationshipModel:
    def test_inferred_field_validated(self) -> None:
        rel = JSONRelationship(
            source="a",
            target="b",
            relationship="causes",
            inferred=True,
        )
        assert rel.inferred is True

    def test_inferred_defaults_to_none(self) -> None:
        rel = JSONRelationship(
            source="a",
            target="b",
            relationship="causes",
        )
        assert rel.inferred is None

    def test_legacy_isInferred_ignored(self) -> None:
        rel = JSONRelationship(
            source="a",
            target="b",
            relationship="causes",
            isInferred=True,
        )
        assert rel.inferred is None
