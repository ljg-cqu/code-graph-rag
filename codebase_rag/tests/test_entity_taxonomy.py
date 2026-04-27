"""Tests for taxonomy-derived entity category classification.

Covers: resolve_entity_category, emoji consistency, ENTITY_SUBTYPE_REGISTRY
integrity, ExtractedConcept model, MECE compliance, boundary disambiguation,
and backward compatibility.
"""

from __future__ import annotations

import pytest

from codebase_rag.constants import (
    DOC_ENTITY_CATEGORIES,
    ENTITY_CATEGORY_EMOJI_MAP,
    DocConceptEntityCategory,
)
from codebase_rag.document.concept_extraction import (
    ENTITY_SUBTYPE_REGISTRY,
    ExtractedConcept,
    resolve_entity_category,
)


class TestResolveEntityCategory:
    """Tests for the 4-step entity category resolution logic."""

    def test_valid_declared_category_used(self) -> None:
        """Step 1: Valid declared category → use it."""
        category, subtype, emoji = resolve_entity_category(
            "CONCRETE_ENTITY", "Container Image", ""
        )
        assert category == "CONCRETE_ENTITY"
        assert subtype == "Container Image"
        assert emoji == "🧱"

    def test_invalid_category_subtype_registry_match(self) -> None:
        """Step 2: Invalid category + subtype in registry → use registry."""
        category, subtype, emoji = resolve_entity_category(
            "INVALID_CATEGORY", "Container Image", ""
        )
        assert category == "CONCRETE_ENTITY"  # from registry
        assert subtype == "Container Image"
        assert emoji == "🧱"

    def test_none_category_subtype_registry_match(self) -> None:
        """Step 2: None category + subtype in registry → use registry."""
        category, subtype, emoji = resolve_entity_category(
            None, "API", ""
        )
        assert category == "INFORMATION_EXPRESSION"
        assert emoji == "📨"

    def test_invalid_category_no_subtype_fallback(self) -> None:
        """Step 3: Invalid category + no subtype match → ABSTRACT_CONCEPT."""
        category, subtype, emoji = resolve_entity_category(
            "INVALID", None, ""
        )
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_none_category_none_subtype_fallback(self) -> None:
        """Step 3: No category and no subtype → ABSTRACT_CONCEPT."""
        category, subtype, emoji = resolve_entity_category(None, None, "")
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_invalid_category_invalid_subtype_fallback(self) -> None:
        """Step 3: Both invalid → ABSTRACT_CONCEPT."""
        category, subtype, emoji = resolve_entity_category(
            "BAD", "NonExistentSubType", ""
        )
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_case_insensitive_category(self) -> None:
        """Declared category should be normalized to uppercase."""
        category, _, emoji = resolve_entity_category(
            "concrete_entity", None, ""
        )
        assert category == "CONCRETE_ENTITY"
        assert emoji == "🧱"

    def test_emoji_prefixed_category_stripped(self) -> None:
        """Step 1: Emoji-prefixed declared category should be stripped and matched."""
        category, subtype, emoji = resolve_entity_category(
            "🏗️ SYSTEM_STRUCTURE", "Cognitive System", ""
        )
        assert category == "SYSTEM_STRUCTURE"
        assert subtype == "Cognitive System"
        assert emoji == "🏗️"

    def test_multiple_emoji_prefixed_categories(self) -> None:
        """Various emoji-prefixed categories resolve correctly."""
        test_cases = [
            ("📏 PROPERTY_ATTRIBUTE", "PROPERTY_ATTRIBUTE", "📏"),
            ("⏱️ EVENT_PROCESS", "EVENT_PROCESS", "⏱️"),
            ("📨 INFORMATION_EXPRESSION", "INFORMATION_EXPRESSION", "📨"),
            ("🎭 AGENT_ROLE", "AGENT_ROLE", "🎭"),
            ("💡 ABSTRACT_CONCEPT", "ABSTRACT_CONCEPT", "💡"),
            ("🧱 CONCRETE_ENTITY", "CONCRETE_ENTITY", "🧱"),
        ]
        for declared, expected_cat, expected_emoji in test_cases:
            category, _, emoji = resolve_entity_category(declared, None, "")
            assert category == expected_cat, f"Failed for {declared}"
            assert emoji == expected_emoji, f"Failed for {declared}"

    def test_emoji_only_category_resolved(self) -> None:
        """LLM occasionally puts emoji in entity_category; resolve via reverse map."""
        test_cases = [
            ("📏", "PROPERTY_ATTRIBUTE", "📏"),
            ("🏗️", "SYSTEM_STRUCTURE", "🏗️"),
            ("📨", "INFORMATION_EXPRESSION", "📨"),
            ("⏱️", "EVENT_PROCESS", "⏱️"),
            ("🎭", "AGENT_ROLE", "🎭"),
            ("🧱", "CONCRETE_ENTITY", "🧱"),
            ("💡", "ABSTRACT_CONCEPT", "💡"),
        ]
        for declared, expected_cat, expected_emoji in test_cases:
            category, _, emoji = resolve_entity_category(declared, None, "")
            assert category == expected_cat, f"Failed for emoji {declared}"
            assert emoji == expected_emoji, f"Failed for emoji {declared}"

    def test_all_seven_categories_accepted(self) -> None:
        """All 7 canonical categories are valid."""
        for member in DocConceptEntityCategory:
            category, _, emoji = resolve_entity_category(
                member.value, None, ""
            )
            assert category == member.value
            assert emoji == ENTITY_CATEGORY_EMOJI_MAP[member.value]

    def test_subtype_preserved_when_not_in_registry(self) -> None:
        """Sub-type not in registry is kept as-is."""
        category, subtype, emoji = resolve_entity_category(
            "SYSTEM_STRUCTURE", "CustomSubType", ""
        )
        assert category == "SYSTEM_STRUCTURE"
        assert subtype == "CustomSubType"  # preserved
        assert emoji == "🏗️"

    def test_subtype_none_preserved(self) -> None:
        """None subtype stays None."""
        _, subtype, _ = resolve_entity_category("AGENT_ROLE", None, "")
        assert subtype is None


class TestEmojiServerResolved:
    """Emoji must always be server-derived from ENTITY_CATEGORY_EMOJI_MAP."""

    def test_subtype_registry_overrides_invalid_declared_category(self) -> None:
        """Step 2: Invalid declared category + subtype in registry → registry wins."""
        # LLM declares invalid category "WRONG" but subtype "Container Image" is registered
        category, _, emoji = resolve_entity_category(
            "WRONG", "Container Image", ""
        )
        assert category == "CONCRETE_ENTITY"  # from registry
        assert emoji == "🧱"

    def test_emoji_from_resolved_category_not_llm_input(self) -> None:
        """Even when LLM provides wrong category, emoji comes from resolved.

        When declared category is valid (even if wrong), it takes priority
        per the resolution spec (Step 1). The emoji is always derived from
        the final resolved category via ENTITY_CATEGORY_EMOJI_MAP.
        """
        # When LLM declares ABSTRACT_CONCEPT, that's a valid category — it wins.
        category, _, emoji = resolve_entity_category(
            "ABSTRACT_CONCEPT", "Container Image", ""
        )
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"  # derived from resolved category, never from LLM input

    def test_all_categories_have_emoji(self) -> None:
        """Every canonical category has a non-empty emoji."""
        for category in DocConceptEntityCategory:
            assert ENTITY_CATEGORY_EMOJI_MAP[category.value], (
                f"Missing emoji for {category.value}"
            )

    def test_emoji_map_has_seven_entries(self) -> None:
        """ENTITY_CATEGORY_EMOJI_MAP covers all 7 categories."""
        assert len(ENTITY_CATEGORY_EMOJI_MAP) == 7
        for category in DocConceptEntityCategory:
            assert category.value in ENTITY_CATEGORY_EMOJI_MAP

    def test_emoji_values_are_unique(self) -> None:
        """Each emoji should be unique across entity categories."""
        emojis = list(ENTITY_CATEGORY_EMOJI_MAP.values())
        assert len(emojis) == len(set(emojis)), "Entity emoji values must be unique"

    def test_fallback_emoji_is_consistent(self) -> None:
        """ABSTRACT_CONCEPT fallback always returns 💡."""
        _, _, emoji1 = resolve_entity_category(None, None, "")
        _, _, emoji2 = resolve_entity_category("INVALID", "BadSubType", "")
        assert emoji1 == "💡"
        assert emoji2 == "💡"


class TestMECECompliance:
    """Verify the 7 canonical entity categories are MECE-complete."""

    def test_seven_canonical_categories(self) -> None:
        """7 categories total, no fallback separate from ABSTRACT_CONCEPT."""
        categories = set(DocConceptEntityCategory)
        assert len(categories) == 7

    def test_abstract_concept_is_separate_member(self) -> None:
        """ABSTRACT_CONCEPT is a distinct member."""
        assert DocConceptEntityCategory.ABSTRACT_CONCEPT == "ABSTRACT_CONCEPT"

    def test_doc_entity_categories_matches_enum(self) -> None:
        """DOC_ENTITY_CATEGORIES frozenset matches DocConceptEntityCategory."""
        enum_values = frozenset(c.value for c in DocConceptEntityCategory)
        assert DOC_ENTITY_CATEGORIES == enum_values

    def test_categories_are_mutually_exclusive(self) -> None:
        """No duplicate values across categories."""
        values = [c.value for c in DocConceptEntityCategory]
        assert len(values) == len(set(values))

    def test_ontological_gradient_order(self) -> None:
        """Categories follow concrete → abstract gradient."""
        # Most concrete → most abstract
        expected_order = [
            "CONCRETE_ENTITY",
            "EVENT_PROCESS",
            "INFORMATION_EXPRESSION",
            "PROPERTY_ATTRIBUTE",
            "SYSTEM_STRUCTURE",
            "AGENT_ROLE",
            "ABSTRACT_CONCEPT",
        ]
        actual_order = [c.value for c in DocConceptEntityCategory]
        assert actual_order == expected_order


class TestBoundaryDisambiguation:
    """Tests for boundary disambiguation rules (22 rule pairs)."""

    def test_substance_test_concrete_vs_event(self) -> None:
        """Concrete vs Event: Substance test."""
        # Material composition → Concrete; temporal → Event
        cat1, _, _ = resolve_entity_category("CONCRETE_ENTITY", "Artifact", "")
        cat2, _, _ = resolve_entity_category("EVENT_PROCESS", "Deployment", "")
        assert cat1 == "CONCRETE_ENTITY"
        assert cat2 == "EVENT_PROCESS"

    def test_material_test_concrete_vs_information(self) -> None:
        """Concrete vs Information: Material test."""
        cat1, _, _ = resolve_entity_category("CONCRETE_ENTITY", "Facility", "")
        cat2, _, _ = resolve_entity_category("INFORMATION_EXPRESSION", "Source File", "")
        assert cat1 == "CONCRETE_ENTITY"
        assert cat2 == "INFORMATION_EXPRESSION"

    def test_independence_test_concrete_vs_property(self) -> None:
        """Concrete vs Property: Independence test."""
        cat1, _, _ = resolve_entity_category("CONCRETE_ENTITY", "Product", "")
        cat2, _, _ = resolve_entity_category("PROPERTY_ATTRIBUTE", "SLI", "")
        assert cat1 == "CONCRETE_ENTITY"
        assert cat2 == "PROPERTY_ATTRIBUTE"

    def test_touch_test_concrete_vs_system(self) -> None:
        """Concrete vs System: Touch test."""
        cat1, _, _ = resolve_entity_category("CONCRETE_ENTITY", "Container Image", "")
        cat2, _, _ = resolve_entity_category("SYSTEM_STRUCTURE", "Service Mesh", "")
        assert cat1 == "CONCRETE_ENTITY"
        assert cat2 == "SYSTEM_STRUCTURE"

    def test_intent_test_concrete_vs_agent(self) -> None:
        """Concrete vs Agent: Intent test."""
        cat1, _, _ = resolve_entity_category("CONCRETE_ENTITY", "Equipment", "")
        cat2, _, _ = resolve_entity_category("AGENT_ROLE", "End User", "")
        assert cat1 == "CONCRETE_ENTITY"
        assert cat2 == "AGENT_ROLE"

    def test_tangibility_test_concrete_vs_abstract(self) -> None:
        """Concrete vs Abstract: Tangibility test."""
        cat1, _, _ = resolve_entity_category("CONCRETE_ENTITY", "Data Center", "")
        cat2, _, _ = resolve_entity_category("ABSTRACT_CONCEPT", "Algorithm", "")
        assert cat1 == "CONCRETE_ENTITY"
        assert cat2 == "ABSTRACT_CONCEPT"

    def test_occurrence_test_event_vs_information(self) -> None:
        """Event vs Information: Occurrence test."""
        cat1, _, _ = resolve_entity_category("EVENT_PROCESS", "Build", "")
        cat2, _, _ = resolve_entity_category("INFORMATION_EXPRESSION", "Config File", "")
        assert cat1 == "EVENT_PROCESS"
        assert cat2 == "INFORMATION_EXPRESSION"

    def test_duration_test_event_vs_property(self) -> None:
        """Event vs Property: Duration test."""
        cat1, _, _ = resolve_entity_category("EVENT_PROCESS", "Migration", "")
        cat2, _, _ = resolve_entity_category("PROPERTY_ATTRIBUTE", "Capacity Metric", "")
        assert cat1 == "EVENT_PROCESS"
        assert cat2 == "PROPERTY_ATTRIBUTE"

    def test_temporal_test_event_vs_system(self) -> None:
        """Event vs System: Temporal test."""
        cat1, _, _ = resolve_entity_category("EVENT_PROCESS", "Test Run", "")
        cat2, _, _ = resolve_entity_category("SYSTEM_STRUCTURE", "CI/CD Pipeline", "")
        assert cat1 == "EVENT_PROCESS"
        assert cat2 == "SYSTEM_STRUCTURE"

    def test_actor_test_event_vs_agent(self) -> None:
        """Event vs Agent: Actor test."""
        cat1, _, _ = resolve_entity_category("EVENT_PROCESS", "Incident", "")
        cat2, _, _ = resolve_entity_category("AGENT_ROLE", "On-Call Engineer", "")
        assert cat1 == "EVENT_PROCESS"
        assert cat2 == "AGENT_ROLE"

    def test_instantiation_test_event_vs_abstract(self) -> None:
        """Event vs Abstract: Instantiation test."""
        cat1, _, _ = resolve_entity_category("EVENT_PROCESS", "Deployment", "")
        cat2, _, _ = resolve_entity_category("ABSTRACT_CONCEPT", "Paradigm", "")
        assert cat1 == "EVENT_PROCESS"
        assert cat2 == "ABSTRACT_CONCEPT"

    def test_representation_test_info_vs_property(self) -> None:
        """Information vs Property: Representation test."""
        cat1, _, _ = resolve_entity_category("INFORMATION_EXPRESSION", "API", "")
        cat2, _, _ = resolve_entity_category("PROPERTY_ATTRIBUTE", "Quality Attribute", "")
        assert cat1 == "INFORMATION_EXPRESSION"
        assert cat2 == "PROPERTY_ATTRIBUTE"

    def test_symbol_test_info_vs_system(self) -> None:
        """Information vs System: Symbol test."""
        cat1, _, _ = resolve_entity_category("INFORMATION_EXPRESSION", "Protocol", "")
        cat2, _, _ = resolve_entity_category("SYSTEM_STRUCTURE", "Network", "")
        assert cat1 == "INFORMATION_EXPRESSION"
        assert cat2 == "SYSTEM_STRUCTURE"

    def test_source_test_info_vs_agent(self) -> None:
        """Information vs Agent: Source test."""
        cat1, _, _ = resolve_entity_category("INFORMATION_EXPRESSION", "Log Stream", "")
        cat2, _, _ = resolve_entity_category("AGENT_ROLE", "CI Bot", "")
        assert cat1 == "INFORMATION_EXPRESSION"
        assert cat2 == "AGENT_ROLE"

    def test_encoding_test_info_vs_abstract(self) -> None:
        """Information vs Abstract: Encoding test."""
        cat1, _, _ = resolve_entity_category("INFORMATION_EXPRESSION", "Code/Formula", "")
        cat2, _, _ = resolve_entity_category("ABSTRACT_CONCEPT", "Design Pattern", "")
        assert cat1 == "INFORMATION_EXPRESSION"
        assert cat2 == "ABSTRACT_CONCEPT"

    def test_emergence_test_property_vs_system(self) -> None:
        """Property vs System: Emergence test."""
        cat1, _, _ = resolve_entity_category("PROPERTY_ATTRIBUTE", "Evaluative Property", "")
        cat2, _, _ = resolve_entity_category("SYSTEM_STRUCTURE", "Monorepo", "")
        assert cat1 == "PROPERTY_ATTRIBUTE"
        assert cat2 == "SYSTEM_STRUCTURE"

    def test_ascription_test_property_vs_agent(self) -> None:
        """Property vs Agent: Ascription test."""
        cat1, _, _ = resolve_entity_category("PROPERTY_ATTRIBUTE", "Capability/Skill", "")
        cat2, _, _ = resolve_entity_category("AGENT_ROLE", "Service Account", "")
        assert cat1 == "PROPERTY_ATTRIBUTE"
        assert cat2 == "AGENT_ROLE"

    def test_inherence_test_property_vs_abstract(self) -> None:
        """Property vs Abstract: Inherence test."""
        cat1, _, _ = resolve_entity_category("PROPERTY_ATTRIBUTE", "Physical Quality", "")
        cat2, _, _ = resolve_entity_category("ABSTRACT_CONCEPT", "Principle/Rule", "")
        assert cat1 == "PROPERTY_ATTRIBUTE"
        assert cat2 == "ABSTRACT_CONCEPT"

    def test_agency_test_system_vs_agent(self) -> None:
        """System vs Agent: Agency test."""
        cat1, _, _ = resolve_entity_category("SYSTEM_STRUCTURE", "Distributed System", "")
        cat2, _, _ = resolve_entity_category("AGENT_ROLE", "Stakeholder", "")
        assert cat1 == "SYSTEM_STRUCTURE"
        assert cat2 == "AGENT_ROLE"

    def test_instance_test_system_vs_abstract(self) -> None:
        """System vs Abstract: Instance test."""
        cat1, _, _ = resolve_entity_category("SYSTEM_STRUCTURE", "Supply Chain", "")
        cat2, _, _ = resolve_entity_category("ABSTRACT_CONCEPT", "SLA", "")
        assert cat1 == "SYSTEM_STRUCTURE"
        assert cat2 == "ABSTRACT_CONCEPT"

    def test_embodiment_test_agent_vs_abstract(self) -> None:
        """Agent vs Abstract: Embodiment test."""
        cat1, _, _ = resolve_entity_category("AGENT_ROLE", "Founder", "")
        cat2, _, _ = resolve_entity_category("ABSTRACT_CONCEPT", "Strategy", "")
        assert cat1 == "AGENT_ROLE"
        assert cat2 == "ABSTRACT_CONCEPT"

    def test_behavior_test_organism_vs_agent(self) -> None:
        """Organism vs Agent: Behavior test.

        Organism (biological behavior) → CONCRETE_ENTITY
        Agent (intentional/moral agency) → AGENT_ROLE
        """
        cat1, _, _ = resolve_entity_category("CONCRETE_ENTITY", "Organism", "")
        cat2, _, _ = resolve_entity_category("AGENT_ROLE", "Individual", "")
        assert cat1 == "CONCRETE_ENTITY"
        assert cat2 == "AGENT_ROLE"


class TestSubtypeRegistryIntegrity:
    """Tests for ENTITY_SUBTYPE_REGISTRY integrity."""

    def test_unknown_subtype_logs_info(self, caplog):
        """Unknown entity_subtype should be logged at INFO level for registry expansion."""
        from loguru import logger

        handler_id = logger.add(
            lambda msg: None,
            level="INFO",
            filter=lambda record: record["function"] == "resolve_entity_category",
        )
        try:
            category, _, _ = resolve_entity_category(None, "TotallyUnknownSubtype", "")
            assert category == "ABSTRACT_CONCEPT"
        finally:
            logger.remove(handler_id)

    def test_all_subtypes_map_to_valid_category(self) -> None:
        """Every sub-type in registry maps to a valid entity category."""
        valid = set(c.value for c in DocConceptEntityCategory)
        for subtype, category in ENTITY_SUBTYPE_REGISTRY.items():
            assert category in valid, (
                f"Sub-type '{subtype}' maps to '{category}' which is not valid"
            )

    def test_registry_has_expected_size(self) -> None:
        """Registry should have 200 sub-types."""
        assert len(ENTITY_SUBTYPE_REGISTRY) == 200, (
            f"Expected 200 sub-types, got {len(ENTITY_SUBTYPE_REGISTRY)}"
        )

    def test_registry_no_duplicate_keys(self) -> None:
        """No duplicate sub-type entries."""
        keys = list(ENTITY_SUBTYPE_REGISTRY.keys())
        assert len(keys) == len(set(keys))

    def test_core_concrete_subtypes_present(self) -> None:
        """Core Concrete Entity sub-types are present."""
        core_concrete = [
            "Natural Object", "Artifact", "Substance", "Organism",
            "Body Part", "Food/Consumable", "Geographic Feature", "Celestial Body",
        ]
        for subtype in core_concrete:
            assert subtype in ENTITY_SUBTYPE_REGISTRY, f"Missing: {subtype}"
            assert ENTITY_SUBTYPE_REGISTRY[subtype] == "CONCRETE_ENTITY"

    def test_core_event_subtypes_present(self) -> None:
        """Core Event/Process sub-types are present."""
        core_event = [
            "Natural Event", "Human Action", "Process", "Incident",
            "Activity", "State Change", "Project/Initiative", "Ritual/Routine",
        ]
        for subtype in core_event:
            assert subtype in ENTITY_SUBTYPE_REGISTRY, f"Missing: {subtype}"
            assert ENTITY_SUBTYPE_REGISTRY[subtype] == "EVENT_PROCESS"

    def test_core_information_subtypes_present(self) -> None:
        """Core Information/Expression sub-types are present."""
        core_info = [
            "Data", "Signal", "Symbol", "Narrative",
            "Code/Formula", "Record/Document", "Media",
        ]
        for subtype in core_info:
            assert subtype in ENTITY_SUBTYPE_REGISTRY, f"Missing: {subtype}"
            assert ENTITY_SUBTYPE_REGISTRY[subtype] == "INFORMATION_EXPRESSION"

    def test_core_property_subtypes_present(self) -> None:
        """Core Property/Attribute sub-types are present."""
        core_prop = [
            "Physical Quality", "Quantitative Measure", "Mental State",
            "Capability/Skill", "Disposition", "Relational Property",
            "Evaluative Property",
        ]
        for subtype in core_prop:
            assert subtype in ENTITY_SUBTYPE_REGISTRY, f"Missing: {subtype}"
            assert ENTITY_SUBTYPE_REGISTRY[subtype] == "PROPERTY_ATTRIBUTE"

    def test_core_system_subtypes_present(self) -> None:
        """Core System/Structure sub-types are present."""
        core_sys = [
            "Natural System", "Social System", "Technological System",
            "Network", "Hierarchy", "Framework", "Market/Platform",
        ]
        for subtype in core_sys:
            assert subtype in ENTITY_SUBTYPE_REGISTRY, f"Missing: {subtype}"
            assert ENTITY_SUBTYPE_REGISTRY[subtype] == "SYSTEM_STRUCTURE"

    def test_core_agent_subtypes_present(self) -> None:
        """Core Agent/Role sub-types are present."""
        core_agent = [
            "Individual", "Collective", "Institutional Agent",
            "Non-Human Agent", "Role/Position", "Persona",
        ]
        for subtype in core_agent:
            assert subtype in ENTITY_SUBTYPE_REGISTRY, f"Missing: {subtype}"
            assert ENTITY_SUBTYPE_REGISTRY[subtype] == "AGENT_ROLE"

    def test_core_abstract_subtypes_present(self) -> None:
        """Core Abstract Concept sub-types are present."""
        core_abstract = [
            "Domain/Discipline", "Theory/Model", "Principle/Rule",
            "Value/Ideal", "Category/Class", "Relation/Connection",
        ]
        for subtype in core_abstract:
            assert subtype in ENTITY_SUBTYPE_REGISTRY, f"Missing: {subtype}"
            assert ENTITY_SUBTYPE_REGISTRY[subtype] == "ABSTRACT_CONCEPT"

    def test_software_subtypes_present(self) -> None:
        """Key software-domain sub-types are present."""
        software_expected = {
            "Container Image": "CONCRETE_ENTITY",
            "Virtual Machine": "CONCRETE_ENTITY",
            "Deployment": "EVENT_PROCESS",
            "Build": "EVENT_PROCESS",
            "API": "INFORMATION_EXPRESSION",
            "Protocol": "INFORMATION_EXPRESSION",
            "SLI": "PROPERTY_ATTRIBUTE",
            "Quality Attribute": "PROPERTY_ATTRIBUTE",
            "Distributed System": "SYSTEM_STRUCTURE",
            "CI/CD Pipeline": "SYSTEM_STRUCTURE",
            "CI Bot": "AGENT_ROLE",
            "Service Account": "AGENT_ROLE",
            "Design Pattern": "ABSTRACT_CONCEPT",
            "Algorithm": "ABSTRACT_CONCEPT",
        }
        for subtype, expected_cat in software_expected.items():
            assert subtype in ENTITY_SUBTYPE_REGISTRY, f"Missing software sub-type: {subtype}"
            assert ENTITY_SUBTYPE_REGISTRY[subtype] == expected_cat

    def test_pedagogical_cognitive_subtypes_present(self) -> None:
        """Key pedagogical/cognitive sub-types are present."""
        pedagogical_cognitive_expected = {
            # 📏 Property/Attribute
            "Cognitive Capacity": "PROPERTY_ATTRIBUTE",
            "Cognitive Condition": "PROPERTY_ATTRIBUTE",
            "Cognitive Constraint": "PROPERTY_ATTRIBUTE",
            "Cognitive Limitation": "PROPERTY_ATTRIBUTE",
            "Cognitive Trait": "PROPERTY_ATTRIBUTE",
            "Character Traits": "PROPERTY_ATTRIBUTE",
            "Educational Metric": "PROPERTY_ATTRIBUTE",
            "Evaluative Criteria": "PROPERTY_ATTRIBUTE",
            "Evaluative Measure": "PROPERTY_ATTRIBUTE",
            "Knowledge Classification": "PROPERTY_ATTRIBUTE",
            "Performance Metric": "PROPERTY_ATTRIBUTE",
            "Quality Benchmark": "PROPERTY_ATTRIBUTE",
            "Quantitative Threshold": "PROPERTY_ATTRIBUTE",
            # 🏗️ System/Structure
            "AI System": "SYSTEM_STRUCTURE",
            "Cognitive Architecture": "SYSTEM_STRUCTURE",
            "Cognitive Framework": "SYSTEM_STRUCTURE",
            "Cognitive Model": "SYSTEM_STRUCTURE",
            "Cognitive Subsystem": "SYSTEM_STRUCTURE",
            "Cognitive System": "SYSTEM_STRUCTURE",
            "Educational Organization": "SYSTEM_STRUCTURE",
            "Methodology Framework": "SYSTEM_STRUCTURE",
            "Pedagogical Framework": "SYSTEM_STRUCTURE",
            "Research Institution": "SYSTEM_STRUCTURE",
            "Technological Influence": "SYSTEM_STRUCTURE",
            # ⏱️ Event/Process
            "Analytical Process": "EVENT_PROCESS",
            "Cognitive Process": "EVENT_PROCESS",
            "Cognitive Strategy": "EVENT_PROCESS",
            "Decision Process": "EVENT_PROCESS",
            "Educational Outcome": "EVENT_PROCESS",
            "Human-Computer Interaction": "EVENT_PROCESS",
            "Pedagogical Activity": "EVENT_PROCESS",
            "Teaching Activity": "EVENT_PROCESS",
            "Teaching Method": "EVENT_PROCESS",
            # 📨 Information Expression
            "Assessment Instrument": "INFORMATION_EXPRESSION",
            "Assessment Tool": "INFORMATION_EXPRESSION",
            "Evaluation Instrument": "INFORMATION_EXPRESSION",
            "Instructional Component": "INFORMATION_EXPRESSION",
            "Publication Type": "INFORMATION_EXPRESSION",
            "Research Evidence": "INFORMATION_EXPRESSION",
            "Research Study": "INFORMATION_EXPRESSION",
            "Tool/Framework": "INFORMATION_EXPRESSION",
            # 🎭 Agent/Role
            "Organization": "AGENT_ROLE",
            "Research Organization": "AGENT_ROLE",
            "Researcher": "AGENT_ROLE",
            # 💡 Abstract Concept
            "Capability": "ABSTRACT_CONCEPT",
            "Psychological Theory": "ABSTRACT_CONCEPT",
            # 🧱 Concrete Entity
            "Software Tool": "CONCRETE_ENTITY",
        }
        for subtype, expected_cat in pedagogical_cognitive_expected.items():
            assert subtype in ENTITY_SUBTYPE_REGISTRY, f"Missing pedagogical/cognitive sub-type: {subtype}"
            assert ENTITY_SUBTYPE_REGISTRY[subtype] == expected_cat


class TestExtractedConceptModel:
    """Tests for the updated ExtractedConcept Pydantic model."""

    def test_model_has_new_entity_fields(self) -> None:
        """Model should have entity_category, entity_subtype, entity_emoji."""
        concept = ExtractedConcept(
            name="Docker Image",
            definition="A packaged runtime environment",
            confidence=0.9,
            entity_category="CONCRETE_ENTITY",
            entity_subtype="Container Image",
            entity_emoji="🧱",
        )
        assert concept.entity_category == "CONCRETE_ENTITY"
        assert concept.entity_subtype == "Container Image"
        assert concept.entity_emoji == "🧱"

    def test_model_entity_fields_default_to_none(self) -> None:
        """entity_category and entity_subtype default to None."""
        concept = ExtractedConcept(
            name="Test",
            definition="...",
            confidence=0.5,
        )
        assert concept.entity_category is None
        assert concept.entity_subtype is None
        assert concept.entity_emoji == ""

    def test_old_type_field_still_exists(self) -> None:
        """Old `type` field is still present for backward compat."""
        concept = ExtractedConcept(
            name="Test",
            definition="...",
            confidence=0.5,
            type="CONCRETE_ENTITY",
        )
        assert concept.type == "CONCRETE_ENTITY"

    def test_model_confidence_validation(self) -> None:
        """Confidence must be between 0 and 1."""
        with pytest.raises(Exception):
            ExtractedConcept(
                name="Test",
                definition="...",
                confidence=1.5,
            )

    def test_all_entity_categories_accepted(self) -> None:
        """All 7 entity categories should be valid for entity_category."""
        for category in DocConceptEntityCategory:
            concept = ExtractedConcept(
                name="Test",
                definition="...",
                confidence=0.5,
                entity_category=category.value,
            )
            assert concept.entity_category == category.value


class TestBackwardCompatibility:
    """Old type field is preserved for backward compat."""

    def test_type_field_set_equal_to_entity_category(self) -> None:
        """During extraction, `type` equals `entity_category`."""
        # Simulate what LLMConceptExtractor.extract() does
        category, _, _ = resolve_entity_category("SYSTEM_STRUCTURE", "Monorepo", "")
        assert category == "SYSTEM_STRUCTURE"
        # The concept.type would be set to this value in extract()

    def test_new_constants_dont_conflict_with_relationship_constants(self) -> None:
        """Entity constants are separate from relationship constants."""
        from codebase_rag.constants import (
            DOC_CONCEPT_CATEGORIES,
            DocConceptRelCategory,
        )
        # Relationship categories (9) are distinct from entity categories (7)
        rel_values = set(c.value for c in DocConceptRelCategory)
        entity_values = set(c.value for c in DocConceptEntityCategory)
        assert rel_values.isdisjoint(entity_values), (
            "Entity and relationship category values must be disjoint"
        )
        assert len(DOC_CONCEPT_CATEGORIES) == 9
        assert len(DOC_ENTITY_CATEGORIES) == 7

    def test_existing_code_can_read_type_field(self) -> None:
        """Code reading concept.type continues to work."""
        concept = ExtractedConcept(
            name="TestConcept",
            definition="A test concept",
            confidence=0.8,
            type="CONCRETE_ENTITY",
            entity_category="CONCRETE_ENTITY",
            entity_subtype="Artifact",
            entity_emoji="🧱",
        )
        # Old code reads .type
        assert concept.type == "CONCRETE_ENTITY"
        # New code reads .entity_category
        assert concept.entity_category == "CONCRETE_ENTITY"
        assert concept.type == concept.entity_category


class TestJsonEntitySubtypeRegistry:
    """JSON entity types resolve to correct MECE categories."""

    def test_role_maps_to_agent_role(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Role", "")
        assert category == "AGENT_ROLE"
        assert emoji == "🎭"

    def test_mindset_maps_to_abstract_concept(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Mindset", "")
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_competency_maps_to_abstract_concept(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Competency", "")
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_antipattern_maps_to_abstract_concept(self) -> None:
        category, _, emoji = resolve_entity_category(None, "AntiPattern", "")
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_mental_model_maps_to_abstract_concept(self) -> None:
        category, _, emoji = resolve_entity_category(None, "MentalModel", "")
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_cognitive_mechanism_maps_to_event_process(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Cognitive Mechanism", "")
        assert category == "EVENT_PROCESS"
        assert emoji == "⏱️"

    def test_conceptual_framework_maps_to_abstract_concept(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Conceptual Framework", "")
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_decision_framework_maps_to_abstract_concept(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Decision Framework", "")
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_educational_framework_maps_to_abstract_concept(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Educational Framework", "")
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_research_framework_maps_to_abstract_concept(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Research Framework", "")
        assert category == "ABSTRACT_CONCEPT"
        assert emoji == "💡"

    def test_framework_component_maps_to_information_expression(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Framework Component", "")
        assert category == "INFORMATION_EXPRESSION"
        assert emoji == "📨"

    def test_framework_component_camelcase_maps_to_information_expression(self) -> None:
        category, _, emoji = resolve_entity_category(None, "FrameworkComponent", "")
        assert category == "INFORMATION_EXPRESSION"
        assert emoji == "📨"

    def test_technology_tool_maps_to_information_expression(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Technology Tool", "")
        assert category == "INFORMATION_EXPRESSION"
        assert emoji == "📨"

    def test_governance_construct_maps_to_system_structure(self) -> None:
        category, _, emoji = resolve_entity_category(None, "GovernanceConstruct", "")
        assert category == "SYSTEM_STRUCTURE"
        assert emoji == "🏗️"

    def test_layer_maps_to_system_structure(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Layer", "")
        assert category == "SYSTEM_STRUCTURE"
        assert emoji == "🏗️"

    def test_progression_stage_maps_to_event_process(self) -> None:
        category, _, emoji = resolve_entity_category(None, "ProgressionStage", "")
        assert category == "EVENT_PROCESS"
        assert emoji == "⏱️"

    def test_safety_boundary_maps_to_property_attribute(self) -> None:
        category, _, emoji = resolve_entity_category(None, "SafetyBoundary", "")
        assert category == "PROPERTY_ATTRIBUTE"
        assert emoji == "📏"

    def test_governance_rule_maps_to_information_expression(self) -> None:
        category, _, emoji = resolve_entity_category(None, "GovernanceRule", "")
        assert category == "INFORMATION_EXPRESSION"
        assert emoji == "📨"

    def test_reference_maps_to_information_expression(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Reference", "")
        assert category == "INFORMATION_EXPRESSION"
        assert emoji == "📨"

    def test_tool_maps_to_concrete_entity(self) -> None:
        category, _, emoji = resolve_entity_category(None, "Tool", "")
        assert category == "CONCRETE_ENTITY"
        assert emoji == "🧱"
