"""Schema consistency tests to prevent drift between enums and definitions."""

import pytest

from codebase_rag.constants import (
    NodeLabel,
    RelationshipType,
    _NODE_LABEL_UNIQUE_KEYS,
    NODE_UNIQUE_CONSTRAINTS,
)
from codebase_rag.types_defs import (
    NODE_SCHEMAS,
    RELATIONSHIP_SCHEMAS,
    NodeSchema,
    RelationshipSchema,
)


class TestNodeSchemaConsistency:
    """Verify all NodeLabel values have corresponding schemas and constraints."""

    def test_all_node_labels_have_schemas(self):
        """Every NodeLabel enum value must have a NodeSchema entry."""
        schema_labels = {schema.label for schema in NODE_SCHEMAS}
        missing = set(NodeLabel) - schema_labels

        assert not missing, (
            f"Missing NodeSchema for labels: {missing}\n"
            f"Add NodeSchema entries in types_defs.py NODE_SCHEMAS tuple."
        )

    def test_all_node_labels_have_unique_keys(self):
        """Every NodeLabel enum value must have a unique key defined."""
        missing = set(NodeLabel) - set(_NODE_LABEL_UNIQUE_KEYS.keys())

        assert not missing, (
            f"Missing unique keys for labels: {missing}\n"
            f"Add entries to _NODE_LABEL_UNIQUE_KEYS in constants.py."
        )

    def test_unique_keys_use_valid_types(self):
        """All unique keys must be valid UniqueKeyType values."""
        from codebase_rag.constants import UniqueKeyType

        valid_types = set(UniqueKeyType)
        for label, key_type in _NODE_LABEL_UNIQUE_KEYS.items():
            assert key_type in valid_types, (
                f"Invalid unique key type '{key_type}' for {label}. "
                f"Must be one of: {valid_types}"
            )

    def test_node_schema_properties_are_strings(self):
        """NodeSchema property definitions must be valid Cypher-like strings."""
        for schema in NODE_SCHEMAS:
            assert isinstance(schema, NodeSchema), (
                f"Invalid schema type: {type(schema)}"
            )
            assert isinstance(schema.properties, str), (
                f"NodeSchema for {schema.label} has non-string properties: "
                f"{type(schema.properties)}"
            )
            assert schema.properties.startswith("{"), (
                f"NodeSchema for {schema.label} properties should start with '{{'"
            )
            assert schema.properties.endswith("}"), (
                f"NodeSchema for {schema.label} properties should end with '}}'"
            )

    def test_unique_constraint_dict_matches_unique_keys(self):
        """NODE_UNIQUE_CONSTRAINTS should match _NODE_LABEL_UNIQUE_KEYS."""
        expected_constraints = {
            label.value: key.value
            for label, key in _NODE_LABEL_UNIQUE_KEYS.items()
        }

        assert NODE_UNIQUE_CONSTRAINTS == expected_constraints, (
            f"NODE_UNIQUE_CONSTRAINTS mismatch:\n"
            f"Expected: {expected_constraints}\n"
            f"Got: {NODE_UNIQUE_CONSTRAINTS}"
        )


class TestRelationshipSchemaConsistency:
    """Verify relationship type coverage and schema validity."""

    def test_all_relationship_types_have_schemas(self):
        """All RelationshipType values must have RelationshipSchema entries."""
        schema_types = {schema.rel_type for schema in RELATIONSHIP_SCHEMAS}
        missing = set(RelationshipType) - schema_types

        assert not missing, (
            f"Missing RelationshipSchema for types: {[rt.value for rt in missing]}\n"
            f"Add RelationshipSchema entries in types_defs.py RELATIONSHIP_SCHEMAS tuple."
        )

    def test_relationship_schema_sources_are_valid_labels(self):
        """Relationship source types must be valid NodeLabel values."""
        valid_labels = set(NodeLabel)

        for schema in RELATIONSHIP_SCHEMAS:
            for source in schema.sources:
                assert source in valid_labels, (
                    f"Invalid source label '{source}' in {schema.rel_type} schema. "
                    f"Must be one of: {[l.value for l in valid_labels]}"
                )

    def test_relationship_schema_targets_are_valid_labels(self):
        """Relationship target types must be valid NodeLabel values."""
        valid_labels = set(NodeLabel)

        for schema in RELATIONSHIP_SCHEMAS:
            for target in schema.targets:
                assert target in valid_labels, (
                    f"Invalid target label '{target}' in {schema.rel_type} schema. "
                    f"Must be one of: {[l.value for l in valid_labels]}"
                )

    def test_no_duplicate_relationship_schemas(self):
        """No duplicate relationship schema definitions."""
        seen = {}
        for i, schema in enumerate(RELATIONSHIP_SCHEMAS):
            key = (tuple(schema.sources), schema.rel_type, tuple(schema.targets))
            if key in seen:
                pytest.fail(
                    f"Duplicate RelationshipSchema for {schema.rel_type}: "
                    f"First at index {seen[key]}, duplicate at index {i}"
                )
            seen[key] = i


class TestSchemaPropertyConsistency:
    """Verify property consistency across node types."""

    def test_embeddable_nodes_have_nullable_embeddings(self):
        """All embeddable node schemas should have nullable embedding properties."""
        from codebase_rag.constants import EMBEDDABLE_CODE_NODE_LABELS

        for schema in NODE_SCHEMAS:
            if schema.label.value in EMBEDDABLE_CODE_NODE_LABELS:
                props = schema.properties
                # Check embedding is nullable (has | null)
                if "embedding" in props:
                    assert "| null" in props or "null |" in props, (
                        f"{schema.label} has embedding but may not be nullable. "
                        f"Properties: {props}"
                    )
