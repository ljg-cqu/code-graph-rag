"""Tests for Pydantic schemas, especially QueryGraphData validator."""

from codebase_rag.schemas import QueryGraphData, _normalize_value
from codebase_rag.types_defs import (
    NODE_SCHEMAS,
    RELATIONSHIP_SCHEMAS,
    NodeLabel,
    RelationshipType,
)


class TestNormalizeValue:
    """Tests for the _normalize_value helper function."""

    def test_returns_none_unchanged(self) -> None:
        assert _normalize_value(None) is None

    def test_returns_primitives_unchanged(self) -> None:
        assert _normalize_value("string") == "string"
        assert _normalize_value(42) == 42
        assert _normalize_value(3.14) == 3.14
        assert _normalize_value(True) is True
        assert _normalize_value(False) is False

    def test_normalizes_list_recursively(self) -> None:
        result = _normalize_value([1, "two", None])
        assert result == [1, "two", None]

    def test_normalizes_dict_recursively(self) -> None:
        result = _normalize_value({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_normalizes_nested_structures(self) -> None:
        """Test the main use case: list of dicts."""
        input_val = [
            {"name": "UniIOTX.sol", "type": "Module"},
            {"name": "UniIOTX", "type": "Contract"},
        ]
        result = _normalize_value(input_val)
        assert result == input_val

    def test_converts_unknown_types_to_string(self) -> None:
        class CustomObj:
            def __str__(self) -> str:
                return "custom"

        result = _normalize_value(CustomObj())
        assert result == "custom"

    def test_normalizes_deeply_nested_structures(self) -> None:
        """Test deeply nested structures are handled recursively."""
        input_val = {
            "outer": [
                {"inner": [{"deep": "value"}]},
            ]
        }
        result = _normalize_value(input_val)
        assert result == input_val


class TestQueryGraphData:
    """Tests for QueryGraphData model."""

    def test_accepts_simple_results(self) -> None:
        data = QueryGraphData(
            query_used="MATCH (n) RETURN n.name AS name",
            results=[{"name": "test"}],
            summary="Found 1 result",
        )
        assert data.results == [{"name": "test"}]

    def test_accepts_nested_list_of_dicts(self) -> None:
        """Test the main failing case from the bug report."""
        results = [
            {
                "filePath": "UniIOTX.sol",
                "fileName": "UniIOTX.sol",
                "elements": [
                    {"name": "UniIOTX.sol", "type": "Module"},
                    {"name": "UniIOTX", "type": "Contract"},
                    {"name": "burnFrom", "type": "Method"},
                ],
            }
        ]
        data = QueryGraphData(
            query_used="MATCH (f:File) RETURN f.path AS filePath",
            results=results,
            summary="Found 1 file",
        )
        assert data.results == results

    def test_accepts_empty_results(self) -> None:
        data = QueryGraphData(
            query_used="MATCH (n) RETURN n", results=[], summary="No results"
        )
        assert data.results == []

    def test_handles_null_values_in_results(self) -> None:
        data = QueryGraphData(
            query_used="MATCH (n) RETURN n.name AS name, n.desc AS desc",
            results=[{"name": "test", "desc": None}],
            summary="Found 1 result",
        )
        assert data.results == [{"name": "test", "desc": None}]

    def test_rejects_non_dict_rows(self) -> None:
        """Non-dict rows should raise ValidationError."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="each result row must be a dict"):
            QueryGraphData(
                query_used="MATCH (n) RETURN n",
                results=[{"valid": "row"}, "invalid", 123],  # type: ignore
                summary="Processed",
            )

    def test_converts_non_standard_types_to_string(self) -> None:
        """Non-standard types should be converted to strings."""
        data = QueryGraphData(
            query_used="MATCH (n) RETURN n",
            results=[{"value": b"bytes"}],  # bytes should become string
            summary="Found 1",
        )
        assert data.results == [{"value": "b'bytes'"}]


class TestRelationshipSchemas:
    """Tests for relationship schema consistency."""

    def test_document_schema_matches_split_graph_metadata(self) -> None:
        document_schema = None
        for schema in NODE_SCHEMAS:
            if schema.label == NodeLabel.DOCUMENT:
                document_schema = schema
                break

        assert document_schema is not None
        assert "path: string" in document_schema.properties
        assert "workspace: string" in document_schema.properties
        assert "resolved_code_references: list[string]" in document_schema.properties
        assert "resolved_code_reference_count: int" in document_schema.properties
        assert "qualified_name" not in document_schema.properties

    def test_chunk_schema_includes_resolved_reference_metadata(self) -> None:
        chunk_schema = None
        for schema in NODE_SCHEMAS:
            if schema.label == NodeLabel.CHUNK:
                chunk_schema = schema
                break

        assert chunk_schema is not None
        assert "token_count: int" in chunk_schema.properties
        assert "section_title: string" in chunk_schema.properties
        assert "resolved_code_references: list[string]" in chunk_schema.properties
        assert "embedding: list[float]" in chunk_schema.properties

    def test_defines_includes_all_module_definitions(self) -> None:
        """DEFINES relationship should include all node types a Module can define."""
        defines_schema = None
        for schema in RELATIONSHIP_SCHEMAS:
            if schema.rel_type == RelationshipType.DEFINES:
                defines_schema = schema
                break

        assert defines_schema is not None
        expected_targets = {
            NodeLabel.CLASS,
            NodeLabel.FUNCTION,
            NodeLabel.INTERFACE,
            NodeLabel.ENUM,
            NodeLabel.TYPE,
            NodeLabel.UNION,
            NodeLabel.CONTRACT,
            NodeLabel.LIBRARY,
        }
        actual_targets = set(defines_schema.targets)
        assert actual_targets == expected_targets, (
            f"DEFINES targets mismatch. Expected: {expected_targets}, Got: {actual_targets}"
        )

    def test_defines_method_includes_all_method_owners(self) -> None:
        """DEFINES_METHOD should include every node type that can own methods."""
        defines_method_schema = None
        for schema in RELATIONSHIP_SCHEMAS:
            if schema.rel_type == RelationshipType.DEFINES_METHOD:
                defines_method_schema = schema
                break

        assert defines_method_schema is not None
        expected_sources = {
            NodeLabel.CLASS,
            NodeLabel.CONTRACT,
            NodeLabel.INTERFACE,
            NodeLabel.LIBRARY,
        }
        actual_sources = set(defines_method_schema.sources)
        assert actual_sources == expected_sources, (
            f"DEFINES_METHOD sources mismatch. Expected: {expected_sources}, Got: {actual_sources}"
        )

    def test_inherits_includes_class_and_contract(self) -> None:
        """INHERITS relationship should include both Class and Contract."""
        inherits_schema = None
        for schema in RELATIONSHIP_SCHEMAS:
            if schema.rel_type == RelationshipType.INHERITS:
                inherits_schema = schema
                break

        assert inherits_schema is not None
        expected = {NodeLabel.CLASS, NodeLabel.CONTRACT}
        actual_sources = set(inherits_schema.sources)
        actual_targets = set(inherits_schema.targets)
        assert actual_sources == expected, (
            f"INHERITS sources mismatch. Expected: {expected}, Got: {actual_sources}"
        )
        assert actual_targets == expected, (
            f"INHERITS targets mismatch. Expected: {expected}, Got: {actual_targets}"
        )

    def test_belongs_to_section_points_from_chunk_to_section(self) -> None:
        belongs_to_section_schema = None
        for schema in RELATIONSHIP_SCHEMAS:
            if schema.rel_type == RelationshipType.BELONGS_TO_SECTION:
                belongs_to_section_schema = schema
                break

        assert belongs_to_section_schema is not None
        assert set(belongs_to_section_schema.sources) == {NodeLabel.CHUNK}
        assert set(belongs_to_section_schema.targets) == {NodeLabel.SECTION}

    def test_all_node_labels_have_unique_keys(self) -> None:
        """Every NodeLabel MUST have a corresponding entry in _NODE_LABEL_UNIQUE_KEYS."""
        from codebase_rag.constants import _NODE_LABEL_UNIQUE_KEYS
        missing = set(NodeLabel) - set(_NODE_LABEL_UNIQUE_KEYS.keys())
        assert not missing, f"NodeLabel(s) missing unique keys: {missing}"

    def test_all_relationship_types_have_schemas(self) -> None:
        """Every RelationshipType used in ingestion MUST have a schema entry."""
        schema_rel_types = {schema.rel_type for schema in RELATIONSHIP_SCHEMAS}
        missing = set(RelationshipType) - schema_rel_types
        assert not missing, f"RelationshipType(s) missing schemas: {missing}"
