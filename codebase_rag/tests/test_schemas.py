"""Tests for Pydantic schemas, especially QueryGraphData validator."""

import pytest

from codebase_rag.schemas import QueryGraphData, _normalize_value
from codebase_rag.types_defs import RELATIONSHIP_SCHEMAS, NodeLabel, RelationshipType


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
            summary="Found 1 result"
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
                ]
            }
        ]
        data = QueryGraphData(
            query_used="MATCH (f:File) RETURN f.path AS filePath",
            results=results,
            summary="Found 1 file"
        )
        assert data.results == results

    def test_accepts_empty_results(self) -> None:
        data = QueryGraphData(
            query_used="MATCH (n) RETURN n",
            results=[],
            summary="No results"
        )
        assert data.results == []

    def test_handles_null_values_in_results(self) -> None:
        data = QueryGraphData(
            query_used="MATCH (n) RETURN n.name AS name, n.desc AS desc",
            results=[{"name": "test", "desc": None}],
            summary="Found 1 result"
        )
        assert data.results == [{"name": "test", "desc": None}]

    def test_skips_non_dict_rows(self) -> None:
        """Non-dict rows should be skipped."""
        data = QueryGraphData(
            query_used="MATCH (n) RETURN n",
            results=[{"valid": "row"}, "invalid", 123],  # type: ignore
            summary="Processed"
        )
        assert data.results == [{"valid": "row"}]

    def test_converts_non_standard_types_to_string(self) -> None:
        """Non-standard types should be converted to strings."""
        data = QueryGraphData(
            query_used="MATCH (n) RETURN n",
            results=[{"value": b"bytes"}],  # bytes should become string
            summary="Found 1"
        )
        assert data.results == [{"value": "b'bytes'"}]


class TestRelationshipSchemas:
    """Tests for relationship schema consistency."""

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

    def test_defines_method_includes_class_and_contract(self) -> None:
        """DEFINES_METHOD relationship should include both Class and Contract as sources."""
        defines_method_schema = None
        for schema in RELATIONSHIP_SCHEMAS:
            if schema.rel_type == RelationshipType.DEFINES_METHOD:
                defines_method_schema = schema
                break

        assert defines_method_schema is not None
        expected_sources = {NodeLabel.CLASS, NodeLabel.CONTRACT}
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