"""Tests for CrossGraphReferenceResolver."""

from __future__ import annotations

from unittest.mock import MagicMock

from codebase_rag.shared.reference_resolver import (
    CrossGraphReferenceResolver,
    ResolvedReference,
    UnresolvedReference,
)


class TestCrossGraphReferenceResolver:
    """Test cross-graph reference resolution."""

    def test_handles_unavailable_code_graph(self) -> None:
        resolver = CrossGraphReferenceResolver(code_graph=None)
        result = resolver.resolve_references(["auth.login"])

        assert len(result.unresolved) == 1
        assert result.unresolved[0].reason == "graph_unavailable"
        assert len(result.warnings) > 0

    def test_handles_empty_references(self) -> None:
        mock_graph = MagicMock()
        resolver = CrossGraphReferenceResolver(code_graph=mock_graph)
        result = resolver.resolve_references([])

        assert len(result.resolved) == 0
        assert len(result.unresolved) == 0

    def test_resolves_exact_match(self) -> None:
        mock_graph = MagicMock()
        mock_graph.fetch_all.return_value = [
            {
                "qualified_name": "auth.login",
                "node_type": "Function",
                "file_path": "auth.py",
                "start_line": 10,
                "end_line": 20,
            }
        ]

        resolver = CrossGraphReferenceResolver(code_graph=mock_graph)
        result = resolver.resolve_references(["auth.login"])

        assert len(result.resolved) == 1
        assert result.resolved[0].qualified_name == "auth.login"
        assert result.resolved[0].file_path == "auth.py"
        assert result.resolved[0].line_range == (10, 20)

    def test_handles_not_found_with_partial_match(self) -> None:
        mock_graph = MagicMock()
        mock_graph.fetch_all.side_effect = [
            [],
            [
                {
                    "qualified_name": "auth.login",
                    "node_type": "Function",
                    "path": "auth.py",
                    "start_line": 10,
                    "end_line": 20,
                }
            ],
        ]

        resolver = CrossGraphReferenceResolver(code_graph=mock_graph)
        result = resolver.resolve_references(["Project.auth.login"])

        assert len(result.resolved) == 1

    def test_handles_not_found_without_partial_match(self) -> None:
        mock_graph = MagicMock()
        mock_graph.fetch_all.side_effect = [[], []]

        resolver = CrossGraphReferenceResolver(code_graph=mock_graph)
        result = resolver.resolve_references(["nonexistent.func"])

        assert len(result.unresolved) == 1
        assert result.unresolved[0].reason == "not_found"

    def test_handles_graph_error(self) -> None:
        mock_graph = MagicMock()
        mock_graph.fetch_all.side_effect = Exception("Connection failed")

        resolver = CrossGraphReferenceResolver(code_graph=mock_graph)
        result = resolver.resolve_references(["auth.login"])

        assert len(result.unresolved) == 1
        assert result.unresolved[0].reason == "resolution_error"
        assert len(result.warnings) > 0

    def test_limits_references(self) -> None:
        mock_graph = MagicMock()
        mock_graph.fetch_all.return_value = []

        resolver = CrossGraphReferenceResolver(code_graph=mock_graph)
        resolver.resolve_references(
            [f"func{i}" for i in range(50)],
            max_references=10,
        )

        call_args = mock_graph.fetch_all.call_args
        assert len(call_args[0][1]["qualified_names"]) == 10


class TestResolvedReference:
    """Test ResolvedReference dataclass."""

    def test_creates_reference(self) -> None:
        ref = ResolvedReference(
            qualified_name="auth.login",
            node_type="Function",
            file_path="auth.py",
            line_range=(10, 20),
        )

        assert ref.qualified_name == "auth.login"
        assert ref.node_type == "Function"
        assert ref.found is True

    def test_creates_without_line_range(self) -> None:
        ref = ResolvedReference(
            qualified_name="auth.login",
            node_type="Function",
            file_path="auth.py",
            line_range=None,
        )

        assert ref.line_range is None


class TestUnresolvedReference:
    """Test UnresolvedReference dataclass."""

    def test_creates_reference(self) -> None:
        ref = UnresolvedReference(
            qualified_name="auth.login",
            reason="not_found",
        )

        assert ref.qualified_name == "auth.login"
        assert ref.reason == "not_found"
