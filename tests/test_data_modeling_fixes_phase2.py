"""Tests for Phase 2 data modeling fixes.

See .specs/data_modeling_fixes_phase2_spec.md for details.
"""
import pytest
from unittest.mock import MagicMock, patch

from codebase_rag.services.graph_service import MemgraphIngestor
from codebase_rag.tools.health_checker import HealthChecker
from codebase_rag.exceptions import QueryExecutionError


class TestEnsureRelationshipBatch:
    """Tests for ensure_relationship_batch behavior."""

    def test_ensure_relationship_batch_does_not_flush_relationships(self):
        """Verify ensure_relationship_batch does not auto-flush relationships.

        When the relationship buffer reaches batch_size, only nodes should be
        flushed, not relationships. Relationships must be deferred until all
        target nodes exist.
        """
        # Create ingestor without actual connection (we're testing buffer behavior)
        ingestor = MemgraphIngestor(
            host="localhost",
            port=7687,
            batch_size=10,
        )

        # Add nodes first
        for i in range(5):
            ingestor.ensure_node("Function", {"qualified_name": f"func_{i}"})

        # Add relationships to trigger batch_size (10)
        for i in range(10):
            ingestor.ensure_relationship_batch(
                ("Function", "qualified_name", "func_0"),
                "CALLS",
                ("Function", "qualified_name", "func_1"),
            )

        # Verify relationships are still in buffer (not flushed)
        assert ingestor._rel_count == 10
        # Nodes may have been flushed (buffer reached batch_size)
        assert len(ingestor.node_buffer) == 0  # Nodes were flushed

    def test_relationship_buffer_warning_threshold(self):
        """Verify memory warning is logged when buffer grows large."""
        # Patch the class attribute to lower threshold for testing
        with patch.object(
            MemgraphIngestor, "_REL_BUFFER_WARNING_THRESHOLD", 100
        ):
            ingestor = MemgraphIngestor(
                host="localhost",
                port=7687,
                batch_size=200,
            )

            # Add nodes
            for i in range(10):
                ingestor.ensure_node("Function", {"qualified_name": f"func_{i}"})

            # Add relationships up to warning threshold
            with patch("codebase_rag.services.graph_service.logger.debug") as mock_debug:
                for i in range(100):
                    ingestor.ensure_relationship_batch(
                        ("Function", "qualified_name", "func_0"),
                        "CALLS",
                        ("Function", "qualified_name", "func_1"),
                    )

                # Check that memory warning was logged
                warning_calls = [
                    call for call in mock_debug.call_args_list
                    if "Relationship buffer size" in str(call)
                ]
                assert len(warning_calls) >= 1


class TestFetchSingleInt:
    """Tests for _fetch_single_int error handling."""

    def test_fetch_single_int_handles_column_exception(self):
        """Verify _fetch_single_int handles mgclient.Column exceptions.

        When direct indexing row[0] fails, the method should try alternative
        extraction via list(row).
        """
        mock_cursor = MagicMock()

        # Create a row-like object that fails on indexing but works with iteration
        class FailingRow:
            def __getitem__(self, index):
                raise RuntimeError(
                    "<class 'mgclient.Column'> returned a result with an exception set"
                )

            def __iter__(self):
                return iter([42])

        mock_cursor.fetchone.return_value = FailingRow()

        # Mock _consume_all_results to prevent infinite loop
        with patch.object(
            HealthChecker, "_consume_all_results", return_value=None
        ):
            result = HealthChecker._fetch_single_int(mock_cursor, "MATCH (n) RETURN 1")
            assert result == 42

    def test_fetch_single_int_returns_zero_for_none_row(self):
        """Verify _fetch_single_int returns 0 when row is None."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        with patch.object(
            HealthChecker, "_consume_all_results", return_value=None
        ):
            result = HealthChecker._fetch_single_int(mock_cursor, "MATCH (n) RETURN count(n)")
            assert result == 0

    def test_fetch_single_int_handles_normal_row(self):
        """Verify _fetch_single_int works with normal row access."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (123,)

        with patch.object(
            HealthChecker, "_consume_all_results", return_value=None
        ):
            result = HealthChecker._fetch_single_int(mock_cursor, "MATCH (n) RETURN count(n)")
            assert result == 123


class TestSafeGetColumn:
    """Tests for _safe_get_column helper method."""

    def test_safe_get_column_normal_access(self):
        """Verify normal row access works."""
        row = (1, 2, 3)
        assert HealthChecker._safe_get_column(row, 0) == 1
        assert HealthChecker._safe_get_column(row, 1) == 2
        assert HealthChecker._safe_get_column(row, 2) == 3

    def test_safe_get_column_none_row(self):
        """Verify None row returns None."""
        assert HealthChecker._safe_get_column(None, 0) is None

    def test_safe_get_column_with_exception_fallback(self):
        """Verify fallback to iteration when indexing fails."""
        # Create a row-like object that fails on indexing but works with iteration
        class FailingRow:
            def __getitem__(self, index):
                raise RuntimeError("Column exception")

            def __iter__(self):
                return iter([10, 20, 30])

        row = FailingRow()
        result = HealthChecker._safe_get_column(row, 0)
        assert result == 10

    def test_safe_get_column_out_of_bounds(self):
        """Verify out of bounds returns None."""
        # With a real tuple, index 5 is out of bounds
        row = (1, 2)
        # When index is out of bounds, it raises IndexError which is caught
        result = HealthChecker._safe_get_column(row, 5)
        assert result is None
