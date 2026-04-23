"""Mocks for Memgraph database connections.

Provides mock implementations for testing database operations
without requiring a running Memgraph instance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock


class MockMemgraphConnection:
    """Mock Memgraph connection for testing.

    Tracks all queries executed and allows configuring responses.
    """

    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any] | None]] = []
        self.results: list[list[dict[str, Any]]] = []
        self._result_index = 0
        self.closed = False

    def execute(
        self,
        query: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a query and return pre-configured results.

        Args:
            query: Cypher query string
            params: Query parameters

        Returns:
            Pre-configured result or empty list
        """
        if self.closed:
            raise RuntimeError("Connection is closed")

        self.queries.append((query, params))

        if self._result_index < len(self.results):
            result = self.results[self._result_index]
            self._result_index += 1
            return result
        return []

    def add_result(self, result: list[dict[str, Any]]) -> None:
        """Add a result to be returned by subsequent execute() calls.

        Args:
            result: List of result dictionaries
        """
        self.results.append(result)

    def fetchall(self) -> list[dict[str, Any]]:
        """Fetch all results from the last query."""
        if self._result_index <= len(self.results):
            return self.results[self._result_index - 1] if self.results else []
        return []

    def close(self) -> None:
        """Close the connection."""
        self.closed = True

    def reset(self) -> None:
        """Reset the mock state."""
        self.queries.clear()
        self.results.clear()
        self._result_index = 0
        self.closed = False


class MockMemgraphCursor:
    """Mock cursor for Memgraph connection."""

    def __init__(self, connection: MockMemgraphConnection) -> None:
        self._connection = connection
        self._last_result: list[dict[str, Any]] = []

    def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Execute a query."""
        self._last_result = self._connection.execute(query, params)

    def fetchall(self) -> list[dict[str, Any]]:
        """Fetch all results."""
        return self._last_result

    def fetchone(self) -> dict[str, Any] | None:
        """Fetch one result."""
        return self._last_result[0] if self._last_result else None


def create_mock_memgraph_fixture() -> MockMemgraphConnection:
    """Create a MockMemgraphConnection for use in tests.

    Returns:
        Configured MockMemgraphConnection instance
    """
    return MockMemgraphConnection()
