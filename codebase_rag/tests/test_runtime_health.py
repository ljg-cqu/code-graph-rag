"""Tests for runtime health monitoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from codebase_rag.tools.health_checker import (
    HealthStatus,
    RuntimeHealthStatus,
    get_runtime_status,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_all_statuses_exist(self) -> None:
        expected = {"HEALTHY", "DEGRADED", "UNHEALTHY"}
        actual = {s.name for s in HealthStatus}
        assert actual == expected


class TestRuntimeHealthStatus:
    """Test RuntimeHealthStatus dataclass."""

    def test_defaults_to_healthy(self) -> None:
        status = RuntimeHealthStatus()
        assert status.vector_search == HealthStatus.HEALTHY
        assert status.graph_traversal == HealthStatus.HEALTHY
        assert status.procedures == HealthStatus.HEALTHY
        assert status.last_error is None

    def test_overall_healthy(self) -> None:
        status = RuntimeHealthStatus()
        assert status.overall == HealthStatus.HEALTHY

    def test_overall_degraded(self) -> None:
        status = RuntimeHealthStatus(
            procedures=HealthStatus.DEGRADED,
        )
        assert status.overall == HealthStatus.DEGRADED

    def test_overall_unhealthy(self) -> None:
        status = RuntimeHealthStatus(
            vector_search=HealthStatus.UNHEALTHY,
        )
        assert status.overall == HealthStatus.UNHEALTHY

    def test_unhealthy_takes_precedence(self) -> None:
        status = RuntimeHealthStatus(
            vector_search=HealthStatus.UNHEALTHY,
            graph_traversal=HealthStatus.DEGRADED,
            procedures=HealthStatus.HEALTHY,
        )
        assert status.overall == HealthStatus.UNHEALTHY


class TestGetRuntimeStatus:
    """Test get_runtime_status function."""

    def test_returns_status_object(self) -> None:
        status = get_runtime_status()
        assert isinstance(status, RuntimeHealthStatus)

    @patch("codebase_rag.tools.health_checker.mgclient")
    def test_handles_graph_unavailable(self, mock_mgclient: MagicMock) -> None:
        mock_mgclient.connect.side_effect = Exception("Connection refused")

        status = get_runtime_status()
        assert status.graph_traversal == HealthStatus.UNHEALTHY
        assert status.last_error is not None

    @patch("codebase_rag.tools.health_checker.mgclient")
    def test_handles_procedure_unavailable(self, mock_mgclient: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_mgclient.connect.return_value = mock_conn

        def execute_side_effect(query: str) -> None:
            if "pagerank" in query:
                raise Exception("Procedure not found")

        mock_cursor.execute.side_effect = execute_side_effect

        status = get_runtime_status()
        assert status.procedures == HealthStatus.DEGRADED
