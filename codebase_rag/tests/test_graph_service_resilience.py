from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.services.graph_service import ConnectionRetryPolicy, MemgraphIngestor


class TestCreateConnectionWithRetry:
    def test_succeeds_on_third_attempt(self) -> None:
        ingestor = MemgraphIngestor(host="localhost", port=7687)
        mock_conn = MagicMock()

        with patch.object(
            MemgraphIngestor, "_create_connection_with_timeout"
        ) as mock_create:
            mock_create.side_effect = [ConnectionError("fail1"), ConnectionError("fail2"), mock_conn]
            with patch("time.sleep") as mock_sleep:
                result = ingestor._create_connection_with_retry(
                    policy=ConnectionRetryPolicy(max_attempts=3, base_delay_seconds=1.0)
                )

        assert result is mock_conn
        assert mock_create.call_count == 3
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    def test_raises_after_exhaustion(self) -> None:
        ingestor = MemgraphIngestor(host="localhost", port=7687)

        with patch.object(
            MemgraphIngestor, "_create_connection_with_timeout"
        ) as mock_create:
            mock_create.side_effect = ConnectionError("always fails")
            with patch("time.sleep"):
                with pytest.raises(ConnectionError, match="Cannot connect to Memgraph") as exc_info:
                    ingestor._create_connection_with_retry(
                        policy=ConnectionRetryPolicy(max_attempts=3)
                    )

        assert exc_info.value.__cause__ is not None
        assert "always fails" in str(exc_info.value.__cause__)
        assert mock_create.call_count == 3


class TestFlushNodesWithStats:
    def test_returns_partial_on_group_failure(self) -> None:
        ingestor = MemgraphIngestor(host="localhost", port=7687)
        ingestor.node_buffer = [
            ("LabelA", {"id": 1}),
            ("LabelA", {"id": 2}),
            ("LabelB", {"id": 3}),
        ]

        with patch.object(
            MemgraphIngestor, "_flush_node_label_group"
        ) as mock_flush:
            mock_flush.side_effect = [
                (2, 0),  # LabelA succeeds
                RuntimeError("LabelB fails"),  # LabelB fails
            ]

            stats = ingestor.flush_nodes_with_stats()

        assert stats["attempted"] == 3
        assert stats["flushed"] == 2
        assert stats["failed"] == 1
        assert ingestor.node_buffer == []

    def test_raises_when_all_groups_fail(self) -> None:
        ingestor = MemgraphIngestor(host="localhost", port=7687)
        ingestor.node_buffer = [
            ("LabelA", {"id": 1}),
            ("LabelB", {"id": 2}),
        ]

        with patch.object(
            MemgraphIngestor, "_flush_node_label_group"
        ) as mock_flush:
            mock_flush.side_effect = RuntimeError("all fail")

            with pytest.raises(RuntimeError, match="all fail"):
                ingestor.flush_nodes_with_stats()

        assert ingestor.node_buffer == []


class TestExitSkipsFlushOnDeadConnection:
    def test_skips_flush_when_connection_dead(self) -> None:
        ingestor = MemgraphIngestor(host="localhost", port=7687)
        ingestor.conn = MagicMock()
        ingestor.conn.cursor.return_value.execute.side_effect = ConnectionError("dead")
        ingestor._executor = None

        with patch.object(MemgraphIngestor, "flush_all") as mock_flush:
            ingestor.__exit__(None, None, None)

        mock_flush.assert_not_called()

    def test_flushes_when_connection_healthy(self) -> None:
        ingestor = MemgraphIngestor(host="localhost", port=7687)
        ingestor.conn = MagicMock()
        ingestor._executor = None

        with patch.object(MemgraphIngestor, "flush_all") as mock_flush:
            ingestor.__exit__(None, None, None)

        mock_flush.assert_called_once()
