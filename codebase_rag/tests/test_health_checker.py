from unittest.mock import MagicMock, patch

from codebase_rag.tools.health_checker import HealthChecker


def test_parse_label_expression_splits_pipe_labels() -> None:
    labels = HealthChecker._parse_label_expression("Function|Method|Class")

    assert labels == ["Function", "Method", "Class"]


def test_validate_ingestion_quality_uses_label_filter_not_pipe_syntax() -> None:
    checker = HealthChecker()
    cursor = MagicMock()
    cursor.fetchone.side_effect = [(4,), (8,), (0,), (4,), (768,), (0,)]
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("codebase_rag.tools.health_checker.mgclient.connect", return_value=conn):
        checker.validate_ingestion_quality(
            embedded_node_label="Function|Method|Class",
        )

    executed_queries = [call.args[0] for call in cursor.execute.call_args_list]

    assert any(
        "ANY(label IN labels(n) WHERE label IN $embedded_labels)" in query
        for query in executed_queries
    )
    assert not any(
        "MATCH (n:Function|Method|Class)" in query for query in executed_queries
    )
