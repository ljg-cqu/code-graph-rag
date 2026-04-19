from unittest.mock import MagicMock, patch

from codebase_rag.tools.health_checker import HealthChecker


def test_consume_all_results_handles_exceptions() -> None:
    """Test _consume_all_results safely handles errors during cleanup."""
    cursor = MagicMock()
    cursor.fetchone.side_effect = RuntimeError("Connection lost")

    # Should not raise - silently ignores errors
    HealthChecker._consume_all_results(cursor)

    assert cursor.fetchone.called


def test_validate_ingestion_quality_cleanup_on_exception() -> None:
    """Test that cleanup succeeds when exception occurs mid-query."""
    checker = HealthChecker()
    cursor = MagicMock()
    # First fetchone returns a row, subsequent calls return None for _consume_all_results
    cursor.fetchone.side_effect = [(4,), None]
    # Second query raises exception mid-execution
    cursor.execute.side_effect = [None, RuntimeError("Query failed")]
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("codebase_rag.tools.health_checker.mgclient.connect", return_value=conn):
        results = checker.validate_ingestion_quality(expected_node_count=4)

    # Should have one failure result from the exception
    assert len(results) >= 1
    # Cursor close should still be called
    cursor.close.assert_called()
    # Connection close should still be called
    conn.close.assert_called()


def test_validate_ingestion_quality_consumes_pending_results_on_error() -> None:
    """Test that pending results are consumed before closing on error."""
    checker = HealthChecker()
    cursor = MagicMock()
    # Raise exception on first execute
    cursor.execute.side_effect = RuntimeError("Query execution error")
    # fetchone returns None for _consume_all_results cleanup
    cursor.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("codebase_rag.tools.health_checker.mgclient.connect", return_value=conn):
        results = checker.validate_ingestion_quality()

    # _consume_all_results should be called before close
    cursor.close.assert_called()
    conn.close.assert_called()


def test_parse_label_expression_splits_pipe_labels() -> None:
    labels = HealthChecker._parse_label_expression("Function|Method|Class")

    assert labels == ["Function", "Method", "Class"]


def test_validate_ingestion_quality_uses_label_filter_not_pipe_syntax() -> None:
    checker = HealthChecker()
    cursor = MagicMock()
    # Each _fetch_single_int needs a value followed by None for _consume_all_results
    # node count, edge count, missing embeddings, embedded node count, dimension, duplicates
    cursor.fetchone.side_effect = [
        (4,), None,      # node count
        (8,), None,      # edge count
        (0,), None,      # missing embeddings count
        (4,), None,      # embedded node count
        (768,), None,    # dimension check
        (0,), None,      # duplicates count
    ]
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


def test_check_vector_indexes_all_present() -> None:
    """Test check_vector_indexes passes when all indexes exist."""
    checker = HealthChecker()
    cursor = MagicMock()
    # Return all expected indexes (matching EMBEDDABLE_CODE_NODE_LABELS)
    cursor.fetchall.return_value = [
        ("function_embedding_index",),
        ("method_embedding_index",),
        ("class_embedding_index",),
        ("interface_embedding_index",),
        ("contract_embedding_index",),
        ("library_embedding_index",),
        ("enum_embedding_index",),
        ("type_embedding_index",),
        ("union_embedding_index",),
        ("event_embedding_index",),
        ("modifier_embedding_index",),
        ("statevariable_embedding_index",),
        ("customerror_embedding_index",),
        ("hotkey_embedding_index",),
        ("hotstring_embedding_index",),
        ("label_embedding_index",),
        ("ahkclass_embedding_index",),
        ("codechunk_embedding_index",),
    ]
    # fetchone returns None for _consume_all_results cleanup
    cursor.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("codebase_rag.tools.health_checker.mgclient.connect", return_value=conn):
        result = checker.check_vector_indexes()

    assert result.passed is True
    assert "Vector indexes exist" in result.name


def test_check_vector_indexes_missing_some() -> None:
    """Test check_vector_indexes fails when some indexes are missing."""
    checker = HealthChecker()
    cursor = MagicMock()
    # Return only some indexes
    cursor.fetchall.return_value = [
        ("function_embedding_index",),
    ]
    # fetchone returns None for _consume_all_results cleanup
    cursor.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("codebase_rag.tools.health_checker.mgclient.connect", return_value=conn):
        result = checker.check_vector_indexes()

    assert result.passed is False
    assert "missing" in result.message.lower()


def test_check_vector_search_with_results() -> None:
    """Test check_vector_search passes when search returns results."""
    checker = HealthChecker()

    mock_provider = MagicMock()
    mock_provider.embed.return_value = [0.1] * 768

    mock_backend = MagicMock()
    mock_backend.search.return_value = [(1, 0.95), (2, 0.88)]
    mock_backend.get_stats.return_value = {"total_embeddings": 100}

    with (
        patch(
            "codebase_rag.embeddings.get_embedding_provider",
            return_value=mock_provider,
        ),
        patch(
            "codebase_rag.vector_backend.get_shared_backend",
            return_value=mock_backend,
        ),
        patch(
            "codebase_rag.tools.health_checker.settings"
        ) as mock_settings,
    ):
        mock_settings.active_embedding_config = MagicMock(
            provider="local", model_id="test-model"
        )
        result = checker.check_vector_search()

    assert result.passed is True
    assert "100 embeddings" in result.message


def test_check_vector_search_no_embeddings() -> None:
    """Test check_vector_search fails when no embeddings exist."""
    checker = HealthChecker()

    mock_provider = MagicMock()
    mock_provider.embed.return_value = [0.1] * 768

    mock_backend = MagicMock()
    mock_backend.search.return_value = []  # No results
    mock_backend.get_stats.return_value = {"total_embeddings": 0}

    with (
        patch(
            "codebase_rag.embeddings.get_embedding_provider",
            return_value=mock_provider,
        ),
        patch(
            "codebase_rag.vector_backend.get_shared_backend",
            return_value=mock_backend,
        ),
        patch(
            "codebase_rag.tools.health_checker.settings"
        ) as mock_settings,
    ):
        mock_settings.active_embedding_config = MagicMock(
            provider="local", model_id="test-model"
        )
        result = checker.check_vector_search()

    assert result.passed is False
    assert "No embeddings" in result.message
