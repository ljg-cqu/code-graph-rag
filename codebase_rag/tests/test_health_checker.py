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


def test_check_vector_indexes_all_present() -> None:
    """Test check_vector_indexes passes when all indexes exist."""
    checker = HealthChecker()
    cursor = MagicMock()
    # Return all expected indexes
    cursor.fetchall.return_value = [
        ("function_embedding_index",),
        ("method_embedding_index",),
        ("class_embedding_index",),
        ("interface_embedding_index",),
    ]
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
            "codebase_rag.tools.health_checker.get_embedding_provider",
            return_value=mock_provider,
        ),
        patch(
            "codebase_rag.tools.health_checker.get_shared_backend",
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
            "codebase_rag.tools.health_checker.get_embedding_provider",
            return_value=mock_provider,
        ),
        patch(
            "codebase_rag.tools.health_checker.get_shared_backend",
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
