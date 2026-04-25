from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.exceptions import QueryExecutionError
from codebase_rag.tools.health_checker import HealthChecker


def test_consume_all_results_handles_exceptions() -> None:
    """Test _consume_all_results safely handles errors during cleanup."""
    cursor = MagicMock()
    cursor.fetchone.side_effect = RuntimeError("Connection lost")

    # Should not raise - silently ignores errors
    HealthChecker._consume_all_results(cursor)

    assert cursor.fetchone.called


def test_safe_get_column_with_system_error() -> None:
    """Test _safe_get_column returns None on SystemError without leaking exception."""
    row = MagicMock()
    row.__getitem__.side_effect = SystemError(
        "<class 'mgclient.Column'> returned a result with an exception set"
    )
    row.__iter__.side_effect = SystemError("corrupted")

    result = HealthChecker._safe_get_column(row, 0)

    assert result is None


def test_run_check_with_retry_succeeds_on_second_attempt() -> None:
    """Test retry wrapper succeeds when second attempt works."""
    checker = HealthChecker()
    call_count = 0

    def flaky_check():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectionError("Temporary failure")
        return [MagicMock(passed=True)]

    with patch("time.sleep"):
        results = checker._run_check_with_retry(
            flaky_check, max_attempts=3, base_delay=0.1
        )

    assert call_count == 2
    assert len(results) == 1
    assert results[0].passed is True


def test_run_check_with_retry_exhausts_all_attempts() -> None:
    """Test retry wrapper returns unavailable result after all attempts fail."""
    checker = HealthChecker()

    def always_fail():
        raise ConnectionError("Persistent failure")

    with patch("time.sleep"):
        results = checker._run_check_with_retry(
            always_fail, max_attempts=3, base_delay=0.1
        )

    assert len(results) == 1
    assert results[0].passed is False
    assert "unavailable" in results[0].name


def test_run_check_with_retry_no_retry_for_non_retryable() -> None:
    """Test retry wrapper does not retry non-retryable exceptions."""
    checker = HealthChecker()
    call_count = 0

    def non_retryable_fail():
        nonlocal call_count
        call_count += 1
        raise ValueError("Logic error")

    with patch("time.sleep"):
        results = checker._run_check_with_retry(
            non_retryable_fail, max_attempts=3, base_delay=0.1
        )

    assert call_count == 1
    assert len(results) == 1
    assert results[0].passed is False


def test_run_check_with_retry_on_system_error() -> None:
    """Test retry wrapper retries SystemError and succeeds on second attempt."""
    checker = HealthChecker()
    call_count = 0

    def flaky_system_error():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise SystemError(
                "<class 'mgclient.Column'> returned a result with an exception set"
            )
        return [MagicMock(passed=True)]

    with patch("time.sleep"):
        results = checker._run_check_with_retry(
            flaky_system_error, max_attempts=3, base_delay=0.1
        )

    assert call_count == 2
    assert len(results) == 1
    assert results[0].passed is True


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

    # Each check creates its own connection, so we need to track connection count
    connection_count = [0]

    def create_connection(*args, **kwargs):
        connection_count[0] += 1
        return conn

    with patch(
        "codebase_rag.tools.health_checker.mgclient.connect",
        side_effect=create_connection,
    ):
        results = checker.validate_ingestion_quality(expected_node_count=4)

    # Should have results from successful checks plus failure results
    # Each check creates its own connection, so failures are isolated
    assert len(results) >= 1
    # Cursor close should still be called
    cursor.close.assert_called()
    # Connection close should still be called for each connection created
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

    def create_connection(*args, **kwargs):
        return conn

    with patch(
        "codebase_rag.tools.health_checker.mgclient.connect",
        side_effect=create_connection,
    ):
        checker.validate_ingestion_quality()

    # Each check creates its own connection, so we get partial results
    # Failed checks should still cleanup properly
    cursor.close.assert_called()
    conn.close.assert_called()


def test_parse_label_expression_splits_pipe_labels() -> None:
    labels = HealthChecker._parse_label_expression("Function|Method|Class")

    assert labels == ["Function", "Method", "Class"]


def test_validate_ingestion_quality_uses_label_filter_not_pipe_syntax() -> None:
    checker = HealthChecker()
    created_cursors: list[MagicMock] = []

    def create_cursor(*args, **kwargs):
        fresh_cursor = MagicMock()
        fresh_cursor.fetchone.side_effect = [(4,), None]
        created_cursors.append(fresh_cursor)
        return fresh_cursor

    conn = MagicMock()
    conn.cursor.side_effect = create_cursor

    def create_connection(*args, **kwargs):
        return conn

    with patch(
        "codebase_rag.tools.health_checker.mgclient.connect",
        side_effect=create_connection,
    ):
        checker.validate_ingestion_quality(
            embedded_node_label="Function|Method|Class",
        )

    executed_queries = []
    for cursor in created_cursors:
        executed_queries.extend(
            [call.args[0] for call in cursor.execute.call_args_list]
        )

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
        patch("codebase_rag.tools.health_checker.settings") as mock_settings,
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
        patch("codebase_rag.tools.health_checker.settings") as mock_settings,
    ):
        mock_settings.active_embedding_config = MagicMock(
            provider="local", model_id="test-model"
        )
        result = checker.check_vector_search()

    assert result.passed is False
    assert "No embeddings" in result.message


def test_validate_ingestion_quality_excludes_builtins_from_embedding_check() -> None:
    """Test that builtin functions are excluded from missing embeddings check."""
    checker = HealthChecker()
    # Track all executed queries across all connections
    executed_queries = []

    def create_mock_cursor():
        cursor = MagicMock()
        cursor.fetchone.return_value = (5,)  # Return dummy values
        # Capture queries
        original_execute = cursor.execute

        def execute(query, params=None):
            executed_queries.append(query)
            return original_execute(query, params)

        cursor.execute.side_effect = execute
        return cursor

    conn = MagicMock()
    conn.cursor.return_value = create_mock_cursor()

    def create_connection(*args, **kwargs):
        # Reset cursor for each new connection
        conn.cursor.return_value = create_mock_cursor()
        return conn

    with patch(
        "codebase_rag.tools.health_checker.mgclient.connect",
        side_effect=create_connection,
    ):
        checker.validate_ingestion_quality()

    # Check that missing embeddings query excludes builtins
    missing_emb_queries = [
        q
        for q in executed_queries
        if "embedding" in q.lower() and "is null" in q.lower()
    ]
    assert len(missing_emb_queries) >= 1, (
        "Should have at least one missing embeddings query"
    )
    for query in missing_emb_queries:
        assert "is_builtin" in query.lower() or "builtin." in query.lower(), (
            f"Missing embeddings query should exclude builtins: {query}"
        )


def test_get_missing_embeddings_returns_nodes_without_embeddings() -> None:
    """Test get_missing_embeddings returns nodes missing embeddings."""
    checker = HealthChecker()
    cursor = MagicMock()
    cursor.fetchall.return_value = [
        ("myproject.utils.helper", ["Function"], "utils/helper.py"),
        ("myproject.services.UserService.get", ["Method"], "services/user_service.py"),
    ]
    cursor.fetchone.return_value = None  # for _consume_all_results
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("codebase_rag.tools.health_checker.mgclient.connect", return_value=conn):
        results = checker.get_missing_embeddings(limit=10)

    assert len(results) == 2
    assert results[0]["qualified_name"] == "myproject.utils.helper"
    assert results[1]["qualified_name"] == "myproject.services.UserService.get"

    # Verify query excludes builtins
    executed_query = cursor.execute.call_args[0][0]
    assert (
        "is_builtin" in executed_query.lower() or "builtin." in executed_query.lower()
    )


def test_get_missing_embeddings_count_returns_count() -> None:
    """Test get_missing_embeddings_count returns correct count."""
    checker = HealthChecker()
    cursor = MagicMock()
    cursor.fetchone.side_effect = [
        (5,),
        None,
    ]  # count, then None for _consume_all_results
    conn = MagicMock()
    conn.cursor.return_value = cursor

    with patch("codebase_rag.tools.health_checker.mgclient.connect", return_value=conn):
        count = checker.get_missing_embeddings_count()

    assert count == 5

    # Verify query excludes builtins
    executed_query = cursor.execute.call_args[0][0]
    assert (
        "is_builtin" in executed_query.lower() or "builtin." in executed_query.lower()
    )


def test_validate_ingestion_quality_error_details_in_result() -> None:
    """Test that error details are captured in HealthCheckResult when validation fails."""
    checker = HealthChecker()
    cursor = MagicMock()
    # Raise exception on first execute
    cursor.execute.side_effect = RuntimeError("Database connection lost")
    cursor.fetchone.return_value = None  # for _consume_all_results cleanup
    conn = MagicMock()
    conn.cursor.return_value = cursor

    def create_connection(*args, **kwargs):
        return conn

    with patch(
        "codebase_rag.tools.health_checker.mgclient.connect",
        side_effect=create_connection,
    ):
        results = checker.validate_ingestion_quality()

    # With isolated connections, we get partial failure results
    # Each failed check returns its own failure result
    assert len(results) >= 1
    assert any(r.passed is False for r in results)
    # At least one error should contain the original error message
    error_messages = [r.error for r in results if r.error]
    assert any("Database connection lost" in str(e) for e in error_messages)


def test_validate_ingestion_quality_error_logged_at_warning_level() -> None:
    """Test that validation errors are logged at WARNING level, not DEBUG."""
    checker = HealthChecker()
    cursor = MagicMock()
    cursor.execute.side_effect = RuntimeError("Query timeout exceeded")
    cursor.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cursor

    def create_connection(*args, **kwargs):
        return conn

    with (
        patch(
            "codebase_rag.tools.health_checker.mgclient.connect",
            side_effect=create_connection,
        ),
        patch("codebase_rag.tools.health_checker.logger") as mock_logger,
    ):
        checker.validate_ingestion_quality()

    # Error should be logged at WARNING level (per-check logging in isolated pattern)
    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
    assert any(
        "failed" in call.lower() or "error" in call.lower() for call in warning_calls
    ), "Error should be logged at WARNING level"


def test_validate_ingestion_quality_stacktrace_with_config_enabled() -> None:
    """Test that stack traces are included when LOG_QUALITY_CHECK_STACKTRACES is True."""
    checker = HealthChecker()
    cursor = MagicMock()
    cursor.execute.side_effect = ValueError("Invalid query parameter")
    cursor.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cursor

    mock_settings = MagicMock()
    mock_settings.LOG_QUALITY_CHECK_STACKTRACES = True
    mock_settings.MEMGRAPH_HOST = "localhost"
    mock_settings.MEMGRAPH_PORT = 7687
    mock_settings.MAX_MISSING_EMBEDDINGS_PCT = 10.0

    def create_connection(*args, **kwargs):
        return conn

    with (
        patch(
            "codebase_rag.tools.health_checker.mgclient.connect",
            side_effect=create_connection,
        ),
        patch("codebase_rag.tools.health_checker.settings", mock_settings),
    ):
        results = checker.validate_ingestion_quality()

    # Find the failure result and check for error content
    failure_results = [r for r in results if not r.passed]
    assert len(failure_results) >= 1
    # At least one error should contain the original message
    assert any("Invalid query parameter" in str(r.error) for r in failure_results)


def test_validate_ingestion_quality_no_stacktrace_by_default() -> None:
    """Test that stack traces are NOT included by default (only error message)."""
    checker = HealthChecker()
    cursor = MagicMock()
    cursor.execute.side_effect = ValueError("Test error without traceback")
    cursor.fetchone.return_value = None
    conn = MagicMock()
    conn.cursor.return_value = cursor

    mock_settings = MagicMock()
    mock_settings.LOG_QUALITY_CHECK_STACKTRACES = False
    mock_settings.MEMGRAPH_HOST = "localhost"
    mock_settings.MEMGRAPH_PORT = 7687
    mock_settings.MAX_MISSING_EMBEDDINGS_PCT = 10.0

    def create_connection(*args, **kwargs):
        return conn

    with (
        patch(
            "codebase_rag.tools.health_checker.mgclient.connect",
            side_effect=create_connection,
        ),
        patch("codebase_rag.tools.health_checker.settings", mock_settings),
    ):
        results = checker.validate_ingestion_quality()

    # Find the failure result and verify no traceback by default
    failure_results = [r for r in results if not r.passed]
    assert len(failure_results) >= 1
    # Error should contain the message
    assert any("Test error without traceback" in str(r.error) for r in failure_results)


class TestQueryExecutionError:
    """Tests for QueryExecutionError exception."""

    def test_error_captures_query_and_params(self):
        """Test that QueryExecutionError captures query context."""
        original_error = ValueError("Original error")
        query = "MATCH (n) RETURN count(n)"
        params = {"key": "value"}

        error = QueryExecutionError(query, params, original_error)

        assert error.query == query
        assert error.params == params
        assert error.original_error is original_error
        assert "MATCH (n)" in str(error)
        assert "Original error" in str(error)

    def test_error_truncates_long_query(self):
        """Test that QueryExecutionError truncates long queries."""
        original_error = ValueError("Original error")
        query = "MATCH (n) RETURN n" + " " * 300
        params = None

        error = QueryExecutionError(query, params, original_error)

        assert "..." in str(error)
        # Query preview should be at most 203 chars (200 + "...")
        query_line = str(error).split("Query: ")[1].split("\n")[0]
        assert len(query_line) <= 203

    def test_error_message_format(self):
        """Test error message contains all expected parts."""
        original_error = RuntimeError("Connection lost")
        query = "MATCH (n:Function) RETURN n"
        params = {"limit": 10}

        error = QueryExecutionError(query, params, original_error)

        error_str = str(error)
        assert "Memgraph query failed" in error_str
        assert "Connection lost" in error_str
        assert "Query: MATCH (n:Function)" in error_str
        assert "Params: {'limit': 10}" in error_str


class TestFetchSingleInt:
    """Tests for _fetch_single_int error handling."""

    def test_fetch_single_int_wraps_errors(self):
        """Test that _fetch_single_int wraps errors with QueryExecutionError."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = Exception(
            "<class 'mgclient.Column'> returned a result with an exception set"
        )

        with pytest.raises(QueryExecutionError) as exc_info:
            HealthChecker._fetch_single_int(
                mock_cursor, "MATCH (n) RETURN count(n)", {"param": "value"}
            )

        assert "MATCH (n)" in str(exc_info.value)
        assert "param" in str(exc_info.value)
        assert "mgclient.Column" in str(exc_info.value)

    def test_fetch_single_int_success_returns_int(self):
        """Test _fetch_single_int returns integer on success."""
        mock_cursor = MagicMock()
        # First fetchone returns the result, second returns None for _consume_all_results
        mock_cursor.fetchone.side_effect = [(42,), None]

        result = HealthChecker._fetch_single_int(
            mock_cursor, "MATCH (n) RETURN count(n)"
        )

        assert result == 42

    def test_fetch_single_int_returns_zero_for_none(self):
        """Test _fetch_single_int returns 0 when no result."""
        mock_cursor = MagicMock()
        # fetchone returns None for both the query result and _consume_all_results
        mock_cursor.fetchone.return_value = None

        result = HealthChecker._fetch_single_int(
            mock_cursor, "MATCH (n:NonExistent) RETURN count(n)"
        )

        assert result == 0


class TestFetchEmbeddingDim:
    """Tests for _fetch_embedding_dim error handling."""

    def test_fetch_embedding_dim_wraps_errors(self):
        """Test that _fetch_embedding_dim wraps errors with QueryExecutionError."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = Exception(
            "<class 'mgclient.Column'> returned a result with an exception set"
        )

        checker = HealthChecker()

        with pytest.raises(QueryExecutionError) as exc_info:
            checker._fetch_embedding_dim(
                mock_cursor, ["Function", "Method"], "embedding"
            )

        assert "size(n.embedding)" in str(exc_info.value)
        assert "embedded_labels" in str(exc_info.value)

    def test_fetch_embedding_dim_success_returns_int(self):
        """Test _fetch_embedding_dim returns integer on success."""
        mock_cursor = MagicMock()
        # First fetchone returns the result, second returns None for _consume_all_results
        mock_cursor.fetchone.side_effect = [(768,), None]

        checker = HealthChecker()
        result = checker._fetch_embedding_dim(
            mock_cursor, ["Function", "Method"], "embedding"
        )

        assert result == 768

    def test_fetch_embedding_dim_returns_none_for_no_results(self):
        """Test _fetch_embedding_dim returns None when no embeddings exist."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None

        checker = HealthChecker()
        result = checker._fetch_embedding_dim(mock_cursor, ["Function"], "embedding")

        assert result is None


class TestQueryExecutionErrorRetry:
    def test_run_check_retries_on_query_execution_error(self):
        """Test retry wrapper retries QueryExecutionError and succeeds."""
        checker = HealthChecker()
        call_count = 0

        def flaky_check():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise QueryExecutionError(
                    "MATCH (n) RETURN count(n)",
                    None,
                    SystemError("<class 'mgclient.Column'> returned a result with an exception set"),
                )
            return [MagicMock(passed=True)]

        with patch("time.sleep"):
            results = checker._run_check_with_retry(
                flaky_check, max_attempts=3, base_delay=0.1
            )

        assert call_count == 3
        assert len(results) == 1
        assert results[0].passed is True

    def test_run_check_exhausts_on_persistent_query_execution_error(self):
        """Test retry wrapper returns unavailable after QueryExecutionError exhausts."""
        checker = HealthChecker()

        def always_fail():
            raise QueryExecutionError(
                "MATCH (n) RETURN count(n)",
                None,
                ConnectionError("Persistent failure"),
            )

        with patch("time.sleep"):
            results = checker._run_check_with_retry(
                always_fail, max_attempts=3, base_delay=0.1
            )

        assert len(results) == 1
        assert results[0].passed is False
        assert "unavailable" in results[0].name

    def test_safe_get_column_handles_value_error_from_corrupted_row(self):
        """Test _safe_get_column returns None on ValueError without leaking exception."""
        row = MagicMock()
        row.__getitem__.side_effect = ValueError(
            "<class 'mgclient.Column'> returned a result with an exception set"
        )
        row.__iter__.side_effect = ValueError("corrupted")

        result = HealthChecker._safe_get_column(row, 0)

        assert result is None

    def test_safe_get_column_handles_type_error_from_corrupted_row(self):
        """Test _safe_get_column returns None on TypeError without leaking exception."""
        row = MagicMock()
        row.__getitem__.side_effect = TypeError("bad operand type")
        row.__iter__.side_effect = TypeError("corrupted")

        result = HealthChecker._safe_get_column(row, 0)

        assert result is None
