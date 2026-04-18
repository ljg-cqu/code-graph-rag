"""Tests for MemgraphFailureClassifier."""

from __future__ import annotations

from codebase_rag.services.failure_classifier import (
    FailureClassification,
    FailureType,
    classify_memgraph_failure,
)


class TestFailureClassification:
    """Test failure classification logic."""

    def test_classifies_syntax_error(self) -> None:
        error = Exception("Syntax error at line 5: unexpected token")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.SYNTAX_ERROR
        assert not classification.should_retry
        assert classification.recovery_action == "repair_query"

    def test_classifies_missing_procedure(self) -> None:
        error = Exception("There is no procedure with name 'pagerank.get'")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.MISSING_PROCEDURE
        assert not classification.should_retry
        assert classification.recovery_action == "fallback_alternative"

    def test_classifies_vector_index_missing(self) -> None:
        error = Exception("Vector index not found for label Function")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.VECTOR_INDEX_MISSING
        assert not classification.should_retry
        assert classification.metadata.get("affected_capability") == "vector_search"

    def test_classifies_vector_dimension_mismatch(self) -> None:
        error = Exception("Dimension mismatch: expected 384, got 768")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.VECTOR_DIMENSION_MISMATCH
        assert not classification.should_retry

    def test_classifies_authentication_failure(self) -> None:
        error = Exception("Authentication failed: invalid credentials")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.AUTHENTICATION_FAILURE
        assert not classification.should_retry
        assert classification.max_retries == 0

    def test_classifies_transient_network(self) -> None:
        error = Exception("Connection reset by peer")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.TRANSIENT_NETWORK
        assert classification.should_retry
        assert classification.max_retries == 3
        assert classification.recovery_action == "reconnect"

    def test_classifies_timeout(self) -> None:
        error = Exception("Query timed out after 30 seconds")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.TRANSIENT_TIMEOUT
        assert classification.should_retry
        assert classification.max_retries == 2

    def test_classifies_unknown_error(self) -> None:
        error = Exception("Some random error")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.UNKNOWN
        assert classification.should_retry
        assert classification.max_retries == 1

    def test_broken_pipe_is_transient(self) -> None:
        error = Exception("Broken pipe")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.TRANSIENT_NETWORK
        assert classification.should_retry

    def test_connection_refused_is_transient(self) -> None:
        error = Exception("Connection refused")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.TRANSIENT_NETWORK


class TestFailureType:
    """Test FailureType enum."""

    def test_all_failure_types_exist(self) -> None:
        expected_types = {
            "TRANSIENT_NETWORK",
            "TRANSIENT_TIMEOUT",
            "SYNTAX_ERROR",
            "MISSING_PROCEDURE",
            "VECTOR_INDEX_MISSING",
            "VECTOR_DIMENSION_MISMATCH",
            "AUTHENTICATION_FAILURE",
            "PERMISSION_DENIED",
            "DATA_INTEGRITY",
            "RESOURCE_EXHAUSTION",
            "UNKNOWN",
        }
        actual_types = {ft.name for ft in FailureType}
        assert actual_types == expected_types


class TestFailureClassificationDataclass:
    """Test FailureClassification dataclass."""

    def test_creates_classification(self) -> None:
        classification = FailureClassification(
            failure_type=FailureType.SYNTAX_ERROR,
            message="Test error",
            should_retry=False,
            max_retries=0,
            recovery_action="repair_query",
        )

        assert classification.failure_type == FailureType.SYNTAX_ERROR
        assert classification.message == "Test error"
        assert not classification.should_retry
        assert classification.recovery_action == "repair_query"
        assert classification.metadata == {}

    def test_creates_with_metadata(self) -> None:
        classification = FailureClassification(
            failure_type=FailureType.VECTOR_INDEX_MISSING,
            message="Test",
            should_retry=False,
            metadata={"affected_capability": "vector_search"},
        )

        assert classification.metadata["affected_capability"] == "vector_search"
