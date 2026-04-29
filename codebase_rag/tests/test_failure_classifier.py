"""Tests for MemgraphFailureClassifier."""

from __future__ import annotations

from codebase_rag.services.failure_classifier import (
    FailureClassification,
    FailureType,
    classify_memgraph_failure,
    is_stdlib_module,
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

    def test_classifies_transaction_conflict(self) -> None:
        error = Exception("Cannot resolve conflicting transactions")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.TRANSACTION_CONFLICT
        assert classification.should_retry
        assert classification.max_retries == 3
        assert classification.recovery_action == "retry_with_backoff"

    def test_transaction_conflict_retry_count(self) -> None:
        error = Exception("Conflicting transactions detected")
        classification = classify_memgraph_failure(error)
        assert classification.max_retries == 3

    def test_broken_pipe_is_transient(self) -> None:
        error = Exception("Broken pipe")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.TRANSIENT_NETWORK
        assert classification.should_retry

    def test_connection_refused_is_transient(self) -> None:
        error = Exception("Connection refused")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.TRANSIENT_NETWORK

    def test_classifies_mgclient_state_corruption(self) -> None:
        """MGCLIENT_STATE_CORRUPTION should be classified from mgclient errors."""
        error = Exception("mgclient.Column returned a result with an exception set")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.MGCLIENT_STATE_CORRUPTION
        assert classification.should_retry
        assert classification.max_retries == 2
        assert classification.recovery_action == "recreate_connection"

    def test_classifies_mgclient_close_during_execution(self) -> None:
        """MGCLIENT_STATE_CORRUPTION should match close during execution error."""
        error = Exception("cannot close connection during execution of a query")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.MGCLIENT_STATE_CORRUPTION
        assert classification.should_retry

    def test_classifies_enterprise_license_error(self) -> None:
        """ENTERPRISE_FEATURE_REQUIRED should match multi-tenancy license errors."""
        error = Exception(
            "Your license has an invalid type. To use multi-tenancy "
            "you need to have an enterprise license."
        )
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.ENTERPRISE_FEATURE_REQUIRED
        assert not classification.should_retry
        assert classification.max_retries == 0
        assert classification.recovery_action == "upgrade_to_enterprise_or_use_community_mode"

    def test_classifies_enterprise_feature_error(self) -> None:
        """ENTERPRISE_FEATURE_REQUIRED should match generic enterprise feature errors."""
        error = Exception("This is an enterprise feature")
        classification = classify_memgraph_failure(error)

        assert classification.failure_type == FailureType.ENTERPRISE_FEATURE_REQUIRED
        assert not classification.should_retry


    def test_from_error_type_enterprise_feature(self) -> None:
        """from_error_type should map enterprise_feature_required correctly."""
        classification = FailureClassification.from_error_type(
            "enterprise_feature_required",
            message="Enterprise license required",
        )
        assert classification.failure_type == FailureType.ENTERPRISE_FEATURE_REQUIRED
        assert not classification.should_retry
        assert classification.message == "Enterprise license required"


class TestIsStdlibModule:
    """Test is_stdlib_module function."""

    def test_identifies_stdlib_os(self) -> None:
        """os should be identified as stdlib."""
        assert is_stdlib_module("os") is True

    def test_identifies_stdlib_typing(self) -> None:
        """typing should be identified as stdlib."""
        assert is_stdlib_module("typing") is True

    def test_identifies_stdlib_with_submodule(self) -> None:
        """Submodules of stdlib should be identified."""
        assert is_stdlib_module("os.path") is True
        assert is_stdlib_module("collections.abc") is True

    def test_identifies_third_party(self) -> None:
        """Third-party packages should not be identified as stdlib."""
        assert is_stdlib_module("pydantic") is False
        assert is_stdlib_module("numpy") is False
        assert is_stdlib_module("requests") is False

    def test_identifies_unknown_as_not_stdlib(self) -> None:
        """Unknown modules should not be identified as stdlib."""
        assert is_stdlib_module("some_unknown_module") is False


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
            "TRANSACTION_CONFLICT",
            "IMPORT_TARGET_MISSING",
            "MGCLIENT_STATE_CORRUPTION",
            "ENTERPRISE_FEATURE_REQUIRED",
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
