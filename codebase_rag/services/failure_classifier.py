"""Memgraph failure classification for granular error handling.

Provides systematic classification of Memgraph failures with appropriate
recovery strategies instead of binary retry/don't-retry handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class FailureType(Enum):
    """Classification of Memgraph failures."""

    TRANSIENT_NETWORK = auto()
    TRANSIENT_TIMEOUT = auto()
    SYNTAX_ERROR = auto()
    MISSING_PROCEDURE = auto()
    VECTOR_INDEX_MISSING = auto()
    VECTOR_DIMENSION_MISMATCH = auto()
    AUTHENTICATION_FAILURE = auto()
    PERMISSION_DENIED = auto()
    DATA_INTEGRITY = auto()
    RESOURCE_EXHAUSTION = auto()
    UNKNOWN = auto()


@dataclass
class FailureClassification:
    """Result of failure classification."""

    failure_type: FailureType
    message: str
    should_retry: bool
    max_retries: int = 0
    recovery_action: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


_SYNTAX_ERROR_MARKERS = frozenset(
    {
        "syntax error",
        "parse error",
        "invalid",
        "expected",
        "unexpected",
        "mismatched input",
        "no viable alternative",
        "cannot match",
    }
)

_MISSING_PROCEDURE_MARKERS = frozenset(
    {
        "there is no procedure",
        "procedure not found",
        "unknown procedure",
        "function not found",
        "doesn't exist",
    }
)

_VECTOR_INDEX_MARKERS = frozenset(
    {
        "vector index",
        "vector_search",
        "index not found",
    }
)

_VECTOR_DIMENSION_MARKERS = frozenset(
    {
        "dimension mismatch",
        "different number of dimensions",
    }
)

_AUTH_FAILURE_MARKERS = frozenset(
    {
        "authentication failed",
        "invalid credentials",
        "access denied",
        "permission denied",
    }
)

_DATA_INTEGRITY_MARKERS = frozenset(
    {
        "constraint violation",
        "unique constraint",
        "duplicate",
        "integrity",
    }
)

_RESOURCE_EXHAUSTION_MARKERS = frozenset(
    {
        "memory limit",
        "out of memory",
        "too many connections",
        "query killed",
        "resource exhausted",
    }
)

_TRANSIENT_NETWORK_MARKERS = frozenset(
    {
        "broken pipe",
        "bad session",
        "connection reset",
        "connection aborted",
        "connection refused",
        "connection closed",
        "server closed the connection",
        "network is unreachable",
        "temporarily unavailable",
        "socket",
        "transport",
        "failed to send chunk data",
        "failed to send message end marker",
    }
)


def classify_memgraph_failure(error: Exception) -> FailureClassification:
    """Classify a Memgraph failure and determine recovery strategy."""
    message = str(error).lower()

    # Not connected errors (ingestor not initialized, don't retry)
    if "not connected" in message:
        return FailureClassification(
            failure_type=FailureType.UNKNOWN,
            message="Not connected to Memgraph",
            should_retry=False,
            max_retries=0,
            recovery_action="connect",
        )

    # Check syntax errors first (most specific)
    if any(m in message for m in _SYNTAX_ERROR_MARKERS):
        return FailureClassification(
            failure_type=FailureType.SYNTAX_ERROR,
            message="Cypher syntax error detected",
            should_retry=False,
            recovery_action="repair_query",
        )

    # Check missing procedure (community vs enterprise)
    if any(m in message for m in _MISSING_PROCEDURE_MARKERS):
        return FailureClassification(
            failure_type=FailureType.MISSING_PROCEDURE,
            message="Memgraph procedure not found (likely community edition)",
            should_retry=False,
            recovery_action="fallback_alternative",
        )

    # Vector index issues
    if any(m in message for m in _VECTOR_INDEX_MARKERS):
        return FailureClassification(
            failure_type=FailureType.VECTOR_INDEX_MISSING,
            message="Vector index not found or invalid",
            should_retry=False,
            recovery_action="recreate_index",
            metadata={"affected_capability": "vector_search"},
        )

    # Vector dimension mismatch
    if any(m in message for m in _VECTOR_DIMENSION_MARKERS):
        return FailureClassification(
            failure_type=FailureType.VECTOR_DIMENSION_MISMATCH,
            message="Embedding dimension mismatch",
            should_retry=False,
            recovery_action="recreate_index_and_reembed",
            metadata={"affected_capability": "vector_search"},
        )

    # Authentication and permission failures (don't retry)
    if "permission denied" in message or "access denied" in message:
        return FailureClassification(
            failure_type=FailureType.PERMISSION_DENIED,
            message="Permission denied",
            should_retry=False,
            max_retries=0,
        )

    if any(m in message for m in _AUTH_FAILURE_MARKERS):
        return FailureClassification(
            failure_type=FailureType.AUTHENTICATION_FAILURE,
            message="Authentication failure",
            should_retry=False,
            max_retries=0,
        )

    # Data integrity issues (don't retry)
    if any(m in message for m in _DATA_INTEGRITY_MARKERS):
        return FailureClassification(
            failure_type=FailureType.DATA_INTEGRITY,
            message="Data integrity error",
            should_retry=False,
            recovery_action="alert_admin",
        )

    # Resource exhaustion (may retry after cleanup)
    if any(m in message for m in _RESOURCE_EXHAUSTION_MARKERS):
        return FailureClassification(
            failure_type=FailureType.RESOURCE_EXHAUSTION,
            message="Resource exhaustion",
            should_retry=True,
            max_retries=1,
            recovery_action="cleanup",
        )

    # Transient network errors (retry with backoff)
    if any(m in message for m in _TRANSIENT_NETWORK_MARKERS):
        return FailureClassification(
            failure_type=FailureType.TRANSIENT_NETWORK,
            message="Transient network error",
            should_retry=True,
            max_retries=3,
            recovery_action="reconnect",
        )

    # Timeouts (retry with longer timeout)
    if "timeout" in message or "timed out" in message:
        return FailureClassification(
            failure_type=FailureType.TRANSIENT_TIMEOUT,
            message="Query timeout",
            should_retry=True,
            max_retries=2,
            recovery_action="increase_timeout",
        )

    # Default: unknown error (retry once)
    return FailureClassification(
        failure_type=FailureType.UNKNOWN,
        message=f"Unknown error: {error}",
        should_retry=True,
        max_retries=1,
    )
