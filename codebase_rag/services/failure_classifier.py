"""Memgraph failure classification for granular error handling.

Provides systematic classification of Memgraph failures with appropriate
recovery strategies instead of binary retry/don't-retry handling.
"""

from __future__ import annotations

import sys
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
    TRANSACTION_CONFLICT = auto()
    IMPORT_TARGET_MISSING = auto()  # Target module for IMPORTS relationship not in graph
    MGCLIENT_STATE_CORRUPTION = auto()  # mgclient internal state error
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

    @classmethod
    def from_exception(cls, error: Exception) -> FailureClassification:
        """Classify any exception using marker matching.

        Args:
            error: The exception to classify

        Returns:
            FailureClassification with appropriate type and recovery strategy
        """
        return classify_memgraph_failure(error)

    @classmethod
    def from_error_type(cls, error_type: str, message: str = "") -> FailureClassification:
        """Create classification from error type string.

        Args:
            error_type: String identifier for the error type
            message: Optional error message

        Returns:
            FailureClassification with mapped failure type
        """
        type_to_failure = {
            "connection_error": FailureType.TRANSIENT_NETWORK,
            "auth_error": FailureType.AUTHENTICATION_FAILURE,
            "timeout": FailureType.TRANSIENT_TIMEOUT,
            "syntax_error": FailureType.SYNTAX_ERROR,
            "permission_denied": FailureType.PERMISSION_DENIED,
            "data_integrity": FailureType.DATA_INTEGRITY,
            "resource_exhaustion": FailureType.RESOURCE_EXHAUSTION,
            "vector_dimension_mismatch": FailureType.VECTOR_DIMENSION_MISMATCH,
            "vector_index_missing": FailureType.VECTOR_INDEX_MISSING,
            "missing_procedure": FailureType.MISSING_PROCEDURE,
        }
        failure_type = type_to_failure.get(error_type, FailureType.UNKNOWN)
        return cls(
            failure_type=failure_type,
            message=message,
            should_retry=failure_type
            in {
                FailureType.TRANSIENT_NETWORK,
                FailureType.TRANSIENT_TIMEOUT,
                FailureType.RESOURCE_EXHAUSTION,
                FailureType.TRANSACTION_CONFLICT,
                FailureType.MGCLIENT_STATE_CORRUPTION,
            },
        )


_SYNTAX_ERROR_MARKERS = frozenset(
    {
        "syntax error",
        "parse error",
        "unexpected token",
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

_TRANSACTION_CONFLICT_MARKERS = frozenset(
    {
        "conflicting transactions",
        "cannot resolve",
        "transaction conflict",
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

# mgclient state corruption markers
_MGCLIENT_STATE_MARKERS = frozenset({
    "mgclient.Column",
    "returned a result with an exception set",
    "cannot close connection during execution",
})

# Known third-party packages for import classification
_KNOWN_THIRD_PARTY = frozenset({
    "pydantic",
    "pydantic_ai",
    "loguru",
    "numpy",
    "pandas",
    "requests",
    "httpx",
    "fastapi",
    "flask",
    "django",
    "pytest",
    "sqlalchemy",
    "tortoise",
    "memgraph",
    "mgclient",
    "tiktoken",
    "openai",
    "anthropic",
    "google",
    "ollama",
    "unixcoder",
    "transformers",
    "torch",
    "tensorflow",
    "jax",
    "qdrant",
    "redis",
    "boto3",
    "botocore",
    "azure",
})


def is_stdlib_module(module_name: str) -> bool:
    """Check if module is Python standard library.

    Args:
        module_name: Full module name (e.g., 'os.path' or 'typing')

    Returns:
        True if module is from Python standard library
    """
    base_name = module_name.split('.')[0]

    # Use sys.stdlib_module_names if available (Python 3.10+)
    if hasattr(sys, 'stdlib_module_names'):
        return base_name in sys.stdlib_module_names

    # Fallback for older Python versions
    import importlib.util
    import importlib.machinery

    # Check if it's a built-in or frozen module
    if base_name in sys.builtin_module_names:
        return True

    # Try to find the module spec
    try:
        spec = importlib.util.find_spec(base_name)
        if spec is None:
            return False
        # If origin is None, it's likely a built-in or namespace package
        if spec.origin is None:
            return True
        # Check if it's in the standard library path
        stdlib_path = importlib.machinery.PathFinder().find_spec('os').origin
        if stdlib_path:
            stdlib_dir = stdlib_path.rsplit('/lib/', 1)[0] + '/lib'
            return spec.origin.startswith(stdlib_dir)
    except (ImportError, ModuleNotFoundError, ValueError):
        pass

    return False


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

    # Vector dimension mismatch (check before syntax - more specific)
    if any(m in message for m in _VECTOR_DIMENSION_MARKERS):
        return FailureClassification(
            failure_type=FailureType.VECTOR_DIMENSION_MISMATCH,
            message="Embedding dimension mismatch",
            should_retry=False,
            recovery_action="recreate_index_and_reembed",
            metadata={"affected_capability": "vector_search"},
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

    # Authentication failures (check before syntax - "invalid credentials" vs "invalid")
    if any(m in message for m in _AUTH_FAILURE_MARKERS):
        return FailureClassification(
            failure_type=FailureType.AUTHENTICATION_FAILURE,
            message="Authentication failure",
            should_retry=False,
            max_retries=0,
        )

    # Permission denied (distinct from auth failure)
    if "permission denied" in message or "access denied" in message:
        return FailureClassification(
            failure_type=FailureType.PERMISSION_DENIED,
            message="Permission denied",
            should_retry=False,
            max_retries=0,
        )

    # Check syntax errors (after more specific checks)
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

    # Transaction conflicts during concurrent writes (retry with backoff)
    if any(m in message for m in _TRANSACTION_CONFLICT_MARKERS):
        return FailureClassification(
            failure_type=FailureType.TRANSACTION_CONFLICT,
            message="Transaction conflict during concurrent write",
            should_retry=True,
            max_retries=3,
            recovery_action="retry_with_backoff",
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

    # mgclient state corruption (retry with fresh connection)
    if any(m in message for m in _MGCLIENT_STATE_MARKERS):
        return FailureClassification(
            failure_type=FailureType.MGCLIENT_STATE_CORRUPTION,
            message="mgclient connection state corrupted",
            should_retry=True,
            max_retries=2,
            recovery_action="recreate_connection",
        )

    # Default: unknown error (retry once)
    return FailureClassification(
        failure_type=FailureType.UNKNOWN,
        message=f"Unknown error: {error}",
        should_retry=True,
        max_retries=1,
    )
