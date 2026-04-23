"""Error handling infrastructure for document extraction.

Provides:
- ErrorType enum for classification
- ExtractionError dataclass
- DeadLetterQueue for failed documents
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class ErrorType(StrEnum):
    """Classification of extraction errors."""

    MALFORMED_FILE = "malformed_file"  # Corrupted file content
    MISSING_DEPENDENCY = "missing_dependency"  # PyPDF2 not installed
    FILE_TOO_LARGE = "file_too_large"  # Exceeds DOC_MAX_FILE_SIZE_MB
    PERMISSION_DENIED = "permission_denied"
    ENCODING_ERROR = "encoding_error"
    PATH_TRAVERSAL = "path_traversal"
    FILE_NOT_FOUND = "file_not_found"
    NOT_A_FILE = "not_a_file"
    EMBEDDING_ERROR = "embedding_error"  # Embedding generation failed
    CHUNKING_ERROR = "chunking_error"  # Document chunking failed
    GRAPH_ERROR = "graph_error"  # Database operation failed
    VERSION_ERROR = "version_error"  # Version cache operation failed
    # Concept extraction error types
    CONCEPT_TIMEOUT = "concept_timeout"
    CONCEPT_PARSING_ERROR = "concept_parsing_error"
    CONCEPT_RATE_LIMIT = "concept_rate_limit"
    CONCEPT_AUTH_ERROR = "concept_auth_error"
    CONCEPT_CONTEXT_OVERFLOW = "concept_context_overflow"
    CONCEPT_NETWORK_ERROR = "concept_network_error"
    CONCEPT_LLM_ERROR = "concept_llm_error"
    # Quota/Rate Limit specific errors (distinguish from transient rate limits)
    CONCEPT_QUOTA_EXCEEDED = "concept_quota_exceeded"  # Monthly/daily quota exhausted
    CONCEPT_RATE_LIMITED = "concept_rate_limited"      # Transient rate limit (per-second/minute)
    UNKNOWN = "unknown"


# Recoverable vs fatal error classification
RECOVERABLE_ERRORS = frozenset(
    {
        ErrorType.MALFORMED_FILE,
        ErrorType.ENCODING_ERROR,
        ErrorType.FILE_TOO_LARGE,
        ErrorType.FILE_NOT_FOUND,
        ErrorType.NOT_A_FILE,
        ErrorType.EMBEDDING_ERROR,
        ErrorType.CHUNKING_ERROR,
        ErrorType.GRAPH_ERROR,
        ErrorType.VERSION_ERROR,
        ErrorType.UNKNOWN,
        # Concept extraction errors - recoverable
        ErrorType.CONCEPT_TIMEOUT,
        ErrorType.CONCEPT_RATE_LIMIT,
        ErrorType.CONCEPT_NETWORK_ERROR,
        ErrorType.CONCEPT_PARSING_ERROR,
        ErrorType.CONCEPT_LLM_ERROR,
        ErrorType.CONCEPT_RATE_LIMITED,      # Transient rate limit - retry with backoff
    }
)

FATAL_ERRORS = frozenset(
    {
        ErrorType.PERMISSION_DENIED,
        ErrorType.PATH_TRAVERSAL,
        ErrorType.MISSING_DEPENDENCY,
        # Concept extraction errors - fatal
        ErrorType.CONCEPT_AUTH_ERROR,
        ErrorType.CONCEPT_CONTEXT_OVERFLOW,
        ErrorType.CONCEPT_QUOTA_EXCEEDED,    # Cannot retry until quota resets
    }
)


class ExtractionException(Exception):
    """
    Exception raised during document extraction.

    This is raised by extractors and converted to ExtractionError
    for storage in the dead letter queue.
    """

    def __init__(
        self,
        path: str,
        error_type: ErrorType,
        message: str,
    ) -> None:
        self.path = path
        self.error_type = error_type
        self.message = message
        self.timestamp = datetime.now(UTC).isoformat()
        super().__init__(f"[{error_type.value}] {path}: {message}")

    def to_extraction_error(self) -> ExtractionError:
        """Convert to ExtractionError for dead letter queue."""
        return ExtractionError(
            path=self.path,
            error_type=self.error_type,
            message=self.message,
            timestamp=self.timestamp,
        )


@dataclass
class ExtractionError:
    """Structured error for failed document extraction."""

    path: str
    error_type: ErrorType
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3
    retry_after: str | None = None  # ISO timestamp for delayed retry
    error_category: str | None = None  # FailureType name for guidance lookup
    # Concept extraction specific fields
    chunk_qn: str | None = None  # For concept extraction failures
    chunk_length: int | None = None
    chunk_preview: str | None = None  # First 200 chars of chunk content
    exception_type: str | None = None

    def __post_init__(self) -> None:
        """Set recoverable based on error type."""
        if self.error_type in FATAL_ERRORS:
            self.recoverable = False
        elif self.error_type in RECOVERABLE_ERRORS:
            self.recoverable = True
        # Set recoverable for concept extraction errors
        concept_recoverable = {
            ErrorType.CONCEPT_TIMEOUT,
            ErrorType.CONCEPT_RATE_LIMIT,
            ErrorType.CONCEPT_NETWORK_ERROR,
            ErrorType.CONCEPT_PARSING_ERROR,
            ErrorType.CONCEPT_LLM_ERROR,
            ErrorType.CONCEPT_RATE_LIMITED,      # Transient rate limit - recoverable
        }
        concept_fatal = {
            ErrorType.CONCEPT_AUTH_ERROR,
            ErrorType.CONCEPT_CONTEXT_OVERFLOW,
            ErrorType.CONCEPT_QUOTA_EXCEEDED,    # Quota exhausted - fatal until reset
        }
        if self.error_type in concept_recoverable:
            self.recoverable = True
        elif self.error_type in concept_fatal:
            self.recoverable = False

    def to_log_message(self) -> str:
        """Generate a detailed log message."""
        base_msg = (
            f"Extraction failed [{self.error_type.value}]\n"
            f"  Path: {self.path}\n"
            f"  Error: {self.exception_type or 'N/A'}: {self.message or '(no message)'}\n"
            f"  Recoverable: {self.recoverable}"
        )
        if self.chunk_qn:
            base_msg += f"\n  Chunk: {self.chunk_qn}\n  Length: {self.chunk_length} chars"
        return base_msg

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "error_type": self.error_type.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "recoverable": self.recoverable,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "retry_after": self.retry_after,
            "error_category": self.error_category,
            "chunk_qn": self.chunk_qn,
            "chunk_length": self.chunk_length,
            "chunk_preview": self.chunk_preview,
            "exception_type": self.exception_type,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExtractionError:
        """Create from dictionary."""
        return cls(
            path=data["path"],
            error_type=ErrorType(data["error_type"]),
            message=data["message"],
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
            recoverable=data.get("recoverable", True),
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3),
            retry_after=data.get("retry_after"),
            error_category=data.get("error_category"),
            chunk_qn=data.get("chunk_qn"),
            chunk_length=data.get("chunk_length"),
            chunk_preview=data.get("chunk_preview"),
            exception_type=data.get("exception_type"),
        )

    def to_failure_classification(self) -> FailureClassification:
        """Convert extraction error to failure classification.

        Maps document error types to the unified failure classification system
        for integration with the error handling pipeline.

        Returns:
            FailureClassification with mapped failure type
        """
        from ..services.failure_classifier import FailureClassification, FailureType

        error_type_mapping = {
            ErrorType.MALFORMED_FILE: FailureType.DATA_INTEGRITY,
            ErrorType.MISSING_DEPENDENCY: FailureType.MISSING_PROCEDURE,
            ErrorType.FILE_TOO_LARGE: FailureType.RESOURCE_EXHAUSTION,
            ErrorType.PERMISSION_DENIED: FailureType.PERMISSION_DENIED,
            ErrorType.ENCODING_ERROR: FailureType.DATA_INTEGRITY,
            ErrorType.EMBEDDING_ERROR: FailureType.VECTOR_DIMENSION_MISMATCH,
            ErrorType.GRAPH_ERROR: FailureType.TRANSIENT_NETWORK,
            ErrorType.VERSION_ERROR: FailureType.DATA_INTEGRITY,
            ErrorType.PATH_TRAVERSAL: FailureType.PERMISSION_DENIED,
            ErrorType.FILE_NOT_FOUND: FailureType.DATA_INTEGRITY,
            ErrorType.NOT_A_FILE: FailureType.DATA_INTEGRITY,
            ErrorType.CHUNKING_ERROR: FailureType.DATA_INTEGRITY,
            ErrorType.UNKNOWN: FailureType.UNKNOWN,
        }

        return FailureClassification(
            failure_type=error_type_mapping.get(self.error_type, FailureType.UNKNOWN),
            message=self.message,
            should_retry=self.recoverable,
            max_retries=self.max_retries,
            metadata={"path": self.path, "error_type": self.error_type.value},
        )


# Default retry delays for transient failures
DEFAULT_RETRY_DELAYS = [
    30,    # 30 seconds
    300,   # 5 minutes
    1800,  # 30 minutes
]


class DeadLetterQueue:
    """
    Dead letter queue for failed document extractions.

    Failed documents are logged and can be retried later.
    Uses file locking for thread safety.

    Features:
    - Retry scheduling with configurable delays
    - Priority-based retry (connection errors get faster retries)
    - Expiration of stale errors after max retries
    """

    def __init__(
        self,
        queue_path: Path,
        retry_delays: list[int] | None = None,
    ) -> None:
        self.queue_path = queue_path
        self.queue_path.mkdir(parents=True, exist_ok=True)
        self.retry_delays = retry_delays or DEFAULT_RETRY_DELAYS

    def _safe_error_filename(self, path: str) -> str:
        """Generate unique, safe filename for error file."""
        path_hash = hashlib.sha256(path.encode()).hexdigest()[:16]
        safe_name = re.sub(r"[^\w\-]", "_", Path(path).name)[:50]
        return f"{path_hash}_{safe_name}.error.json"

    def enqueue(self, error: ExtractionError) -> Path:
        """
        Add failed document to dead letter queue.

        Sets retry_after timestamp based on retry_count and configured delays.

        Returns:
            Path to the error file
        """
        filename = self._safe_error_filename(error.path)
        error_file = self.queue_path / filename

        # Calculate retry_after if not set and retries remaining
        if error.retry_after is None and error.retry_count < error.max_retries:
            delay_index = min(error.retry_count, len(self.retry_delays) - 1)
            delay_seconds = self.retry_delays[delay_index]
            retry_after = datetime.now(UTC).timestamp() + delay_seconds
            error.retry_after = datetime.fromtimestamp(retry_after, UTC).isoformat()

        # Use file locking for thread safety
        with open(error_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(error.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return error_file

    def get_pending(self, include_scheduled: bool = True) -> list[ExtractionError]:
        """Get all pending errors for retry.

        Args:
            include_scheduled: If False, only return errors ready for retry

        Returns:
            List of ExtractionError objects
        """
        errors = []
        now = datetime.now(UTC).timestamp()

        for error_file in self.queue_path.glob("*.error.json"):
            try:
                data = json.loads(error_file.read_text())
                error = ExtractionError.from_dict(data)

                # Skip if not ready for retry
                if not include_scheduled and error.retry_after:
                    retry_after_ts = datetime.fromisoformat(error.retry_after).timestamp()
                    if retry_after_ts > now:
                        continue

                errors.append(error)
            except (json.JSONDecodeError, KeyError):
                # Skip corrupted error files
                continue
        return errors

    def get_ready_for_retry(self) -> list[ExtractionError]:
        """Get errors that are ready for retry (retry_after has passed)."""
        return self.get_pending(include_scheduled=False)

    def remove(self, error: ExtractionError) -> bool:
        """Remove error from queue after successful retry."""
        filename = self._safe_error_filename(error.path)
        error_file = self.queue_path / filename
        if error_file.exists():
            error_file.unlink()
            return True
        return False

    def mark_retry_attempt(self, error: ExtractionError) -> ExtractionError:
        """
        Increment retry count and schedule next retry.

        Returns:
            Updated ExtractionError with new retry_after timestamp
        """
        error.retry_count += 1

        if error.retry_count < error.max_retries:
            delay_index = min(error.retry_count, len(self.retry_delays) - 1)
            delay_seconds = self.retry_delays[delay_index]
            retry_after = datetime.now(UTC).timestamp() + delay_seconds
            error.retry_after = datetime.fromtimestamp(retry_after, UTC).isoformat()

        return error

    async def retry_with_backoff(
        self, extractor, max_concurrent: int = 5
    ) -> dict[str, bool]:
        """
        Retry all pending errors with exponential backoff.

        Args:
            extractor: Document extractor to use for retry
            max_concurrent: Maximum concurrent retries

        Returns:
            Dict mapping path -> success status
        """
        import asyncio

        results: dict[str, bool] = {}
        pending = self.get_ready_for_retry()

        # Filter out exhausted retries
        to_retry = [e for e in pending if e.retry_count < e.max_retries]

        # Process with rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def retry_one(error: ExtractionError) -> tuple[str, bool]:
            async with semaphore:
                try:
                    await extractor.extract_async(Path(error.path))
                    self.remove(error)
                    return error.path, True
                except Exception:
                    # Increment retry count and reschedule
                    updated = self.mark_retry_attempt(error)
                    self.enqueue(updated)
                    return error.path, False

        # Run retries concurrently
        tasks = [retry_one(e) for e in to_retry]
        for coro in asyncio.as_completed(tasks):
            path, success = await coro
            results[path] = success

        return results

    def clear(self) -> int:
        """Clear all pending errors. Returns count of removed files."""
        count = 0
        for error_file in self.queue_path.glob("*.error.json"):
            error_file.unlink()
            count += 1
        return count

    def size(self) -> int:
        """Get number of pending errors."""
        return len(list(self.queue_path.glob("*.error.json")))

    def stats(self) -> dict:
        """Get queue statistics."""
        pending = self.get_pending()
        now = datetime.now(UTC).timestamp()

        ready_count = 0
        scheduled_count = 0
        exhausted_count = 0

        for error in pending:
            if error.retry_count >= error.max_retries:
                exhausted_count += 1
            elif error.retry_after:
                retry_after_ts = datetime.fromisoformat(error.retry_after).timestamp()
                if retry_after_ts <= now:
                    ready_count += 1
                else:
                    scheduled_count += 1
            else:
                ready_count += 1

        return {
            "total": len(pending),
            "ready_for_retry": ready_count,
            "scheduled": scheduled_count,
            "exhausted": exhausted_count,
        }


__all__ = [
    "ErrorType",
    "ExtractionError",
    "ExtractionException",
    "DeadLetterQueue",
    "RECOVERABLE_ERRORS",
    "FATAL_ERRORS",
    "DEFAULT_RETRY_DELAYS",
]
