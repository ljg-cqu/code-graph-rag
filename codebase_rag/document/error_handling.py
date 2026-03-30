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
    UNKNOWN = "unknown"


# Recoverable vs fatal error classification
RECOVERABLE_ERRORS = frozenset(
    {
        ErrorType.MALFORMED_FILE,
        ErrorType.ENCODING_ERROR,
        ErrorType.FILE_TOO_LARGE,
        ErrorType.FILE_NOT_FOUND,
        ErrorType.NOT_A_FILE,
        ErrorType.UNKNOWN,
    }
)

FATAL_ERRORS = frozenset(
    {
        ErrorType.PERMISSION_DENIED,
        ErrorType.PATH_TRAVERSAL,
        ErrorType.MISSING_DEPENDENCY,
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

    def __post_init__(self) -> None:
        """Set recoverable based on error type."""
        if self.error_type in FATAL_ERRORS:
            self.recoverable = False
        elif self.error_type in RECOVERABLE_ERRORS:
            self.recoverable = True

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
        )


class DeadLetterQueue:
    """
    Dead letter queue for failed document extractions.

    Failed documents are logged and can be retried later.
    Uses file locking for thread safety.
    """

    def __init__(self, queue_path: Path) -> None:
        self.queue_path = queue_path
        self.queue_path.mkdir(parents=True, exist_ok=True)

    def _safe_error_filename(self, path: str) -> str:
        """Generate unique, safe filename for error file."""
        path_hash = hashlib.sha256(path.encode()).hexdigest()[:16]
        safe_name = re.sub(r"[^\w\-]", "_", Path(path).name)[:50]
        return f"{path_hash}_{safe_name}.error.json"

    def enqueue(self, error: ExtractionError) -> Path:
        """
        Add failed document to dead letter queue.

        Returns:
            Path to the error file
        """
        filename = self._safe_error_filename(error.path)
        error_file = self.queue_path / filename

        # Use file locking for thread safety
        with open(error_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(error.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return error_file

    def get_pending(self) -> list[ExtractionError]:
        """Get all pending errors for retry."""
        errors = []
        for error_file in self.queue_path.glob("*.error.json"):
            try:
                data = json.loads(error_file.read_text())
                errors.append(ExtractionError.from_dict(data))
            except (json.JSONDecodeError, KeyError) as e:
                # Skip corrupted error files
                continue
        return errors

    def remove(self, error: ExtractionError) -> bool:
        """Remove error from queue after successful retry."""
        filename = self._safe_error_filename(error.path)
        error_file = self.queue_path / filename
        if error_file.exists():
            error_file.unlink()
            return True
        return False

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
        pending = self.get_pending()

        # Filter out exhausted retries
        to_retry = [e for e in pending if e.retry_count < e.max_retries]

        # Process with rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def retry_one(error: ExtractionError) -> tuple[str, bool]:
            async with semaphore:
                # Exponential backoff
                delay = 2**error.retry_count
                await asyncio.sleep(delay)

                try:
                    await extractor.extract_async(Path(error.path))
                    self.remove(error)
                    return error.path, True
                except Exception:
                    error.retry_count += 1
                    self.enqueue(error)
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


__all__ = [
    "ErrorType",
    "ExtractionError",
    "ExtractionException",
    "DeadLetterQueue",
    "RECOVERABLE_ERRORS",
    "FATAL_ERRORS",
]