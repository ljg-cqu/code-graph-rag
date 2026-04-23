"""Tests for document error handling."""

from __future__ import annotations

from datetime import datetime

from codebase_rag.document.concept_extraction import (
    _parse_quota_reset_time,
    classify_concept_extraction_error,
    get_user_facing_message,
)
from codebase_rag.document.error_handling import (
    FATAL_ERRORS,
    RECOVERABLE_ERRORS,
    DeadLetterQueue,
    ErrorType,
    ExtractionError,
    ExtractionException,
)


class TestErrorType:
    """Tests for ErrorType enum."""

    def test_error_type_values(self):
        """Test all error types have string values."""
        assert ErrorType.MALFORMED_FILE == "malformed_file"
        assert ErrorType.MISSING_DEPENDENCY == "missing_dependency"
        assert ErrorType.FILE_TOO_LARGE == "file_too_large"
        assert ErrorType.PERMISSION_DENIED == "permission_denied"
        assert ErrorType.ENCODING_ERROR == "encoding_error"
        assert ErrorType.PATH_TRAVERSAL == "path_traversal"
        assert ErrorType.FILE_NOT_FOUND == "file_not_found"
        assert ErrorType.NOT_A_FILE == "not_a_file"
        assert ErrorType.UNKNOWN == "unknown"


class TestErrorClassification:
    """Tests for error classification."""

    def test_recoverable_errors(self):
        """Test recoverable error set."""
        assert ErrorType.MALFORMED_FILE in RECOVERABLE_ERRORS
        assert ErrorType.ENCODING_ERROR in RECOVERABLE_ERRORS
        assert ErrorType.FILE_TOO_LARGE in RECOVERABLE_ERRORS
        assert ErrorType.FILE_NOT_FOUND in RECOVERABLE_ERRORS
        assert ErrorType.NOT_A_FILE in RECOVERABLE_ERRORS
        assert ErrorType.UNKNOWN in RECOVERABLE_ERRORS

    def test_fatal_errors(self):
        """Test fatal error set."""
        assert ErrorType.PERMISSION_DENIED in FATAL_ERRORS
        assert ErrorType.PATH_TRAVERSAL in FATAL_ERRORS
        assert ErrorType.MISSING_DEPENDENCY in FATAL_ERRORS

    def test_no_overlap(self):
        """Ensure no error is in both sets."""
        overlap = RECOVERABLE_ERRORS & FATAL_ERRORS
        assert len(overlap) == 0


class TestExtractionError:
    """Tests for ExtractionError dataclass."""

    def test_create_extraction_error(self):
        """Test creating an extraction error."""
        error = ExtractionError(
            path="/test/doc.pdf",
            error_type=ErrorType.MALFORMED_FILE,
            message="File is corrupted",
        )
        assert error.path == "/test/doc.pdf"
        assert error.error_type == ErrorType.MALFORMED_FILE
        assert error.message == "File is corrupted"
        assert error.recoverable is True  # MALFORMED_FILE is recoverable

    def test_fatal_error_recoverable_false(self):
        """Test that fatal errors have recoverable=False."""
        error = ExtractionError(
            path="/test/doc.pdf",
            error_type=ErrorType.PATH_TRAVERSAL,
            message="Path traversal detected",
        )
        assert error.recoverable is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        error = ExtractionError(
            path="/test/doc.pdf",
            error_type=ErrorType.ENCODING_ERROR,
            message="Cannot decode",
        )
        data = error.to_dict()
        assert data["path"] == "/test/doc.pdf"
        assert data["error_type"] == "encoding_error"
        assert data["message"] == "Cannot decode"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "path": "/test/doc.pdf",
            "error_type": "file_too_large",
            "message": "File exceeds 50MB",
        }
        error = ExtractionError.from_dict(data)
        assert error.path == "/test/doc.pdf"
        assert error.error_type == ErrorType.FILE_TOO_LARGE
        assert error.message == "File exceeds 50MB"


class TestExtractionException:
    """Tests for ExtractionException."""

    def test_create_exception(self):
        """Test creating an extraction exception."""
        exc = ExtractionException(
            path="/test/doc.pdf",
            error_type=ErrorType.FILE_NOT_FOUND,
            message="File does not exist",
        )
        assert exc.path == "/test/doc.pdf"
        assert exc.error_type == ErrorType.FILE_NOT_FOUND
        assert "file_not_found" in str(exc)

    def test_to_extraction_error(self):
        """Test converting exception to error."""
        exc = ExtractionException(
            path="/test/doc.pdf",
            error_type=ErrorType.ENCODING_ERROR,
            message="Bad encoding",
        )
        error = exc.to_extraction_error()
        assert isinstance(error, ExtractionError)
        assert error.path == exc.path
        assert error.error_type == exc.error_type


class TestDeadLetterQueue:
    """Tests for DeadLetterQueue."""

    def test_enqueue_error(self, tmp_path):
        """Test enqueueing an error."""
        queue = DeadLetterQueue(tmp_path / "errors")
        error = ExtractionError(
            path="/test/doc.pdf",
            error_type=ErrorType.MALFORMED_FILE,
            message="Corrupted",
        )
        error_path = queue.enqueue(error)
        assert error_path.exists()
        assert error_path.suffix == ".json"

    def test_get_pending(self, tmp_path):
        """Test getting pending errors."""
        queue = DeadLetterQueue(tmp_path / "errors")
        error = ExtractionError(
            path="/test/doc.pdf",
            error_type=ErrorType.MALFORMED_FILE,
            message="Corrupted",
        )
        queue.enqueue(error)

        pending = queue.get_pending()
        assert len(pending) == 1
        assert pending[0].path == "/test/doc.pdf"

    def test_remove_error(self, tmp_path):
        """Test removing an error after retry."""
        queue = DeadLetterQueue(tmp_path / "errors")
        error = ExtractionError(
            path="/test/doc.pdf",
            error_type=ErrorType.MALFORMED_FILE,
            message="Corrupted",
        )
        queue.enqueue(error)
        assert queue.size() == 1

        queue.remove(error)
        assert queue.size() == 0

    def test_clear_queue(self, tmp_path):
        """Test clearing all errors."""
        queue = DeadLetterQueue(tmp_path / "errors")
        for i in range(3):
            queue.enqueue(
                ExtractionError(
                    path=f"/test/doc{i}.pdf",
                    error_type=ErrorType.MALFORMED_FILE,
                    message=f"Error {i}",
                )
            )
        assert queue.size() == 3

        count = queue.clear()
        assert count == 3
        assert queue.size() == 0


class MockModelHTTPError(Exception):
    """Mock ModelHTTPError for testing."""

    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self.body = body
        super().__init__(f"status_code: {status_code}, body: {body}")


class TestQuotaErrorClassification:
    """Test classification of quota and rate limit errors."""

    def test_account_quota_exceeded_detection(self):
        """Properly detect AccountQuotaExceeded error."""
        exc = MockModelHTTPError(
            status_code=429,
            body={
                "code": "AccountQuotaExceeded",
                "message": "You have exceeded the monthly usage quota. "
                "It will reset at 2026-05-09 23:59:59 +0800 CST.",
            },
        )

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_QUOTA_EXCEEDED
        assert error.recoverable is False
        assert error.retry_after is not None

    def test_rate_limit_transient_detection(self):
        """Properly detect transient rate limit."""
        exc = MockModelHTTPError(
            status_code=429,
            body={
                "code": "RateLimitExceeded",
                "message": "Rate limit exceeded. Please retry after 60 seconds.",
            },
        )

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_RATE_LIMITED
        assert error.recoverable is True

    def test_generic_429_defaults_to_rate_limited(self):
        """Generic 429 should be treated as transient rate limit."""
        exc = MockModelHTTPError(
            status_code=429, body={"error": "Too many requests"}
        )

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_RATE_LIMITED
        assert error.recoverable is True

    def test_quota_in_message_detection(self):
        """Detect quota errors from message text."""
        exc = Exception("API Error: You have exceeded your quota. Please upgrade.")

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_QUOTA_EXCEEDED
        assert error.recoverable is False

    def test_error_types_in_recoverable_set(self):
        """CONCEPT_RATE_LIMITED should be in RECOVERABLE_ERRORS."""
        assert ErrorType.CONCEPT_RATE_LIMITED in RECOVERABLE_ERRORS

    def test_error_types_in_fatal_set(self):
        """CONCEPT_QUOTA_EXCEEDED should be in FATAL_ERRORS."""
        assert ErrorType.CONCEPT_QUOTA_EXCEEDED in FATAL_ERRORS


class TestQuotaResetTimeParsing:
    """Test parsing of quota reset time from error messages."""

    def test_parse_reset_time_with_timezone(self):
        """Parse reset time with timezone offset."""
        message = "It will reset at 2026-05-09 23:59:59 +0800 CST"
        result = _parse_quota_reset_time(message)

        assert result is not None
        assert result.year == 2026
        assert result.month == 5
        assert result.day == 9

    def test_parse_reset_time_none_on_invalid(self):
        """Return None for messages without reset time."""
        message = "Some random error message"
        result = _parse_quota_reset_time(message)

        assert result is None

    def test_parse_reset_time_various_formats(self):
        """Parse various reset time formats."""
        # Different timezone offsets
        messages = [
            "reset at 2026-12-31 23:59:59 +0000 UTC",
            "reset at 2026-01-15 12:30:45 -0500 EST",
            "Quota will reset at 2026-06-20 08:00:00 +0530 IST",
        ]

        for message in messages:
            result = _parse_quota_reset_time(message)
            assert result is not None, f"Failed to parse: {message}"
            assert isinstance(result, datetime)


class TestUserFacingMessages:
    """Test user-facing error message generation."""

    def test_quota_exceeded_message(self):
        """Test quota exceeded user message."""
        error = ExtractionError(
            path="test::chunk",
            error_type=ErrorType.CONCEPT_QUOTA_EXCEEDED,
            message="Quota exceeded",
            retry_after="2026-05-09T23:59:59+08:00",
        )

        message = get_user_facing_message(error)

        assert "quota has been exceeded" in message.lower()
        assert "resets at:" in message.lower()
        assert "upgrade" in message.lower()

    def test_rate_limited_message(self):
        """Test rate limited user message."""
        error = ExtractionError(
            path="test::chunk",
            error_type=ErrorType.CONCEPT_RATE_LIMITED,
            message="Rate limit exceeded",
        )

        message = get_user_facing_message(error)

        assert "too many requests" in message.lower()
        assert "backoff" in message.lower()

    def test_auth_error_message(self):
        """Test auth error user message."""
        error = ExtractionError(
            path="test::chunk",
            error_type=ErrorType.CONCEPT_AUTH_ERROR,
            message="Invalid API key",
        )

        message = get_user_facing_message(error)

        assert "authentication" in message.lower() or "api key" in message.lower()

    def test_context_overflow_message(self):
        """Test context overflow user message."""
        error = ExtractionError(
            path="test::chunk",
            error_type=ErrorType.CONCEPT_CONTEXT_OVERFLOW,
            message="Context too large",
        )

        message = get_user_facing_message(error)

        assert "chunk" in message.lower() or "context" in message.lower()

    def test_fallback_message(self):
        """Test fallback message for unknown error types."""
        error = ExtractionError(
            path="test::chunk",
            error_type=ErrorType.UNKNOWN,
            message="Something went wrong",
        )

        message = get_user_facing_message(error)

        assert "failed" in message.lower()
        assert "something went wrong" in message.lower() or "unknown" in message.lower()
