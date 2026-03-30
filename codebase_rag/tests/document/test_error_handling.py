"""Tests for document error handling."""

from __future__ import annotations

import pytest

from codebase_rag.document.error_handling import (
    ErrorType,
    ExtractionError,
    ExtractionException,
    DeadLetterQueue,
    RECOVERABLE_ERRORS,
    FATAL_ERRORS,
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