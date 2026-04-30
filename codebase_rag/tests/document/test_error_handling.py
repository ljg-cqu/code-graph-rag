"""Tests for document error handling."""

from __future__ import annotations

from datetime import datetime

import pytest

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

    def test_output_token_limit_recoverable(self):
        """CONCEPT_OUTPUT_TOKEN_LIMIT should be recoverable."""
        assert ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT in RECOVERABLE_ERRORS
        assert ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT not in FATAL_ERRORS


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

    def test_output_token_limit_recoverable_true(self):
        """CONCEPT_OUTPUT_TOKEN_LIMIT should have recoverable=True."""
        error = ExtractionError(
            path="/test/doc.pdf",
            error_type=ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT,
            message="Model token limit exceeded before any response was generated",
        )
        assert error.recoverable is True

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
        exc = MockModelHTTPError(status_code=429, body={"error": "Too many requests"})

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


class TestContextOverflowErrorClassification:
    """Test classification of context overflow vs output token limit."""

    def test_context_overflow_without_response_pattern(self):
        """Generic token error should be CONCEPT_CONTEXT_OVERFLOW (fatal)."""
        exc = Exception("Context window exceeded: token limit reached")

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_CONTEXT_OVERFLOW
        assert error.recoverable is False

    def test_output_token_limit_with_response_pattern(self):
        """Before-any-response pattern should be CONCEPT_OUTPUT_TOKEN_LIMIT (recoverable)."""
        exc = Exception(
            "Model token limit (2048) exceeded before any response was generated"
        )

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT
        assert error.recoverable is True

    def test_output_token_limit_case_insensitive(self):
        """Pattern matching should be case-insensitive."""
        exc = Exception("MODEL TOKEN LIMIT EXCEEDED BEFORE ANY RESPONSE WAS GENERATED")

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT

    def test_output_token_limit_with_output_token_phrasing(self):
        """Output token limit exceeded should be CONCEPT_OUTPUT_TOKEN_LIMIT."""
        exc = Exception("output token limit exceeded for model gpt-4")

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT
        assert error.recoverable is True

    def test_output_token_limit_with_maximum_output_phrasing(self):
        """Maximum output length should be CONCEPT_OUTPUT_TOKEN_LIMIT."""
        exc = Exception("maximum output length reached")

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT
        assert error.recoverable is True

    def test_output_token_limit_with_response_limit_phrasing(self):
        """Response limit reached should be CONCEPT_OUTPUT_TOKEN_LIMIT."""
        exc = Exception("response limit reached, try reducing output")

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT
        assert error.recoverable is True

    def test_context_overflow_still_fatal_without_output_pattern(self):
        """Generic token errors without output-specific phrasing remain fatal."""
        exc = Exception("context window exceeded: token limit reached")

        error = classify_concept_extraction_error(exc, "test content", "test::chunk")

        assert error.error_type == ErrorType.CONCEPT_CONTEXT_OVERFLOW
        assert error.recoverable is False


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

    def test_output_token_limit_message(self):
        """Test output token limit user message."""
        error = ExtractionError(
            path="test::chunk",
            error_type=ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT,
            message="Model token limit exceeded before any response was generated",
        )

        message = get_user_facing_message(error)

        assert "output token" in message.lower() or "limit" in message.lower()

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


class TestDeadLetterQueueCleanup:
    """Tests for DLQ stale error cleanup."""

    def test_cleanup_removes_files_older_than_ttl(self, tmp_path):
        from datetime import UTC, datetime, timedelta

        dlq = DeadLetterQueue(tmp_path)
        old_error = ExtractionError(
            path="/test/old.pdf",
            error_type=ErrorType.MALFORMED_FILE,
            message="Old error",
        )
        dlq.enqueue(old_error)

        # Manually set mtime to 10 days ago
        old_file = list(tmp_path.glob("*.error.json"))[0]
        old_time = (datetime.now(UTC) - timedelta(days=10)).timestamp()
        old_file.touch()
        import os

        os.utime(old_file, (old_time, old_time))

        removed = dlq.cleanup_stale_errors(max_age_days=7, max_files=100)
        assert removed == 1
        assert dlq.size() == 0

    def test_cleanup_respects_max_files(self, tmp_path):
        dlq = DeadLetterQueue(tmp_path)
        for i in range(5):
            error = ExtractionError(
                path=f"/test/doc{i}.pdf",
                error_type=ErrorType.MALFORMED_FILE,
                message="Error",
            )
            dlq.enqueue(error)

        removed = dlq.cleanup_stale_errors(max_age_days=365, max_files=2)
        assert removed == 3
        assert dlq.size() == 2

    def test_cleanup_triggered_after_processing(self, tmp_path):
        from unittest.mock import MagicMock, patch

        from codebase_rag.document.document_updater import DocumentGraphUpdater

        provider = MagicMock()
        ingestor = MagicMock()
        ingestor.node_buffer = []
        ingestor._rel_count = 0
        ingestor.fetch_all.return_value = [{"count": 0}]

        with patch(
            "codebase_rag.document.document_updater.get_embedding_provider",
            return_value=provider,
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

        with patch.object(updater, "_embedding_provider", provider):
            with patch(
                "codebase_rag.document.document_updater.MemgraphIngestor.__enter__",
                return_value=ingestor,
            ):
                with patch(
                    "codebase_rag.document.document_updater._check_graph_availability"
                ):
                    with patch.object(updater, "_collect_documents", return_value=[]):
                        with patch.object(
                            updater.dead_letter_queue,
                            "cleanup_stale_errors",
                            return_value=0,
                        ) as mock_cleanup:
                            updater.run()
                            mock_cleanup.assert_called_once()

    def test_warning_emitted_when_size_exceeds_threshold(self, tmp_path):
        from unittest.mock import patch

        from codebase_rag.config import settings

        dlq = DeadLetterQueue(tmp_path)
        for i in range(settings.DOC_ERRORS_WARNING_THRESHOLD + 1):
            error = ExtractionError(
                path=f"/test/doc{i}.pdf",
                error_type=ErrorType.MALFORMED_FILE,
                message="Error",
            )
            dlq.enqueue(error)

        with patch("loguru.logger.warning") as mock_warning:
            # Simulate _cleanup_dlq behavior: check size after cleanup
            dlq.cleanup_stale_errors(max_age_days=365, max_files=10000)
            queue_size = dlq.size()
            if queue_size > settings.DOC_ERRORS_WARNING_THRESHOLD:
                from loguru import logger

                from codebase_rag.document import logs as doc_ls

                logger.warning(doc_ls.DOC_DLQ_SIZE_WARNING.format(count=queue_size))
            mock_warning.assert_called_once()


class TestBoundedDeadLetterQueue:
    """Tests for bounded DLQ with deduplication."""

    def test_bounded_dlq_drops_after_limit(self, tmp_path):
        dlq = DeadLetterQueue(tmp_path)
        dlq._max_size = 2
        dlq._seen_paths.clear()
        for i in range(5):
            error = ExtractionError(
                path=f"/test/doc{i}.pdf",
                error_type=ErrorType.MALFORMED_FILE,
                message="Error",
            )
            dlq.enqueue(error)
        assert dlq.size() == 2
        assert len(dlq._seen_paths) == 2

    def test_dlq_dedup_same_path(self, tmp_path):
        dlq = DeadLetterQueue(tmp_path)
        error = ExtractionError(
            path="/test/doc.pdf",
            error_type=ErrorType.MALFORMED_FILE,
            message="Error 1",
        )
        dlq.enqueue(error)
        dlq.enqueue(error)
        assert dlq.size() == 1


class TestDocumentUpdaterSkipValidation:
    """Tests that DocumentGraphUpdater passes skip_validation=True to extractors."""

    def test_process_document_passes_skip_validation(self, tmp_path):
        """_process_document should pass skip_validation=True after pre-check."""
        from unittest.mock import MagicMock, patch

        from codebase_rag.document.document_updater import DocumentGraphUpdater

        existing_file = tmp_path / "doc.md"
        existing_file.write_text("# Test")

        mock_extractor = MagicMock()
        mock_extractor.extract.return_value = MagicMock(
            code_references=[], sections=[], metadata={}
        )

        with (
            patch(
                "codebase_rag.document.document_updater.get_extractor_for_file",
                return_value=mock_extractor,
            ),
            patch.object(
                DocumentGraphUpdater, "_resolve_code_reference_names", return_value=[]
            ),
            patch.object(
                DocumentGraphUpdater,
                "_prepare_embeddings_with_fallback",
                return_value={},
            ),
            patch.object(DocumentGraphUpdater, "_delete_document_nodes"),
            patch.object(
                DocumentGraphUpdater,
                "_store_document",
                return_value=({"sections": 1}, {}, "2024-01-01"),
            ),
            patch.object(
                DocumentGraphUpdater, "_store_chunks_with_embeddings", return_value=0
            ),
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)
            updater.version_cache = MagicMock()
            updater.version_tracker = MagicMock()
            updater.version_tracker.needs_reindex.return_value = (True, None)
            updater.chunker = MagicMock()
            updater.chunker.chunk_document.return_value = []

            result = updater._process_document(
                existing_file, MagicMock(), concept_ingestor=MagicMock()
            )

        assert result == "indexed"
        mock_extractor.extract.assert_called_once_with(
            existing_file, skip_validation=True
        )

    def test_process_document_async_passes_skip_validation(self, tmp_path):
        """_process_document_async should pass skip_validation=True after pre-check."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from codebase_rag.document.document_updater import DocumentGraphUpdater

        existing_file = tmp_path / "doc.md"
        existing_file.write_text("# Test")

        mock_extractor = MagicMock()
        mock_extractor.extract_async = AsyncMock(
            return_value=MagicMock(code_references=[], sections=[], metadata={})
        )

        with (
            patch(
                "codebase_rag.document.document_updater.get_extractor_for_file",
                return_value=mock_extractor,
            ),
            patch.object(
                DocumentGraphUpdater, "_resolve_code_reference_names", return_value=[]
            ),
            patch.object(
                DocumentGraphUpdater,
                "_prepare_embeddings_with_fallback",
                return_value={},
            ),
            patch.object(DocumentGraphUpdater, "_delete_document_nodes"),
            patch.object(
                DocumentGraphUpdater,
                "_store_document",
                return_value=({"sections": 1}, {}, "2024-01-01"),
            ),
            patch.object(
                DocumentGraphUpdater, "_store_chunks_with_embeddings", return_value=0
            ),
        ):
            updater = DocumentGraphUpdater("localhost", 7688, tmp_path)
            updater.version_cache = MagicMock()
            updater.version_tracker = MagicMock()
            updater.version_tracker.needs_reindex.return_value = (True, None)
            updater.chunker = MagicMock()
            updater.chunker.chunk_document.return_value = []

            import asyncio

            result = asyncio.run(
                updater._process_document_async(
                    existing_file, MagicMock(), concept_ingestor=MagicMock()
                )
            )

        assert result == "indexed"
        mock_extractor.extract_async.assert_called_once_with(
            existing_file, skip_validation=True
        )


class TestDeadLetterQueueRetryWithBackoff:
    """Tests for DLQ retry_with_backoff behavior."""

    @pytest.mark.asyncio
    async def test_retry_skips_missing_files_gracefully(self, tmp_path):
        """DLQ retry should remove entries for files that no longer exist."""
        from datetime import UTC, datetime, timedelta
        from unittest.mock import MagicMock

        from codebase_rag.document.extractors.base import BaseDocumentExtractor

        dlq = DeadLetterQueue(tmp_path)
        error = ExtractionError(
            path="/nonexistent/path/doc.pdf",
            error_type=ErrorType.CONCEPT_CONTEXT_OVERFLOW,
            message="Context overflow",
            retry_after=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        )
        dlq.enqueue(error)

        class MockExtractor(BaseDocumentExtractor):
            def supported_extensions(self) -> list[str]:
                return [".pdf"]

            def _extract(self, file_path):
                return MagicMock()

            async def _extract_async(self, file_path):
                return MagicMock()

        extractor = MockExtractor()
        results = await dlq.retry_with_backoff(extractor, max_concurrent=1)

        assert results == {"/nonexistent/path/doc.pdf": True}
        assert dlq.size() == 0

    @pytest.mark.asyncio
    async def test_retry_extracts_existing_files(self, tmp_path):
        """DLQ retry should process entries for files that still exist."""
        from datetime import UTC, datetime, timedelta
        from unittest.mock import MagicMock

        from codebase_rag.document.extractors.base import BaseDocumentExtractor

        dlq = DeadLetterQueue(tmp_path)
        existing_file = tmp_path / "existing.pdf"
        existing_file.write_text("test content")

        error = ExtractionError(
            path=str(existing_file),
            error_type=ErrorType.CONCEPT_CONTEXT_OVERFLOW,
            message="Context overflow",
            retry_after=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
        )
        dlq.enqueue(error)

        class MockExtractor(BaseDocumentExtractor):
            def supported_extensions(self) -> list[str]:
                return [".pdf"]

            def _extract(self, file_path):
                return MagicMock()

            async def _extract_async(self, file_path):
                return MagicMock()

        extractor = MockExtractor()
        results = await dlq.retry_with_backoff(extractor, max_concurrent=1)

        assert results == {str(existing_file): True}
        assert dlq.size() == 0
