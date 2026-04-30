"""Tests for proactive chunk splitting in concept extraction."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from codebase_rag.document.concept_extraction import (
    ExtractedConcept,
    ExtractionResult,
    LLMConceptExtractor,
    _should_log_verb_warning,
    _should_proactive_split,
    estimate_input_tokens,
    resolve_category,
)


class TestEstimateInputTokens:
    """Input token estimation using tiktoken."""

    def test_empty_content(self) -> None:
        assert estimate_input_tokens("") == 0

    def test_short_text(self) -> None:
        text = "Hello world"
        tokens = estimate_input_tokens(text)
        assert tokens > 0
        assert tokens < 10

    def test_long_text(self) -> None:
        text = "word " * 1000
        tokens = estimate_input_tokens(text)
        assert tokens > 500
        assert tokens < 2000


class TestProactiveSplit:
    """Proactive split behavior in extract_with_retry."""

    @pytest.fixture
    def mock_settings(self):
        with patch("codebase_rag.config.settings") as mock_settings:
            mock_settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT = 100
            mock_settings.DOC_CONCEPT_MAX_SPLIT_DEPTH = 5
            mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 100
            mock_settings.DOC_CHUNK_SIZE = 4000
            mock_settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS = 4096
            mock_settings.DOC_CONCEPT_MIN_OUTPUT_TOKENS = 1024
            mock_settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS = 16384
            mock_settings.DOC_CONCEPT_TOKENS_PER_CHAR = 0.5
            mock_settings.DOC_CONCEPT_EXTRACTION_MAX_RETRIES = 2
            mock_settings.DOC_CONCEPT_EXTRACTION_RETRY_DELAY = 1.0
            mock_settings.DOC_CONCEPT_FAST_FAIL_TIMEOUT = 10.0
            mock_settings.DOC_CONCEPT_TIMEOUT_RETRY_MULTIPLIER = 1.5
            mock_settings.DOC_CONCEPT_TIMEOUT_RETRY_CAP_MULTIPLIER = 3.0
            mock_settings.DOC_CONCEPT_CONSECUTIVE_TIMEOUT_THRESHOLD = 3
            mock_settings.DOC_CONCEPT_TIMEOUT_RETRY_DELAY_MULTIPLIER = 1.5
            mock_settings.DOC_CONCEPT_MIN_CONFIDENCE = 0.7
            mock_settings.DOC_CONCEPT_MIN_RELATIONSHIP_STRENGTH = 0.6
            yield mock_settings

    @pytest.fixture
    def success_result(self):
        return ExtractionResult(
            concepts=[
                ExtractedConcept(
                    name="Concept1",
                    definition="A test concept",
                    confidence=0.9,
                    entity_category="ABSTRACT_CONCEPT",
                )
            ],
            relationships=[],
        )

    @pytest.fixture
    def empty_result(self):
        return ExtractionResult()

    async def _fake_extract(
        self, chunk_content, chunk_qn, timeout=None, max_tokens=None
    ):
        return ExtractionResult(
            concepts=[
                ExtractedConcept(
                    name="Concept1",
                    definition="A test concept",
                    confidence=0.9,
                    entity_category="ABSTRACT_CONCEPT",
                )
            ],
            relationships=[],
        )

    def test_proactive_split_triggered_when_over_limit(
        self, mock_settings, success_result
    ):
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        call_count = 0

        async def counting_extract(
            self, chunk_content, chunk_qn, timeout=None, max_tokens=None
        ):
            nonlocal call_count
            call_count += 1
            return success_result

        with patch.object(LLMConceptExtractor, "extract", counting_extract):
            import asyncio

            large_content = "\n\n".join(
                f"Paragraph {i} with substantial content to ensure token count is high enough. "
                "Each paragraph needs to exceed the minimum chunk size to avoid filtering. "
                * 50
                for i in range(200)
            )

            result = asyncio.run(
                extractor.extract_with_retry(large_content, "test_chunk")
            )

        assert result.concepts[0].name == "Concept1"
        assert call_count > 1

    def test_no_proactive_split_when_under_limit(self, mock_settings, success_result):
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        call_count = 0

        async def counting_extract(
            self, chunk_content, chunk_qn, timeout=None, max_tokens=None
        ):
            nonlocal call_count
            call_count += 1
            return success_result

        with patch.object(LLMConceptExtractor, "extract", counting_extract):
            import asyncio

            result = asyncio.run(
                extractor.extract_with_retry("Short content.", "test_chunk")
            )

        assert result.concepts[0].name == "Concept1"
        assert call_count == 1

    def test_recursive_split_respects_max_depth(self, mock_settings, empty_result):
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        mock_settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT = 10
        mock_settings.DOC_CONCEPT_MAX_SPLIT_DEPTH = 2
        mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 50

        call_count = 0

        async def counting_extract(
            self, chunk_content, chunk_qn, timeout=None, max_tokens=None
        ):
            nonlocal call_count
            call_count += 1
            return empty_result

        with patch.object(LLMConceptExtractor, "extract", counting_extract):
            import asyncio

            content = "A. " * 5000
            asyncio.run(extractor.extract_with_retry(content, "test_chunk"))

        assert call_count <= 8

    def test_proactive_split_returns_results_when_successful(
        self, mock_settings, success_result
    ):
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        mock_settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT = 5
        mock_settings.DOC_CONCEPT_MAX_SPLIT_DEPTH = 5
        mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 10

        with patch.object(LLMConceptExtractor, "extract", self._fake_extract):
            import asyncio

            content = "Paragraph one with enough content.\n\nParagraph two with enough content."
            result = asyncio.run(extractor.extract_with_retry(content, "test_chunk"))

        assert len(result.concepts) > 0
        assert result.concepts[0].name == "Concept1"

    def test_proactive_split_fallback_to_dlq_when_all_fail(
        self, mock_settings, empty_result
    ):
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        mock_settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT = 5
        mock_settings.DOC_CONCEPT_MAX_SPLIT_DEPTH = 2
        mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 10
        mock_settings.DOC_CONCEPT_EXTRACTION_MAX_RETRIES = 0

        async def failing_extract(
            self, chunk_content, chunk_qn, timeout=None, max_tokens=None
        ):
            raise RuntimeError("Simulated extraction failure")

        dlq_calls = []

        class MockDLQ:
            def enqueue(self, error):
                dlq_calls.append(error)

        mock_dlq = MockDLQ()

        with patch.object(LLMConceptExtractor, "extract", failing_extract):
            import asyncio

            content = "Paragraph one with enough content to trigger split.\n\nParagraph two with enough content."
            result = asyncio.run(
                extractor.extract_with_retry(
                    content, "test_chunk", dead_letter_queue=mock_dlq
                )
            )

        assert result.concepts == []
        assert result.relationships == []
        assert len(dlq_calls) == 1
        assert dlq_calls[0].chunk_qn == "test_chunk"
        assert dlq_calls[0].recoverable is True


class TestOutputTokenLimitAdaptiveRetry:
    """Test symmetric adaptive token budget for CONCEPT_OUTPUT_TOKEN_LIMIT."""

    @pytest.fixture
    def mock_settings(self):
        with patch("codebase_rag.config.settings") as mock_settings:
            mock_settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT = 100000
            mock_settings.DOC_CONCEPT_MAX_SPLIT_DEPTH = 5
            mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 100
            mock_settings.DOC_CHUNK_SIZE = 4000
            mock_settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS = 4096
            mock_settings.DOC_CONCEPT_MIN_OUTPUT_TOKENS = 1024
            mock_settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS = 16384
            mock_settings.DOC_CONCEPT_TOKENS_PER_CHAR = 0.5
            mock_settings.DOC_CONCEPT_EXTRACTION_MAX_RETRIES = 2
            mock_settings.DOC_CONCEPT_EXTRACTION_RETRY_DELAY = 0.1
            mock_settings.DOC_CONCEPT_FAST_FAIL_TIMEOUT = 10.0
            mock_settings.DOC_CONCEPT_TIMEOUT_RETRY_MULTIPLIER = 1.5
            mock_settings.DOC_CONCEPT_TIMEOUT_RETRY_CAP_MULTIPLIER = 3.0
            mock_settings.DOC_CONCEPT_CONSECUTIVE_TIMEOUT_THRESHOLD = 3
            mock_settings.DOC_CONCEPT_TIMEOUT_RETRY_DELAY_MULTIPLIER = 1.5
            mock_settings.DOC_CONCEPT_MIN_CONFIDENCE = 0.7
            mock_settings.DOC_CONCEPT_MIN_RELATIONSHIP_STRENGTH = 0.6
            yield mock_settings

    def test_output_token_limit_doubles_max_tokens_on_retry(self, mock_settings):
        """CONCEPT_OUTPUT_TOKEN_LIMIT should double max_tokens like CONTEXT_OVERFLOW."""
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        call_count = 0
        token_values = []

        async def token_tracking_extract(
            self, chunk_content, chunk_qn, timeout=None, max_tokens=None
        ):
            nonlocal call_count
            call_count += 1
            token_values.append(max_tokens)
            if call_count < 2:
                raise Exception("output token limit exceeded for this model")
            return ExtractionResult(
                concepts=[
                    ExtractedConcept(
                        name="Concept1",
                        definition="A test concept",
                        confidence=0.9,
                        entity_category="ABSTRACT_CONCEPT",
                    )
                ],
                relationships=[],
            )

        with patch.object(LLMConceptExtractor, "extract", token_tracking_extract):
            import asyncio

            result = asyncio.run(
                extractor.extract_with_retry("Some content here.", "test_chunk")
            )

        assert result.concepts[0].name == "Concept1"
        assert len(token_values) == 2
        assert token_values[1] > token_values[0]

    def test_output_token_limit_respects_model_max_cap(self, mock_settings):
        """CONCEPT_OUTPUT_TOKEN_LIMIT retry should cap at DOC_CONCEPT_MAX_OUTPUT_TOKENS."""
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        call_count = 0
        token_values = []

        async def token_tracking_extract(
            self, chunk_content, chunk_qn, timeout=None, max_tokens=None
        ):
            nonlocal call_count
            call_count += 1
            token_values.append(max_tokens)
            raise Exception("output token limit exceeded")

        with patch.object(LLMConceptExtractor, "extract", token_tracking_extract):
            import asyncio

            asyncio.run(
                extractor.extract_with_retry("Some content here.", "test_chunk")
            )

        assert all(
            t <= mock_settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS for t in token_values
        )

    def test_output_token_limit_exhausts_retries_and_stops(self, mock_settings):
        """CONCEPT_OUTPUT_TOKEN_LIMIT must not retry beyond max_retries."""
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        call_count = 0

        async def counting_extract(
            self, chunk_content, chunk_qn, timeout=None, max_tokens=None
        ):
            nonlocal call_count
            call_count += 1
            raise Exception("output token limit exceeded")

        with patch.object(LLMConceptExtractor, "extract", counting_extract):
            import asyncio

            asyncio.run(
                extractor.extract_with_retry("Some content here.", "test_chunk")
            )

        assert call_count == mock_settings.DOC_CONCEPT_EXTRACTION_MAX_RETRIES + 1


class TestShouldLogVerbWarning:
    """Tests for warning de-duplication in resolve_category."""

    @pytest.fixture(autouse=True)
    def reset_warning_set(self):
        """Reset the module-level warning set before each test for hermeticity."""
        from codebase_rag.document.concept_extraction import _logged_verb_warnings

        _logged_verb_warnings.clear()

    def test_logs_first_occurrence(self):
        """First occurrence of a verb conflict should be logged."""
        assert _should_log_verb_warning("supports", "CAUSAL", "ATTRIBUTIVE") is True

    def test_suppresses_exact_duplicate(self):
        """Exact same conflict should not be logged twice."""
        _should_log_verb_warning("supports", "CAUSAL", "ATTRIBUTIVE")
        assert _should_log_verb_warning("supports", "CAUSAL", "ATTRIBUTIVE") is False

    def test_different_declared_category_still_logged(self):
        """Same verb with different declared category is a distinct conflict."""
        _should_log_verb_warning("supports", "CAUSAL", "ATTRIBUTIVE")
        assert _should_log_verb_warning("supports", "SEQUENTIAL", "ATTRIBUTIVE") is True

    def test_different_registry_category_still_logged(self):
        """Same verb+declared with different registry category is distinct."""
        _should_log_verb_warning("supports", "CAUSAL", "ATTRIBUTIVE")
        assert _should_log_verb_warning("supports", "CAUSAL", "HIERARCHICAL") is True

    def test_different_verb_is_distinct(self):
        """Different verbs are always distinct conflicts."""
        _should_log_verb_warning("supports", "CAUSAL", "ATTRIBUTIVE")
        assert _should_log_verb_warning("constrains", "CONTEXTUAL", "CAUSAL") is True

    def test_case_insensitive_deduplication(self):
        """Deduplication should be case-insensitive."""
        _should_log_verb_warning("Supports", "causal", "attributive")
        assert _should_log_verb_warning("SUPPORTS", "CAUSAL", "ATTRIBUTIVE") is False


class TestShouldProactiveSplit:
    """Tests for pattern-based proactive split detection."""

    def test_long_single_line_triggers_split(self):
        """A single line longer than 2000 chars should trigger proactive split."""
        content = "x" * 2500
        assert _should_proactive_split(content) is True

    def test_normal_markdown_no_split(self):
        """Normal markdown with short lines should not trigger split."""
        content = "# Heading\n\nShort paragraph.\n\n- Item 1\n- Item 2"
        assert _should_proactive_split(content) is False

    def test_dense_content_triggers_split(self):
        """Very long content with many commas should trigger split.

        The length gate (15000 chars) prevents false positives on short
        dense content, so a 180-char string with 60 commas is ignored.
        """
        short_dense = "a, " * 60
        assert _should_proactive_split(short_dense) is False

        long_dense = "value1, value2, value3, " * 600
        assert _should_proactive_split(long_dense) is True

    def test_table_like_content_triggers_split(self):
        """Wide table-like lines should trigger split."""
        lines = ["| " + "col |" * 500 for _ in range(5)]
        content = "\n".join(lines)
        assert _should_proactive_split(content) is True

    def test_short_content_no_split(self):
        """Short content should never trigger split."""
        content = "Just a short sentence."
        assert _should_proactive_split(content) is False


class TestDebugLogOnSuppressedWarning:
    """Integration tests for DEBUG log emission when warnings are suppressed."""

    @pytest.fixture(autouse=True)
    def reset_warning_set(self):
        """Reset the module-level warning set before each test for hermeticity."""
        from codebase_rag.document.concept_extraction import _logged_verb_warnings

        _logged_verb_warnings.clear()

    def test_suppressed_warning_emits_debug_log(self):
        """Second occurrence of same verb conflict should emit DEBUG log with template."""
        from unittest.mock import patch

        with (
            patch(
                "codebase_rag.document.concept_extraction._load_learned_verbs",
                return_value={},
            ),
            patch("loguru.logger") as mock_logger,
        ):
            resolve_category("causes", "COMPOSITIONAL")
            resolve_category("causes", "COMPOSITIONAL")

        assert mock_logger.warning.call_count == 1
        mock_logger.debug.assert_called_once_with(
            "Verb 'causes' registry override (suppressed): "
            "LLM declared 'COMPOSITIONAL', registry says 'CAUSAL'"
        )


class TestProactiveSplitUnderTokenLimit:
    """Proactive split triggers even when token count is under limit."""

    @pytest.fixture
    def mock_settings(self):
        with patch("codebase_rag.config.settings") as mock_settings:
            mock_settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT = 100000
            mock_settings.DOC_CONCEPT_MAX_SPLIT_DEPTH = 5
            mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 100
            mock_settings.DOC_CHUNK_SIZE = 4000
            mock_settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS = 4096
            mock_settings.DOC_CONCEPT_MIN_OUTPUT_TOKENS = 1024
            mock_settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS = 16384
            mock_settings.DOC_CONCEPT_TOKENS_PER_CHAR = 0.5
            mock_settings.DOC_CONCEPT_EXTRACTION_MAX_RETRIES = 2
            mock_settings.DOC_CONCEPT_EXTRACTION_RETRY_DELAY = 1.0
            mock_settings.DOC_CONCEPT_FAST_FAIL_TIMEOUT = 10.0
            mock_settings.DOC_CONCEPT_TIMEOUT_RETRY_MULTIPLIER = 1.5
            mock_settings.DOC_CONCEPT_TIMEOUT_RETRY_CAP_MULTIPLIER = 3.0
            mock_settings.DOC_CONCEPT_CONSECUTIVE_TIMEOUT_THRESHOLD = 3
            mock_settings.DOC_CONCEPT_TIMEOUT_RETRY_DELAY_MULTIPLIER = 1.5
            mock_settings.DOC_CONCEPT_MIN_CONFIDENCE = 0.7
            mock_settings.DOC_CONCEPT_MIN_RELATIONSHIP_STRENGTH = 0.6
            yield mock_settings

    def test_proactive_split_triggers_when_under_token_limit(self, mock_settings):
        """_should_proactive_split should trigger split even when estimated_tokens <= input_limit."""
        import asyncio

        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        call_args = []

        async def tracking_extract(
            self, chunk_content, chunk_qn, timeout=None, max_tokens=None
        ):
            call_args.append((chunk_content, chunk_qn))
            return ExtractionResult(
                concepts=[
                    ExtractedConcept(
                        name="Concept1",
                        definition="A test concept",
                        confidence=0.9,
                        entity_category="ABSTRACT_CONCEPT",
                    )
                ],
                relationships=[],
            )

        with patch.object(LLMConceptExtractor, "extract", tracking_extract):
            content = "x" * 2500
            assert estimate_input_tokens(content) < mock_settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT
            result = asyncio.run(
                extractor.extract_with_retry(content, "test_chunk")
            )

        assert result.concepts[0].name == "Concept1"
        assert any("_part" in qn for _, qn in call_args)
