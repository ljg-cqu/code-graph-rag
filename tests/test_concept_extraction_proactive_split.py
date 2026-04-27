"""Tests for proactive chunk splitting in concept extraction."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from codebase_rag.document.concept_extraction import (
    ExtractedConcept,
    ExtractionResult,
    LLMConceptExtractor,
    estimate_input_tokens,
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

    async def _fake_extract(self, chunk_content, chunk_qn, timeout=None, max_tokens=None):
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

    def test_proactive_split_triggered_when_over_limit(self, mock_settings, success_result):
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        call_count = 0

        async def counting_extract(self, chunk_content, chunk_qn, timeout=None, max_tokens=None):
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

        async def counting_extract(self, chunk_content, chunk_qn, timeout=None, max_tokens=None):
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

        async def counting_extract(self, chunk_content, chunk_qn, timeout=None, max_tokens=None):
            nonlocal call_count
            call_count += 1
            return empty_result

        with patch.object(LLMConceptExtractor, "extract", counting_extract):
            import asyncio

            content = "A. " * 5000
            result = asyncio.run(
                extractor.extract_with_retry(content, "test_chunk")
            )

        assert call_count <= 8

    def test_proactive_split_returns_results_when_successful(self, mock_settings, success_result):
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        mock_settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT = 5
        mock_settings.DOC_CONCEPT_MAX_SPLIT_DEPTH = 5
        mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 10

        with patch.object(LLMConceptExtractor, "extract", self._fake_extract):
            import asyncio

            content = "Paragraph one with enough content.\n\nParagraph two with enough content."
            result = asyncio.run(
                extractor.extract_with_retry(content, "test_chunk")
            )

        assert len(result.concepts) > 0
        assert result.concepts[0].name == "Concept1"

    def test_proactive_split_fallback_to_dlq_when_all_fail(self, mock_settings, empty_result):
        extractor = LLMConceptExtractor(timeout=30.0, max_timeout=120.0)
        extractor._circuit_breaker = None

        mock_settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT = 5
        mock_settings.DOC_CONCEPT_MAX_SPLIT_DEPTH = 2
        mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 10
        mock_settings.DOC_CONCEPT_EXTRACTION_MAX_RETRIES = 0

        async def failing_extract(self, chunk_content, chunk_qn, timeout=None, max_tokens=None):
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
                extractor.extract_with_retry(content, "test_chunk", dead_letter_queue=mock_dlq)
            )

        assert result.concepts == []
        assert result.relationships == []
        assert len(dlq_calls) == 1
        assert dlq_calls[0].chunk_qn == "test_chunk"
        assert dlq_calls[0].recoverable is True
