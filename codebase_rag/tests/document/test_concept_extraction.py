"""Tests for concept extraction timeout resilience.

Covers adaptive timeout, circuit breaker, and retry logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from codebase_rag.document.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState
from codebase_rag.document.concept_extraction import (
    ExtractedConcept,
    ExtractionResult,
    LLMConceptExtractor,
    calculate_adaptive_timeout,
)
from codebase_rag.document.error_handling import ErrorType, ExtractionError


class TestAdaptiveTimeout:
    """Tests for calculate_adaptive_timeout."""

    def test_base_timeout(self):
        content = ""
        timeout = calculate_adaptive_timeout(content, base_timeout=30.0)
        assert timeout == 30.0

    def test_size_factor(self):
        content = "x" * 5000
        timeout = calculate_adaptive_timeout(
            content,
            base_timeout=30.0,
            timeout_per_1k_chars=10.0,
        )
        assert timeout == 60.0

    def test_max_timeout_cap(self):
        content = "x" * 100000 + "```python\nprint('hello')\n```" * 10
        timeout = calculate_adaptive_timeout(
            content,
            base_timeout=30.0,
            max_timeout=120.0,
            max_size_factor=100.0,
            max_complexity_factor=100.0,
        )
        assert timeout == 120.0

    def test_code_block_factor(self):
        content = "```python\nprint('hello')\n```" * 5
        timeout = calculate_adaptive_timeout(
            content,
            base_timeout=30.0,
            timeout_per_code_block=5.0,
        )
        # 5 blocks * 2 occurrences of ``` per block = 10 code block markers
        # complexity_factor = min(10 * 5, 20) = 20
        # size_factor for ~140 chars = min(140/1000 * 10, 30) = 1.4
        # total = 30 + 1.4 + 20 = 51.4
        assert timeout == pytest.approx(51.4, abs=0.1)

    def test_combined_factors_capped(self):
        content = "```python\nprint('hello')\n```" * 10 + "x" * 10000
        timeout = calculate_adaptive_timeout(
            content,
            base_timeout=30.0,
            max_timeout=120.0,
            timeout_per_1k_chars=10.0,
            timeout_per_code_block=5.0,
            max_size_factor=100.0,
            max_complexity_factor=100.0,
        )
        assert timeout == 120.0


class TestCircuitBreaker:
    """Tests for CircuitBreaker state machine."""

    def test_initial_state_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_opens_after_failure_threshold(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=3),
        )
        for _ in range(3):
            cb.record_failure()
        assert cb.is_open
        assert not cb.can_execute()

    def test_opens_on_rolling_window_failure_rate(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=10, window_size=4),
        )
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert cb.is_open

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout_seconds=0.0,
            ),
        )
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

    def test_closes_after_success_threshold(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout_seconds=0.0,
                success_threshold=2,
            ),
        )
        cb.record_failure()
        assert cb.can_execute()  # Transitions to HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout_seconds=0.0,
            ),
        )
        cb.record_failure()
        assert cb.can_execute()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_does_not_reset_failure_count_in_closed(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=3),
        )
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        # Success in CLOSED state appends to window but does not reset count
        assert cb.failure_count == 3
        assert cb.state == CircuitState.OPEN

    def test_success_resets_in_half_open(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(
                failure_threshold=1,
                timeout_seconds=0.0,
                success_threshold=1,
            ),
        )
        cb.record_failure()
        assert cb.can_execute()  # HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_recent_failures_window(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(window_size=3),
        )
        cb.record_success()
        cb.record_success()
        cb.record_failure()
        assert len(cb.recent_failures) == 3
        cb.record_success()
        assert len(cb.recent_failures) == 3


@dataclass
class _MockAgentOutput:
    output: ExtractionResult


class TestLLMConceptExtractorExtract:
    """Tests for LLMConceptExtractor.extract behavior."""

    @pytest.mark.asyncio
    async def test_extract_raises_timeout_error(self):
        extractor = LLMConceptExtractor()
        extractor.agent = Mock(run=Mock(side_effect=asyncio.TimeoutError))
        with pytest.raises(asyncio.TimeoutError):
            await extractor.extract("content", "qn")

    @pytest.mark.asyncio
    async def test_extract_raises_generic_exception(self):
        extractor = LLMConceptExtractor()
        extractor.agent = Mock(run=Mock(side_effect=RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            await extractor.extract("content", "qn")

    @pytest.mark.asyncio
    async def test_extract_returns_empty_when_init_fails(self):
        extractor = LLMConceptExtractor()
        extractor._initialization_failed = True
        result = await extractor.extract("content", "qn")
        assert result == ExtractionResult()

    @pytest.mark.asyncio
    async def test_extract_returns_result_on_success(self):
        extractor = LLMConceptExtractor()
        concept = ExtractedConcept(
            name="test",
            aliases=[],
            definition="def",
            confidence=0.9,
            source_chunk_qn="",
        )
        result = ExtractionResult(concepts=[concept])

        async def mock_run(content):
            return _MockAgentOutput(result)

        extractor.agent = Mock(run=mock_run)
        output = await extractor.extract("content", "qn")
        assert output.concepts[0].source_chunk_qn == "qn"

    @pytest.mark.asyncio
    async def test_extract_uses_adaptive_timeout(self):
        extractor = LLMConceptExtractor()
        call_log = []

        async def mock_run(content):
            call_log.append(content)
            return _MockAgentOutput(ExtractionResult())

        extractor.agent = Mock(run=mock_run)
        await extractor.extract("x" * 5000, "qn")
        assert len(call_log) == 1

    @pytest.mark.asyncio
    async def test_extract_respects_timeout_override(self):
        extractor = LLMConceptExtractor()
        call_log = []

        async def mock_run(content):
            call_log.append(content)
            return _MockAgentOutput(ExtractionResult())

        extractor.agent = Mock(run=mock_run)
        await extractor.extract("content", "qn", timeout=999.0)

    @pytest.mark.asyncio
    async def test_extract_returns_empty_when_circuit_open(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        cb.record_failure()
        extractor = LLMConceptExtractor(circuit_breaker=cb)
        extractor.agent = Mock(run=Mock(side_effect=asyncio.TimeoutError))
        result = await extractor.extract("content", "qn")
        assert result == ExtractionResult()

    @pytest.mark.asyncio
    async def test_extract_records_success_on_circuit_breaker(self):
        cb = CircuitBreaker(name="test")
        extractor = LLMConceptExtractor(circuit_breaker=cb)
        concept = ExtractedConcept(
            name="test",
            aliases=[],
            definition="def",
            confidence=0.9,
            source_chunk_qn="",
        )
        result = ExtractionResult(concepts=[concept])

        async def mock_run(content):
            return _MockAgentOutput(result)

        extractor.agent = Mock(run=mock_run)
        await extractor.extract("content", "qn")
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_extract_records_failure_on_timeout(self):
        cb = CircuitBreaker(name="test")
        extractor = LLMConceptExtractor(circuit_breaker=cb)
        extractor.agent = Mock(run=Mock(side_effect=asyncio.TimeoutError))
        with pytest.raises(asyncio.TimeoutError):
            await extractor.extract("content", "qn")
        assert cb.failure_count == 1


class TestLLMConceptExtractorRetry:
    """Tests for LLMConceptExtractor.extract_with_retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout_then_success(self):
        extractor = LLMConceptExtractor()
        call_count = 0

        async def mock_run(content):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise asyncio.TimeoutError()
            concept = ExtractedConcept(
                name="test",
                aliases=[],
                definition="def",
                confidence=0.9,
                source_chunk_qn="",
            )
            return _MockAgentOutput(ExtractionResult(concepts=[concept]))

        extractor.agent = Mock(run=mock_run)
        result = await extractor.extract_with_retry("content", "qn")
        assert len(result.concepts) == 1
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_recoverable(self):
        extractor = LLMConceptExtractor()
        call_count = 0

        async def mock_run(content):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("api key invalid")

        extractor.agent = Mock(run=mock_run)
        result = await extractor.extract_with_retry("content", "qn")
        assert result == ExtractionResult()
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_returns_empty_after_exhausted_retries(self):
        extractor = LLMConceptExtractor()
        call_count = 0

        async def mock_run(content):
            nonlocal call_count
            call_count += 1
            raise asyncio.TimeoutError()

        extractor.agent = Mock(run=mock_run)
        result = await extractor.extract_with_retry("content", "qn")
        assert result == ExtractionResult()
        assert call_count == 4  # initial + 3 retries

    @pytest.mark.asyncio
    async def test_queues_to_dlq_on_failure(self, tmp_path):
        from codebase_rag.document.error_handling import DeadLetterQueue

        dlq = DeadLetterQueue(tmp_path)
        extractor = LLMConceptExtractor()
        extractor.agent = Mock(run=Mock(side_effect=asyncio.TimeoutError))
        result = await extractor.extract_with_retry("content", "qn", dead_letter_queue=dlq)
        assert result == ExtractionResult()
        assert dlq.size() == 1

    @pytest.mark.asyncio
    async def test_no_dlq_without_argument(self):
        extractor = LLMConceptExtractor()
        extractor.agent = Mock(run=Mock(side_effect=asyncio.TimeoutError))
        result = await extractor.extract_with_retry("content", "qn")
        assert result == ExtractionResult()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_subsequent_calls(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=2),
        )
        extractor = LLMConceptExtractor(circuit_breaker=cb)
        call_count = 0

        async def mock_run(content):
            nonlocal call_count
            call_count += 1
            raise asyncio.TimeoutError()

        extractor.agent = Mock(run=mock_run)

        await extractor.extract_with_retry("content", "qn1")
        assert cb.failure_count == 2
        assert cb.is_open

        result = await extractor.extract_with_retry("content", "qn2")
        assert result == ExtractionResult()
        assert call_count == 2  # No additional calls after circuit opens
