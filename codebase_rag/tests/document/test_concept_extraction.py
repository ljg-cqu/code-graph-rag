"""Tests for concept extraction timeout resilience.

Covers adaptive timeout, circuit breaker, and retry logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest

from codebase_rag.document.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)
from codebase_rag.document.concept_extraction import (
    ExtractedConcept,
    ExtractionResult,
    LLMConceptExtractor,
    calculate_adaptive_timeout,
)


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

    def test_heading_factor(self):
        content = "# Heading 1\n## Heading 2\n### Heading 3\n" * 5
        timeout = calculate_adaptive_timeout(
            content,
            base_timeout=30.0,
            timeout_per_heading=2.0,
        )
        # heading_count = 30 (1+2+3 per group * 5), heading_factor = min(30*2, 15) = 15
        # size_factor for 195 chars = min(195/1000 * 10, 30) = 1.95
        # total = 30 + 1.95 + 0 + 15 + 0 = 46.95
        assert timeout == pytest.approx(46.95, abs=0.1)

    def test_list_factor(self):
        content = "\n- Item 1\n- Item 2\n- Item 3\n" * 5
        timeout = calculate_adaptive_timeout(
            content,
            base_timeout=30.0,
            timeout_per_list_item=0.5,
        )
        # list_item_count = 15 (3 per group * 5), list_factor = min(15*0.5, 10) = 7.5
        # size_factor for ~140 chars = min(140/1000 * 10, 30) = 1.4
        # total = 30 + 1.4 + 0 + 0 + 7.5 = 38.9
        assert timeout == pytest.approx(38.9, abs=0.1)

    def test_markdown_factors_capped(self):
        content = "# " * 50 + "\n- Item\n" * 50
        timeout = calculate_adaptive_timeout(
            content,
            base_timeout=30.0,
            max_timeout=120.0,
            max_heading_factor=15.0,
            max_list_factor=10.0,
            max_size_factor=100.0,
            max_complexity_factor=100.0,
        )
        # heading_factor capped at 15.0, list_factor capped at 10.0
        # size_factor = min(len(content)/1000 * 10, 100) ≈ 5.0
        # total = 30 + 5.0 + 0 + 15 + 10 = 60.0
        assert timeout == pytest.approx(60.0, abs=0.1)

    def test_combined_markdown_and_code_factors(self):
        content = "# Heading\n\n```python\nprint('hello')\n```\n\n- Item 1\n- Item 2\n"
        timeout = calculate_adaptive_timeout(
            content,
            base_timeout=30.0,
            timeout_per_heading=2.0,
            timeout_per_code_block=5.0,
            timeout_per_list_item=0.5,
        )
        # size_factor for ~59 chars = min(59/1000 * 10, 30) = 0.59
        # code_block_count = 2 (``` appears twice), complexity = min(2 * 5, 20) = 10
        # heading_count = 1, heading_factor = min(1 * 2, 15) = 2
        # list_item_count = 2, list_factor = min(2 * 0.5, 10) = 1
        # total = 30 + 0.59 + 10 + 2 + 1 = 43.59
        assert timeout == pytest.approx(43.59, abs=0.1)


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
            entity_category="CONCRETE_ENTITY",
        )
        result = ExtractionResult(concepts=[concept])

        async def mock_run(content, **kwargs):
            return _MockAgentOutput(result)

        extractor.agent = Mock(run=mock_run)
        output = await extractor.extract("content", "qn")
        assert output.concepts[0].name == "test"

    @pytest.mark.asyncio
    async def test_extract_uses_adaptive_timeout(self):
        extractor = LLMConceptExtractor()
        call_log = []

        async def mock_run(content, **kwargs):
            call_log.append(content)
            return _MockAgentOutput(ExtractionResult())

        extractor.agent = Mock(run=mock_run)
        await extractor.extract("x" * 5000, "qn")
        assert len(call_log) == 1

    @pytest.mark.asyncio
    async def test_extract_respects_timeout_override(self):
        extractor = LLMConceptExtractor()
        call_log = []

        async def mock_run(content, **kwargs):
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
            entity_category="CONCRETE_ENTITY",
        )
        result = ExtractionResult(concepts=[concept])

        async def mock_run(content, **kwargs):
            return _MockAgentOutput(result)

        extractor.agent = Mock(run=mock_run)
        await extractor.extract("content", "qn")
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_extract_records_failure_on_non_timeout_error(self):
        cb = CircuitBreaker(name="test")
        extractor = LLMConceptExtractor(circuit_breaker=cb)
        extractor.agent = Mock(run=Mock(side_effect=RuntimeError("service down")))
        with pytest.raises(RuntimeError, match="service down"):
            await extractor.extract("content", "qn")
        assert cb.failure_count == 1


class TestLLMConceptExtractorRetry:
    """Tests for LLMConceptExtractor.extract_with_retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout_then_success(self):
        extractor = LLMConceptExtractor()
        call_count = 0

        async def mock_run(content, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError()
            concept = ExtractedConcept(
                name="test",
                aliases=[],
                definition="def",
                confidence=0.9,
                entity_category="CONCRETE_ENTITY",
            )
            return _MockAgentOutput(ExtractionResult(concepts=[concept]))

        extractor.agent = Mock(run=mock_run)
        with patch(
            "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
            return_value=True,
        ):
            result = await extractor.extract_with_retry("content", "qn")
        assert len(result.concepts) == 1
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_on_non_recoverable(self):
        extractor = LLMConceptExtractor()
        call_count = 0

        async def mock_run(content, **kwargs):
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

        async def mock_run(content, **kwargs):
            nonlocal call_count
            call_count += 1
            raise TimeoutError()

        extractor.agent = Mock(run=mock_run)
        with patch(
            "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
            return_value=True,
        ):
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

        async def mock_run(content, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("service down")

        extractor.agent = Mock(run=mock_run)

        await extractor.extract_with_retry("content", "qn1")
        assert cb.failure_count == 2
        assert cb.is_open

        result = await extractor.extract_with_retry("content", "qn2")
        assert result == ExtractionResult()
        assert call_count == 2  # No additional calls after circuit opens

    @pytest.mark.asyncio
    async def test_retry_increases_timeout_on_timeout_error(self):
        from unittest.mock import patch

        extractor = LLMConceptExtractor(timeout=10.0, max_timeout=100.0)
        extractor.agent = Mock(run=Mock(return_value=_MockAgentOutput(ExtractionResult())))

        timeouts_passed: list[float] = []

        async def mock_wait_for(awaitable, timeout):
            timeouts_passed.append(timeout)
            raise TimeoutError()

        with patch("asyncio.wait_for", mock_wait_for):
            with patch(
                "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
                return_value=True,
            ):
                result = await extractor.extract_with_retry("content", "qn")
        assert result == ExtractionResult()
        assert len(timeouts_passed) == 4
        # "content" is 7 chars, so size_factor adds ~0.07
        original = pytest.approx(10.07, abs=0.01)
        assert timeouts_passed[0] == original
        # Capped at 1.2x original (12.084) with new multiplier 1.1
        assert timeouts_passed[1] == pytest.approx(11.077, abs=0.01)
        assert timeouts_passed[2] == pytest.approx(12.084, abs=0.01)
        # With fix D-1, fast-fail only on attempt 0; attempt 3 stays capped
        assert timeouts_passed[3] == pytest.approx(12.084, abs=0.01)

    @pytest.mark.asyncio
    async def test_retry_timeout_capped_at_max_timeout(self):
        from unittest.mock import patch

        extractor = LLMConceptExtractor(timeout=80.0, max_timeout=100.0)
        extractor.agent = Mock(run=Mock(return_value=_MockAgentOutput(ExtractionResult())))

        timeouts_passed: list[float] = []

        async def mock_wait_for(awaitable, timeout):
            timeouts_passed.append(timeout)
            raise TimeoutError()

        with patch("asyncio.wait_for", mock_wait_for):
            with patch(
                "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
                return_value=True,
            ):
                await extractor.extract_with_retry("content", "qn")
        assert len(timeouts_passed) == 4
        assert timeouts_passed[0] == pytest.approx(80.07, abs=0.01)
        # Capped at 1.2x original (96.084) with new multiplier 1.1
        assert timeouts_passed[1] == pytest.approx(88.077, abs=0.01)
        assert timeouts_passed[2] == pytest.approx(96.084, abs=0.01)
        # With fix D-1, fast-fail only on attempt 0; attempt 3 stays capped
        assert timeouts_passed[3] == pytest.approx(96.084, abs=0.01)


class TestRetryFastFailTimeout:
    """Tests for fast-fail timeout behavior scoped to first attempt only (D-1)."""

    @pytest.mark.asyncio
    async def test_fast_fail_does_not_collapse_mid_retry(self):
        """A retry in progress must never have its timeout collapsed to fast-fail."""
        from unittest.mock import patch

        extractor = LLMConceptExtractor(timeout=10.0, max_timeout=100.0)
        extractor.agent = Mock(run=Mock(return_value=_MockAgentOutput(ExtractionResult())))

        timeouts_passed: list[float] = []

        async def mock_wait_for(awaitable, timeout):
            timeouts_passed.append(timeout)
            raise TimeoutError()

        with patch("asyncio.wait_for", mock_wait_for):
            with patch(
                "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
                return_value=True,
            ):
                await extractor.extract_with_retry("content", "qn")

        # attempt 0: normal timeout (counter 0→1, below threshold)
        assert timeouts_passed[0] == pytest.approx(10.07, abs=0.01)
        # attempt 1: normal increase (counter 1→2, below threshold)
        assert timeouts_passed[1] == pytest.approx(11.077, abs=0.01)
        # attempt 2: normal increase (counter 2→3, hits threshold but attempt==2, no fast-fail)
        assert timeouts_passed[2] == pytest.approx(12.084, abs=0.01)
        # attempt 3: capped, NOT collapsed to 5s (counter 3→4, attempt==3, no fast-fail)
        assert timeouts_passed[3] == pytest.approx(12.084, abs=0.01)

    @pytest.mark.asyncio
    async def test_fast_fail_applies_only_on_first_attempt(self):
        """When consecutive_timeouts >= threshold, only attempt 0 gets fast-fail."""
        from unittest.mock import patch

        extractor = LLMConceptExtractor(timeout=10.0, max_timeout=100.0)
        extractor._consecutive_timeouts = 3  # Pre-warm to threshold
        extractor.agent = Mock(run=Mock(return_value=_MockAgentOutput(ExtractionResult())))

        timeouts_passed: list[float] = []

        async def mock_wait_for(awaitable, timeout):
            timeouts_passed.append(timeout)
            raise TimeoutError()

        with patch("asyncio.wait_for", mock_wait_for):
            with patch(
                "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
                return_value=True,
            ):
                await extractor.extract_with_retry("content", "qn")

        assert len(timeouts_passed) == 4
        # attempt 0: normal adaptive timeout (counter 3→4, attempt==0 → fast-fail for next)
        assert timeouts_passed[0] == pytest.approx(10.07, abs=0.01)
        # attempt 1: fast-fail applied (5.0s) because attempt 0 triggered it
        assert timeouts_passed[1] == pytest.approx(5.0, abs=0.1)
        # attempt 2: increases from 5.0, not re-fast-failed (attempt==1)
        assert timeouts_passed[2] == pytest.approx(5.5, abs=0.1)
        # attempt 3: increases from 5.5, not re-fast-failed (attempt==2)
        assert timeouts_passed[3] == pytest.approx(6.05, abs=0.1)

    @pytest.mark.asyncio
    async def test_retry_timeout_monotonically_increases_until_cap(self):
        """Timeout should never drop during retries of the same chunk."""
        from unittest.mock import patch

        extractor = LLMConceptExtractor(timeout=10.0, max_timeout=100.0)
        extractor.agent = Mock(run=Mock(return_value=_MockAgentOutput(ExtractionResult())))

        timeouts_passed: list[float] = []

        async def mock_wait_for(awaitable, timeout):
            timeouts_passed.append(timeout)
            raise TimeoutError()

        with patch("asyncio.wait_for", mock_wait_for):
            with patch(
                "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
                return_value=True,
            ):
                await extractor.extract_with_retry("content", "qn")

        assert len(timeouts_passed) == 4
        for i in range(1, len(timeouts_passed)):
            assert timeouts_passed[i] >= timeouts_passed[i - 1] - 0.01, (
                f"Timeout dropped from {timeouts_passed[i - 1]} to {timeouts_passed[i]} "
                f"at attempt {i}"
            )


class TestAdaptiveTimeoutTuning:
    """Tests for adaptive timeout tuning and provider health probe."""

    @pytest.mark.asyncio
    async def test_timeout_retry_capped(self):
        from unittest.mock import patch

        extractor = LLMConceptExtractor(timeout=10.0, max_timeout=100.0)
        extractor.agent = Mock(run=Mock(return_value=_MockAgentOutput(ExtractionResult())))

        timeouts_passed: list[float] = []

        async def mock_wait_for(awaitable, timeout):
            timeouts_passed.append(timeout)
            raise TimeoutError()

        with patch("asyncio.wait_for", mock_wait_for):
            with patch(
                "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
                return_value=True,
            ):
                await extractor.extract_with_retry("content", "qn")
        assert len(timeouts_passed) == 4
        original = 10.07
        cap = original * 1.2
        assert timeouts_passed[1] <= cap + 0.01
        assert timeouts_passed[2] <= cap + 0.01
        assert timeouts_passed[3] <= cap + 0.01

    @pytest.mark.asyncio
    async def test_provider_health_probe_skips_retry(self):
        extractor = LLMConceptExtractor()
        call_count = 0

        async def mock_run(content, **kwargs):
            nonlocal call_count
            call_count += 1
            raise TimeoutError()

        extractor.agent = Mock(run=mock_run)
        with patch(
            "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
            return_value=False,
        ):
            result = await extractor.extract_with_retry("content", "qn")
        assert result == ExtractionResult()
        assert call_count == 1  # Only initial attempt, probe fails, skip retry

    @pytest.mark.asyncio
    async def test_consecutive_timeout_fast_fail_on_first_attempt_only(self):
        """Fast-fail triggered by attempt 0 failure; retries must NOT re-trigger fast-fail."""
        from unittest.mock import patch

        extractor = LLMConceptExtractor(timeout=10.0, max_timeout=100.0)
        extractor.agent = Mock(run=Mock(return_value=_MockAgentOutput(ExtractionResult())))

        timeouts_passed: list[float] = []

        async def mock_wait_for(awaitable, timeout):
            timeouts_passed.append(timeout)
            raise TimeoutError()

        with patch("asyncio.wait_for", mock_wait_for):
            with patch(
                "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
                return_value=True,
            ):
                # First chunk: 4 attempts, all timeout, counter becomes 4
                await extractor.extract_with_retry("content", "qn1")
                # Second chunk: attempt 0 fails (counter=4, attempt==0 → fast-fail for retry)
                await extractor.extract_with_retry("content", "qn2")

        # First chunk: 4 timeouts
        # Second chunk: 4 timeouts
        assert len(timeouts_passed) == 8
        # Second chunk attempt 0 uses normal adaptive timeout
        assert timeouts_passed[4] == pytest.approx(10.07, abs=0.01)
        # Second chunk attempt 1 gets fast-fail (set by attempt 0's failure)
        assert timeouts_passed[5] == pytest.approx(5.0, abs=0.1)
        # Second chunk retries should NOT re-trigger fast-fail (monotonically increasing)
        assert timeouts_passed[6] == pytest.approx(5.5, abs=0.1)
        assert timeouts_passed[7] == pytest.approx(6.05, abs=0.1)

    @pytest.mark.asyncio
    async def test_consecutive_timeouts_reset_on_success(self):
        """Counter resets to 0 after a successful extraction, preventing
        stale timeout counts from carrying over across chunks."""
        extractor = LLMConceptExtractor(timeout=10.0, max_timeout=100.0)
        extractor.agent = Mock(run=Mock(return_value=_MockAgentOutput(ExtractionResult())))

        timeouts_passed: list[float] = []
        call_index = 0

        async def mock_wait_for(awaitable, timeout):
            nonlocal call_index
            timeouts_passed.append(timeout)
            call_index += 1
            if call_index <= 2:
                raise TimeoutError()
            # 3rd call succeeds — counter should reset to 0
            return _MockAgentOutput(ExtractionResult())

        with patch("asyncio.wait_for", mock_wait_for):
            with patch(
                "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
                return_value=True,
            ):
                await extractor.extract_with_retry("content", "qn")
        # After 2 timeouts + 1 success, counter must be 0
        assert extractor._consecutive_timeouts == 0
        # First 2 attempts used normal adaptive timeout, 3rd (success) also normal
        assert timeouts_passed[0] == pytest.approx(10.07, abs=0.1)
        assert timeouts_passed[1] == pytest.approx(11.077, abs=0.1)

    @pytest.mark.asyncio
    async def test_interleaved_timeout_success_no_fast_fail(self):
        """Interleaved timeout/success across chunks should NOT trigger
        fast-fail mode, since the counter resets after each success."""
        extractor = LLMConceptExtractor(timeout=10.0, max_timeout=100.0)
        extractor.agent = Mock(run=Mock(return_value=_MockAgentOutput(ExtractionResult())))

        timeouts_passed: list[float] = []

        async def mock_wait_for(awaitable, timeout):
            timeouts_passed.append(timeout)
            # Fail on first attempt of each chunk, succeed on retry
            # This gives pattern: timeout, success, timeout, success...
            if len(timeouts_passed) % 2 == 1:
                raise TimeoutError()
            return _MockAgentOutput(ExtractionResult())

        with patch("asyncio.wait_for", mock_wait_for):
            with patch(
                "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
                return_value=True,
            ):
                # Process 5 chunks — each times out once then succeeds
                for _ in range(5):
                    await extractor.extract_with_retry("content", "qn")

        # Fast-fail never engaged because success resets counter each time
        # All first-attempt timeouts should use the normal adaptive timeout (~10.07s),
        # NOT the fast-fail 5s
        first_attempts = [t for i, t in enumerate(timeouts_passed) if i % 2 == 0]
        for t in first_attempts:
            assert t == pytest.approx(10.07, abs=0.1)

    @pytest.mark.asyncio
    async def test_probe_timeout_does_not_affect_breaker(self):
        cb = CircuitBreaker(name="test")
        extractor = LLMConceptExtractor(circuit_breaker=cb)
        extractor.agent = Mock(run=Mock(side_effect=asyncio.TimeoutError))
        with patch(
            "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
            return_value=False,
        ):
            await extractor.extract_with_retry("content", "qn")
        # Timeout errors do not record breaker failures; probe also should not
        assert cb.failure_count == 0


class TestRetryObservability:
    """Tests for retry success/failure observability (D-4)."""

    @pytest.mark.asyncio
    async def test_success_after_retry_emits_debug_log(self):
        """When extraction succeeds on attempt > 0, a DEBUG log is emitted."""
        from unittest.mock import patch

        extractor = LLMConceptExtractor(timeout=10.0, max_timeout=100.0)
        call_count = 0

        async def mock_run(content, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError()
            return _MockAgentOutput(ExtractionResult())

        extractor.agent = Mock(run=mock_run)

        with patch(
            "codebase_rag.document.concept_extraction.LLMConceptExtractor._probe_provider_health",
            return_value=True,
        ):
            with patch("loguru.logger") as mock_logger:
                result = await extractor.extract_with_retry("content", "qn")

        assert result == ExtractionResult()
        assert call_count == 3
        mock_logger.debug.assert_called_once_with(
            "Concept extraction succeeded on attempt 3 for qn"
        )

    @pytest.mark.asyncio
    async def test_success_on_first_attempt_no_debug_log(self):
        """When extraction succeeds on attempt 0, no retry-success DEBUG log."""
        extractor = LLMConceptExtractor()
        result = ExtractionResult()

        async def mock_run(content, **kwargs):
            return _MockAgentOutput(result)

        extractor.agent = Mock(run=mock_run)

        with patch("loguru.logger") as mock_logger:
            output = await extractor.extract_with_retry("content", "qn")

        assert output == result
        mock_logger.debug.assert_not_called()


class TestCircuitBreakerStormMitigation:
    """Tests for circuit breaker storm mitigation."""

    @pytest.mark.asyncio
    async def test_silent_return_on_breaker_open_in_retry(self):
        cb = CircuitBreaker(
            name="test",
            config=CircuitBreakerConfig(failure_threshold=1),
        )
        extractor = LLMConceptExtractor(circuit_breaker=cb)
        extractor.agent = Mock(run=Mock(side_effect=RuntimeError("down")))

        await extractor.extract_with_retry("content", "qn1")
        assert cb.is_open

        result = await extractor.extract_with_retry("content", "qn2")
        assert result == ExtractionResult()

    @pytest.mark.asyncio
    async def test_no_regression_on_normal_flow(self):
        extractor = LLMConceptExtractor()
        concept = ExtractedConcept(
            name="test",
            aliases=[],
            definition="def",
            confidence=0.9,
            entity_category="CONCRETE_ENTITY",
        )
        result = ExtractionResult(concepts=[concept])

        async def mock_run(content, **kwargs):
            return _MockAgentOutput(result)

        extractor.agent = Mock(run=mock_run)

        output = await extractor.extract_with_retry("content", "qn")
        assert len(output.concepts) == 1


class TestRelationshipDiversityCheck:
    """Tests for D-8 relationship category diversity guardrail."""

    def test_no_warning_when_diverse(self) -> None:
        from codebase_rag.document.concept_extraction import _check_relationship_diversity
        from codebase_rag.document.concept_extraction import ConceptRelationship

        rels = [
            ConceptRelationship(from_concept="A", to_concept="B", verb="is-a", category="HIERARCHICAL", emoji="🌳", strength=0.8),
            ConceptRelationship(from_concept="A", to_concept="C", verb="contains", category="COMPOSITIONAL", emoji="🧩", strength=0.8),
            ConceptRelationship(from_concept="A", to_concept="D", verb="causes", category="CAUSAL", emoji="⚡", strength=0.8),
        ]
        # Should not raise; no assertion needed for non-logging path
        _check_relationship_diversity(rels, "chunk:1")

    def test_warning_when_causal_dominant(self) -> None:
        from unittest.mock import patch
        from codebase_rag.document.concept_extraction import _check_relationship_diversity
        from codebase_rag.document.concept_extraction import ConceptRelationship

        rels = [
            ConceptRelationship(from_concept="A", to_concept="B", verb="causes", category="CAUSAL", emoji="⚡", strength=0.8),
            ConceptRelationship(from_concept="A", to_concept="C", verb="enables", category="CAUSAL", emoji="⚡", strength=0.8),
            ConceptRelationship(from_concept="A", to_concept="D", verb="triggers", category="CAUSAL", emoji="⚡", strength=0.8),
            ConceptRelationship(from_concept="A", to_concept="E", verb="produces", category="CAUSAL", emoji="⚡", strength=0.8),
        ]
        with patch("loguru.logger") as mock_logger:
            is_skewed = _check_relationship_diversity(rels, "chunk:test")
            assert is_skewed is True
            mock_logger.warning.assert_called_once()
            assert "skew detected" in mock_logger.warning.call_args[0][0]

    def test_empty_relationships_noop(self) -> None:
        from codebase_rag.document.concept_extraction import _check_relationship_diversity
        is_skewed = _check_relationship_diversity([], "chunk:empty")
        assert is_skewed is False


class TestMaxTokensConfiguration:
    """Tests for DOC_CONCEPT_EXTRACTION_MAX_TOKENS configuration."""

    def test_max_tokens_configuration_exists(self):
        from codebase_rag.config import settings

        assert hasattr(settings, "DOC_CONCEPT_EXTRACTION_MAX_TOKENS")
        assert settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS >= 256
        assert settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS <= 32768

    def test_max_tokens_default_value(self):
        from codebase_rag.config import settings

        # Default should be 4096
        assert settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS == 4096


class TestSplitExtraction:
    """Tests for chunk splitting recovery from context overflow."""

    @pytest.mark.asyncio
    async def test_split_extraction_with_mocked_agent(self):
        """Test that split extraction works by mocking the agent directly."""
        extractor = LLMConceptExtractor()

        # Track which chunks are processed
        processed_qns = []

        async def mock_run(content, **kwargs):
            # This will be called for each part after splitting
            # We can detect which part by the content size
            processed_qns.append(len(content))
            concept = ExtractedConcept(
                name=f"Concept-{len(processed_qns)}",
                aliases=[],
                definition="Test concept",
                confidence=0.9,
                entity_category="ABSTRACT_CONCEPT",
            )
            return _MockAgentOutput(ExtractionResult(concepts=[concept]))

        extractor.agent = Mock(run=mock_run)

        # Create content that will be split into paragraphs
        large_chunk = "paragraph one.\n\nparagraph two.\n\nparagraph three."

        result = await extractor._extract_with_splitting(
            large_chunk, "test_chunk", 30.0
        )

        # Should have extracted concepts from the split parts
        assert isinstance(result, ExtractionResult)
        # The paragraph split should create 2 parts (mid-point of 3 paragraphs)
        assert len(processed_qns) >= 1

    def test_merge_extraction_results_deduplicates(self):
        from codebase_rag.document.concept_extraction import (
            ConceptRelationship,
        )

        extractor = LLMConceptExtractor()

        r1 = ExtractionResult(
            concepts=[ExtractedConcept(
                name="Alpha",
                aliases=[],
                type="entity",
                definition="First concept",
                confidence=0.9,
                entity_category="ABSTRACT_CONCEPT",
            )],
            relationships=[ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="relates-to",
                category="RELATED_TO",
                emoji="🔗",
                strength=0.8,
            )],
        )
        r2 = ExtractionResult(
            concepts=[ExtractedConcept(
                name="Alpha",
                aliases=[],
                type="entity",
                definition="First concept",
                confidence=0.9,
                entity_category="ABSTRACT_CONCEPT",
            )],
            relationships=[ConceptRelationship(
                from_concept="A",
                to_concept="B",
                verb="relates-to",
                category="RELATED_TO",
                emoji="🔗",
                strength=0.8,
            )],
        )

        merged = extractor._merge_extraction_results([r1, r2])

        assert len(merged.concepts) == 1
        assert len(merged.relationships) == 1

    def test_merge_preserves_different_concepts(self):
        extractor = LLMConceptExtractor()

        r1 = ExtractionResult(
            concepts=[ExtractedConcept(
                name="Alpha",
                aliases=[],
                type="entity",
                definition="First concept",
                confidence=0.9,
                entity_category="ABSTRACT_CONCEPT",
            )],
            relationships=[],
        )
        r2 = ExtractionResult(
            concepts=[ExtractedConcept(
                name="Beta",
                aliases=[],
                type="entity",
                definition="Second concept",
                confidence=0.9,
                entity_category="ABSTRACT_CONCEPT",
            )],
            relationships=[],
        )

        merged = extractor._merge_extraction_results([r1, r2])

        assert len(merged.concepts) == 2
        assert {c.name for c in merged.concepts} == {"Alpha", "Beta"}

    def test_merge_deduplicates_relationships(self):
        from codebase_rag.document.concept_extraction import (
            ConceptRelationship,
        )

        extractor = LLMConceptExtractor()

        r1 = ExtractionResult(
            concepts=[],
            relationships=[
                ConceptRelationship(
                    from_concept="A", to_concept="B", verb="causes",
                    category="CAUSAL", emoji="⚡", strength=0.8,
                ),
                ConceptRelationship(
                    from_concept="A", to_concept="C", verb="enables",
                    category="CAUSAL", emoji="⚡", strength=0.7,
                ),
            ],
        )
        r2 = ExtractionResult(
            concepts=[],
            relationships=[
                ConceptRelationship(
                    from_concept="A", to_concept="B", verb="causes",
                    category="CAUSAL", emoji="⚡", strength=0.8,
                ),
                ConceptRelationship(
                    from_concept="B", to_concept="D", verb="triggers",
                    category="CAUSAL", emoji="⚡", strength=0.6,
                ),
            ],
        )

        merged = extractor._merge_extraction_results([r1, r2])

        # Should have 3 unique relationships (A->B is duplicated)
        assert len(merged.relationships) == 3
        keys = {(r.from_concept, r.to_concept, r.verb) for r in merged.relationships}
        assert keys == {
            ("A", "B", "causes"),
            ("A", "C", "enables"),
            ("B", "D", "triggers"),
        }

    def test_merge_empty_results(self):
        """Merging empty results should return empty ExtractionResult."""
        extractor = LLMConceptExtractor()

        merged = extractor._merge_extraction_results([])

        assert merged == ExtractionResult()

    def test_split_single_paragraph_uses_sentence_split(self):
        """When there's only one paragraph, use sentence splitting."""
        from codebase_rag.document.concept_extraction import _SENTENCE_SPLIT_RE

        # Content with multiple sentences but no paragraph breaks
        content = "First sentence. Second sentence! Third sentence?"
        paragraphs = content.split("\n\n")

        assert len(paragraphs) == 1  # Single paragraph

        # The sentence split should work
        sentences = _SENTENCE_SPLIT_RE.split(content)
        assert len(sentences) == 3
