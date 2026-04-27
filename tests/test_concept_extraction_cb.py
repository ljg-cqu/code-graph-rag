"""Tests for circuit breaker behavior in concept extraction.

Verifies that fatal errors (context overflow, auth, quota) do NOT trigger
circuit breaker failures, while transient errors (timeout, network, LLM) DO.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from codebase_rag.document.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from codebase_rag.document.concept_extraction import LLMConceptExtractor


class MockAgent:
    """Mock pydantic-ai agent for controlled exception raising."""

    def __init__(self, exception: Exception | None = None) -> None:
        self.exception = exception

    async def run(self, user_prompt: str | None = None, **kwargs):
        if self.exception is not None:
            raise self.exception
        output = MagicMock()
        output.concepts = []
        output.relationships = []
        result = MagicMock()
        result.output = output
        return result


class TestCircuitBreakerErrorClassification:
    """Test circuit breaker records only transient errors."""

    @pytest.fixture
    def extractor(self):
        cb = CircuitBreaker(name="test", config=CircuitBreakerConfig(failure_threshold=3))
        ext = LLMConceptExtractor(circuit_breaker=cb)
        return ext, cb

    @pytest.mark.asyncio
    async def test_context_overflow_does_not_trigger_cb(self, extractor):
        """CONCEPT_CONTEXT_OVERFLOW should not count as circuit breaker failure."""
        ext, cb = extractor
        ext.agent = MockAgent(Exception("context window exceeded: token limit"))

        with pytest.raises(Exception, match="token limit"):
            await ext.extract("x" * 100000, "test_chunk")

        assert cb.failure_count == 0
        assert cb.can_execute() is True

    @pytest.mark.asyncio
    async def test_auth_error_does_not_trigger_cb(self, extractor):
        """CONCEPT_AUTH_ERROR should not count as circuit breaker failure."""
        ext, cb = extractor
        ext.agent = MockAgent(Exception("invalid api key"))

        with pytest.raises(Exception, match="api key"):
            await ext.extract("content", "test_chunk")

        assert cb.failure_count == 0
        assert cb.can_execute() is True

    @pytest.mark.asyncio
    async def test_quota_exceeded_does_not_trigger_cb(self, extractor):
        """CONCEPT_QUOTA_EXCEEDED should not count as circuit breaker failure."""
        ext, cb = extractor
        ext.agent = MockAgent(Exception("quota exceeded"))

        with pytest.raises(Exception, match="quota"):
            await ext.extract("content", "test_chunk")

        assert cb.failure_count == 0
        assert cb.can_execute() is True

    @pytest.mark.asyncio
    async def test_timeout_triggers_cb(self, extractor):
        """TimeoutError should count as circuit breaker failure."""
        ext, cb = extractor
        ext.agent = MockAgent(TimeoutError("request timed out"))

        with pytest.raises(TimeoutError):
            await ext.extract("content", "test_chunk")

        assert cb.failure_count == 1
        assert cb.can_execute() is True

    @pytest.mark.asyncio
    async def test_network_error_triggers_cb(self, extractor):
        """Network errors should count as circuit breaker failure."""
        ext, cb = extractor
        ext.agent = MockAgent(Exception("connection reset by peer"))

        with pytest.raises(Exception, match="connection"):
            await ext.extract("content", "test_chunk")

        assert cb.failure_count == 1
        assert cb.can_execute() is True

    @pytest.mark.asyncio
    async def test_llm_error_triggers_cb(self, extractor):
        """Generic LLM errors should count as circuit breaker failure."""
        ext, cb = extractor
        ext.agent = MockAgent(Exception("model returned empty response"))

        with pytest.raises(Exception, match="empty response"):
            await ext.extract("content", "test_chunk")

        assert cb.failure_count == 1
        assert cb.can_execute() is True

    @pytest.mark.asyncio
    async def test_mixed_errors_only_transient_count(self, extractor):
        """Only transient errors increment failure count across multiple calls."""
        ext, cb = extractor

        # 1. Context overflow - should NOT count
        ext.agent = MockAgent(Exception("context window exceeded"))
        with pytest.raises(Exception):
            await ext.extract("x" * 100000, "chunk1")
        assert cb.failure_count == 0

        # 2. Timeout - should count
        ext.agent = MockAgent(TimeoutError("timeout"))
        with pytest.raises(TimeoutError):
            await ext.extract("content", "chunk2")
        assert cb.failure_count == 1

        # 3. Auth error - should NOT count
        ext.agent = MockAgent(Exception("invalid api key"))
        with pytest.raises(Exception):
            await ext.extract("content", "chunk3")
        assert cb.failure_count == 1

        # 4. Another timeout - should count
        ext.agent = MockAgent(TimeoutError("timeout"))
        with pytest.raises(TimeoutError):
            await ext.extract("content", "chunk4")
        assert cb.failure_count == 2

    @pytest.mark.asyncio
    async def test_success_resets_cb_state(self, extractor):
        """Successful extraction records success on the circuit breaker."""
        ext, cb = extractor
        ext.agent = MockAgent()

        # First cause a transient failure
        ext.agent = MockAgent(TimeoutError("timeout"))
        with pytest.raises(TimeoutError):
            await ext.extract("content", "chunk1")
        assert cb.failure_count == 1

        # Then succeed
        ext.agent = MockAgent()
        result = await ext.extract("content", "chunk2")
        assert result.concepts == []
        assert cb.failure_count == 1  # failure_count stays until reset threshold
