"""Tests for LLM rate limiting and quota management.

This module tests the TokenBucket, LLMRateLimiter, and rate_limited decorator.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from codebase_rag.rate_limiter import (
    LLMRateLimiter,
    QuotaExceededError,
    QuotaInfo,
    QuotaStatus,
    TokenBucket,
    get_rate_limiter,
    rate_limited,
)


class TestTokenBucket:
    """Test TokenBucket rate limiting."""

    def test_initial_capacity(self):
        """Test bucket starts at full capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=1)
        assert bucket.tokens == 5

    def test_consume_decreases_tokens(self):
        """Test consuming tokens decreases count."""
        bucket = TokenBucket(capacity=5, refill_rate=1)
        result = bucket.consume()
        assert result is True
        assert bucket.tokens == 4

    def test_consume_multiple_tokens(self):
        """Test consuming multiple tokens at once."""
        bucket = TokenBucket(capacity=5, refill_rate=1)
        result = bucket.consume(3)
        assert result is True
        assert bucket.tokens == 2

    def test_consume_fails_when_empty(self):
        """Test consuming fails when bucket is empty."""
        bucket = TokenBucket(capacity=1, refill_rate=1)
        bucket.consume()  # Empty the bucket
        result = bucket.consume()
        assert result is False

    def test_refill_over_time(self):
        """Test tokens refill over time."""
        bucket = TokenBucket(capacity=5, refill_rate=10)  # 10 tokens per second
        bucket.tokens = 0  # Start empty

        # Wait for refill
        time.sleep(0.2)

        # Should have some tokens now
        result = bucket.consume()
        assert result is True

    def test_burst_capacity(self):
        """Test burst capacity is allowed."""
        bucket = TokenBucket(capacity=10, refill_rate=1)

        # Should be able to consume all at once
        for _ in range(10):
            assert bucket.consume() is True

        # Bucket should be empty now
        assert bucket.consume() is False

    def test_does_not_exceed_capacity(self):
        """Test refill does not exceed capacity."""
        bucket = TokenBucket(capacity=5, refill_rate=100)  # Fast refill

        # Wait for potential over-fill
        time.sleep(0.1)

        # Should still be at capacity, not over
        assert bucket.tokens <= 5


class TestQuotaInfo:
    """Test QuotaInfo quota tracking."""

    def test_healthy_status(self):
        """Test HEALTHY status when usage is low."""
        info = QuotaInfo(
            provider="test",
            model="model",
            used_requests=10,
            total_requests=100,
        )
        assert info.quota_status == QuotaStatus.HEALTHY

    def test_warning_status(self):
        """Test WARNING status when usage exceeds 80%."""
        info = QuotaInfo(
            provider="test",
            model="model",
            used_requests=85,
            total_requests=100,
        )
        assert info.quota_status == QuotaStatus.WARNING

    def test_critical_status(self):
        """Test CRITICAL status when usage exceeds 95%."""
        info = QuotaInfo(
            provider="test",
            model="model",
            used_requests=97,
            total_requests=100,
        )
        assert info.quota_status == QuotaStatus.CRITICAL

    def test_exhausted_by_requests(self):
        """Test EXHAUSTED when requests quota exceeded."""
        info = QuotaInfo(
            provider="test",
            model="model",
            used_requests=100,
            total_requests=100,
        )
        assert info.quota_status == QuotaStatus.EXHAUSTED

    def test_exhausted_by_tokens(self):
        """Test EXHAUSTED when tokens quota exceeded."""
        info = QuotaInfo(
            provider="test",
            model="model",
            used_tokens=1000,
            total_tokens=1000,
        )
        assert info.quota_status == QuotaStatus.EXHAUSTED

    def test_calculate_usage_ratio(self):
        """Test usage ratio calculation."""
        info = QuotaInfo(
            provider="test",
            model="model",
            used_requests=50,
            total_requests=100,
            used_tokens=250,
            total_tokens=500,
        )
        assert info._calculate_usage_ratio() == 0.5

    def test_usage_ratio_returns_max(self):
        """Test usage ratio returns max of request and token ratios."""
        info = QuotaInfo(
            provider="test",
            model="model",
            used_requests=90,  # 90%
            total_requests=100,
            used_tokens=50,  # 50%
            total_tokens=100,
        )
        assert info._calculate_usage_ratio() == 0.9


class TestLLMRateLimiter:
    """Test LLMRateLimiter functionality."""

    def test_register_provider(self):
        """Test registering a provider."""
        limiter = LLMRateLimiter()
        limiter.register_provider("openai", "gpt-4", requests_per_minute=60)

        # Should be able to acquire
        assert limiter.acquire("openai", "gpt-4") is True

    def test_acquire_blocks_on_exhausted_quota(self):
        """Test acquire returns False when quota exhausted."""
        limiter = LLMRateLimiter()
        limiter.register_provider("test", "model", requests_per_minute=60)

        # Simulate quota exhaustion
        limiter.update_quota("test", "model", used_requests=1000, total_requests=1000)

        result = limiter.acquire("test", "model", timeout=0.1)
        assert result is False

    def test_record_error_tracks_429s(self):
        """Test recording 429 errors tracks quota exhaustion."""
        limiter = LLMRateLimiter()
        limiter.register_provider("test", "model", requests_per_minute=60)

        # Set up quota info for tracking
        limiter.update_quota("test", "model", total_requests=100)

        # Record multiple 429s (more than threshold of 5)
        for _ in range(6):
            limiter.record_error("test", "model", 429)

        status = limiter.check_quota("test", "model")
        assert status == QuotaStatus.EXHAUSTED

    def test_update_quota(self):
        """Test updating quota information."""
        limiter = LLMRateLimiter()
        limiter.update_quota(
            "test",
            "model",
            used_requests=50,
            total_requests=100,
            used_tokens=1000,
            total_tokens=5000,
        )

        info = limiter._quota_info["test/model"]
        assert info.used_requests == 50
        assert info.total_requests == 100
        assert info.used_tokens == 1000
        assert info.total_tokens == 5000

    def test_check_quota_returns_healthy_for_unknown(self):
        """Test check_quota returns HEALTHY for unknown providers."""
        limiter = LLMRateLimiter()
        status = limiter.check_quota("unknown", "model")
        assert status == QuotaStatus.HEALTHY

    def test_acquire_with_timeout(self):
        """Test acquire respects timeout."""
        limiter = LLMRateLimiter()
        limiter.register_provider("test", "model", requests_per_minute=1)

        # Exhaust bucket
        limiter.acquire("test", "model")

        # Next acquire should timeout
        start = time.monotonic()
        result = limiter.acquire("test", "model", timeout=0.1)
        elapsed = time.monotonic() - start

        assert result is False
        assert elapsed >= 0.1


class TestRateLimitedDecorator:
    """Test rate_limited decorator."""

    def test_decorator_allows_call_when_healthy(self):
        """Test decorator allows call when quota healthy."""
        limiter = LLMRateLimiter()
        limiter.register_provider("test", "model", requests_per_minute=60)

        @rate_limited("test", "model")
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

    def test_decorator_raises_on_exhausted_quota(self):
        """Test decorator raises QuotaExceededError when quota exhausted."""
        # Use a fresh limiter via patching
        fresh_limiter = LLMRateLimiter()
        fresh_limiter.register_provider("test", "model", requests_per_minute=60)
        fresh_limiter.update_quota("test", "model", used_requests=100, total_requests=100)

        with patch("codebase_rag.rate_limiter._rate_limiter", fresh_limiter):
            @rate_limited("test", "model", timeout=0.01)
            def test_func():
                return "success"

            with pytest.raises(QuotaExceededError) as exc_info:
                test_func()

            assert "test/model" in str(exc_info.value)
            assert "Rate limit exceeded" in str(exc_info.value)

    def test_decorator_records_429_errors(self):
        """Test decorator records 429 errors from exceptions."""
        # Use a fresh limiter via patching
        fresh_limiter = LLMRateLimiter()
        fresh_limiter.register_provider("test", "model", requests_per_minute=60)

        class FakeException(Exception):
            status_code = 429

        with patch("codebase_rag.rate_limiter._rate_limiter", fresh_limiter):
            @rate_limited("test", "model")
            def failing_func():
                raise FakeException("Rate limited")

            with pytest.raises(FakeException):
                failing_func()

            # Should have recorded the error
            assert len(fresh_limiter._error_counts.get("test/model", [])) > 0


class TestGetRateLimiter:
    """Test get_rate_limiter singleton."""

    def test_returns_singleton(self):
        """Test get_rate_limiter returns the same instance."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()
        assert limiter1 is limiter2

    def test_global_limiter_shared_state(self):
        """Test global limiter has shared state."""
        limiter = get_rate_limiter()
        limiter.register_provider("global_test", "model", requests_per_minute=60)

        # Get again and verify state persists
        limiter2 = get_rate_limiter()
        assert "global_test/model" in limiter2._buckets
