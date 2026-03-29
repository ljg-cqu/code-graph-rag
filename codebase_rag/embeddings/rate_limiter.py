"""Rate limiting infrastructure for embedding API calls.

This module provides adaptive rate limiting to handle API throttling
and prevent overwhelming external embedding services.
"""

from __future__ import annotations

import random
import time


class TokenBucket:
    """Token bucket for rate limiting.

    Allows bursts up to the bucket capacity, then refills at a steady rate.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        """Initialize token bucket.

        Args:
            rate: Tokens added per second.
            capacity: Maximum tokens in bucket.
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_time = time.monotonic()

    def acquire(self, tokens: int = 1) -> float:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire.

        Returns:
            Time to wait before tokens are available (0 if immediate).
        """
        now = time.monotonic()
        elapsed = now - self._last_time
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
        self._last_time = now

        if self._tokens >= tokens:
            self._tokens -= tokens
            return 0.0

        wait_time = (tokens - self._tokens) / self._rate
        return wait_time


class AdaptiveRateLimiter:
    """Token bucket rate limiter with adaptive backoff.

    Handles rate limit responses (HTTP 429) by backing off exponentially
    and gradually recovering after successful requests.

    Attributes:
        rpm_bucket: Token bucket for requests per minute.
        tpm_bucket: Token bucket for tokens per minute.
    """

    def __init__(
        self,
        requests_per_minute: int = 500,
        tokens_per_minute: int = 300_000,
    ) -> None:
        """Initialize adaptive rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute.
            tokens_per_minute: Maximum tokens per minute.
        """
        # Convert to per-second rates
        rpm_rate = requests_per_minute / 60.0
        tpm_rate = tokens_per_minute / 60.0

        self.rpm_bucket = TokenBucket(rpm_rate, requests_per_minute)
        self.tpm_bucket = TokenBucket(tpm_rate, tokens_per_minute)
        self._consecutive_429s: int = 0
        self._max_consecutive_429s: int = 10

    def acquire(self, tokens: int) -> float:
        """Wait until capacity is available.

        Args:
            tokens: Estimated token count for the request.

        Returns:
            Total wait time in seconds.
        """
        wait_time = max(
            self.rpm_bucket.acquire(1),
            self.tpm_bucket.acquire(tokens)
        )
        if wait_time > 0:
            time.sleep(wait_time)
        return wait_time

    def handle_429(self, retry_after: float | None) -> float:
        """Handle rate limit response with adaptive backoff.

        Args:
            retry_after: Optional retry-after header value in seconds.

        Returns:
            Backoff time in seconds.
        """
        self._consecutive_429s += 1

        if retry_after and retry_after > 0:
            backoff = retry_after
        else:
            # Exponential backoff with jitter
            base = min(60.0, 2.0 ** self._consecutive_429s)
            backoff = base + random.uniform(0, 1)

        time.sleep(backoff)
        return backoff

    def reset_429_counter(self) -> None:
        """Reset the consecutive 429 counter after a successful request."""
        self._consecutive_429s = 0

    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (too many consecutive 429s)."""
        return self._consecutive_429s >= self._max_consecutive_429s


__all__ = ["TokenBucket", "AdaptiveRateLimiter"]