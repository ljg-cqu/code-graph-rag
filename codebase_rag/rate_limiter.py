"""Rate limiting and quota management for LLM calls.

This module provides centralized rate limiting for LLM providers,
quota tracking, and provider fallback chains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from functools import wraps
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from loguru import logger

if TYPE_CHECKING:
    from .config import Settings

T = TypeVar("T")


class QuotaStatus(Enum):
    """Quota status for an LLM provider."""

    HEALTHY = auto()
    WARNING = auto()  # > 80% quota used
    CRITICAL = auto()  # > 95% quota used
    EXHAUSTED = auto()  # Quota exceeded


@dataclass
class QuotaInfo:
    """Quota information for an LLM provider."""

    provider: str
    model: str
    used_requests: int = 0
    total_requests: int | None = None
    used_tokens: int = 0
    total_tokens: int | None = None
    reset_time: datetime | None = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def quota_status(self) -> QuotaStatus:
        """Calculate current quota status."""
        if self.total_requests and self.used_requests >= self.total_requests:
            return QuotaStatus.EXHAUSTED
        if self.total_tokens and self.used_tokens >= self.total_tokens:
            return QuotaStatus.EXHAUSTED

        usage_ratio = self._calculate_usage_ratio()
        if usage_ratio > 0.95:
            return QuotaStatus.CRITICAL
        if usage_ratio > 0.80:
            return QuotaStatus.WARNING
        return QuotaStatus.HEALTHY

    def _calculate_usage_ratio(self) -> float:
        """Calculate the highest usage ratio across request and token quotas."""
        ratios = []
        if self.total_requests:
            ratios.append(self.used_requests / self.total_requests)
        if self.total_tokens:
            ratios.append(self.used_tokens / self.total_tokens)
        return max(ratios) if ratios else 0.0


class TokenBucket:
    """Token bucket for rate limiting.

    Allows bursts up to capacity, then refills at a steady rate.
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        name: str = "default",
    ):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()
        self.name = name
        self._lock = Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        """Try to consume tokens. Returns True if successful."""
        with self._lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.refill_rate,
        )
        self.last_refill = now


class QuotaExceededError(RuntimeError):
    """Raised when rate limit or quota is exceeded."""

    def __init__(self, message: str, provider: str = "", model: str = ""):
        super().__init__(message)
        self.provider = provider
        self.model = model


class LLMRateLimiter:
    """Centralized rate limiter for LLM calls."""

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._quota_info: dict[str, QuotaInfo] = {}
        self._error_counts: dict[str, list[float]] = {}  # timestamp tracking
        self._lock = Lock()

    def register_provider(
        self,
        provider: str,
        model: str,
        requests_per_minute: float = 60,
        requests_per_day: float | None = None,
    ) -> None:
        """Register a provider with rate limits.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name (e.g., "gpt-4o-mini")
            requests_per_minute: Maximum requests per minute
            requests_per_day: Maximum requests per day (optional)
        """
        key = f"{provider}/{model}"
        with self._lock:
            self._buckets[key] = TokenBucket(
                capacity=requests_per_minute,
                refill_rate=requests_per_minute / 60,
                name=key,
            )
            # Initialize quota info if not exists
            if key not in self._quota_info:
                self._quota_info[key] = QuotaInfo(
                    provider=provider,
                    model=model,
                )

    def check_quota(self, provider: str, model: str) -> QuotaStatus:
        """Check current quota status for a provider."""
        key = f"{provider}/{model}"
        with self._lock:
            info = self._quota_info.get(key)
            if info:
                return info.quota_status
            return QuotaStatus.HEALTHY

    def record_error(self, provider: str, model: str, error_code: int) -> None:
        """Record an error for tracking.

        Args:
            provider: Provider name
            model: Model name
            error_code: HTTP error code
        """
        key = f"{provider}/{model}"
        now = time.monotonic()

        with self._lock:
            if key not in self._error_counts:
                self._error_counts[key] = []

            self._error_counts[key].append(now)

            # Clean old entries (> 1 minute)
            self._error_counts[key] = [
                t for t in self._error_counts[key] if now - t < 60
            ]

            # If too many 429s, mark quota as exhausted
            if error_code == 429 and len(self._error_counts[key]) > 5:
                logger.warning(f"Quota exceeded detected for {key}")
                if key in self._quota_info:
                    info = self._quota_info[key]
                    if info.total_requests:
                        info.used_requests = info.total_requests
                    else:
                        # Mark as exhausted without a total
                        info.used_requests = 999999

    def acquire(
        self,
        provider: str,
        model: str,
        timeout: float = 30,
    ) -> bool:
        """Acquire permission to make an LLM call.

        Args:
            provider: Provider name
            model: Model name
            timeout: Maximum time to wait for permission

        Returns:
            True if permission granted, False if timeout
        """
        key = f"{provider}/{model}"

        # Check quota status first
        if self.check_quota(provider, model) == QuotaStatus.EXHAUSTED:
            logger.warning(f"Quota exhausted for {key}, blocking request")
            return False

        bucket = self._buckets.get(key)
        if not bucket:
            return True  # No limit configured

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if bucket.consume():
                return True
            time.sleep(0.1)

        return False

    def update_quota(
        self,
        provider: str,
        model: str,
        used_requests: int | None = None,
        total_requests: int | None = None,
        used_tokens: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Update quota information for a provider.

        This can be called after receiving quota info from API responses
        or when manually configuring quota limits.
        """
        key = f"{provider}/{model}"
        with self._lock:
            if key not in self._quota_info:
                self._quota_info[key] = QuotaInfo(
                    provider=provider,
                    model=model,
                )

            info = self._quota_info[key]
            if used_requests is not None:
                info.used_requests = used_requests
            if total_requests is not None:
                info.total_requests = total_requests
            if used_tokens is not None:
                info.used_tokens = used_tokens
            if total_tokens is not None:
                info.total_tokens = total_tokens
            info.last_updated = datetime.now(UTC)


# Global rate limiter instance
_rate_limiter = LLMRateLimiter()


def get_rate_limiter() -> LLMRateLimiter:
    """Get the global rate limiter instance."""
    return _rate_limiter


def rate_limited(
    provider: str,
    model: str,
    timeout: float = 30,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to rate-limit LLM operations.

    Args:
        provider: Provider name
        model: Model name
        timeout: Maximum time to wait for permission

    Example:
        @rate_limited("openai", "gpt-4o-mini")
        def generate_text(prompt: str) -> str:
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            limiter = get_rate_limiter()

            if not limiter.acquire(provider, model, timeout):
                raise QuotaExceededError(
                    f"Rate limit exceeded for {provider}/{model}. "
                    f"Quota may be exhausted or too many concurrent requests.",
                    provider=provider,
                    model=model,
                )

            try:
                return func(*args, **kwargs)
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                if status_code == 429:
                    limiter.record_error(provider, model, 429)
                raise

        return wrapper

    return decorator


def rate_limited_async(
    provider: str,
    model: str,
    timeout: float = 30,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to rate-limit async LLM operations.

    Args:
        provider: Provider name
        model: Model name
        timeout: Maximum time to wait for permission
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            limiter = get_rate_limiter()

            if not limiter.acquire(provider, model, timeout):
                raise QuotaExceededError(
                    f"Rate limit exceeded for {provider}/{model}. "
                    f"Quota may be exhausted or too many concurrent requests.",
                    provider=provider,
                    model=model,
                )

            try:
                return await func(*args, **kwargs)
            except Exception as e:
                status_code = getattr(e, "status_code", None)
                if status_code == 429:
                    limiter.record_error(provider, model, 429)
                raise

        return wrapper

    return decorator
