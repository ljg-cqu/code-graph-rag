"""Provider fallback chain for resilient LLM calls.

Manages fallback between multiple LLM providers when the primary
provider is unavailable or quota is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from loguru import logger

from ..config import settings
from ..rate_limiter import QuotaStatus, get_rate_limiter

T = TypeVar("T")


@dataclass
class ProviderConfig:
    """Configuration for a provider in the fallback chain."""

    provider: str
    model: str
    priority: int  # Lower = higher priority


class ProviderFallbackChain:
    """Manages fallback between multiple LLM providers.

    Example:
        chain = ProviderFallbackChain([
            ProviderConfig("openai", "gpt-4o-mini", priority=1),
            ProviderConfig("anthropic", "claude-3-haiku", priority=2),
        ])

        result = chain.execute(lambda p, m: generate_with_provider(p, m, prompt))
    """

    def __init__(self, providers: list[ProviderConfig]):
        self.providers = sorted(providers, key=lambda p: p.priority)
        self._rate_limiter = get_rate_limiter()

    def execute(self, operation: Callable[[str, str], T]) -> T:
        """Execute operation with fallback chain.

        Args:
            operation: Function that takes (provider, model) and returns T

        Returns:
            Result from the first successful provider

        Raises:
            RuntimeError: If all providers fail
        """
        last_error: Exception | None = None

        for config in self.providers:
            status = self._rate_limiter.check_quota(config.provider, config.model)

            if status == QuotaStatus.EXHAUSTED:
                logger.info(
                    f"Skipping {config.provider}/{config.model}: quota exhausted"
                )
                continue

            if status == QuotaStatus.CRITICAL:
                logger.warning(
                    f"Using {config.provider}/{config.model}: quota critical"
                )

            try:
                return operation(config.provider, config.model)
            except Exception as e:
                last_error = e
                error_code = getattr(e, "status_code", None)
                self._rate_limiter.record_error(
                    config.provider,
                    config.model,
                    error_code or 500,
                )
                logger.warning(
                    f"Provider {config.provider}/{config.model} failed: {e}"
                )

        raise RuntimeError(
            f"All providers exhausted. Last error: {last_error}"
        )

    async def execute_async(
        self,
        operation: Callable[[str, str], Any],
    ) -> Any:
        """Execute async operation with fallback chain.

        Args:
            operation: Async function that takes (provider, model)

        Returns:
            Result from the first successful provider

        Raises:
            RuntimeError: If all providers fail
        """
        last_error: Exception | None = None

        for config in self.providers:
            status = self._rate_limiter.check_quota(config.provider, config.model)

            if status == QuotaStatus.EXHAUSTED:
                logger.info(
                    f"Skipping {config.provider}/{config.model}: quota exhausted"
                )
                continue

            if status == QuotaStatus.CRITICAL:
                logger.warning(
                    f"Using {config.provider}/{config.model}: quota critical"
                )

            try:
                return await operation(config.provider, config.model)
            except Exception as e:
                last_error = e
                error_code = getattr(e, "status_code", None)
                self._rate_limiter.record_error(
                    config.provider,
                    config.model,
                    error_code or 500,
                )
                logger.warning(
                    f"Provider {config.provider}/{config.model} failed: {e}"
                )

        raise RuntimeError(
            f"All providers exhausted. Last error: {last_error}"
        )


def build_fallback_chain_from_config() -> ProviderFallbackChain:
    """Build fallback chain from application configuration.

    Uses settings to determine primary provider and fallbacks.
    """
    # Map provider names to default models
    model_map = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-haiku",
        "doubao": "doubao-seed-2-0-pro-260215",
        "ollama": "llama3",
        "gemini": "gemini-pro",
    }

    providers: list[ProviderConfig] = []

    # Primary provider from config
    primary_provider = getattr(settings, "PRIMARY_LLM_PROVIDER", None)
    primary_model = getattr(settings, "PRIMARY_LLM_MODEL", None)

    if primary_provider:
        providers.append(
            ProviderConfig(
                provider=primary_provider,
                model=primary_model or model_map.get(primary_provider, "default"),
                priority=1,
            )
        )

    # Add fallback providers from config
    fallback_providers = getattr(settings, "FALLBACK_PROVIDERS", [])
    for i, provider_name in enumerate(fallback_providers, start=2):
        providers.append(
            ProviderConfig(
                provider=provider_name,
                model=model_map.get(provider_name, "default"),
                priority=i,
            )
        )

    return ProviderFallbackChain(providers)


# Default fallback chain (built from config)
DEFAULT_QUERY_CHAIN: ProviderFallbackChain | None = None


def get_default_query_chain() -> ProviderFallbackChain:
    """Get or create the default fallback chain."""
    global DEFAULT_QUERY_CHAIN
    if DEFAULT_QUERY_CHAIN is None:
        DEFAULT_QUERY_CHAIN = build_fallback_chain_from_config()
    return DEFAULT_QUERY_CHAIN
