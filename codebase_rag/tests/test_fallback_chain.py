"""Tests for provider fallback chain.

This module tests the ProviderFallbackChain for resilient LLM calls.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from codebase_rag.providers.fallback_chain import (
    ProviderConfig,
    ProviderFallbackChain,
    build_fallback_chain_from_config,
)
from codebase_rag.rate_limiter import QuotaStatus, get_rate_limiter


class TestProviderConfig:
    """Test ProviderConfig dataclass."""

    def test_provider_config_creation(self):
        """Test creating ProviderConfig."""
        config = ProviderConfig(
            provider="openai",
            model="gpt-4",
            priority=1,
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.priority == 1


class TestProviderFallbackChain:
    """Test ProviderFallbackChain execution."""

    def test_providers_sorted_by_priority(self):
        """Test providers are sorted by priority."""
        chain = ProviderFallbackChain([
            ProviderConfig("provider_c", "model", priority=3),
            ProviderConfig("provider_a", "model", priority=1),
            ProviderConfig("provider_b", "model", priority=2),
        ])

        providers = [p.provider for p in chain.providers]
        assert providers == ["provider_a", "provider_b", "provider_c"]

    def test_execute_with_successful_provider(self):
        """Test execute with successful provider."""
        chain = ProviderFallbackChain([
            ProviderConfig("openai", "gpt-4", priority=1),
        ])

        def operation(provider, model):
            return f"success from {provider}"

        result = chain.execute(operation)
        assert result == "success from openai"

    def test_execute_fallback_on_failure(self):
        """Test fallback when primary provider fails."""
        chain = ProviderFallbackChain([
            ProviderConfig("failing", "model", priority=1),
            ProviderConfig("backup", "model", priority=2),
        ])

        def operation(provider, model):
            if provider == "failing":
                raise RuntimeError("Primary failed")
            return f"success from {provider}"

        result = chain.execute(operation)
        assert result == "success from backup"

    def test_execute_raises_when_all_fail(self):
        """Test execute raises when all providers fail."""
        chain = ProviderFallbackChain([
            ProviderConfig("failing1", "model", priority=1),
            ProviderConfig("failing2", "model", priority=2),
        ])

        def operation(provider, model):
            raise RuntimeError(f"{provider} failed")

        with pytest.raises(RuntimeError) as exc_info:
            chain.execute(operation)

        assert "All providers exhausted" in str(exc_info.value)

    def test_execute_skips_exhausted_providers(self):
        """Test execute skips providers with exhausted quota."""
        limiter = get_rate_limiter()
        limiter.register_provider("exhausted", "model", requests_per_minute=60)
        limiter.update_quota("exhausted", "model", used_requests=100, total_requests=100)

        chain = ProviderFallbackChain([
            ProviderConfig("exhausted", "model", priority=1),
            ProviderConfig("available", "model", priority=2),
        ])

        def operation(provider, model):
            return f"success from {provider}"

        result = chain.execute(operation)
        assert result == "success from available"

    def test_execute_uses_critical_providers(self):
        """Test execute still uses providers with critical quota."""
        limiter = get_rate_limiter()
        limiter.register_provider("critical", "model", requests_per_minute=60)
        limiter.update_quota("critical", "model", used_requests=97, total_requests=100)

        chain = ProviderFallbackChain([
            ProviderConfig("critical", "model", priority=1),
        ])

        def operation(provider, model):
            return f"success from {provider}"

        result = chain.execute(operation)
        assert result == "success from critical"


class TestBuildFallbackChainFromConfig:
    """Test building fallback chain from config."""

    def test_build_chain_with_primary_provider(self):
        """Test building chain with primary provider."""
        with patch("codebase_rag.providers.fallback_chain.settings") as mock_settings:
            mock_settings.PRIMARY_LLM_PROVIDER = "openai"
            mock_settings.PRIMARY_LLM_MODEL = "gpt-4"
            mock_settings.FALLBACK_PROVIDERS = []

            chain = build_fallback_chain_from_config()

            assert len(chain.providers) == 1
            assert chain.providers[0].provider == "openai"
            assert chain.providers[0].model == "gpt-4"

    def test_build_chain_with_fallback_providers(self):
        """Test building chain with fallback providers."""
        with patch("codebase_rag.providers.fallback_chain.settings") as mock_settings:
            mock_settings.PRIMARY_LLM_PROVIDER = "openai"
            mock_settings.PRIMARY_LLM_MODEL = "gpt-4"
            mock_settings.FALLBACK_PROVIDERS = ["anthropic", "google"]

            chain = build_fallback_chain_from_config()

            assert len(chain.providers) == 3
            assert chain.providers[0].provider == "openai"
            assert chain.providers[1].provider == "anthropic"
            assert chain.providers[2].provider == "google"

    def test_build_chain_without_primary(self):
        """Test building chain without primary provider."""
        with patch("codebase_rag.providers.fallback_chain.settings") as mock_settings:
            mock_settings.PRIMARY_LLM_PROVIDER = ""
            mock_settings.PRIMARY_LLM_MODEL = ""
            mock_settings.FALLBACK_PROVIDERS = ["openai"]

            chain = build_fallback_chain_from_config()

            # Should only have fallback providers
            assert len(chain.providers) == 1
            assert chain.providers[0].provider == "openai"

    def test_build_chain_uses_model_map(self):
        """Test building chain uses model map for known providers."""
        with patch("codebase_rag.providers.fallback_chain.settings") as mock_settings:
            mock_settings.PRIMARY_LLM_PROVIDER = ""
            mock_settings.PRIMARY_LLM_MODEL = ""
            mock_settings.FALLBACK_PROVIDERS = ["anthropic", "doubao"]

            chain = build_fallback_chain_from_config()

            assert chain.providers[0].model == "claude-3-haiku"
            assert chain.providers[1].model == "doubao-seed-2-0-pro-260215"


class TestAsyncExecution:
    """Test async execution in fallback chain."""

    @pytest.mark.asyncio
    async def test_execute_async_with_success(self):
        """Test async execute with successful provider."""
        chain = ProviderFallbackChain([
            ProviderConfig("openai", "gpt-4", priority=1),
        ])

        async def operation(provider, model):
            return f"async success from {provider}"

        result = await chain.execute_async(operation)
        assert result == "async success from openai"

    @pytest.mark.asyncio
    async def test_execute_async_fallback(self):
        """Test async execute falls back on failure."""
        chain = ProviderFallbackChain([
            ProviderConfig("failing", "model", priority=1),
            ProviderConfig("backup", "model", priority=2),
        ])

        async def operation(provider, model):
            if provider == "failing":
                raise RuntimeError("Failed")
            return f"async success from {provider}"

        result = await chain.execute_async(operation)
        assert result == "async success from backup"
