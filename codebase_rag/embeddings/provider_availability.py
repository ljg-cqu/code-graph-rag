"""Lightweight provider availability checking (deterministic, no LLM).

This module provides deterministic checks for embedding provider availability.
Per LLM-First design: these are infrastructure checks, not semantic decisions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .base import EmbeddingProvider


@dataclass
class ProviderStatus:
    """Status of an embedding provider.

    Attributes:
        name: Provider name (local, openai, google, ollama).
        is_available: Whether the provider can be used.
        reason: Reason if not available.
        has_api_key: Whether API key is configured.
        error_type: Classification of error (DEPENDENCY_MISSING, AUTH_MISSING, etc.).
        install_command: Command to install missing dependencies.
        fallback_available: Whether a fallback provider is available.
        fallback_provider: Name of fallback provider.
    """

    name: str
    is_available: bool
    reason: str | None = None
    has_api_key: bool = False
    error_type: str | None = None  # 'DEPENDENCY_MISSING', 'AUTH_MISSING', etc.
    install_command: str | None = None
    fallback_available: bool = False
    fallback_provider: str | None = None


def check_provider_availability(provider_name: str) -> ProviderStatus:
    """Check if a specific provider is available.

    Deterministic checks only - no LLM needed. This is infrastructure
    validation, not semantic decision making.

    Args:
        provider_name: Provider name to check (local, openai, google, ollama).

    Returns:
        ProviderStatus with availability information.
    """
    from ..config import settings
    from . import _EMBEDDING_PROVIDER_REGISTRY

    provider_lower = provider_name.lower()

    # Check if provider is registered
    if provider_lower not in _EMBEDDING_PROVIDER_REGISTRY:
        return ProviderStatus(
            name=provider_name,
            is_available=False,
            reason="Provider not installed",
            has_api_key=False,
            error_type="NOT_REGISTERED",
        )

    # Local provider: check dependencies
    if provider_lower == "local":
        from .local import check_local_embedding_available

        available, reason = check_local_embedding_available()
        if not available:
            return ProviderStatus(
                name="local",
                is_available=False,
                reason=reason,
                has_api_key=True,
                error_type="DEPENDENCY_MISSING",
                install_command="uv sync --extra semantic  # or: pip install torch transformers",
            )
        return ProviderStatus(
            name="local",
            is_available=True,
            reason=None,
            has_api_key=True,
        )

    # Ollama: check if running (no API key needed)
    if provider_lower == "ollama":
        try:
            import requests

            response = requests.get(
                f"{settings.OLLAMA_BASE_URL}/api/tags",
                timeout=settings.OLLAMA_HEALTH_TIMEOUT,
            )
            if response.status_code == 200:
                return ProviderStatus(
                    name="ollama",
                    is_available=True,
                    reason=None,
                    has_api_key=True,
                )
            return ProviderStatus(
                name="ollama",
                is_available=False,
                reason=f"Ollama returned status {response.status_code}",
                has_api_key=True,
                error_type="ENDPOINT_UNREACHABLE",
                install_command="ollama serve",
            )
        except Exception as e:
            return ProviderStatus(
                name="ollama",
                is_available=False,
                reason=f"Ollama not running: {e}",
                has_api_key=True,
                error_type="ENDPOINT_UNREACHABLE",
                install_command="ollama serve",
            )

    # OpenAI: check API key
    if provider_lower == "openai":
        api_key = (
            settings.EMBEDDING_API_KEY
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            return ProviderStatus(
                name="openai",
                is_available=False,
                reason="API key not set",
                has_api_key=False,
                error_type="AUTH_MISSING",
                install_command="export OPENAI_API_KEY='your-key-here'",
            )
        return ProviderStatus(
            name="openai",
            is_available=True,
            reason=None,
            has_api_key=True,
        )

    # Google: check API key
    if provider_lower == "google":
        api_key = (
            settings.EMBEDDING_API_KEY
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            return ProviderStatus(
                name="google",
                is_available=False,
                reason="API key not set",
                has_api_key=False,
                error_type="AUTH_MISSING",
                install_command="export GOOGLE_API_KEY='your-key-here'",
            )
        return ProviderStatus("google", True, None, has_api_key=True)

    # Unknown provider
    return ProviderStatus(provider_name, False, "Unknown provider", has_api_key=False)


def get_best_available_provider(
    preferred_order: list[str] | None = None,
) -> str | None:
    """Get the best available provider deterministically.

    No LLM - just check availability in order. This is a mechanical
    decision, not a semantic one.

    Args:
        preferred_order: Order to check providers. If None, uses config default
            followed by common fallbacks.

    Returns:
        Name of the first available provider, or None if none available.
    """
    from ..config import settings

    if preferred_order is None:
        # Default order: configured provider first, then fallbacks
        preferred_order = [
            settings.EMBEDDING_PROVIDER,
            "openai",
            "ollama",
            "google",
            "local",
        ]

    # Remove duplicates while preserving order
    seen = set()
    unique_order = []
    for provider in preferred_order:
        provider_lower = provider.lower()
        if provider_lower not in seen:
            seen.add(provider_lower)
            unique_order.append(provider_lower)

    for provider in unique_order:
        status = check_provider_availability(provider)
        if status.is_available:
            logger.debug(f"Selected available embedding provider: {provider}")
            return provider
        logger.debug(f"Provider {provider} not available: {status.reason}")

    logger.warning("No embedding providers available")
    return None


def get_best_available_provider_with_status(
    preferred_order: list[str] | None = None,
) -> tuple[str | None, ProviderStatus | None]:
    """Get the best available provider with full status.

    Extends get_best_available_provider() to return detailed status.
    No LLM - just check availability in order.

    Args:
        preferred_order: Order to check providers. If None, uses config default
            followed by common fallbacks.

    Returns:
        Tuple of (provider_name, status). Provider name is None if none available.
    """
    from ..config import settings

    if preferred_order is None:
        # Default order: configured provider first, then fallbacks
        preferred_order = [
            settings.EMBEDDING_PROVIDER,
            "openai",
            "ollama",
            "google",
            "local",
        ]

    # Remove duplicates while preserving order
    seen = set()
    unique_order = []
    for provider in preferred_order:
        provider_lower = provider.lower()
        if provider_lower not in seen:
            seen.add(provider_lower)
            unique_order.append(provider_lower)

    primary_status = None
    for provider in unique_order:
        status = check_provider_availability(provider)
        if primary_status is None:
            primary_status = status  # Save primary (first) status

        if status.is_available:
            # Mark if this is a fallback
            if provider != unique_order[0]:
                status.fallback_available = True
                status.fallback_provider = provider
            return provider, status

        logger.debug(f"Provider {provider} not available: {status.reason}")

    logger.warning("No embedding providers available")
    return None, primary_status


def get_all_provider_statuses() -> dict[str, ProviderStatus]:
    """Get status of all known providers.

    Returns:
        Dict mapping provider name to status.
    """
    from . import _EMBEDDING_PROVIDER_REGISTRY

    statuses = {}
    for provider in _EMBEDDING_PROVIDER_REGISTRY:
        statuses[provider] = check_provider_availability(provider)

    return statuses


__all__ = [
    "ProviderStatus",
    "check_provider_availability",
    "get_best_available_provider",
    "get_best_available_provider_with_status",
    "get_all_provider_statuses",
]
