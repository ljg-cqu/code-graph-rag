"""Embedding provider registry and factory functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .. import constants as cs
from ..exceptions import EmbeddingProviderNotFoundError

if TYPE_CHECKING:
    from .base import EmbeddingProvider
    from .local import LocalEmbeddingProvider

# Provider registry - maps provider name to implementation class
_EMBEDDING_PROVIDER_REGISTRY: dict[str, type[EmbeddingProvider]] = {}


def _register_provider(name: str, cls: type[EmbeddingProvider]) -> None:
    """Register an embedding provider implementation."""
    _EMBEDDING_PROVIDER_REGISTRY[name.lower()] = cls


def get_embedding_provider_class(provider: str) -> type[EmbeddingProvider]:
    """Get the embedding provider class by name.

    Args:
        provider: Provider name (local, openai, google, ollama).

    Returns:
        The provider class.

    Raises:
        EmbeddingProviderNotFoundError: If provider is not registered.
    """
    provider_lower = provider.lower()
    if provider_lower not in _EMBEDDING_PROVIDER_REGISTRY:
        available = ", ".join(sorted(_EMBEDDING_PROVIDER_REGISTRY.keys()))
        raise EmbeddingProviderNotFoundError(
            f"Unknown embedding provider '{provider}'. Available: {available}"
        )
    return _EMBEDDING_PROVIDER_REGISTRY[provider_lower]


def get_embedding_provider(
    provider: str,
    model_id: str,
    dimension: int | None = None,
    **config: str | int | None,
) -> EmbeddingProvider:
    """Factory function to create an embedding provider.

    Args:
        provider: Provider name (local, openai, google, ollama).
        model_id: Model identifier.
        dimension: Optional dimension override.
        **config: Additional provider-specific configuration.

    Returns:
        Configured embedding provider instance.

    Raises:
        EmbeddingProviderNotFoundError: If provider is not registered.
    """
    cls = get_embedding_provider_class(provider)
    return cls(model_id=model_id, dimension=dimension, **config)


# Import and register providers after registry is defined
# Local provider is always available (if torch/transformers are installed)
from .local import LocalEmbeddingProvider

_register_provider(cs.EmbeddingProvider.LOCAL, LocalEmbeddingProvider)

# External providers - imported only when needed
# OpenAI
try:
    from .openai import OpenAIEmbeddingProvider

    _register_provider(cs.EmbeddingProvider.OPENAI, OpenAIEmbeddingProvider)
except ImportError:
    pass

# Google
try:
    from .google import GoogleEmbeddingProvider

    _register_provider(cs.EmbeddingProvider.GOOGLE, GoogleEmbeddingProvider)
except ImportError:
    pass

# Ollama
try:
    from .ollama import OllamaEmbeddingProvider

    _register_provider(cs.EmbeddingProvider.OLLAMA, OllamaEmbeddingProvider)
except ImportError:
    pass


def get_local_embedding_provider(
    model_id: str = cs.UNIXCODER_MODEL,
    device: str = "auto",
) -> LocalEmbeddingProvider:
    """Get the local embedding provider.

    Convenience function for getting the default local provider.

    Args:
        model_id: Model identifier. Defaults to UniXcoder.
        device: Device to use (auto, cpu, cuda).

    Returns:
        Configured LocalEmbeddingProvider instance.
    """
    return LocalEmbeddingProvider(model_id=model_id, device=device)


__all__ = [
    "get_embedding_provider",
    "get_embedding_provider_class",
    "get_local_embedding_provider",
    "_EMBEDDING_PROVIDER_REGISTRY",
    "_register_provider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GoogleEmbeddingProvider",
    "OllamaEmbeddingProvider",
    # Switching utilities
    "SwitchResult",
    "switch_embedding_provider",
    "reembed_all_vectors",
    "get_embedding_status",
    # Rate limiting
    "TokenBucket",
    "AdaptiveRateLimiter",
    # Error messages
    "USER_FACING_MESSAGES",
    "AUTH_SOLUTIONS",
    "get_user_facing_message",
    "get_auth_solutions",
]

# Lazy import for switching utilities to avoid circular imports
def __getattr__(name: str):
    if name in ("SwitchResult", "switch_embedding_provider", "reembed_all_vectors", "get_embedding_status"):
        from .switching import SwitchResult, switch_embedding_provider, reembed_all_vectors, get_embedding_status
        return locals()[name]
    if name in ("TokenBucket", "AdaptiveRateLimiter"):
        from .rate_limiter import TokenBucket, AdaptiveRateLimiter
        return locals()[name]
    if name in ("USER_FACING_MESSAGES", "AUTH_SOLUTIONS", "get_user_facing_message", "get_auth_solutions"):
        from .errors import USER_FACING_MESSAGES, AUTH_SOLUTIONS, get_user_facing_message, get_auth_solutions
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")