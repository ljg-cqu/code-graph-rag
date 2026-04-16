"""Embedding provider registry and factory functions."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .. import constants as cs
from ..config import settings
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
    effective_dimension = (
        dimension if dimension is not None else settings.get_effective_vector_dim()
    )
    return cls(model_id=model_id, dimension=effective_dimension, **config)


def _bootstrap_providers() -> None:
    from .local import LocalEmbeddingProvider as _LocalEmbeddingProvider

    globals()["LocalEmbeddingProvider"] = _LocalEmbeddingProvider
    _register_provider(cs.EmbeddingProvider.LOCAL, _LocalEmbeddingProvider)

    optional_providers = (
        (".openai", "OpenAIEmbeddingProvider", cs.EmbeddingProvider.OPENAI),
        (".google", "GoogleEmbeddingProvider", cs.EmbeddingProvider.GOOGLE),
        (".ollama", "OllamaEmbeddingProvider", cs.EmbeddingProvider.OLLAMA),
    )
    for module_name, class_name, provider_name in optional_providers:
        try:
            module = import_module(module_name, __name__)
        except ImportError:
            continue
        provider_cls = getattr(module, class_name)
        globals()[class_name] = provider_cls
        _register_provider(provider_name, provider_cls)


_bootstrap_providers()


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

for provider_name in (
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "GoogleEmbeddingProvider",
    "OllamaEmbeddingProvider",
):
    if provider_name in globals():
        __all__.append(provider_name)


# Lazy import for switching utilities to avoid circular imports
def __getattr__(name: str):
    if name in (
        "SwitchResult",
        "switch_embedding_provider",
        "reembed_all_vectors",
        "get_embedding_status",
    ):
        from . import switching

        return getattr(switching, name)
    if name in ("TokenBucket", "AdaptiveRateLimiter"):
        from . import rate_limiter

        return getattr(rate_limiter, name)
    if name in (
        "USER_FACING_MESSAGES",
        "AUTH_SOLUTIONS",
        "get_user_facing_message",
        "get_auth_solutions",
    ):
        from . import errors

        return getattr(errors, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
