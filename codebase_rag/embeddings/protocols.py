"""Type protocols for embedding providers."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProviderProtocol(Protocol):
    """Protocol for embedding provider implementations."""

    @property
    def model_id(self) -> str:
        """Return the model identifier."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        ...

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    def validate_config(self) -> None:
        """Validate provider configuration."""
        ...