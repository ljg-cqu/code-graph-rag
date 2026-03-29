"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .. import constants as cs


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers.

    Design Divergence from LLM Providers:
    Unlike the LLM provider system which uses a create_model() factory method
    to instantiate model objects, embedding providers directly implement embed()
    methods. This is because:
    1. Embedding is a single operation, not a stateful conversation
    2. No need for separate model instances - providers are stateless
    3. Simpler API: get_provider() -> embed() vs get_provider() -> create_model() -> embed()
    4. Batching is handled internally by embed_batch() rather than separate model instances
    """

    __slots__ = ("_config", "model_id", "dimension")

    def __init__(
        self,
        model_id: str,
        dimension: int,
        **config: str | int | None,
    ) -> None:
        self._config: dict[str, str | int | None] = config
        self.model_id = model_id
        self.dimension = dimension

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Chunking Behavior:
        When len(texts) > batch_size, this method automatically chunks the input
        into multiple batches of size batch_size (last batch may be smaller).
        Each chunk is processed sequentially with rate limiting applied between
        chunks. The returned list maintains the same order as the input texts.

        Example:
            texts = ["a", "b", "c", "d", "e"] with batch_size=2
            -> Processes ["a", "b"], then ["c", "d"], then ["e"]
            -> Returns [[emb_a], [emb_b], [emb_c], [emb_d], [emb_e]]

        Args:
            texts: List of strings to embed.
            batch_size: Maximum number of texts per API call. Defaults to 32.

        Returns:
            List of embeddings in the same order as input texts.
        """
        ...

    @abstractmethod
    def validate_config(self) -> None:
        """Validate provider configuration.

        Raises:
            EmbeddingAuthenticationError: If required credentials are missing.
            EmbeddingConnectionError: If connection cannot be established.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> cs.EmbeddingProvider:
        """Return provider identifier as EmbeddingProvider enum."""
        ...

    def get_config(self, key: str, default: str | int | None = None) -> str | int | None:
        """Get a configuration value."""
        return self._config.get(key, default)