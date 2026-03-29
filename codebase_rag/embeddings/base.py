"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .. import constants as cs

if TYPE_CHECKING:
    from ..utils.token_utils import count_tokens


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

    def embed_batch_with_token_limit(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_batch_tokens: int | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts with token-aware batching.

        This method creates batches that respect both text count and token limits,
        providing more efficient API usage and preventing limit violations.

        Args:
            texts: List of strings to embed.
            batch_size: Maximum number of texts per batch.
            max_batch_tokens: Maximum tokens per batch. If None, uses provider default.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        if max_batch_tokens is None:
            max_batch_tokens = self._default_max_batch_tokens()

        # Group texts into token-aware batches
        batches = self._create_token_aware_batches(
            texts,
            max_texts=batch_size,
            max_tokens=max_batch_tokens,
        )

        all_embeddings: list[list[float]] = []
        for batch in batches:
            batch_embeddings = self.embed_batch(batch, batch_size=len(batch))
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _create_token_aware_batches(
        self,
        texts: list[str],
        max_texts: int,
        max_tokens: int,
    ) -> list[list[str]]:
        """Create batches that respect both text count and token limits.

        Uses a greedy algorithm to pack texts efficiently.

        Args:
            texts: List of texts to batch.
            max_texts: Maximum number of texts per batch.
            max_tokens: Maximum tokens per batch.

        Returns:
            List of batches, each containing texts.
        """
        from ..utils.token_utils import count_tokens

        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            text_tokens = count_tokens(text)

            # Check if adding this text would exceed limits
            would_exceed_count = len(current_batch) >= max_texts
            would_exceed_tokens = current_tokens + text_tokens > max_tokens

            if would_exceed_count or would_exceed_tokens:
                # Start new batch
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        # Don't forget the last batch
        if current_batch:
            batches.append(current_batch)

        return batches

    def _default_max_batch_tokens(self) -> int:
        """Default max tokens per batch based on provider.

        Returns conservative estimates based on typical provider limits.
        """
        provider = self.provider_name.value if hasattr(self.provider_name, "value") else str(self.provider_name)

        # Conservative estimates based on typical provider limits
        match provider:
            case "openai":
                return 500_000  # OpenAI allows ~500k tokens per batch
            case "google":
                # Distinguish between GLA and Vertex AI
                provider_type = self.get_config("provider_type", "gla")
                if provider_type == "vertex":
                    return 7_500_000  # Vertex AI: 250 texts * 30k tokens each
                return 2_000_000  # GLA: 100 texts * 20k tokens each
            case "ollama":
                return 100_000  # Conservative for local models
            case "local":
                return 100_000  # Conservative for local models
            case _:
                return 100_000  # Default conservative value