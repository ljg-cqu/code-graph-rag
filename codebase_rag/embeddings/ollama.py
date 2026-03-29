"""Ollama embedding provider for local model hosting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .. import constants as cs
from ..exceptions import EmbeddingConnectionError, EmbeddingGenerationError
from .base import EmbeddingProvider

if TYPE_CHECKING:
    import httpx

# Known Ollama embedding model dimensions
OLLAMA_MODEL_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
    "snowflake-arctic-embed": 1024,
}

# Default model
DEFAULT_OLLAMA_MODEL = "nomic-embed-text"


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider for local model hosting.

    This provider uses Ollama's local API to generate embeddings.
    No API key required - Ollama runs locally.

    Attributes:
        model_id: Ollama model name (e.g., nomic-embed-text).
        dimension: Embedding dimension (auto-detected from model).
        endpoint: Ollama API endpoint URL.
        keep_alive: Duration to keep model loaded (e.g., "5m", "1h").
    """

    def __init__(
        self,
        model_id: str = DEFAULT_OLLAMA_MODEL,
        dimension: int | None = None,
        endpoint: str | None = None,
        keep_alive: str | None = None,
    ) -> None:
        # Determine dimension from known models or default
        if dimension is None:
            dimension = OLLAMA_MODEL_DIMENSIONS.get(model_id, 768)

        super().__init__(model_id, dimension, endpoint=endpoint, keep_alive=keep_alive)
        self._endpoint = endpoint or "http://localhost:11434/api/embeddings"
        self._keep_alive = keep_alive or "5m"
        self._client: httpx.Client | None = None

    @property
    def provider_name(self) -> cs.EmbeddingProvider:
        return cs.EmbeddingProvider.OLLAMA

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=120.0)  # Longer timeout for local model loading
        return self._client

    def validate_config(self) -> None:
        """Validate Ollama provider configuration.

        Raises:
            EmbeddingConnectionError: If Ollama server is not reachable.
        """
        client = self._get_client()
        try:
            # Check if Ollama is running
            base_url = self._endpoint.rsplit("/api/embeddings", 1)[0]
            response = client.get(f"{base_url}/api/tags", timeout=5.0)
            response.raise_for_status()
        except Exception as e:
            raise EmbeddingConnectionError(
                f"Ollama server is not reachable at {self._endpoint}. "
                f"Make sure Ollama is running: ollama serve. Error: {e}",
                provider="ollama",
                model=self.model_id,
            ) from e

    def _make_request(
        self, texts: list[str], batch_size: int
    ) -> list[list[float]]:
        """Make embedding request to Ollama API.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per request.

        Returns:
            List of embedding vectors.
        """
        client = self._get_client()
        all_embeddings: list[list[float]] = []

        # Ollama's /api/embeddings endpoint processes one text at a time
        # The /api/embed (newer) endpoint supports batch, but we use single for compatibility
        for i, text in enumerate(texts):
            if i > 0 and i % batch_size == 0:
                logger.debug(f"Ollama embedding progress: {i}/{len(texts)}")

            payload = {
                "model": self.model_id,
                "prompt": text,
                "keep_alive": self._keep_alive,
            }

            try:
                response = client.post(
                    self._endpoint,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                embedding = data.get("embedding", [])
                if not embedding:
                    raise EmbeddingGenerationError(
                        f"Ollama returned empty embedding for text",
                        provider="ollama",
                        model=self.model_id,
                    )

                all_embeddings.append(embedding)

            except Exception as e:
                raise EmbeddingGenerationError(
                    f"Ollama embedding request failed: {e}",
                    provider="ollama",
                    model=self.model_id,
                ) from e

        return all_embeddings

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        embeddings = self._make_request([text], batch_size=1)
        return embeddings[0]

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of strings to embed.
            batch_size: Progress reporting interval. Defaults to 32.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        return self._make_request(texts, batch_size)