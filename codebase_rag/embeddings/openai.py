"""OpenAI embedding provider using text-embedding models.

This provider supports OpenAI's embedding API as well as OpenAI-compatible APIs
like Aliyun Dashscope, Azure OpenAI, and other compatible services.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .. import constants as cs
from ..exceptions import EmbeddingAuthenticationError, EmbeddingGenerationError
from .base import EmbeddingProvider

if TYPE_CHECKING:
    import httpx

# Known OpenAI embedding model dimensions
OPENAI_MODEL_DIMENSIONS: dict[str, int] = {
    # OpenAI native models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Aliyun/Dashscope models (OpenAI-compatible)
    "text-embedding-v1": 1536,
    "text-embedding-v2": 1536,
    "text-embedding-v3": 1024,
    "text-embedding-v4": 1024,
}

# Default models
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

# Batch size limits by provider/endpoint
# OpenAI's limit is 2048, but compatible APIs may have lower limits
MAX_BATCH_SIZE = 2048  # OpenAI's actual limit

# Known limits for OpenAI-compatible APIs (matched by URL pattern)
# Format: (url_pattern, max_batch_size)
COMPATIBLE_API_BATCH_LIMITS = [
    ("dashscope.aliyuncs.com", 10),  # Aliyun DashScope
    ("azure", 16),  # Azure OpenAI (conservative default)
]


def _get_batch_limit_for_endpoint(endpoint: str) -> int:
    """Get the batch size limit for a given endpoint URL.

    Args:
        endpoint: The embedding API endpoint URL.

    Returns:
        Maximum batch size for that endpoint.
    """
    endpoint_lower = endpoint.lower()
    for pattern, limit in COMPATIBLE_API_BATCH_LIMITS:
        if pattern in endpoint_lower:
            return limit
    return MAX_BATCH_SIZE


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider using text-embedding models.

    This provider uses OpenAI's embedding API to generate embeddings.
    Also supports OpenAI-compatible APIs like Aliyun Dashscope.
    Requires an API key set via EMBEDDING_API_KEY or OPENAI_API_KEY.

    Attributes:
        model_id: OpenAI model identifier (e.g., text-embedding-3-small).
        dimension: Embedding dimension (auto-detected from model).
        api_key: OpenAI API key.
        endpoint: Custom endpoint URL (optional).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_OPENAI_MODEL,
        dimension: int | None = None,
        api_key: str | None = None,
        endpoint: str | None = None,
        # Additional config parameters that may be passed by factory (ignored)
        keep_alive: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        provider_type: str | None = None,
        service_account_file: str | None = None,
        device: str | None = None,
    ) -> None:
        # Determine dimension from known models or default
        if dimension is None:
            dimension = OPENAI_MODEL_DIMENSIONS.get(model_id, 1536)

        super().__init__(model_id, dimension, api_key=api_key, endpoint=endpoint)
        self._api_key = api_key
        self._endpoint = endpoint or "https://api.openai.com/v1/embeddings"
        self._client: httpx.Client | None = None

    @property
    def provider_name(self) -> cs.EmbeddingProvider:
        return cs.EmbeddingProvider.OPENAI

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=60.0)
        return self._client

    def validate_config(self) -> None:
        """Validate OpenAI provider configuration.

        Raises:
            EmbeddingAuthenticationError: If API key is missing.
        """
        if not self._api_key or not self._api_key.strip():
            raise EmbeddingAuthenticationError(
                "OpenAI embedding requires an API key. "
                "Set EMBEDDING_API_KEY or OPENAI_API_KEY environment variable.",
                provider="openai",
                model=self.model_id,
            )

    def _make_request(
        self, texts: list[str], batch_size: int
    ) -> list[list[float]]:
        """Make embedding request to OpenAI API.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per request.

        Returns:
            List of embedding vectors.
        """
        client = self._get_client()

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # Determine effective batch size based on endpoint
        endpoint_limit = _get_batch_limit_for_endpoint(self._endpoint)
        effective_batch_size = min(batch_size, endpoint_limit)
        if batch_size > endpoint_limit:
            logger.debug(
                f"Batch size capped from {batch_size} to {effective_batch_size} for endpoint {self._endpoint}"
            )

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), effective_batch_size):
            batch = texts[start : start + effective_batch_size]

            payload = {
                "model": self.model_id,
                "input": batch,
            }

            # encoding_format is only supported by OpenAI's native API
            # Many OpenAI-compatible APIs (DashScope, Azure, etc.) reject this parameter
            if self._endpoint.startswith("https://api.openai.com"):
                payload["encoding_format"] = "float"

            try:
                response = client.post(
                    self._endpoint,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                # Extract embeddings in order
                embeddings_data = data.get("data", [])
                # Sort by index to maintain order
                embeddings_data.sort(key=lambda x: x.get("index", 0))

                for item in embeddings_data:
                    embedding = item.get("embedding", [])
                    all_embeddings.append(embedding)

            except Exception as e:
                raise EmbeddingGenerationError(
                    f"OpenAI embedding request failed: {e}",
                    provider="openai",
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
            batch_size: Number of texts per API call. Defaults to 32.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        return self._make_request(texts, batch_size)