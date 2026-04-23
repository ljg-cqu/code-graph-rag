"""Mocks for embedding providers.

Provides mock implementations for testing embedding operations
without requiring actual embedding models or API connections.
"""

from __future__ import annotations

from typing import Any


class MockEmbeddingProvider:
    """Mock embedding provider for testing.

    Returns deterministic embeddings based on input text.
    """

    def __init__(
        self,
        dimension: int = 768,
        name: str = "mock",
    ) -> None:
        self.dimension = dimension
        self.name = name
        self.call_count = 0
        self.last_texts: list[str] | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for texts.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (deterministic based on text hash)
        """
        self.call_count += 1
        self.last_texts = texts

        return [self._generate_embedding(text) for text in texts]

    async def embed_async(self, texts: list[str]) -> list[list[float]]:
        """Async version of embed."""
        return self.embed(texts)

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate a deterministic embedding for text.

        Uses simple hash-based approach for reproducible embeddings.

        Args:
            text: Input text

        Returns:
            Embedding vector of self.dimension length
        """
        # Simple deterministic embedding based on text hash
        text_hash = hash(text)
        embedding = []
        for i in range(self.dimension):
            # Generate value between -1 and 1 based on hash and position
            value = ((text_hash + i * 12345) % 10000) / 10000.0 * 2 - 1
            embedding.append(value)
        return embedding

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimension

    def reset(self) -> None:
        """Reset the mock state."""
        self.call_count = 0
        self.last_texts = None


def create_mock_embedding_fixture(dimension: int = 768) -> MockEmbeddingProvider:
    """Create a MockEmbeddingProvider for use in tests.

    Args:
        dimension: Embedding dimension (default 768)

    Returns:
        Configured MockEmbeddingProvider instance
    """
    return MockEmbeddingProvider(dimension=dimension)
