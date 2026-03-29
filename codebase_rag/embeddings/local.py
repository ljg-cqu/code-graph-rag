"""Local transformers-based embedding provider (UniXcoder)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from .. import constants as cs
from ..exceptions import EmbeddingError, EmbeddingGenerationError
from .base import EmbeddingProvider

if TYPE_CHECKING:
    import torch

# Known model dimensions for local providers
KNOWN_MODEL_DIMENSIONS: dict[str, int] = {
    "microsoft/unixcoder-base": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-large-en-v1.5": 1024,
}


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local transformers-based embedding using UniXcoder or similar models.

    This provider uses the local torch/transformers installation to generate
    embeddings. It's the default provider and requires no API keys.

    Attributes:
        model_id: The HuggingFace model identifier.
        dimension: The embedding vector dimension.
        device: The device to use (auto, cpu, cuda).
    """

    def __init__(
        self,
        model_id: str = cs.UNIXCODER_MODEL,
        device: str = "auto",
        dimension: int | None = None,
        # Additional config parameters that may be passed by factory (ignored)
        api_key: str | None = None,
        endpoint: str | None = None,
        keep_alive: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        provider_type: str | None = None,
        service_account_file: str | None = None,
    ) -> None:
        # Determine dimension from known models or default
        if dimension is None:
            dimension = KNOWN_MODEL_DIMENSIONS.get(model_id, 768)

        super().__init__(model_id, dimension, device=device)
        self._device = device
        self._model: object | None = None
        self._torch: type[torch] | None = None

    @property
    def provider_name(self) -> cs.EmbeddingProvider:
        return cs.EmbeddingProvider.LOCAL

    def _get_device(self) -> str:
        """Determine the device to use for inference."""
        if self._device != "auto":
            return self._device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except ImportError:
            return "cpu"

    def _ensure_model_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return

        try:
            import torch
            from ..unixcoder import UniXcoder

            self._torch = torch
            device = self._get_device()

            logger.info(f"Loading embedding model {self.model_id} on {device}...")

            model = UniXcoder(self.model_id)
            model.eval()

            if device == "cuda":
                model = model.cuda()

            self._model = model
            logger.info(f"Embedding model {self.model_id} loaded successfully")

        except ImportError as e:
            raise EmbeddingGenerationError(
                f"Failed to load embedding model {self.model_id}: {e}. "
                "Install semantic dependencies with: uv sync --extra semantic",
                provider="local",
                model=self.model_id,
            ) from e

    def validate_config(self) -> None:
        """Validate local provider configuration.

        For local providers, this checks that the required dependencies
        (torch, transformers) are available.
        """
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer  # noqa: F401
        except ImportError as e:
            raise EmbeddingGenerationError(
                f"Local embedding requires torch and transformers: {e}. "
                "Install with: uv sync --extra semantic",
                provider="local",
                model=self.model_id,
            ) from e

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        self._ensure_model_loaded()

        import numpy as np

        assert self._model is not None
        assert self._torch is not None

        device = self._get_device()
        model = self._model
        torch = self._torch

        tokens = model.tokenize([text], max_length=1024)
        tokens_tensor = torch.tensor(tokens).to(device)

        with torch.no_grad():
            _, sentence_embeddings = model(tokens_tensor)
            embedding = sentence_embeddings.cpu().numpy()

        return embedding[0].tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of strings to embed.
            batch_size: Number of texts to process at once. Defaults to 32.

        Returns:
            List of embeddings in the same order as input texts.
        """
        if not texts:
            return []

        self._ensure_model_loaded()

        import numpy as np

        assert self._model is not None
        assert self._torch is not None

        device = self._get_device()
        model = self._model
        torch = self._torch

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens_list = model.tokenize(batch, max_length=1024, padding=True)
            tokens_tensor = torch.tensor(tokens_list).to(device)

            with torch.no_grad():
                _, sentence_embeddings = model(tokens_tensor)
                batch_np = sentence_embeddings.cpu().numpy()

            for row in batch_np:
                all_embeddings.append(row.tolist())

        return all_embeddings