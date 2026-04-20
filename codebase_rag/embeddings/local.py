"""Local transformers-based embedding provider (UniXcoder)."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol, cast

from loguru import logger

from .. import constants as cs
from ..config import settings
from ..exceptions import EmbeddingGenerationError
from .base import EmbeddingProvider

if TYPE_CHECKING:
    import torch

# Module-level singleton state for preventing redundant model loads.
# When OpenAI auth fails (401), the local fallback loads the model from
# scratch each time. This singleton ensures the model is loaded once
# and reused across all fallback invocations.
_module_model_lock = threading.Lock()
_module_model_instance: LocalEmbeddingProvider | None = None


def get_local_embedding_provider(
    model_id: str = "BAAI/bge-large-en-v1.5",
    device: str = "cpu",
) -> LocalEmbeddingProvider:
    """Get or create the process-level singleton LocalEmbeddingProvider.

    Thread-safe: uses module-level lock to prevent concurrent model loads.
    The model is loaded once and reused across all fallback invocations,
    eliminating the redundant loads observed when OpenAI auth fails.

    Args:
        model_id: HuggingFace model identifier. Defaults to "BAAI/bge-large-en-v1.5".
        device: Device for inference (auto, cpu, cuda). Defaults to "cpu".

    Returns:
        The singleton LocalEmbeddingProvider instance.
    """
    global _module_model_instance
    if _module_model_instance is not None:
        return _module_model_instance
    with _module_model_lock:
        if _module_model_instance is None:
            logger.info(f"Creating singleton LocalEmbeddingProvider with model {model_id}")
            _module_model_instance = LocalEmbeddingProvider(
                model_id=model_id, device=device
            )
            _module_model_instance._ensure_model_loaded()
        return _module_model_instance


class TokenizerOutput(Protocol):
    def to(self, device: str) -> dict[str, torch.Tensor]: ...


class BatchTokenizer(Protocol):
    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> TokenizerOutput: ...


class UniXcoderLikeModel(Protocol):
    def tokenize(
        self,
        inputs: list[str],
        max_length: int = 512,
        padding: bool = False,
    ) -> list[list[int]]: ...

    def __call__(
        self, tokens_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

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
        ssl_verify: bool = True,
        proxy: str | None = None,
        fallback_to_local: bool = True,
        fallback_model: str = "BAAI/bge-large-en-v1.5",
    ) -> None:
        # Determine dimension from known models or default
        if dimension is None:
            dimension = KNOWN_MODEL_DIMENSIONS.get(model_id, 768)

        super().__init__(model_id, dimension, device=device)
        self._device = device
        self._model: object | None = None
        self._tokenizer: object | None = None

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
            from transformers import AutoModel, AutoTokenizer

            device = self._get_device()

            logger.info(f"Loading embedding model {self.model_id} on {device}...")

            # Load model type based on ID
            if "bge-" in self.model_id.lower():
                # BGE model uses standard transformers
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
                model = AutoModel.from_pretrained(self.model_id)
            else:
                # UniXcoder model
                from ..unixcoder import UniXcoder

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
        from ..config import settings

        self._ensure_model_loaded()

        assert self._model is not None, "Local embedding model failed to load"
        import torch

        device = self._get_device()
        model = self._model

        # Truncate long texts to avoid tokenization issues (consistent with OpenAI provider)
        max_chars = settings.EMBEDDING_MAX_LENGTH * 4
        truncated_text = text[:max_chars]

        if "bge-" in self.model_id.lower():
            # BGE model processing
            tokenizer = self._tokenizer
            if tokenizer is None or not callable(tokenizer):
                raise EmbeddingGenerationError(
                    f"Tokenizer for {self.model_id} is not callable",
                    provider="local",
                    model=self.model_id,
                )
            inputs = cast(BatchTokenizer, tokenizer)(
                [truncated_text],
                padding=True,
                truncation=True,
                max_length=settings.EMBEDDING_MAX_LENGTH,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                model_output = model(**inputs)
                # Mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = model_output[0]
                input_mask = (
                    attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                )
                sentence_embeddings = torch.sum(
                    token_embeddings * input_mask, 1
                ) / torch.clamp(input_mask.sum(1), min=1e-9)
                # Normalize
                sentence_embeddings = torch.nn.functional.normalize(
                    sentence_embeddings, p=2, dim=1
                )
                embedding = sentence_embeddings.cpu().numpy()
            return embedding[0].tolist()
        else:
            # UniXcoder model processing
            tokens = cast(UniXcoderLikeModel, model).tokenize(
                [truncated_text], max_length=settings.EMBEDDING_MAX_LENGTH
            )
            tokens_tensor = torch.tensor(tokens).to(device)

            with torch.no_grad():
                _, sentence_embeddings = cast(UniXcoderLikeModel, model)(
                    tokens_tensor
                )
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

        assert self._model is not None, "Local embedding model failed to load"
        import torch

        device = self._get_device()
        model = self._model

        all_embeddings: list[list[float]] = []

        # Truncate long texts to avoid tokenization issues (consistent with OpenAI provider)
        max_chars = settings.EMBEDDING_MAX_LENGTH * 4
        truncated_texts = [text[:max_chars] for text in texts]

        if "bge-" in self.model_id.lower():
            # BGE model batch processing
            tokenizer = self._tokenizer
            if tokenizer is None or not callable(tokenizer):
                raise EmbeddingGenerationError(
                    f"Tokenizer for {self.model_id} is not callable",
                    provider="local",
                    model=self.model_id,
                )
            for start in range(0, len(truncated_texts), batch_size):
                batch = truncated_texts[start : start + batch_size]
                inputs = cast(BatchTokenizer, tokenizer)(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=settings.EMBEDDING_MAX_LENGTH,
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    model_output = model(**inputs)
                    # Mean pooling
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = model_output[0]
                    input_mask = (
                        attention_mask.unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    sentence_embeddings = torch.sum(
                        token_embeddings * input_mask, 1
                    ) / torch.clamp(input_mask.sum(1), min=1e-9)
                    # Normalize
                    sentence_embeddings = torch.nn.functional.normalize(
                        sentence_embeddings, p=2, dim=1
                    )
                    batch_np = sentence_embeddings.cpu().numpy()

                for row in batch_np:
                    all_embeddings.append(row.tolist())
        else:
            # UniXcoder model batch processing
            for start in range(0, len(truncated_texts), batch_size):
                batch = truncated_texts[start : start + batch_size]
                tokens_list = cast(UniXcoderLikeModel, model).tokenize(
                    batch, max_length=settings.EMBEDDING_MAX_LENGTH, padding=True
                )
                tokens_tensor = torch.tensor(tokens_list).to(device)

                with torch.no_grad():
                    _, sentence_embeddings = cast(UniXcoderLikeModel, model)(
                        tokens_tensor
                    )
                    batch_np = sentence_embeddings.cpu().numpy()

                for row in batch_np:
                    all_embeddings.append(row.tolist())

        return all_embeddings
