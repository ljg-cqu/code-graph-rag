"""Embedding generation with configurable providers.

This module provides embedding functionality with a pluggable provider system.
The default provider is 'local' using UniXcoder, which requires torch/transformers.
Other providers (openai, google, ollama) can be configured via environment variables.

Configuration:
    EMBEDDING_PROVIDER: Provider name (local, openai, google, ollama). Default: local
    EMBEDDING_MODEL: Model identifier. Default: microsoft/unixcoder-base
    EMBEDDING_API_KEY: API key for external providers
    EMBEDDING_ENDPOINT: Custom endpoint URL
    EMBEDDING_DEVICE: Device for local models (auto/cpu/cuda)
"""

from __future__ import annotations

import fcntl
import functools
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from loguru import logger

from . import constants as cs
from . import exceptions as ex
from . import logs as ls
from .config import settings

# Backward compatibility imports for test mocking
# These are imported at module level so tests can patch them
try:
    import torch
    from .unixcoder import UniXcoder

    _HAS_SEMANTIC_DEPS = True
except ImportError:
    torch = None  # type: ignore[assignment]
    UniXcoder = None  # type: ignore[assignment]
    _HAS_SEMANTIC_DEPS = False

if TYPE_CHECKING:
    from .embeddings.base import EmbeddingProvider


class EmbeddingCache:
    """Per-model embedding cache with file locking and version compatibility.

    Cache files are stored per-model to avoid dimension mismatches when
    switching providers. Each cache file is URL-encoded to handle special
    characters in model identifiers (e.g., "microsoft/unixcoder-base" becomes
    "microsoft%2Funixcoder-base").

    File Format (v2):
        {
            "_metadata": {
                "version": 2,
                "model_id": "model-identifier",
                "dimension": 768,
                "count": 1234
            },
            "embeddings": {
                "hash1": [0.1, 0.2, ...],
                ...
            }
        }

    File Locking:
        Uses fcntl.flock for cross-process safety. The lock is held during
        both read and write operations.

    Attributes:
        CACHE_VERSION: Current cache format version.
        LEGACY_VERSION: Version before model-specific caches.
        MIN_SUPPORTED_VERSION: Minimum version this code can read.
    """

    __slots__ = ("_cache", "_path", "_model_id", "_dimension", "_dirty")
    CACHE_VERSION = 2
    LEGACY_VERSION = 1
    MIN_SUPPORTED_VERSION = 1

    def __init__(
        self, path: Path | None = None, model_id: str | None = None, dimension: int | None = None
    ) -> None:
        self._cache: dict[str, list[float]] = {}
        self._path = path
        self._model_id = model_id or cs.UNIXCODER_MODEL
        self._dimension = dimension or 768
        self._dirty: bool = False

    @staticmethod
    def _content_hash(content: str, model_id: str = "") -> str:
        """Generate cache key hash for content + model_id combination.

        Args:
            content: The text content to hash.
            model_id: The model identifier to include in hash.

        Returns:
            SHA256 hash string for use as cache key.
        """
        if model_id:
            combined = f"{model_id}:{content}"
        else:
            combined = content
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    @staticmethod
    def get_cache_path_for_model(base_dir: Path, model_id: str) -> Path:
        """Get cache file path for a specific model.

        Args:
            base_dir: Base directory for cache files.
            model_id: Model identifier (URL-encoded for filename safety).

        Returns:
            Path to the model-specific cache file.
        """
        # URL-encode model_id to handle special characters like "/"
        safe_model_id = quote(model_id, safe="")
        return base_dir / f"embeddings_{safe_model_id}.json"

    def get(self, content: str, model_id: str = "") -> list[float] | None:
        """Get cached embedding for content."""
        cache_key = self._content_hash(content, model_id or self._model_id)
        return self._cache.get(cache_key)

    def put(self, content: str, embedding: list[float], model_id: str = "") -> None:
        """Store embedding in cache."""
        cache_key = self._content_hash(content, model_id or self._model_id)
        self._cache[cache_key] = embedding
        self._dirty = True

    def get_many(self, snippets: list[str], model_id: str = "") -> dict[int, list[float]]:
        """Get multiple cached embeddings."""
        results: dict[int, list[float]] = {}
        for i, snippet in enumerate(snippets):
            if (cached := self.get(snippet, model_id)) is not None:
                results[i] = cached
        return results

    def put_many(
        self, snippets: list[str], embeddings: list[list[float]], model_id: str = ""
    ) -> None:
        """Store multiple embeddings in cache."""
        for snippet, embedding in zip(snippets, embeddings):
            self.put(snippet, embedding, model_id)

    def _acquire_lock(self, fd: int, exclusive: bool = True) -> None:
        """Acquire file lock for thread/process safety.

        Args:
            fd: File descriptor.
            exclusive: If True, acquire exclusive (write) lock.
        """
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, lock_type)

    def _release_lock(self, fd: int) -> None:
        """Release file lock.

        Args:
            fd: File descriptor.
        """
        fcntl.flock(fd, fcntl.LOCK_UN)

    def save(self) -> None:
        """Save cache to disk with atomic write and file locking.

        Uses atomic write pattern: write to temp file, then rename.
        This ensures readers never see a partial write.
        """
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "_metadata": {
                    "model_id": self._model_id,
                    "dimension": self._dimension,
                    "version": self.CACHE_VERSION,
                    "count": len(self._cache),
                },
                "embeddings": self._cache,
            }

            # Write to temp file first (atomic write pattern)
            fd, temp_path = tempfile.mkstemp(
                dir=str(self._path.parent),
                prefix=".tmp_cache_",
                suffix=".json"
            )
            try:
                self._acquire_lock(fd, exclusive=True)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                # Atomic rename
                os.replace(temp_path, str(self._path))
                self._dirty = False
            except Exception:
                # Clean up temp file on error
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning(ls.EMBEDDING_CACHE_SAVE_FAILED, path=self._path, error=e)

    def _migrate_legacy(self, legacy_path: Path) -> None:
        """Migrate legacy cache format to new model-specific format.

        Legacy caches (v1) were stored as a single file with no metadata.
        This method loads the legacy cache and marks it for migration.

        Args:
            legacy_path: Path to the legacy .embedding_cache.json file.
        """
        if not legacy_path.exists():
            return

        try:
            with legacy_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "embeddings" not in data:
                # Legacy format: direct dict of content_hash -> embedding
                self._cache = data
                self._dirty = True
                logger.info(
                    f"Migrated legacy cache with {len(self._cache)} entries "
                    f"to model {self._model_id}"
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to migrate legacy cache: {e}")

    def load(self) -> None:
        """Load cache from disk with file locking and version validation.

        Handles both legacy format (direct dict) and new format (with metadata).
        Validates version compatibility and dimension if available.
        """
        if self._path is None or not self._path.exists():
            # Check for legacy cache to migrate
            if self._path is not None:
                legacy_path = self._path.parent / cs.EMBEDDING_CACHE_FILENAME
                if legacy_path.exists() and legacy_path != self._path:
                    self._migrate_legacy(legacy_path)
            return

        fd = None
        try:
            fd = os.open(str(self._path), os.O_RDONLY)
            self._acquire_lock(fd, exclusive=False)

            with os.fdopen(fd, "r", encoding="utf-8") as f:
                data = json.load(f)
            fd = None  # File closed by context manager

            # Handle both old format (dict of embeddings) and new format
            if isinstance(data, dict) and "embeddings" in data:
                meta = data.get("_metadata", {})
                cache_version = meta.get("version", self.LEGACY_VERSION)

                if cache_version < self.MIN_SUPPORTED_VERSION:
                    logger.warning(
                        f"Cache version {cache_version} too old, starting fresh"
                    )
                    self._cache = {}
                    return

                # Validate dimension if we have one configured
                cached_dim = meta.get("dimension")
                if cached_dim is not None and self._dimension != cached_dim:
                    logger.warning(
                        f"Cache dimension {cached_dim} differs from configured "
                        f"{self._dimension}, cache may be stale"
                    )

                self._cache = data.get("embeddings", {})
            else:
                # Legacy format: direct dict of embeddings
                self._cache = data

            logger.debug(
                ls.EMBEDDING_CACHE_LOADED, count=len(self._cache), path=self._path
            )
        except Exception as e:
            logger.warning(ls.EMBEDDING_CACHE_LOAD_FAILED, path=self._path, error=e)
            self._cache = {}
        finally:
            if fd is not None:
                try:
                    self._release_lock(fd)
                    os.close(fd)
                except OSError:
                    pass

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._dirty = True

    def __len__(self) -> int:
        return len(self._cache)


# Global cache instance
_embedding_cache: EmbeddingCache | None = None

# Global provider instance
_embedding_provider: EmbeddingProvider | None = None


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache.

    Uses per-model cache files to avoid dimension mismatches when
    switching embedding providers. The cache file path is derived
    from the model identifier.
    """
    global _embedding_cache
    if _embedding_cache is None:
        cache_dir = Path(settings.EMBEDDING_CACHE_DIR)
        model_id = settings.EMBEDDING_MODEL
        cache_path = EmbeddingCache.get_cache_path_for_model(cache_dir, model_id)
        _embedding_cache = EmbeddingCache(
            path=cache_path,
            model_id=model_id,
            dimension=settings.get_effective_vector_dim(),
        )
        _embedding_cache.load()
    return _embedding_cache


def clear_embedding_cache() -> None:
    """Clear the global embedding cache."""
    global _embedding_cache, _embedding_provider
    if _embedding_cache is not None:
        _embedding_cache.clear()
        _embedding_cache = None
    # Also reset provider to pick up new settings
    _embedding_provider = None
    # Also clear the legacy model cache
    get_model.cache_clear()


# Backward compatibility: cached model getter for test mocking
# This function is kept for backward compatibility with existing tests
@functools.lru_cache(maxsize=1)
def get_model() -> object:
    """Get or create the cached UniXcoder model instance.

    This function is provided for backward compatibility with existing tests.
    New code should use get_embedding_provider_instance() instead.

    Returns:
        UniXcoder model instance.

    Raises:
        ImportError: If torch/transformers are not installed.
    """
    if not _HAS_SEMANTIC_DEPS:
        raise ImportError("torch/transformers not installed")

    assert UniXcoder is not None  # for type checker
    assert torch is not None  # for type checker

    model = UniXcoder(cs.UNIXCODER_MODEL)
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def get_embedding_provider_instance() -> EmbeddingProvider:
    """Get or create the global embedding provider singleton.

    This function lazily initializes the embedding provider based on
    the current configuration. If the provider changes, call
    clear_embedding_cache() to reset.

    Returns:
        Configured EmbeddingProvider instance.
    """
    global _embedding_provider
    if _embedding_provider is None:
        from .embeddings import get_embedding_provider

        config = settings.active_embedding_config
        dimension = config.dimension or settings.get_effective_vector_dim()

        _embedding_provider = get_embedding_provider(
            provider=config.provider,
            model_id=config.model_id,
            dimension=dimension,
            api_key=config.api_key,
            endpoint=config.endpoint,
            keep_alive=config.keep_alive,
            project_id=config.project_id,
            region=config.region,
            provider_type=config.provider_type,
            service_account_file=config.service_account_file,
            device=config.device,
        )
        _embedding_provider.validate_config()

    return _embedding_provider


def _check_semantic_dependencies() -> bool:
    """Check if semantic dependencies are available."""
    from .utils.dependencies import has_torch, has_transformers

    # For local provider, need torch/transformers
    if settings.EMBEDDING_PROVIDER.lower() == "local":
        return has_torch() and has_transformers()
    # For external providers, we don't need local ML dependencies
    return True


def _embed_with_local_model(
    code: str, model: object, max_length: int = 512
) -> list[float]:
    """Generate embedding using local UniXcoder model.

    This internal function implements the backward-compatible embedding
    path using the get_model() cached function.

    Args:
        code: The code string to embed.
        model: The UniXcoder model instance.
        max_length: Maximum token length.

    Returns:
        List of floats representing the embedding vector.
    """
    assert torch is not None  # for type checker
    import numpy as np

    # Determine device from model parameters
    device = "cpu"
    for param in model.parameters():
        device = str(param.device)
        break

    # Tokenize and generate embedding
    tokens = model.tokenize([code], max_length=max_length)
    tokens_tensor = torch.tensor(tokens).to(device)

    with torch.no_grad():
        _, sentence_embeddings = model(tokens_tensor)
        embedding = sentence_embeddings.cpu().numpy()

    return embedding[0].tolist()


def _embed_batch_with_local_model(
    snippets: list[str], model: object, max_length: int = 512, batch_size: int = 32
) -> list[list[float]]:
    """Generate embeddings batch using local UniXcoder model.

    This internal function implements the backward-compatible batch embedding
    path using the get_model() cached function.

    Args:
        snippets: List of code strings to embed.
        model: The UniXcoder model instance.
        max_length: Maximum token length.
        batch_size: Number of snippets to process per batch.

    Returns:
        List of embedding vectors in the same order as input snippets.
    """
    assert torch is not None  # for type checker
    import numpy as np

    # Determine device from model parameters
    device = "cpu"
    for param in model.parameters():
        device = str(param.device)
        break

    all_embeddings: list[list[float]] = []

    for start in range(0, len(snippets), batch_size):
        batch = snippets[start : start + batch_size]
        tokens_list = model.tokenize(batch, max_length=max_length, padding=True)
        tokens_tensor = torch.tensor(tokens_list).to(device)

        with torch.no_grad():
            _, sentence_embeddings = model(tokens_tensor)
            batch_np = sentence_embeddings.cpu().numpy()

        for row in batch_np:
            all_embeddings.append(row.tolist())

    return all_embeddings


def embed_code(code: str, max_length: int | None = None) -> list[float]:
    """Generate embedding for a single code snippet.

    This function uses the configured embedding provider to generate
    embeddings. The default provider is 'local' using UniXcoder.

    For backward compatibility with existing tests, the local provider
    uses get_model() which can be mocked.

    Args:
        code: The code string to embed.
        max_length: Maximum token length (used by local provider).

    Returns:
        List of floats representing the embedding vector.

    Raises:
        RuntimeError: If semantic dependencies are not installed for local provider.
        EmbeddingError: If embedding generation fails.
    """
    effective_max_length = max_length or settings.EMBEDDING_MAX_LENGTH
    cache = get_embedding_cache()

    # For backward compatibility with tests, use get_model() for local provider
    # This allows tests to patch get_model and mock the model behavior
    if settings.EMBEDDING_PROVIDER.lower() == "local":
        if not _check_semantic_dependencies():
            raise RuntimeError(
                "Semantic search requires torch and transformers. "
                "Install with: uv sync --extra semantic"
            )

        model_id = settings.EMBEDDING_MODEL

        # Check cache first
        if (cached := cache.get(code, model_id)) is not None:
            return cached

        # Use backward-compatible get_model() path
        model = get_model()
        embedding = _embed_with_local_model(code, model, effective_max_length)

        # Cache the result
        cache.put(code, embedding, model_id)
        return embedding

    # For other providers, use the provider system
    provider = get_embedding_provider_instance()

    # Check cache first
    if (cached := cache.get(code, provider.model_id)) is not None:
        return cached

    # Generate embedding
    embedding = provider.embed(code)

    # Cache the result
    cache.put(code, embedding, provider.model_id)

    return embedding


def embed_code_batch(
    snippets: list[str],
    max_length: int | None = None,
    batch_size: int = cs.EMBEDDING_DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Generate embeddings for multiple code snippets.

    This function batches multiple snippets for efficient processing.
    It uses caching to avoid regenerating embeddings for previously seen code.

    For backward compatibility with existing tests, the local provider
    uses get_model() which can be mocked.

    Args:
        snippets: List of code strings to embed.
        max_length: Maximum token length (used by local provider).
        batch_size: Number of snippets to process per batch.

    Returns:
        List of embedding vectors in the same order as input snippets.

    Raises:
        RuntimeError: If semantic dependencies are not installed for local provider.
        EmbeddingError: If embedding generation fails.
    """
    if not snippets:
        return []

    effective_max_length = max_length or settings.EMBEDDING_MAX_LENGTH
    cache = get_embedding_cache()

    # For backward compatibility with tests, use get_model() for local provider
    if settings.EMBEDDING_PROVIDER.lower() == "local":
        if not _check_semantic_dependencies():
            raise RuntimeError(
                "Semantic search requires torch and transformers. "
                "Install with: uv sync --extra semantic"
            )

        model_id = settings.EMBEDDING_MODEL

        # Check cache for all snippets
        cached_results = cache.get_many(snippets, model_id)

        if len(cached_results) == len(snippets):
            logger.debug(ls.EMBEDDING_CACHE_HIT, count=len(snippets))
            return [cached_results[i] for i in range(len(snippets))]

        # Find uncached snippets
        uncached_indices = [i for i in range(len(snippets)) if i not in cached_results]
        uncached_snippets = [snippets[i] for i in uncached_indices]

        # Use backward-compatible get_model() path
        model = get_model()
        new_embeddings = _embed_batch_with_local_model(
            uncached_snippets, model, effective_max_length, batch_size
        )

        # Cache new embeddings
        cache.put_many(uncached_snippets, new_embeddings, model_id)

        # Combine results in original order
        results: list[list[float]] = [[] for _ in snippets]
        for i, emb in cached_results.items():
            results[i] = emb
        for idx, orig_i in enumerate(uncached_indices):
            results[orig_i] = new_embeddings[idx]

        return results

    # For other providers, use the provider system
    provider = get_embedding_provider_instance()

    # Check cache for all snippets
    cached_results = cache.get_many(snippets, provider.model_id)

    if len(cached_results) == len(snippets):
        logger.debug(ls.EMBEDDING_CACHE_HIT, count=len(snippets))
        return [cached_results[i] for i in range(len(snippets))]

    # Find uncached snippets
    uncached_indices = [i for i in range(len(snippets)) if i not in cached_results]
    uncached_snippets = [snippets[i] for i in uncached_indices]

    # Generate embeddings for uncached snippets
    new_embeddings = provider.embed_batch(uncached_snippets, batch_size=batch_size)

    # Cache new embeddings
    cache.put_many(uncached_snippets, new_embeddings, provider.model_id)

    # Combine results in original order
    results: list[list[float]] = [[] for _ in snippets]
    for i, emb in cached_results.items():
        results[i] = emb
    for idx, orig_i in enumerate(uncached_indices):
        results[orig_i] = new_embeddings[idx]

    return results