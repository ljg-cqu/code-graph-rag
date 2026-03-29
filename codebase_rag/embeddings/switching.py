"""Embedding provider switching utilities.

This module provides functionality for switching between embedding providers
and re-embedding vectors when the provider changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .. import constants as cs
from .. import logs as ls
from ..config import settings
from ..exceptions import DimensionMismatchError

if TYPE_CHECKING:
    from ..embeddings.base import EmbeddingProvider


@dataclass
class SwitchResult:
    """Result of switching embedding providers.

    Attributes:
        success: Whether the switch was successful.
        old_provider: Previous provider name.
        old_model: Previous model identifier.
        old_dimension: Previous embedding dimension.
        new_provider: New provider name.
        new_model: New model identifier.
        new_dimension: New embedding dimension.
        vectors_reembedded: Number of vectors re-embedded (if applicable).
        error: Error message if switch failed.
    """

    success: bool
    old_provider: str
    old_model: str
    old_dimension: int
    new_provider: str
    new_model: str
    new_dimension: int
    vectors_reembedded: int = 0
    error: str | None = None


def switch_embedding_provider(
    new_provider: str,
    new_model: str,
    dimension: int | None = None,
    force: bool = False,
    reembed: bool = False,
    **config: str | int | None,
) -> SwitchResult:
    """Switch to a new embedding provider.

    This function validates the new provider configuration, optionally
    re-embeds all vectors, and updates the global configuration.

    Args:
        new_provider: New provider name (local, openai, google, ollama).
        new_model: New model identifier.
        dimension: Optional dimension override.
        force: If True, proceed even with dimension mismatch (acknowledges data loss).
        reembed: If True, re-embed all vectors after switching.
        **config: Additional provider configuration (api_key, endpoint, etc.).

    Returns:
        SwitchResult with details about the switch operation.

    Raises:
        DimensionMismatchError: If dimension changes and force=False and reembed=False.
        EmbeddingError: If the new provider fails validation.
    """
    from ..embeddings import get_embedding_provider

    # Get current configuration
    old_provider = settings.EMBEDDING_PROVIDER
    old_model = settings.EMBEDDING_MODEL
    old_dimension = settings.get_effective_vector_dim()

    # Create new provider to validate configuration
    new_provider_instance = get_embedding_provider(
        provider=new_provider,
        model_id=new_model,
        dimension=dimension,
        **config,
    )
    new_provider_instance.validate_config()

    new_dimension = new_provider_instance.dimension

    # Check for dimension mismatch
    if old_dimension != new_dimension and not reembed and not force:
        raise DimensionMismatchError(
            existing_dim=old_dimension,
            configured_dim=new_dimension,
            message=(
                f"Embedding dimension changed from {old_dimension} to {new_dimension}. "
                "Set force=True to acknowledge data loss, or reembed=True to re-embed all vectors."
            ),
            provider=new_provider,
            model=new_model,
        )

    # Log warning if force is used with dimension mismatch
    if old_dimension != new_dimension and force and not reembed:
        logger.warning(
            f"Dimension mismatch forced: {old_dimension}d -> {new_dimension}d. "
            "Existing vectors may be incompatible. Acknowledged by force=True."
        )

    # Update configuration
    settings.set_embedding(
        provider=new_provider,
        model_id=new_model,
        dimension=dimension,
        **config,
    )

    # Clear cache to use new provider
    from ..embedder import clear_embedding_cache, get_embedding_cache

    clear_embedding_cache()

    result = SwitchResult(
        success=True,
        old_provider=old_provider,
        old_model=old_model,
        old_dimension=old_dimension,
        new_provider=new_provider,
        new_model=new_model,
        new_dimension=new_dimension,
    )

    if reembed:
        try:
            vectors_reembedded = reembed_all_vectors()
            result.vectors_reembedded = vectors_reembedded
            logger.info(
                f"Switched embedding provider from {old_provider}/{old_model} to "
                f"{new_provider}/{new_model} ({vectors_reembedded} vectors re-embedded)"
            )
        except Exception as e:
            result.success = False
            result.error = f"Re-embedding failed: {e}"
            logger.error(f"Failed to re-embed vectors after provider switch: {e}")

    return result


def reembed_all_vectors() -> int:
    """Re-embed all vectors in the database with the current provider.

    This function fetches all embedded nodes from the database and
    re-embeds them using the current embedding provider configuration.

    Returns:
        Number of vectors re-embedded.

    Raises:
        EmbeddingError: If embedding or database operations fail.
    """
    from ..embedder import embed_code_batch, get_embedding_cache
    from ..vector_store import get_vector_backend

    backend = get_vector_backend()
    cache = get_embedding_cache()

    # Get all embedded nodes
    # This depends on the vector backend implementation
    # For Memgraph, we query nodes with embeddings
    try:
        embedded_nodes = backend.get_all_embedded_nodes()
    except AttributeError:
        logger.warning("Vector backend does not support get_all_embedded_nodes")
        return 0

    if not embedded_nodes:
        logger.info("No embedded nodes found to re-embed")
        return 0

    # Extract code snippets
    snippets = []
    node_ids = []
    for node in embedded_nodes:
        code = node.get("code") or node.get("source_code") or node.get("text", "")
        if code:
            snippets.append(code)
            node_ids.append(node.get("id"))

    if not snippets:
        logger.info("No code snippets found to re-embed")
        return 0

    # Clear cache to ensure fresh embeddings
    cache.clear()

    # Batch embed
    logger.info(f"Re-embedding {len(snippets)} nodes...")
    embeddings = embed_code_batch(snippets, batch_size=settings.VECTOR_EMBEDDING_BATCH_SIZE)

    # Update vectors in backend
    updated = 0
    for node_id, embedding in zip(node_ids, embeddings):
        try:
            backend.update_embedding(node_id, embedding)
            updated += 1
        except Exception as e:
            logger.warning(f"Failed to update embedding for node {node_id}: {e}")

    # Save cache
    cache.save()

    logger.info(f"Re-embedded {updated} vectors")
    return updated


def get_embedding_status() -> dict:
    """Get current embedding configuration status.

    Returns:
        Dictionary with current embedding configuration.
    """
    from ..embeddings import _EMBEDDING_PROVIDER_REGISTRY

    return {
        "provider": settings.EMBEDDING_PROVIDER,
        "model": settings.EMBEDDING_MODEL,
        "dimension": settings.get_effective_vector_dim(),
        "available_providers": list(_EMBEDDING_PROVIDER_REGISTRY.keys()),
        "cache_dir": settings.EMBEDDING_CACHE_DIR,
    }


__all__ = [
    "SwitchResult",
    "switch_embedding_provider",
    "reembed_all_vectors",
    "get_embedding_status",
]