from __future__ import annotations

import importlib.util
from collections.abc import Sequence

from codebase_rag.constants import (
    MODULE_TORCH,
    MODULE_TRANSFORMERS,
)

_dependency_cache: dict[str, bool] = {}


def _check_dependency(module_name: str) -> bool:
    if module_name not in _dependency_cache:
        _dependency_cache[module_name] = (
            importlib.util.find_spec(module_name) is not None
        )
    return _dependency_cache[module_name]


def has_torch() -> bool:
    return _check_dependency(MODULE_TORCH)


def has_transformers() -> bool:
    return _check_dependency(MODULE_TRANSFORMERS)


def has_semantic_dependencies() -> bool:
    return has_torch() and has_transformers()


def has_embedding_provider() -> bool:
    """Check if an embedding provider is configured and available.

    This is separate from has_semantic_dependencies() which checks for
    local embedding generation capabilities (torch/transformers).
    Query-time retrieval can work with pre-indexed embeddings without torch.
    """
    try:
        from ..config import settings
        from ..embeddings import get_embedding_provider

        config = settings.active_embedding_config
        provider = get_embedding_provider(config=config)
        # Test with simple embedding
        provider.embed("test")
        return True
    except Exception:
        return False


def check_dependencies(required_modules: Sequence[str]) -> bool:
    return all(_check_dependency(module) for module in required_modules)


def get_missing_dependencies(required_modules: Sequence[str]) -> list[str]:
    return [module for module in required_modules if not _check_dependency(module)]
