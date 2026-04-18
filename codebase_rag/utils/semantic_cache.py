"""LRU cache for semantic search results."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class CacheEntry:
    """Single cache entry."""

    results: Any
    timestamp: float
    access_count: int = 0

    def touch(self) -> None:
        self.access_count += 1
        self.timestamp = time.time()


class SemanticSearchCache:
    """LRU cache for semantic search results.

    Caches results based on query hash to avoid redundant embedding generation
    and vector search for identical queries.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_query(query: str, top_k: int) -> str:
        """Create deterministic hash for query parameters."""
        content = f"{query.lower().strip()}|{top_k}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, query: str, top_k: int) -> Any | None:
        """Get cached result if available and not expired."""
        key = self._hash_query(query, top_k)

        if key not in self._cache:
            self._misses += 1
            return None

        entry = self._cache[key]

        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self._cache[key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._hits += 1

        logger.debug(f"Semantic cache hit for query: {query[:50]}...")
        return entry.results

    def put(self, query: str, top_k: int, results: Any) -> None:
        """Store result in cache."""
        key = self._hash_query(query, top_k)

        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove least recently used

        self._cache[key] = CacheEntry(
            results=results,
            timestamp=time.time(),
        )
        logger.debug(f"Semantic cache stored: {query[:50]}...")

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


# Module-level singleton
_CACHE: SemanticSearchCache | None = None


def get_semantic_cache() -> SemanticSearchCache:
    """Get shared semantic search cache."""
    global _CACHE
    if _CACHE is None:
        from ..config import settings

        _CACHE = SemanticSearchCache(
            max_size=settings.CACHE_MAX_ENTRIES,
            ttl_seconds=3600,  # 1 hour default
        )
    return _CACHE


def reset_semantic_cache() -> None:
    """Reset the cache (e.g., when embeddings change)."""
    global _CACHE
    if _CACHE is not None:
        _CACHE.clear()
    _CACHE = None
