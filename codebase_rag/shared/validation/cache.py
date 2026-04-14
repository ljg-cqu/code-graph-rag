"""Validation result caching.

Caches validation results to avoid redundant LLM calls
for unchanged content.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..query_router import ValidationReport


@dataclass
class CachedValidation:
    """Cached validation result."""

    report: ValidationReport
    cached_at: datetime
    expires_at: datetime
    document_path: str
    mode: str
    document_hash: str
    code_graph_hash: str

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "cached_at": self.cached_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "document_path": self.document_path,
            "mode": self.mode,
            "document_hash": self.document_hash,
            "code_graph_hash": self.code_graph_hash,
        }


class ValidationCache:
    """
    Cache validation results to avoid redundant LLM calls.

    Cache invalidation:
    - Document content changes → invalidate all for that document
    - Code graph changes → invalidate all CODE_VS_DOC results
    - TTL expires → lazy refresh
    """

    DEFAULT_TTL_HOURS = 24
    MAX_CACHE_SIZE = 1000

    def __init__(self, backend: dict | None = None) -> None:
        self._cache: dict[str, CachedValidation] = backend or {}
        # Structured index for O(1) document invalidation
        self._document_index: dict[str, set[str]] = {}  # document_path -> cache_keys
        # Structured index for O(1) mode invalidation
        self._mode_index: dict[str, set[str]] = {}  # mode -> cache_keys
        # Async lock for thread safety in concurrent contexts
        self._lock = asyncio.Lock()

    def compute_key(
        self,
        document_path: str,
        document_hash: str,
        code_graph_hash: str,
        mode: str,
        scope: str,
    ) -> str:
        """Compute cache key from validation parameters."""
        content = f"{document_path}|{document_hash}|{code_graph_hash}|{mode}|{scope}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get(
        self,
        document_path: str,
        document_hash: str,
        code_graph_hash: str,
        mode: str,
        scope: str,
    ) -> CachedValidation | None:
        """Get cached validation if not expired."""
        key = self.compute_key(
            document_path, document_hash, code_graph_hash, mode, scope
        )
        async with self._lock:
            cached = self._cache.get(key)
            if cached and cached.expires_at > datetime.now(UTC):
                return cached
            return None

    async def set(
        self,
        document_path: str,
        document_hash: str,
        code_graph_hash: str,
        mode: str,
        scope: str,
        report: ValidationReport,
        ttl_hours: int | None = None,
    ) -> None:
        """Cache validation result with TTL and document index."""
        ttl = ttl_hours or self.DEFAULT_TTL_HOURS
        now = datetime.now(UTC)
        key = self.compute_key(
            document_path, document_hash, code_graph_hash, mode, scope
        )

        async with self._lock:
            self._cache[key] = CachedValidation(
                report=report,
                cached_at=now,
                expires_at=now + timedelta(hours=ttl),
                document_path=document_path,
                mode=mode,
                document_hash=document_hash,
                code_graph_hash=code_graph_hash,
            )

            # Track document -> keys mapping for O(1) invalidation
            if document_path not in self._document_index:
                self._document_index[document_path] = set()
            self._document_index[document_path].add(key)

            # Track mode -> keys mapping for O(1) mode invalidation
            if mode not in self._mode_index:
                self._mode_index[mode] = set()
            self._mode_index[mode].add(key)

            # Evict old entries if over limit
            if len(self._cache) > self.MAX_CACHE_SIZE:
                await self._evict_oldest()

    async def invalidate_document(self, document_path: str) -> int:
        """Invalidate all cached validations for a document."""
        async with self._lock:
            keys_to_remove = self._document_index.get(document_path, set()).copy()
            if not keys_to_remove:
                return 0

            for k in keys_to_remove:
                cached = self._cache.pop(k, None)
                if cached and cached.mode in self._mode_index:
                    # Clean up mode index
                    self._mode_index[cached.mode].discard(k)
                    if not self._mode_index[cached.mode]:
                        del self._mode_index[cached.mode]

            self._document_index.pop(document_path, None)
            return len(keys_to_remove)

    async def invalidate_code_graph(self) -> int:
        """Invalidate all CODE_VS_DOC results when code graph changes.

        Note: CODE_VS_DOC compares code against docs, so code changes affect it.
        """
        async with self._lock:
            keys_to_remove = self._mode_index.get("CODE_VS_DOC", set()).copy()
            if not keys_to_remove:
                return 0

            for key in keys_to_remove:
                cached = self._cache.pop(key, None)
                if cached:
                    # Clean up document index
                    if cached.document_path in self._document_index:
                        self._document_index[cached.document_path].discard(key)
                        if not self._document_index[cached.document_path]:
                            del self._document_index[cached.document_path]
                    # Clean up mode index
                    self._mode_index["CODE_VS_DOC"].discard(key)

            if not self._mode_index["CODE_VS_DOC"]:
                del self._mode_index["CODE_VS_DOC"]

            return len(keys_to_remove)

    async def invalidate_all_code_dependent(self) -> int:
        """Invalidate both CODE_VS_DOC and DOC_VS_CODE when code graph changes.

        Both validation directions depend on code state:
        - CODE_VS_DOC: Code is being validated against docs
        - DOC_VS_CODE: Docs are being validated against code

        Call this when code graph is updated.
        """
        async with self._lock:
            total_removed = 0
            for mode in ("CODE_VS_DOC", "DOC_VS_CODE"):
                keys_to_remove = self._mode_index.get(mode, set()).copy()
                if not keys_to_remove:
                    continue

                for key in keys_to_remove:
                    cached = self._cache.pop(key, None)
                    if cached:
                        # Clean up document index
                        if cached.document_path in self._document_index:
                            self._document_index[cached.document_path].discard(key)
                            if not self._document_index[cached.document_path]:
                                del self._document_index[cached.document_path]
                        # Clean up mode index
                        self._mode_index[mode].discard(key)

                if not self._mode_index[mode]:
                    del self._mode_index[mode]

                total_removed += len(keys_to_remove)

            return total_removed

    async def _evict_oldest(self) -> None:
        """Evict oldest cached entries.

        NOTE: Must only be called while already holding self._lock.
        """
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].cached_at,
        )

        for key, cached in sorted_entries[:100]:  # Remove 100 oldest
            self._cache.pop(key, None)
            # Clean up document index
            if cached.document_path in self._document_index:
                self._document_index[cached.document_path].discard(key)
                if not self._document_index[cached.document_path]:
                    del self._document_index[cached.document_path]
            # Clean up mode index
            if cached.mode in self._mode_index:
                self._mode_index[cached.mode].discard(key)
                if not self._mode_index[cached.mode]:
                    del self._mode_index[cached.mode]

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._cache.clear()
            self._document_index.clear()

    async def size(self) -> int:
        """Get current cache size."""
        async with self._lock:
            return len(self._cache)

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        async with self._lock:
            now = datetime.now(UTC)
            expired = sum(1 for c in self._cache.values() if c.expires_at <= now)
            valid = len(self._cache) - expired

            return {
                "total_entries": len(self._cache),
                "valid_entries": valid,
                "expired_entries": expired,
                "document_index_size": len(self._document_index),
            }


__all__ = ["CachedValidation", "ValidationCache"]
