"""Validation result caching.

Caches validation results to avoid redundant LLM calls
for unchanged content.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
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

    def get(
        self,
        document_path: str,
        document_hash: str,
        code_graph_hash: str,
        mode: str,
        scope: str,
    ) -> CachedValidation | None:
        """Get cached validation if not expired."""
        key = self.compute_key(document_path, document_hash, code_graph_hash, mode, scope)
        cached = self._cache.get(key)

        if cached and cached.expires_at > datetime.now(UTC):
            return cached

        return None

    def set(
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
        key = self.compute_key(document_path, document_hash, code_graph_hash, mode, scope)

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

        # Evict old entries if over limit
        if len(self._cache) > self.MAX_CACHE_SIZE:
            self._evict_oldest()

    def invalidate_document(self, document_path: str) -> int:
        """Invalidate all cached validations for a document."""
        keys_to_remove = self._document_index.get(document_path, set())
        for k in keys_to_remove:
            self._cache.pop(k, None)
        self._document_index.pop(document_path, None)
        return len(keys_to_remove)

    def invalidate_code_graph(self) -> int:
        """Invalidate all CODE_VS_DOC results when code graph changes.

        Note: CODE_VS_DOC compares code against docs, so code changes affect it.
        """
        keys_to_remove = []
        for key, cached in self._cache.items():
            if cached.mode == "CODE_VS_DOC":
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self._cache.pop(key, None)
            # Clean up document index
            for doc_path, keys in self._document_index.items():
                keys.discard(key)

        return len(keys_to_remove)

    def invalidate_all_code_dependent(self) -> int:
        """Invalidate both CODE_VS_DOC and DOC_VS_CODE when code graph changes.

        Both validation directions depend on code state:
        - CODE_VS_DOC: Code is being validated against docs
        - DOC_VS_CODE: Docs are being validated against code

        Call this when code graph is updated.
        """
        keys_to_remove = []
        for key, cached in self._cache.items():
            if cached.mode in ("CODE_VS_DOC", "DOC_VS_CODE"):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            self._cache.pop(key, None)
            # Clean up document index
            for doc_path, keys in self._document_index.items():
                keys.discard(key)

        return len(keys_to_remove)

    def _evict_oldest(self) -> None:
        """Evict oldest cached entries."""
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].cached_at,
        )

        for key, _ in sorted_entries[:100]:  # Remove 100 oldest
            self._cache.pop(key, None)
            # Clean up document index
            for doc_path, keys in self._document_index.items():
                keys.discard(key)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._document_index.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def get_stats(self) -> dict:
        """Get cache statistics."""
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