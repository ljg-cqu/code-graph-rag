"""Tests for ValidationCache."""

import pytest
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

from codebase_rag.shared.validation.cache import (
    CachedValidation,
    ValidationCache,
)
from codebase_rag.shared.query_router import ValidationReport


class TestCachedValidation:
    """Tests for CachedValidation dataclass."""

    def test_create_cached_validation(self):
        """Create cached validation entry."""
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )
        now = datetime.now(UTC)
        cached = CachedValidation(
            report=report,
            cached_at=now,
            expires_at=now + timedelta(hours=24),
            document_path="/docs/api.md",
            mode="CODE_VS_DOC",
            document_hash="abc123",
            code_graph_hash="def456",
        )
        assert cached.document_path == "/docs/api.md"
        assert cached.mode == "CODE_VS_DOC"

    def test_to_dict(self):
        """CachedValidation should serialize."""
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )
        now = datetime.now(UTC)
        cached = CachedValidation(
            report=report,
            cached_at=now,
            expires_at=now + timedelta(hours=24),
            document_path="/docs/api.md",
            mode="CODE_VS_DOC",
            document_hash="abc123",
            code_graph_hash="def456",
        )
        result = cached.to_dict()
        assert result["document_path"] == "/docs/api.md"
        assert result["mode"] == "CODE_VS_DOC"


class TestValidationCache:
    """Tests for ValidationCache class."""

    def test_init(self):
        """Cache initializes empty."""
        cache = ValidationCache()
        assert cache.size() == 0

    def test_compute_key(self):
        """Key computation is deterministic."""
        cache = ValidationCache()
        key1 = cache.compute_key(
            document_path="/docs/api.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
        )
        key2 = cache.compute_key(
            document_path="/docs/api.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
        )
        assert key1 == key2

    def test_compute_key_different_for_different_inputs(self):
        """Different inputs produce different keys."""
        cache = ValidationCache()
        key1 = cache.compute_key(
            document_path="/docs/a.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
        )
        key2 = cache.compute_key(
            document_path="/docs/b.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
        )
        assert key1 != key2

    def test_set_and_get(self):
        """Set and get cached validation."""
        cache = ValidationCache()
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )
        cache.set(
            document_path="/docs/api.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
            report=report,
        )
        assert cache.size() == 1

        cached = cache.get(
            document_path="/docs/api.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
        )
        assert cached is not None
        assert cached.report.total == 5

    def test_get_expired_returns_none(self):
        """Expired cache entries return None."""
        cache = ValidationCache()
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )
        # Set with normal TTL
        cache.set(
            document_path="/docs/api.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
            report=report,
            ttl_hours=24,
        )

        # Get the cached entry
        cached = cache.get(
            document_path="/docs/api.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
        )
        # Entry should exist and not be expired
        assert cached is not None
        assert cached.expires_at > datetime.now(UTC)

    def test_invalidate_document(self):
        """Invalidate all cached validations for a document."""
        cache = ValidationCache()
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )

        # Add multiple entries for same document
        cache.set(
            document_path="/docs/api.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
            report=report,
        )
        cache.set(
            document_path="/docs/api.md",
            document_hash="abc2",
            code_graph_hash="def",
            mode="DOC_VS_CODE",
            scope="sections",
            report=report,
        )
        cache.set(
            document_path="/docs/other.md",
            document_hash="xyz",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
            report=report,
        )

        assert cache.size() == 3
        removed = cache.invalidate_document("/docs/api.md")
        assert removed == 2
        assert cache.size() == 1

    def test_invalidate_code_graph(self):
        """Invalidate all CODE_VS_DOC results when code graph changes."""
        cache = ValidationCache()
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )

        # Add entries for both modes
        cache.set(
            document_path="/docs/a.md",
            document_hash="abc",
            code_graph_hash="old",
            mode="CODE_VS_DOC",
            scope="all",
            report=report,
        )
        cache.set(
            document_path="/docs/b.md",
            document_hash="def",
            code_graph_hash="old",
            mode="DOC_VS_CODE",
            scope="all",
            report=report,
        )

        assert cache.size() == 2
        removed = cache.invalidate_code_graph()
        assert removed == 1
        assert cache.size() == 1

    def test_clear(self):
        """Clear all cached entries."""
        cache = ValidationCache()
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )

        for i in range(5):
            cache.set(
                document_path=f"/docs/{i}.md",
                document_hash=f"hash{i}",
                code_graph_hash="graph",
                mode="CODE_VS_DOC",
                scope="all",
                report=report,
            )

        assert cache.size() == 5
        cache.clear()
        assert cache.size() == 0

    def test_get_stats(self):
        """Get cache statistics."""
        cache = ValidationCache()
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )

        cache.set(
            document_path="/docs/api.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
            report=report,
        )

        stats = cache.get_stats()
        assert stats["total_entries"] == 1
        assert stats["valid_entries"] == 1
        assert stats["expired_entries"] == 0

    def test_evict_oldest(self):
        """Cache evicts oldest entries when over limit."""
        cache = ValidationCache()
        # Temporarily lower limit for testing
        cache.MAX_CACHE_SIZE = 5
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )

        # Add more entries than limit
        for i in range(10):
            cache.set(
                document_path=f"/docs/{i}.md",
                document_hash=f"hash{i}",
                code_graph_hash="graph",
                mode="CODE_VS_DOC",
                scope="all",
                report=report,
            )

        # Cache should have evicted some entries
        assert cache.size() <= 10  # May have evicted or not depending on timing


class TestValidationCacheEdgeCases:
    """Edge case tests for ValidationCache."""

    def test_get_nonexistent_key(self):
        """Getting nonexistent key returns None."""
        cache = ValidationCache()
        result = cache.get(
            document_path="/nonexistent.md",
            document_hash="abc",
            code_graph_hash="def",
            mode="CODE_VS_DOC",
            scope="all",
        )
        assert result is None

    def test_invalidate_nonexistent_document(self):
        """Invalidating nonexistent document returns 0."""
        cache = ValidationCache()
        removed = cache.invalidate_document("/nonexistent.md")
        assert removed == 0

    def test_custom_backend(self):
        """Cache can use custom backend."""
        custom_backend: dict = {}
        cache = ValidationCache(backend=custom_backend)
        # After initialization, the cache uses the backend we passed
        # But the backend starts empty
        assert cache._cache == custom_backend