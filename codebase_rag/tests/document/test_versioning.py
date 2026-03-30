"""Tests for document versioning."""

from __future__ import annotations

import pytest
from pathlib import Path

from codebase_rag.document.versioning import (
    DocumentVersion,
    ContentVersionTracker,
    VersionCache,
)
from codebase_rag.document.extractors.base import ExtractedDocument, ExtractedSection


class TestDocumentVersion:
    """Tests for DocumentVersion dataclass."""

    def test_create_version(self):
        """Test creating a document version."""
        version = DocumentVersion(
            path="/test/doc.md",
            content_hash="abc123",
            modified_date="2024-01-01T00:00:00",
            indexed_at="2024-01-01T01:00:00",
            section_hashes=["hash1", "hash2"],
        )
        assert version.path == "/test/doc.md"
        assert version.content_hash == "abc123"
        assert version.extractor_version == "1.0"

    def test_default_extractor_version(self):
        """Test default extractor version."""
        version = DocumentVersion(
            path="/test/doc.md",
            content_hash="hash",
            modified_date="2024-01-01",
            indexed_at="2024-01-01",
            section_hashes=[],
        )
        assert version.extractor_version == "1.0"


class TestContentVersionTracker:
    """Tests for ContentVersionTracker."""

    def test_compute_hash(self):
        """Test content hash computation."""
        tracker = ContentVersionTracker()

        hash1 = tracker.compute_hash("content")
        hash2 = tracker.compute_hash("content")
        hash3 = tracker.compute_hash("different")

        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 64  # SHA-256 hex length

    def test_needs_reindex_new_document(self):
        """Test new document always needs indexing."""
        tracker = ContentVersionTracker()

        needs_reindex, _ = tracker.needs_reindex(
            Path("/test/new.md"),
            None,  # No stored version
        )

        assert needs_reindex is True

    def test_needs_reindex_unchanged(self, tmp_path):
        """Test unchanged document doesn't need reindex."""
        tracker = ContentVersionTracker()

        # Create file
        doc_file = tmp_path / "doc.md"
        doc_file.write_text("content")

        # Create stored version
        content_hash = tracker.compute_hash("content")
        stored = DocumentVersion(
            path=str(doc_file),
            content_hash=content_hash,
            modified_date="2024-01-01",
            indexed_at="2024-01-01",
            section_hashes=[],
        )

        needs_reindex, _ = tracker.needs_reindex(doc_file, stored)

        assert needs_reindex is False

    def test_needs_reindex_changed(self, tmp_path):
        """Test changed document needs reindex."""
        tracker = ContentVersionTracker()

        doc_file = tmp_path / "doc.md"
        doc_file.write_text("new content")

        # Old version with different hash
        stored = DocumentVersion(
            path=str(doc_file),
            content_hash="old_hash",
            modified_date="2024-01-01",
            indexed_at="2024-01-01",
            section_hashes=[],
        )

        needs_reindex, _ = tracker.needs_reindex(doc_file, stored)

        assert needs_reindex is True

    def test_needs_reindex_extractor_changed(self, tmp_path):
        """Test document needs reindex when extractor version changes."""
        tracker = ContentVersionTracker()

        doc_file = tmp_path / "doc.md"
        doc_file.write_text("content")

        stored = DocumentVersion(
            path=str(doc_file),
            content_hash=tracker.compute_hash("content"),
            modified_date="2024-01-01",
            indexed_at="2024-01-01",
            section_hashes=[],
            extractor_version="0.9",  # Old version
        )

        needs_reindex, _ = tracker.needs_reindex(
            doc_file, stored, extractor_version="1.0"
        )

        assert needs_reindex is True

    def test_create_version_from_document(self):
        """Test creating version from extracted document."""
        tracker = ContentVersionTracker()

        doc = ExtractedDocument(
            path="/test/doc.md",
            file_type=".md",
            content="Test content",
            sections=[
                ExtractedSection(
                    title="Section 1",
                    level=1,
                    start_line=0,
                    end_line=5,
                    content="Section content",
                    subsections=[],
                ),
            ],
            code_blocks=[],
            code_references=[],
            word_count=2,
            modified_date="2024-01-01",
        )

        version = tracker.create_version(doc)

        assert version.path == "/test/doc.md"
        assert version.content_hash == tracker.compute_hash("Test content")
        assert len(version.section_hashes) == 1

    def test_compute_section_hashes(self):
        """Test computing hashes for sections."""
        tracker = ContentVersionTracker()

        sections = [
            ExtractedSection(
                title="S1",
                level=1,
                start_line=0,
                end_line=5,
                content="Content 1",
                subsections=[],
            ),
            ExtractedSection(
                title="S2",
                level=1,
                start_line=6,
                end_line=10,
                content="Content 2",
                subsections=[],
            ),
        ]

        hashes = tracker.compute_section_hashes(sections)

        assert len(hashes) == 2
        assert hashes[0] == tracker.compute_hash("Content 1")
        assert hashes[1] == tracker.compute_hash("Content 2")

    def test_compare_versions(self):
        """Test comparing two document versions."""
        tracker = ContentVersionTracker()

        old = DocumentVersion(
            path="/test/doc.md",
            content_hash="old",
            modified_date="2024-01-01",
            indexed_at="2024-01-01",
            section_hashes=["h1", "h2"],
        )

        new = DocumentVersion(
            path="/test/doc.md",
            content_hash="new",
            modified_date="2024-01-02",
            indexed_at="2024-01-02",
            section_hashes=["h1", "h3"],
        )

        changes = tracker.compare_versions(old, new)

        assert changes["content_changed"] is True
        assert "h3" in changes["sections_added"]
        assert "h2" in changes["sections_removed"]


class TestVersionCache:
    """Tests for VersionCache."""

    def test_set_and_get(self, tmp_path):
        """Test setting and getting versions."""
        cache = VersionCache(tmp_path / "versions.json")

        version = DocumentVersion(
            path="/test/doc.md",
            content_hash="hash",
            modified_date="2024-01-01",
            indexed_at="2024-01-01",
            section_hashes=[],
        )

        cache.set(version)
        retrieved = cache.get("/test/doc.md")

        assert retrieved is not None
        assert retrieved.path == version.path
        assert retrieved.content_hash == version.content_hash

    def test_remove(self, tmp_path):
        """Test removing a version."""
        cache = VersionCache(tmp_path / "versions.json")

        version = DocumentVersion(
            path="/test/doc.md",
            content_hash="hash",
            modified_date="2024-01-01",
            indexed_at="2024-01-01",
            section_hashes=[],
        )

        cache.set(version)
        assert cache.get("/test/doc.md") is not None

        cache.remove("/test/doc.md")
        assert cache.get("/test/doc.md") is None

    def test_persistence(self, tmp_path):
        """Test cache persistence to disk."""
        cache_path = tmp_path / "versions.json"

        # Create and save
        cache1 = VersionCache(cache_path)
        version = DocumentVersion(
            path="/test/doc.md",
            content_hash="hash",
            modified_date="2024-01-01",
            indexed_at="2024-01-01",
            section_hashes=[],
        )
        cache1.set(version)
        cache1.save(cache_path)

        # Load in new instance
        cache2 = VersionCache(cache_path)
        retrieved = cache2.get("/test/doc.md")

        assert retrieved is not None
        assert retrieved.content_hash == "hash"

    def test_clear(self, tmp_path):
        """Test clearing cache."""
        cache = VersionCache(tmp_path / "versions.json")

        for i in range(3):
            cache.set(
                DocumentVersion(
                    path=f"/test/doc{i}.md",
                    content_hash=f"hash{i}",
                    modified_date="2024-01-01",
                    indexed_at="2024-01-01",
                    section_hashes=[],
                )
            )

        cache.clear()
        assert cache.get("/test/doc0.md") is None
        assert cache.get("/test/doc1.md") is None
        assert cache.get("/test/doc2.md") is None