"""Content versioning for incremental document updates.

Documents are tracked with content fingerprints for incremental updates
to avoid full re-indexing.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .extractors.base import ExtractedDocument, ExtractedSection


@dataclass
class DocumentVersion:
    """Version tracking for incremental updates."""

    path: str
    content_hash: str  # SHA-256 of content
    modified_date: str  # File modification time
    indexed_at: str  # Last indexing timestamp
    section_hashes: list[str]  # Hash of each section
    extractor_version: str = "1.0"  # Algorithm version for re-index trigger


class ContentVersionTracker:
    """
    Track document versions for incremental updates.

    Strategy:
    1. Compute content hash at extraction
    2. Compare with stored hash
    3. Only re-index changed documents/sections
    """

    HASH_ALGORITHM = "sha256"
    CURRENT_EXTRACTOR_VERSION = "1.0"

    def compute_hash(self, content: str) -> str:
        """Compute content fingerprint."""
        return hashlib.sha256(content.encode()).hexdigest()

    def compute_section_hashes(
        self, sections: list[ExtractedSection]
    ) -> list[str]:
        """Compute hashes for each section."""
        return [self.compute_hash(s.content) for s in sections]

    def needs_reindex(
        self,
        file_path: Path,
        stored_version: DocumentVersion | None,
        extractor_version: str | None = None,
    ) -> tuple[bool, list[int]]:
        """
        Determine if document needs re-indexing and which sections changed.

        Args:
            file_path: Path to the document
            stored_version: Previously stored version (None for new docs)
            extractor_version: Current extractor version

        Returns:
            Tuple of (needs_full_reindex, changed_section_indices)
        """
        # New document - always needs indexing
        if stored_version is None:
            return True, []

        # Extractor version changed - need full reindex
        current_version = extractor_version or self.CURRENT_EXTRACTOR_VERSION
        if stored_version.extractor_version != current_version:
            return True, []

        # Check content hash
        try:
            current_hash = self.compute_hash(file_path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, OSError):
            # Can't read - needs reindex attempt
            return True, []

        if current_hash != stored_version.content_hash:
            # Content changed - identify which sections
            # (Full reindex for now; section-level diff is future work)
            return True, []

        # No changes detected
        return False, []

    def create_version(
        self,
        doc: ExtractedDocument,
        extractor_version: str | None = None,
    ) -> DocumentVersion:
        """
        Create version metadata for a document.

        Args:
            doc: Extracted document
            extractor_version: Version of the extractor used

        Returns:
            DocumentVersion with tracking metadata
        """
        return DocumentVersion(
            path=doc.path,
            content_hash=self.compute_hash(doc.content),
            modified_date=doc.modified_date,
            indexed_at=datetime.now(UTC).isoformat(),
            section_hashes=self.compute_section_hashes(doc.sections),
            extractor_version=extractor_version or self.CURRENT_EXTRACTOR_VERSION,
        )

    def compare_versions(
        self,
        old_version: DocumentVersion,
        new_version: DocumentVersion,
    ) -> dict:
        """
        Compare two versions to identify changes.

        Returns:
            Dict with change details
        """
        changes = {
            "content_changed": old_version.content_hash != new_version.content_hash,
            "sections_added": [],
            "sections_removed": [],
            "sections_modified": [],
        }

        old_hashes = set(old_version.section_hashes)
        new_hashes = set(new_version.section_hashes)

        # Added sections
        changes["sections_added"] = list(new_hashes - old_hashes)

        # Removed sections
        changes["sections_removed"] = list(old_hashes - new_hashes)

        return changes


class VersionCache:
    """
    Cache for document versions.

    Stores version metadata for quick lookup during incremental updates.
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        self._cache: dict[str, DocumentVersion] = {}
        self._cache_path = cache_path

        if cache_path and cache_path.exists():
            self._load(cache_path)

    def _load(self, path: Path) -> None:
        """Load version cache from disk with file locking."""
        try:
            with open(path) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    data = json.loads(f.read())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            for path_str, version_data in data.items():
                self._cache[path_str] = DocumentVersion(
                    path=version_data["path"],
                    content_hash=version_data["content_hash"],
                    modified_date=version_data["modified_date"],
                    indexed_at=version_data["indexed_at"],
                    section_hashes=version_data.get("section_hashes", []),
                    extractor_version=version_data.get("extractor_version", "1.0"),
                )
        except (json.JSONDecodeError, KeyError):
            # Start fresh if cache is corrupted
            self._cache = {}

    def save(self, path: Path | None = None) -> None:
        """Save version cache to disk with file locking."""
        save_path = path or self._cache_path
        if not save_path:
            return

        data = {}
        for path_str, version in self._cache.items():
            data[path_str] = {
                "path": version.path,
                "content_hash": version.content_hash,
                "modified_date": version.modified_date,
                "indexed_at": version.indexed_at,
                "section_hashes": version.section_hashes,
                "extractor_version": version.extractor_version,
            }

        # Use file locking for thread safety
        with open(save_path, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(data, indent=2))
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def get(self, path: str) -> DocumentVersion | None:
        """Get version for a path."""
        return self._cache.get(path)

    def set(self, version: DocumentVersion) -> None:
        """Set version for a path."""
        self._cache[version.path] = version

    def remove(self, path: str) -> bool:
        """Remove version for a path."""
        if path in self._cache:
            del self._cache[path]
            return True
        return False

    def clear(self) -> None:
        """Clear all versions."""
        self._cache.clear()


__all__ = [
    "DocumentVersion",
    "ContentVersionTracker",
    "VersionCache",
]