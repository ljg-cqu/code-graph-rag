"""Abstract base class for document extractors.

Pattern follows EmbeddingProvider from codebase_rag/embeddings/base.py:
- Registry pattern in __init__.py
- Factory function get_extractor_for_file()
- Stateless extraction
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..error_handling import ErrorType, ExtractionError, ExtractionException

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class ExtractedSection:
    """Represents a document section."""

    title: str
    level: int  # Header level (1-6)
    start_line: int
    end_line: int
    content: str
    subsections: list[ExtractedSection] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "level": self.level,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
        }


@dataclass
class ExtractedDocument:
    """Represents an extracted document."""

    path: str
    file_type: str
    content: str
    sections: list[ExtractedSection]
    code_blocks: list[str]
    code_references: list[str]  # Extracted function/class names
    word_count: int
    modified_date: str
    content_hash: str = ""

    def __post_init__(self) -> None:
        """Compute content hash if not provided."""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "file_type": self.file_type,
            "content": self.content,
            "sections": [s.to_dict() for s in self.sections],
            "code_blocks": self.code_blocks,
            "code_references": self.code_references,
            "word_count": self.word_count,
            "modified_date": self.modified_date,
            "content_hash": self.content_hash,
        }


class BaseDocumentExtractor(ABC):
    """
    Abstract base class for document extractors.

    Pattern follows EmbeddingProvider from codebase_rag/embeddings/base.py:
    - Registry pattern in __init__.py
    - Factory function get_extractor_for_file()
    - Stateless extraction
    """

    __slots__ = ("_config",)

    def __init__(self, **config: str | int | None) -> None:
        self._config: dict[str, str | int | None] = config

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """List of supported file extensions (e.g., ['.md', '.rst'])."""
        pass

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Extract content from document.

        Args:
            file_path: Path to document file

        Returns:
            ExtractedDocument with all extracted content

        Raises:
            ExtractionException: If extraction fails
        """
        pass

    @abstractmethod
    async def extract_async(self, file_path: Path) -> ExtractedDocument:
        """
        Async extraction for large files.

        REQUIRED: All extractors must implement async variant.
        Sync extract() can delegate to async for small files.

        Args:
            file_path: Path to document file

        Returns:
            ExtractedDocument with all extracted content
        """
        pass

    async def extract_batch_async(
        self, file_paths: list[Path]
    ) -> list[ExtractedDocument | ExtractionError]:
        """
        Batch extraction with concurrent processing.

        Uses asyncio.gather for parallel extraction.

        Args:
            file_paths: List of paths to extract

        Returns:
            List of extracted documents or errors (same order as input)
        """
        import asyncio

        results = await asyncio.gather(
            *[self._extract_with_error_boundary(p) for p in file_paths],
            return_exceptions=True,
        )

        # Convert exceptions to ExtractionError
        processed: list[ExtractedDocument | ExtractionError] = []
        for i, result in enumerate(results):
            if isinstance(result, ExtractedDocument):
                processed.append(result)
            elif isinstance(result, ExtractionError):
                processed.append(result)
            elif isinstance(result, Exception):
                processed.append(
                    ExtractionError(
                        path=str(file_paths[i]),
                        error_type=ErrorType.UNKNOWN,
                        message=str(result),
                    )
                )
            else:
                processed.append(
                    ExtractionError(
                        path=str(file_paths[i]),
                        error_type=ErrorType.UNKNOWN,
                        message=f"Unexpected result type: {type(result)}",
                    )
                )

        return processed

    async def _extract_with_error_boundary(
        self, file_path: Path
    ) -> ExtractedDocument | ExtractionError:
        """
        Extraction with error boundary - never crashes pipeline.

        Returns either extracted document or structured error.
        """
        try:
            return await self.extract_async(file_path)
        except ExtractionException as e:
            return e.to_extraction_error()
        except Exception as e:
            return ExtractionError(
                path=str(file_path),
                error_type=ErrorType.UNKNOWN,
                message=str(e),
            )

    def _extract_code_references(self, content: str) -> list[str]:
        """
        Extract potential function/class references from content.

        Uses deterministic patterns (regex), not LLM.
        """
        patterns = [
            r"`([a-zA-Z_][\w]*(?:\.[\w]+)*)\(\)`",  # Function calls
            r"`([a-zA-Z_][\w]*(?:\.[\w]+)*)`",  # Generic code references
            r"class `(\w+)`",  # Class references
            r"method `(\w+)`",  # Method references
            r"function `(\w+)`",  # Function references
            r"\[([a-zA-Z_][\w]*(?:\.[\w]+)*)\]",  # Bracketed references
        ]

        refs = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            refs.extend(matches)

        # Deduplicate and limit
        unique_refs = list(dict.fromkeys(refs))[:50]
        return unique_refs

    MAX_SYMLINK_DEPTH = 10

    def _validate_path(self, file_path: Path, repo_root: Path | None = None, depth: int = 0) -> Path:
        """
        Validate path is safe and exists.

        Args:
            file_path: Path to validate
            repo_root: Optional repository root for path traversal check
            depth: Current symlink recursion depth (internal use)

        Returns:
            Resolved path

        Raises:
            ExtractionException: If path is invalid or unsafe
        """
        if depth > self.MAX_SYMLINK_DEPTH:
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.PATH_TRAVERSAL,
                message="Symlink chain too deep (potential cycle)",
            )

        if not file_path.exists():
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.FILE_NOT_FOUND,
                message=f"File does not exist: {file_path}",
            )

        if not file_path.is_file():
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.NOT_A_FILE,
                message=f"Path is not a file: {file_path}",
            )

        resolved = file_path.resolve()

        # Path traversal check if repo_root provided
        # Use proper path containment check (Python 3.9+)
        if repo_root:
            repo_resolved = repo_root.resolve()
            try:
                if not resolved.is_relative_to(repo_resolved):
                    raise ExtractionException(
                        path=str(file_path),
                        error_type=ErrorType.PATH_TRAVERSAL,
                        message=f"Path traversal detected: {file_path}",
                    )
            except (OSError, ValueError):
                # Fallback for edge cases
                try:
                    resolved.relative_to(repo_resolved)
                except ValueError:
                    raise ExtractionException(
                        path=str(file_path),
                        error_type=ErrorType.PATH_TRAVERSAL,
                        message=f"Path traversal detected: {file_path}",
                    )

        # Symlink check - recursively validate with depth limit
        if resolved.is_symlink():
            return self._validate_path(resolved.resolve(), repo_root, depth + 1)

        return resolved

    def get_config(self, key: str, default: str | int | None = None) -> str | int | None:
        """Get a configuration value."""
        return self._config.get(key, default)


__all__ = [
    "BaseDocumentExtractor",
    "ExtractedDocument",
    "ExtractedSection",
    "ExtractionError",
    "ExtractionException",
]