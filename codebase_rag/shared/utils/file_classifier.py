"""File classification utility.

Determines if a file is code or document for routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class FileType(StrEnum):
    """Classification of file type."""

    CODE = "code"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


@dataclass
class FileClassification:
    """Result of file classification."""

    file_type: FileType
    extension: str
    language: str | None  # For code files
    document_format: str | None  # For documents
    confidence: float  # 0.0 to 1.0


# Known code file extensions
CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rs": "rust",
    ".java": "java",
    ".php": "php",
    ".lua": "lua",
    ".go": "go",
    ".scala": "scala",
    ".cs": "csharp",
    ".sol": "solidity",
    ".vy": "vyper",
    ".kt": "kotlin",
    ".swift": "swift",
    ".rb": "ruby",
}

# Known document file extensions
DOCUMENT_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".rest": "rst",
    ".txt": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".doc": "doc",
    ".adoc": "asciidoc",
    ".asciidoc": "asciidoc",
    ".org": "org",
    ".tex": "latex",
    ".html": "html",
    ".htm": "html",
}


def classify_file(file_path: Path) -> FileClassification:
    """
    Classify a file as code or document.

    Args:
        file_path: Path to the file

    Returns:
        FileClassification with type and metadata
    """
    extension = file_path.suffix.lower()

    # Check code extensions
    if extension in CODE_EXTENSIONS:
        return FileClassification(
            file_type=FileType.CODE,
            extension=extension,
            language=CODE_EXTENSIONS[extension],
            document_format=None,
            confidence=1.0,
        )

    # Check document extensions
    if extension in DOCUMENT_EXTENSIONS:
        return FileClassification(
            file_type=FileType.DOCUMENT,
            extension=extension,
            language=None,
            document_format=DOCUMENT_EXTENSIONS[extension],
            confidence=1.0,
        )

    # Unknown extension
    return FileClassification(
        file_type=FileType.UNKNOWN,
        extension=extension,
        language=None,
        document_format=None,
        confidence=0.0,
    )


def is_code_file(file_path: Path) -> bool:
    """Quick check if file is code."""
    return classify_file(file_path).file_type == FileType.CODE


def is_document_file(file_path: Path) -> bool:
    """Quick check if file is document."""
    return classify_file(file_path).file_type == FileType.DOCUMENT


def get_supported_code_extensions() -> set[str]:
    """Get all supported code file extensions."""
    return set(CODE_EXTENSIONS.keys())


def get_supported_document_extensions() -> set[str]:
    """Get all supported document file extensions."""
    return set(DOCUMENT_EXTENSIONS.keys())


__all__ = [
    "FileType",
    "FileClassification",
    "classify_file",
    "is_code_file",
    "is_document_file",
    "get_supported_code_extensions",
    "get_supported_document_extensions",
]