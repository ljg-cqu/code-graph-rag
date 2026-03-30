"""Document extractors registry and factory.

Pattern follows codebase_rag/embeddings/__init__.py:
- Registry pattern with _EXTRACTOR_REGISTRY
- Factory function get_extractor_for_file()
- Lazy loading of optional extractors
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseDocumentExtractor, ExtractedDocument

# Registry pattern following codebase_rag/embeddings/__init__.py
_EXTRACTOR_REGISTRY: dict[str, type[BaseDocumentExtractor]] = {}


def _register_extractor(cls: type[BaseDocumentExtractor]) -> None:
    """Register an extractor for all its supported extensions."""
    instance = cls()
    for ext in instance.supported_extensions:
        _EXTRACTOR_REGISTRY[ext.lower()] = cls


def get_extractor_for_file(file_path: Path) -> BaseDocumentExtractor | None:
    """
    Get appropriate extractor for file type.

    Args:
        file_path: Path to the document file

    Returns:
        Extractor instance or None if no extractor supports this file type
    """
    ext = file_path.suffix.lower()
    extractor_cls = _EXTRACTOR_REGISTRY.get(ext)
    return extractor_cls() if extractor_cls else None


def get_supported_extensions() -> list[str]:
    """Get list of all supported file extensions."""
    return list(_EXTRACTOR_REGISTRY.keys())


# Import and register extractors
from .base import BaseDocumentExtractor, ExtractedDocument, ExtractedSection, ExtractionError
from .markdown_extractor import MarkdownExtractor

_register_extractor(MarkdownExtractor)

# Optional extractors (import if dependencies available)
try:
    from .pdf_extractor import PDFExtractor

    _register_extractor(PDFExtractor)
except ImportError:
    pass

try:
    from .docx_extractor import DocxExtractor

    _register_extractor(DocxExtractor)
except ImportError:
    pass


__all__ = [
    "get_extractor_for_file",
    "get_supported_extensions",
    "BaseDocumentExtractor",
    "ExtractedDocument",
    "ExtractedSection",
    "ExtractionError",
    "MarkdownExtractor",
    "_EXTRACTOR_REGISTRY",
]