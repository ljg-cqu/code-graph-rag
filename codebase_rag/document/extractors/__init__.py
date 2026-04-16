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
    from .base import (
        BaseDocumentExtractor,
        ExtractedDocument,
        ExtractedSection,
        ExtractionError,
    )

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


def _bootstrap_extractors() -> None:
    from .base import (
        BaseDocumentExtractor as _BaseDocumentExtractor,
    )
    from .base import (
        ExtractedDocument as _ExtractedDocument,
    )
    from .base import (
        ExtractedSection as _ExtractedSection,
    )
    from .base import (
        ExtractionError as _ExtractionError,
    )
    from .markdown_extractor import MarkdownExtractor as _MarkdownExtractor

    globals().update(
        {
            "BaseDocumentExtractor": _BaseDocumentExtractor,
            "ExtractedDocument": _ExtractedDocument,
            "ExtractedSection": _ExtractedSection,
            "ExtractionError": _ExtractionError,
            "MarkdownExtractor": _MarkdownExtractor,
        }
    )
    _register_extractor(_MarkdownExtractor)

    try:
        from .pdf_extractor import PDFExtractor
    except ImportError:
        pass
    else:
        globals()["PDFExtractor"] = PDFExtractor
        _register_extractor(PDFExtractor)

    try:
        from .docx_extractor import DocxExtractor
    except ImportError:
        pass
    else:
        globals()["DocxExtractor"] = DocxExtractor
        _register_extractor(DocxExtractor)


_bootstrap_extractors()


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
