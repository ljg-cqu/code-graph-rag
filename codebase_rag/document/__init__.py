"""Document GraphRAG module for Code-Graph-RAG.

This module extends Code-Graph-RAG to support document indexing, search, and
validation while maintaining strict separation between code and document graphs.

Key components:
- extractors: Document format extractors (Markdown, PDF, DOCX)
- tools: Document-specific MCP tools
- utils: Document utilities (text extraction, reference extraction)
"""

from .chunking import DocumentChunk, SemanticDocumentChunker
from .document_updater import DocumentGraphUpdater
from .error_handling import (
    DeadLetterQueue,
    ErrorType,
    ExtractionError,
    ExtractionException,
)
from .versioning import ContentVersionTracker, DocumentVersion, VersionCache

__all__ = [
    "DocumentGraphUpdater",
    "SemanticDocumentChunker",
    "DocumentChunk",
    "ErrorType",
    "ExtractionError",
    "ExtractionException",
    "DeadLetterQueue",
    "ContentVersionTracker",
    "VersionCache",
    "DocumentVersion",
]
