"""Document GraphRAG module for Code-Graph-RAG.

This module extends Code-Graph-RAG to support document indexing, search, and
validation while maintaining strict separation between code and document graphs.

Key components:
- extractors: Document format extractors (Markdown, PDF, DOCX)
- tools: Document-specific MCP tools
- utils: Document utilities (text extraction, reference extraction)
"""

from .document_updater import DocumentGraphUpdater
from .chunking import SemanticDocumentChunker, DocumentChunk
from .error_handling import ErrorType, ExtractionError, ExtractionException, DeadLetterQueue
from .versioning import ContentVersionTracker, VersionCache, DocumentVersion

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