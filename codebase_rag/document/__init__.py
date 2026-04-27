"""Document GraphRAG module for Code-Graph-RAG.

This module extends Code-Graph-RAG to support document indexing, search, and
validation while maintaining strict separation between code and document graphs.

Key components:
- extractors: Document format extractors (Markdown, PDF, DOCX)
- tools: Document-specific MCP tools
- utils: Document utilities (text extraction, reference extraction)
"""

from .chunking import DocumentChunk, SemanticDocumentChunker
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from .concept_extraction import (
    VERB_REGISTRY,
    ENTITY_SUBTYPE_REGISTRY,
    ConceptExtractionStats,
    ExtractedConcept,
    ConceptRelationship,
    ExtractionResult,
    LLMConceptExtractor,
    calculate_adaptive_timeout,
    classify_concept_extraction_error,
    get_user_facing_message,
    resolve_category,
    resolve_entity_category,
)
from .concept_runner import (
    ConceptExtractionRunner,
    StandaloneExtractionStats,
)
from .document_updater import (
    DocumentGraphUnavailableError,
    DocumentGraphUpdater,
)
from .error_handling import (
    DeadLetterQueue,
    ErrorType,
    ExtractionError,
    ExtractionException,
)
from .versioning import ContentVersionTracker, DocumentVersion, VersionCache

__all__ = [
    "DocumentGraphUpdater",
    "DocumentGraphUnavailableError",
    "SemanticDocumentChunker",
    "DocumentChunk",
    "ErrorType",
    "ExtractionError",
    "ExtractionException",
    "DeadLetterQueue",
    "ContentVersionTracker",
    "VersionCache",
    "DocumentVersion",
    # Concept extraction
    "ConceptExtractionStats",
    "ExtractedConcept",
    "ConceptRelationship",
    "ExtractionResult",
    "LLMConceptExtractor",
    "calculate_adaptive_timeout",
    "classify_concept_extraction_error",
    "get_user_facing_message",
    "resolve_category",
    "resolve_entity_category",
    "VERB_REGISTRY",
    "ENTITY_SUBTYPE_REGISTRY",
    # Standalone concept extraction
    "ConceptExtractionRunner",
    "StandaloneExtractionStats",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
]
