"""Shared infrastructure for Code-Graph-RAG.

This module provides shared utilities for both code and document graphs:
- query_router: Query mode routing
- validation: Bidirectional validation (code↔doc)
- utils: File classification, source attribution
"""

from .query_router import (
    QueryMode,
    QueryRequest,
    QueryResponse,
    QueryRouter,
    Source,
    ValidationResult,
    ValidationReport,
)
from .validation import (
    ValidationTriggerAPI,
    ValidationCache,
    CodeVsDocValidator,
    DocVsCodeValidator,
)
from .utils import classify_file, get_source_label

__all__ = [
    "QueryMode",
    "QueryRequest",
    "QueryResponse",
    "QueryRouter",
    "Source",
    "ValidationResult",
    "ValidationReport",
    "ValidationTriggerAPI",
    "ValidationCache",
    "CodeVsDocValidator",
    "DocVsCodeValidator",
    "classify_file",
    "get_source_label",
]