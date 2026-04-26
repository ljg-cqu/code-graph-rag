"""Schema validation and data models for code-graph-rag.

This module provides:
- Pydantic data models for query results, ingestion, and tools
- Auto-generated ingestion models from JSON Schema
- Structured validation with error codes
- Legacy adapter for backward compatibility

Example usage:
    from codebase_rag.schemas import (
        validate_ingestion_payload,
        validate_json_input_legacy,
        IngestionResult,
        JSONEntity,
    )

    # New structured API
    result = validate_ingestion_payload(data)
    if not result.is_valid:
        for error in result.errors:
            print(f"{error.code}: {error.message}")

    # Legacy API (backward compatible)
    valid, data, errors = validate_json_input_legacy(data)
"""

from __future__ import annotations

# Re-export everything from models.py for backward compatibility
from .models import (
    CodeSnippet,
    EditResult,
    FileCreationResult,
    FileReadResult,
    HealthCheckResult,
    IngestionResult,
    JSONEntity,
    JSONMetadata,
    JSONRelationship,
    PythonObjectInfo,
    QueryGraphData,
    ShellCommandResult,
    UpdateResult,
    _normalize_value,
)

# Import new validation types
from .types import ErrorCodes, Severity
from .validation import (
    INGESTION_SCHEMA,
    ValidationResult,
    ValidationContext,
    ValidationError,
    canonical_entity_id,
    validate_ingestion_payload,
    validate_json_input_legacy,
)

__all__ = [
    # Models (from models.py - backward compatible)
    "CodeSnippet",
    "EditResult",
    "FileCreationResult",
    "FileReadResult",
    "HealthCheckResult",
    "IngestionResult",
    "JSONEntity",
    "JSONMetadata",
    "JSONRelationship",
    "PythonObjectInfo",
    "QueryGraphData",
    "ShellCommandResult",
    "UpdateResult",
    "_normalize_value",
    # Types
    "ErrorCodes",
    "Severity",
    "ValidationError",
    # Validation
    "ValidationResult",
    "ValidationContext",
    "INGESTION_SCHEMA",
    "canonical_entity_id",
    "validate_ingestion_payload",
    "validate_json_input_legacy",
]
