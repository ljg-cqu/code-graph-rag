"""Error codes and shared types for schema validation.

This module defines namespaced error codes for structured error handling
and common type definitions used across the schemas package.
"""

from __future__ import annotations

from typing import Literal


class ErrorCodes:
    """Namespaced error codes for validation errors.

    Format: {domain}.{field}.{issue} for machine-readable parsing.

    Error codes are used in ValidationError.code field and can be
    programmatically handled by downstream consumers.
    """

    class Schema:
        """Schema-level error codes."""

        VERSION_MISMATCH = "schema.version.mismatch"
        INVALID_JSON = "schema.invalid_json"
        VALIDATION_FAILED = "schema.validation_failed"

    class Entity:
        """Entity-level error codes."""

        ID_MISSING = "entity.id.missing"
        ID_INVALID = "entity.id.invalid"
        ID_DUPLICATE = "entity.id.duplicate"
        ID_AUTO_GENERATED = "entity.id.auto_generated"
        NAME_MISSING = "entity.name.missing"
        NAME_EMPTY = "entity.name.empty"
        NAME_DUPLICATE = "entity.name.duplicate"
        NAME_AMBIGUOUS = "entity.name.ambiguous"

    class Relationship:
        """Relationship-level error codes."""

        SOURCE_MISSING = "relationship.source.missing"
        TARGET_MISSING = "relationship.target.missing"
        TYPE_MISSING = "relationship.type.missing"
        TYPE_INVALID = "relationship.type.invalid"
        SOURCE_NOT_FOUND = "relationship.source.not_found"
        TARGET_NOT_FOUND = "relationship.target.not_found"
        COMMA_IN_TARGET = "relationship.target.comma_separated"
        SELF_REFERENCE = "relationship.self_reference"

    class Reference:
        """Reference-related error codes."""

        CIRCULAR = "reference.circular"
        SELF = "reference.self"

    class Type:
        """Type-related error codes."""

        MISMATCH = "type.mismatch"
        FORMAT_INVALID = "type.format.invalid"


# Severity levels for validation issues
Severity = Literal["error", "warning"]
