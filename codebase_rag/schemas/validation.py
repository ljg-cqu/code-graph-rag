"""Unified validation logic for JSON ingestion payloads.

This module provides structured validation with:
- ValidationError dataclass with path, code, message, severity
- ValidationResult with legacy tuple adapter
- validate_ingestion_payload() for new code
- validate_json_input_legacy() for backward compatibility
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import ValidationError as PydanticValidationError

from .types import ErrorCodes, Severity

SCHEMA_PATH = Path(__file__).parent.parent / "schema.json"


def _load_schema() -> dict[str, Any]:
    """Load the ingestion schema from disk."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


INGESTION_SCHEMA = _load_schema()


@dataclass(frozen=True)
class ValidationError:
    """Structured validation error with full context.

    Attributes:
        path: JSON path to the error location (e.g., "entities[0].name")
        code: Machine-readable error code (e.g., "entity.id.duplicate")
        message: Human-readable error message
        severity: Either "error" or "warning"
        context: Optional additional context for debugging
    """

    path: str
    code: str
    message: str
    severity: Severity
    context: dict[str, Any] | None = None


@dataclass
class ValidationResult:
    """Complete validation result with structured errors.

    Attributes:
        is_valid: True if validation passed with no errors
        data: Normalized data if valid, None otherwise
        errors: List of validation errors
        warnings: List of validation warnings
    """

    is_valid: bool = False
    data: dict[str, Any] | None = None
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    def has_error_code(self, code: str) -> bool:
        """Check if a specific error code exists in errors."""
        return any(e.code == code for e in self.errors)

    def errors_for_path(self, path: str) -> list[ValidationError]:
        """Get all errors for a specific path prefix."""
        return [e for e in self.errors if e.path.startswith(path)]

    def to_tuple(self) -> tuple[bool, dict[str, Any] | None, list[str]]:
        """Convert to legacy tuple format for backward compatibility.

        Returns:
            Tuple of (is_valid, normalized_data, error_messages)
        """
        error_messages = [f"{e.path}: {e.message}" for e in self.errors]
        return (self.is_valid, self.data, error_messages)


@dataclass
class ValidationContext:
    """Mutable context for collecting validation results."""

    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    fail_fast: bool = False

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def add_error(
        self,
        path: str,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.errors.append(
            ValidationError(
                path=path,
                code=code,
                message=message,
                severity="error",
                context=context,
            )
        )

    def add_warning(
        self,
        path: str,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        self.warnings.append(
            ValidationError(
                path=path,
                code=code,
                message=message,
                severity="warning",
                context=context,
            )
        )


def canonical_entity_id(name: str) -> str:
    """Generate a canonical entity ID from a name.

    This is the public API for canonical ID generation. It mirrors
    the existing private _canonical_entity_id() logic in json_ingestion.py.

    Args:
        name: Entity name to convert

    Returns:
        URL-safe entity ID (lowercase, alphanumeric + underscore)

    Raises:
        ValueError: If name is empty or whitespace

    Example:
        >>> canonical_entity_id("User Service")
        'user_service'
        >>> canonical_entity_id("API-Gateway")
        'api_gateway'
    """
    if not name or not name.strip():
        raise ValueError("Entity name cannot be empty")
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return slug or "entity"


def validate_ingestion_payload(
    data: dict[str, Any],
    strict: bool = False,
    allow_external_references: bool = False,
) -> ValidationResult:
    """Validate an ingestion payload with structured error reporting.

    Two-phase validation:
    1. JSON Schema validation (structure + types)
    2. Custom business logic validation (references, uniqueness, etc.)

    Args:
        data: Raw JSON data to validate
        strict: If True, reject deprecated features (comma-separated targets,
            missing IDs). If False, emit warnings instead.
        allow_external_references: If True, skip reference validation

    Returns:
        ValidationResult with errors, warnings, and normalized data
    """
    ctx = ValidationContext()

    # Phase 1: JSON Schema validation
    normalized_data: dict[str, Any] | None = None
    try:
        normalized_data = copy.deepcopy(data)
        jsonschema.validate(instance=normalized_data, schema=INGESTION_SCHEMA)
    except jsonschema.exceptions.ValidationError as e:
        ctx.add_error(
            path=".".join(str(p) for p in e.path) or "root",
            code=ErrorCodes.Schema.VALIDATION_FAILED,
            message=e.message,
            context={"validator": e.validator, "schema_path": list(e.schema_path)},
        )
        return ValidationResult(
            is_valid=False,
            data=None,
            errors=ctx.errors,
            warnings=ctx.warnings,
        )
    except Exception as e:
        ctx.add_error(
            path="root",
            code=ErrorCodes.Schema.INVALID_JSON,
            message=str(e),
        )
        return ValidationResult(
            is_valid=False,
            data=None,
            errors=ctx.errors,
            warnings=ctx.warnings,
        )

    # Phase 2: Business logic validation
    _normalize_and_validate_entities(normalized_data, ctx, strict)
    _normalize_and_validate_relationships(normalized_data, ctx, strict)

    if not allow_external_references:
        _validate_relationship_references(normalized_data, ctx)

    return ValidationResult(
        is_valid=not ctx.has_errors,
        data=normalized_data if not ctx.has_errors else None,
        errors=ctx.errors,
        warnings=ctx.warnings,
    )


def _normalize_and_validate_entities(
    data: dict[str, Any],
    ctx: ValidationContext,
    strict: bool,
) -> None:
    """Normalize and validate entity constraints."""
    entities = data.get("entities", [])
    default_labels = list(data.get("metadata", {}).get("default_entity_labels") or [])
    batch_operation = _batch_operation(data)
    batch_last_updated = _batch_last_updated(data)
    seen_ids: set[str] = set()
    name_to_id: dict[str, str] = {}

    for idx, entity in enumerate(entities):
        path = f"entities[{idx}]"

        # Initialize properties if missing
        if entity.get("properties") is None:
            entity["properties"] = {}

        # Apply default labels
        if default_labels and not entity.get("labels"):
            entity["labels"] = list(default_labels)

        # Apply batch-level operation
        if not entity.get("operation"):
            entity["operation"] = batch_operation

        # Apply batch-level timestamp
        if batch_last_updated and not entity.get("last_updated"):
            entity["last_updated"] = batch_last_updated

        # Handle missing entity ID
        entity_id = entity.get("id")
        name = entity.get("name")

        if not entity_id and name:
            entity_id = canonical_entity_id(str(name))
            entity["id"] = entity_id
            if strict:
                ctx.add_error(
                    path=f"{path}.id",
                    code=ErrorCodes.Entity.ID_MISSING,
                    message=f"Entity ID is required. Auto-generated: '{entity_id}'",
                    context={"name": name, "generated_id": entity_id},
                )
            else:
                ctx.add_warning(
                    path=f"{path}.id",
                    code=ErrorCodes.Entity.ID_AUTO_GENERATED,
                    message=(
                        f"Entity ID auto-generated from name '{name}'. "
                        "Explicit IDs will be required in v2.0."
                    ),
                    context={"name": name, "generated_id": entity_id},
                )

        # Check ID uniqueness
        if entity_id:
            if entity_id in seen_ids:
                ctx.add_error(
                    path=f"{path}.id",
                    code=ErrorCodes.Entity.ID_DUPLICATE,
                    message=f"Duplicate entity ID: {entity_id}",
                    context={"duplicate_id": entity_id},
                )
            seen_ids.add(str(entity_id))

        # Check name uniqueness
        if name:
            if name in name_to_id:
                ctx.add_error(
                    path=f"{path}.name",
                    code=ErrorCodes.Entity.NAME_DUPLICATE,
                    message=f"Entity name '{name}' is already used by entity '{name_to_id[name]}'",
                    context={"name": name, "existing_id": name_to_id[name]},
                )
            else:
                name_to_id[name] = str(entity_id) if entity_id else ""


def _normalize_and_validate_relationships(
    data: dict[str, Any],
    ctx: ValidationContext,
    strict: bool,
) -> None:
    """Normalize and validate relationship constraints."""
    raw_relationships = data.get("relationships", [])

    # Handle wrapped relationships format
    if isinstance(raw_relationships, dict):
        relationship_items = raw_relationships.get("relationships", [])
    else:
        relationship_items = raw_relationships

    batch_operation = _batch_operation(data)
    batch_last_updated = _batch_last_updated(data)

    normalized_relationships: list[dict[str, Any]] = []
    for idx, relationship in enumerate(relationship_items):
        target = relationship.get("target")

        # Handle comma-separated targets (deprecated)
        if isinstance(target, str) and "," in target:
            targets = [t.strip() for t in target.split(",") if t.strip()]
            if strict:
                ctx.add_error(
                    path=f"relationships[{idx}].target",
                    code=ErrorCodes.Relationship.COMMA_IN_TARGET,
                    message=(
                        f"Comma-separated targets are not allowed: '{target}'. "
                        "Create multiple relationship objects instead."
                    ),
                    context={"target": target, "suggested_split": targets},
                )
            else:
                ctx.add_warning(
                    path=f"relationships[{idx}].target",
                    code=ErrorCodes.Relationship.COMMA_IN_TARGET,
                    message=(
                        f"Comma-separated targets are deprecated: '{target}'. "
                        "Create multiple relationship objects instead. "
                        "This will be an error in v2.0."
                    ),
                    context={"target": target, "suggested_split": targets},
                )
            # Split into multiple relationships
            for t in targets:
                new_rel = copy.deepcopy(relationship)
                new_rel["target"] = t
                normalized_relationships.append(new_rel)
        else:
            normalized_relationships.append(copy.deepcopy(relationship))

    # Apply batch-level defaults to normalized relationships
    for relationship in normalized_relationships:
        if relationship.get("properties") is None:
            relationship["properties"] = {}

        if not relationship.get("operation"):
            relationship["operation"] = batch_operation

        if batch_last_updated and not relationship.get("last_updated"):
            relationship["last_updated"] = batch_last_updated

    data["relationships"] = normalized_relationships


def _validate_relationship_references(
    data: dict[str, Any],
    ctx: ValidationContext,
) -> None:
    """Ensure all relationship references exist in entities."""
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    # Build set of valid entity refs (both IDs and names)
    entity_refs = {
        str(entity_ref)
        for entity in entities
        for entity_ref in (entity.get("id"), entity.get("name"))
        if entity_ref is not None
    }

    for idx, relationship in enumerate(relationships):
        path = f"relationships[{idx}]"
        source = str(relationship.get("source", ""))
        target = str(relationship.get("target", ""))

        if source and source not in entity_refs:
            ctx.add_error(
                path=f"{path}.source",
                code=ErrorCodes.Relationship.SOURCE_NOT_FOUND,
                message=f"Source entity '{source}' not found in entities",
                context={"source": source, "available_refs": sorted(entity_refs)},
            )

        if target and target not in entity_refs:
            ctx.add_error(
                path=f"{path}.target",
                code=ErrorCodes.Relationship.TARGET_NOT_FOUND,
                message=f"Target entity '{target}' not found in entities",
                context={"target": target, "available_refs": sorted(entity_refs)},
            )


def _batch_operation(data: dict[str, Any]) -> str:
    """Extract batch-level operation with default fallback."""
    raw_operation = data.get("operation") or data.get("metadata", {}).get("operation")
    if raw_operation is None:
        return "add"
    return str(raw_operation).strip().lower() or "add"


def _batch_last_updated(data: dict[str, Any]) -> str | None:
    """Extract batch-level last_updated timestamp."""
    raw_last_updated = data.get("last_updated") or data.get("metadata", {}).get(
        "last_updated"
    )
    if raw_last_updated is None:
        return None
    value = str(raw_last_updated).strip()
    return value or None


def validate_json_input_legacy(
    data: dict[str, Any],
    allow_external_references: bool = False,
    errors: list[str] | None = None,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    """Legacy adapter for backward compatibility.

    Maintains exact same signature as json_ingestion.validate_json_input()
    for drop-in replacement.

    Args:
        data: Raw JSON data as dict (matches current API)
        allow_external_references: If True, skip reference validation
        errors: Optional list to append errors to (for chaining)

    Returns:
        Tuple of (is_valid, normalized_data, error_messages)
    """
    if errors is None:
        errors = []

    result = validate_ingestion_payload(
        data,
        strict=False,
        allow_external_references=allow_external_references,
    )

    # Convert structured errors to legacy string format
    for error in result.errors:
        errors.append(f"{error.path}: {error.message}")

    return (result.is_valid, result.data, errors)
