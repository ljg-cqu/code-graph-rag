# Validation and Error Handling Spec

## Problem Statement

Current validation in `json_ingestion.py` has these issues:

1. **Error messages are strings**, not structured objects
2. **Path information is lost** - errors don't indicate which field failed
3. **No error codes** - difficult to programmatically handle specific errors
4. **Silent mutations** - data is normalized without logging
5. **Aggregation complexity** - mutable lists passed through multiple layers

---

## Proposed Solution

### Structured Error Model

```python
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ValidationError:
    """Structured validation error with full context."""
    
    path: str  # JSON Path: "entities[0].name"
    code: str  # Machine-readable: "required_field_missing"
    message: str  # Human-readable
    severity: Literal["error", "warning"]
    context: dict[str, Any] | None = None  # Additional context


@dataclass
class ValidationResult:
    """Complete validation result."""
    
    is_valid: bool
    data: dict[str, Any] | None  # Normalized data if valid
    errors: list[ValidationError]
    warnings: list[ValidationError]
    
    def has_error_code(self, code: str) -> bool:
        return any(e.code == code for e in self.errors)
    
    def errors_for_path(self, path: str) -> list[ValidationError]:
        return [e for e in self.errors if e.path.startswith(path)]
```

---

## Error Code Taxonomy

```python
class ErrorCodes:
    """Namespaced error codes for validation errors.
    
    Format: {domain}.{field}.{issue} for machine-readable parsing.
    """
    
    class Schema:
        VERSION_MISMATCH = "schema.version.mismatch"
        INVALID_JSON = "schema.invalid_json"
        VALIDATION_FAILED = "schema.validation_failed"
    
    class Entity:
        ID_MISSING = "entity.id.missing"
        ID_INVALID = "entity.id.invalid"  # Pattern mismatch
        ID_DUPLICATE = "entity.id.duplicate"
        ID_AUTO_GENERATED = "entity.id.auto_generated"  # Warning, not error
        NAME_MISSING = "entity.name.missing"
        NAME_EMPTY = "entity.name.empty"
        NAME_DUPLICATE = "entity.name.duplicate"
        NAME_AMBIGUOUS = "entity.name.ambiguous"
    
    class Relationship:
        SOURCE_MISSING = "relationship.source.missing"
        TARGET_MISSING = "relationship.target.missing"
        TYPE_MISSING = "relationship.type.missing"
        TYPE_INVALID = "relationship.type.invalid"  # Pattern mismatch
        SOURCE_NOT_FOUND = "relationship.source.not_found"
        TARGET_NOT_FOUND = "relationship.target.not_found"
        COMMA_IN_TARGET = "relationship.target.comma_separated"  # Deprecated feature
        SELF_REFERENCE = "relationship.self_reference"  # Warning
    
    class Reference:
        CIRCULAR = "reference.circular"
        SELF = "reference.self"
    
    class Type:
        MISMATCH = "type.mismatch"
        FORMAT_INVALID = "type.format.invalid"
```

---

## Integration with Existing Code

### Migration from `validate_json_input()`

The existing function in `json_ingestion.py:286` has this signature:

```python
def validate_json_input(
    data: dict[str, Any],
    allow_external_references: bool = False,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
```

**Migration Strategy**:

1. Create new `ValidationPipeline` in `schemas/validation.py`
2. Create adapter function that wraps new validation with legacy interface:

```python
# codebase_rag/schemas/validation.py

def validate_json_input_legacy(
    data: dict[str, Any],  # Matches existing signature
    allow_external_references: bool = False,  # Maintains feature parity
    errors: list[str] | None = None,  # Optional: append to existing errors
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
        allow_external_references=allow_external_references
    )

    # Convert new structured errors to legacy string format
    for error in result.errors:
        errors.append(f"{error.path}: {error.message}")

    return (result.is_valid, result.data, errors)
```

3. Update `json_ingestion.py` to import from new module:

```python
# json_ingestion.py
from .schemas.validation import validate_json_input_legacy as validate_json_input
```

---

## Validation Pipeline

```python
from typing import Callable


class ValidationPipeline:
    """Composable validation pipeline."""
    
    def __init__(self):
        self.steps: list[Callable[[dict, ValidationContext], None]] = []
    
    def add(self, step: Callable[[dict, ValidationContext], None]) -> "ValidationPipeline":
        self.steps.append(step)
        return self
    
    def run(self, data: dict) -> ValidationResult:
        context = ValidationContext()
        
        for step in self.steps:
            step(data, context)
            if context.has_errors and context.fail_fast:
                break
        
        return ValidationResult(
            is_valid=not context.has_errors,
            data=data if not context.has_errors else None,
            errors=context.errors,
            warnings=context.warnings
        )


class ValidationContext:
    def __init__(self, fail_fast: bool = False):
        self.errors: list[ValidationError] = []
        self.warnings: list[ValidationError] = []
        self.fail_fast = fail_fast
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def add_error(
        self,
        path: str,
        code: str,
        message: str,
        context: dict | None = None
    ) -> None:
        self.errors.append(ValidationError(
            path=path,
            code=code,
            message=message,
            severity="error",
            context=context
        ))
    
    def add_warning(
        self,
        path: str,
        code: str,
        message: str,
        context: dict | None = None
    ) -> None:
        self.warnings.append(ValidationError(
            path=path,
            code=code,
            message=message,
            severity="warning",
            context=context
        ))
```

---

## Validation Steps

### Step 1: Schema Validation

```python
def validate_schema(data: dict, ctx: ValidationContext) -> None:
    """Validate against JSON Schema."""
    try:
        jsonschema.validate(data, INGESTION_SCHEMA)
    except jsonschema.ValidationError as e:
        ctx.add_error(
            path=".".join(str(p) for p in e.path) or "root",
            code="schema_validation_failed",
            message=e.message,
            context={"validator": e.validator, "schema_path": list(e.schema_path)}
        )
```

### Step 2: Entity Validation

```python
def validate_entities(data: dict, ctx: ValidationContext) -> None:
    """Validate entity constraints beyond schema."""
    entities = data.get("entities", [])
    seen_ids: set[str] = set()
    name_to_id: dict[str, str] = {}
    
    for idx, entity in enumerate(entities):
        path = f"entities[{idx}]"
        entity_id = entity.get("id")
        name = entity.get("name")
        
        # Check for auto-generated ID (deprecation warning)
        if not entity_id and name:
            ctx.add_warning(
                path=f"{path}.id",
                code="entity.id.auto_generated",
                message=(
                    f"Entity ID will be auto-generated from name '{name}'. "
                    "Explicit IDs will be required in v2.0."
                ),
                context={"name": name, "generated_id": _canonical_entity_id(name)}
            )
            entity_id = _canonical_entity_id(name)

        # Check ID uniqueness
        if entity_id in seen_ids:
            ctx.add_error(
                path=f"{path}.id",
                code="entity.id.duplicate",
                message=f"Duplicate entity ID: {entity_id}",
                context={"duplicate_id": entity_id}
            )
        seen_ids.add(entity_id)

        # Check name uniqueness
        if name in name_to_id:
            ctx.add_error(
                path=f"{path}.name",
                code="entity.name.duplicate",
                message=f"Entity name '{name}' is already used by entity '{name_to_id[name]}'",
                context={"name": name, "existing_id": name_to_id[name]}
            )
        else:
            name_to_id[name] = entity_id
```

### Step 3: Relationship Validation

```python
def validate_relationships(data: dict, ctx: ValidationContext) -> None:
    """Validate relationship constraints."""
    entities = data.get("entities", [])
    relationships = data.get("relationships", [])
    entity_ids = {e.get("id") or _canonical_entity_id(e.get("name")) for e in entities}
    
    for idx, rel in enumerate(relationships):
        path = f"relationships[{idx}]"
        source = rel.get("source")
        target = rel.get("target")
        rel_type = rel.get("relationship")
        
        # Check for comma-separated targets (deprecation error)
        if target and "," in target:
            ctx.add_error(
                path=f"{path}.target",
                code="relationship.target.comma_separated",
                message=(
                    f"Comma-separated targets are deprecated: '{target}'. "
                    f"Create multiple relationship objects instead."
                ),
                context={
                    "target": target,
                    "suggested_split": [t.strip() for t in target.split(",")]
                }
            )

        # Check source exists
        if source and source not in entity_ids:
            ctx.add_error(
                path=f"{path}.source",
                code="relationship.source.not_found",
                message=f"Source entity '{source}' not found",
                context={"source": source, "available_ids": list(entity_ids)}
            )

        # Check target exists
        if target and target not in entity_ids and "," not in target:
            ctx.add_error(
                path=f"{path}.target",
                code="relationship.target.not_found",
                message=f"Target entity '{target}' not found",
                context={"target": target, "available_ids": list(entity_ids)}
            )

        # Check relationship type format
        if rel_type and not re.match(r'^[A-Z][a-zA-Z0-9_]*$', rel_type):
            ctx.add_error(
                path=f"{path}.relationship",
                code="relationship.type.invalid",
                message=f"Relationship type '{rel_type}' must be PascalCase",
                context={"relationship": rel_type, "pattern": "^[A-Z][a-zA-Z0-9_]*$"}
            )

        # Check for self-reference
        if source == target:
            ctx.add_warning(
                path=path,
                code="relationship.self_reference",
                message=f"Self-referencing relationship detected: {source} -> {source}",
                context={"entity": source}
            )
```

---

## Error Reporting

### Console Output

```
Validation failed with 3 errors:

ERROR: entities[0].id
  Code: entity.id.duplicate
  Message: Duplicate entity ID: user-service
  Context: {"duplicate_id": "user-service"}

ERROR: relationships[2].target
  Code: relationship.target.comma_separated
  Message: Comma-separated targets are deprecated: 'db-1, db-2'. Create multiple relationship objects instead.
  Context: {
    "target": "db-1, db-2",
    "suggested_split": ["db-1", "db-2"]
  }

ERROR: relationships[5].source
  Code: relationship.source.not_found
  Message: Source entity 'auth-service' not found
  Context: {
    "source": "auth-service",
    "available_ids": ["user-service", "api-gateway", "payment-service"]
  }

WARNING: entities[3]
  Code: entity.id.auto_generated
  Message: Entity ID will be auto-generated from name 'Order Processor'. Explicit IDs will be required in v2.0.
```

### JSON Output

```json
{
  "valid": false,
  "errors": [
    {
      "path": "entities[0].id",
      "code": "entity.id.duplicate",
      "message": "Duplicate entity ID: user-service",
      "severity": "error",
      "context": {"duplicate_id": "user-service"}
    }
  ],
  "warnings": [
    {
      "path": "entities[3]",
      "code": "entity.id.auto_generated",
      "message": "Entity ID will be auto-generated...",
      "severity": "warning",
      "context": {"name": "Order Processor"}
    }
  ]
}
```

---

## Migration Utilities

### Canonical ID Generation (Public API)

The `_canonical_entity_id()` function is currently private in `json_ingestion.py`. Make it part of the public API:

```python
# codebase_rag/schemas/validation.py

import re

def canonical_entity_id(name: str) -> str:
    """Generate a canonical entity ID from a name.

    This is a public wrapper for the canonical ID generation logic.

    Args:
        name: Entity name to convert

    Returns:
        URL-safe entity ID (lowercase, alphanumeric + underscore)

    Example:
        >>> canonical_entity_id("User Service")
        'user_service'
        >>> canonical_entity_id("API-Gateway")
        'api_gateway'
    """
    if not name:
        raise ValueError("Entity name cannot be empty")
    return re.sub(r'[^a-zA-Z0-9_]+', '_', name.strip().lower())
```

### Auto-Fixer Script

```python
#!/usr/bin/env python3
"""Auto-fix common data modeling issues for migration to v2.0."""

import json
from pathlib import Path

# Import from public API
from codebase_rag.schemas.validation import canonical_entity_id


def fix_comma_separated_targets(data: dict) -> dict:
    """Split comma-separated relationship targets."""
    new_relationships = []
    
    for rel in data.get("relationships", []):
        target = rel.get("target", "")
        if "," in target:
            for t in target.split(","):
                new_rel = rel.copy()
                new_rel["target"] = t.strip()
                new_relationships.append(new_rel)
        else:
            new_relationships.append(rel)
    
    data["relationships"] = new_relationships
    return data


def fix_missing_entity_ids(data: dict) -> dict:
    """Add explicit IDs to entities that only have names."""
    for entity in data.get("entities", []):
        if not entity.get("id") and entity.get("name"):
            entity["id"] = canonical_entity_id(entity["name"])
    return data


def fix_duplicate_names(data: dict) -> dict:
    """Add numeric suffixes to duplicate entity names."""
    name_counts: dict[str, int] = {}
    
    for entity in data.get("entities", []):
        name = entity.get("name", "")
        if name in name_counts:
            name_counts[name] += 1
            entity["name"] = f"{name} ({name_counts[name]})"
        else:
            name_counts[name] = 1
    
    return data


def migrate_file(input_path: Path, output_path: Path) -> dict:
    """Migrate a single JSON file."""
    with open(input_path) as f:
        data = json.load(f)
    
    fixes_applied = []
    
    # Check for comma-separated targets
    has_comma_targets = any(
        "," in str(r.get("target", ""))
        for r in data.get("relationships", [])
    )
    if has_comma_targets:
        data = fix_comma_separated_targets(data)
        fixes_applied.append("comma_separated_targets")
    
    # Check for missing IDs
    has_missing_ids = any(
        not e.get("id") for e in data.get("entities", [])
    )
    if has_missing_ids:
        data = fix_missing_entity_ids(data)
        fixes_applied.append("missing_entity_ids")
    
    # Write migrated file
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return {
        "input": str(input_path),
        "output": str(output_path),
        "fixes_applied": fixes_applied
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: migrate.py <input.json> <output.json>")
        sys.exit(1)
    
    result = migrate_file(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps(result, indent=2))
```

---

## Implementation Checklist

- [ ] Create `ValidationError` dataclass
- [ ] Create `ValidationResult` dataclass with `to_tuple()` method
- [ ] Implement `ValidationContext` class
- [ ] Implement `ValidationPipeline` class
- [ ] Define namespaced error code taxonomy
- [ ] Create `canonical_entity_id()` public function
- [ ] Port existing validation to new pipeline
- [ ] Create `validate_json_input_legacy()` adapter
- [ ] Update `json_ingestion.py` to use new module
- [ ] Add structured error reporting
- [ ] Create migration utilities
- [ ] Update tests
- [ ] Add documentation

---

*Spec Version: 1.3.0*
*Status: Ready for Implementation*
*Updated: 2026-04-26*

## Changelog

- **1.3.0**: Fixed `any` → `Any` type hints for Python 3.11+ compatibility
- **1.2.0**: Added deprecation timeline, validation steps
- **1.0.0**: Initial spec
