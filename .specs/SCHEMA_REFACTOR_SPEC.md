# Schema Refactoring Design Spec

## Overview

Refactor the data modeling layer to eliminate duplication, enforce consistency, and establish a single source of truth for all schemas.

### Important: Schema Scope

This spec addresses **JSON Ingestion Schema only** - the schema for validating external JSON input. The graph database schema (`types_defs.py` - NODE_SCHEMAS, RELATIONSHIP_SCHEMAS) defines Memgraph node labels and relationship types, and is **out of scope** for this refactor. The two schema systems should remain separate but coordinated.

---

## Current State Analysis

### Schema Locations

```
code-graph-rag/
├── ingestion_schema.json          # JSON Schema (198 lines) - claimed SSOT for JSON input
├── codebase_rag/
│   ├── schemas.py                 # Pydantic models (190 lines) - JSONEntity, JSONRelationship, JSONMetadata
│   ├── json_ingestion.py          # Validation logic (1891 lines)
│   ├── types_defs.py              # Graph schema (897 lines) - NODE_SCHEMAS, RELATIONSHIP_SCHEMAS (OUT OF SCOPE)
│   └── config.py                  # Settings (~200 lines)
```

**Why Move to `codebase_rag/schema.json`?**

The current root-level `ingestion_schema.json` works but creates an organizational gap:
1. **Package self-containment**: Moving the schema into `codebase_rag/` allows the package to be installed and used without relying on a file outside the package directory.
2. **Build-time generation**: `datamodel-codegen` can run as part of package build when the schema is inside the package.
3. **Import simplicity**: Generated Pydantic models can reference the schema via relative path.

**Decision**: Move the schema to `codebase_rag/schema.json` in Phase 1. This provides consistency from the start and avoids confusion about which file is the source of truth. The root-level `ingestion_schema.json` will be deprecated with a symlink or deprecation notice pointing to the new location.

### Issues Identified

1. **Duplication**: Entity definition in both JSON Schema and Pydantic
2. **Divergence Risk**: Changes require manual sync across files
3. **Validation Split**: Logic scattered between jsonschema and Pydantic

### Out of Scope

- `types_defs.py` NODE_SCHEMAS and RELATIONSHIP_SCHEMAS - these define graph database structure, not JSON input validation

---

## Proposed Architecture

### Prerequisites

Add `datamodel-codegen` to project dependencies:

```toml
# pyproject.toml
[project.optional-dependencies]
dev = [
    "datamodel-code-generator>=0.25.0",
]
```

Or install directly:
```bash
pip install datamodel-code-generator
```

### Directory Structure

```
codebase_rag/
├── schemas/
│   ├── __init__.py               # Re-export all schemas
│   ├── ingestion.py              # Auto-generated from JSON Schema
│   ├── validation.py             # Unified validation logic
│   └── types.py                  # Shared type definitions
├── schema.json                   # Moved from root, SSOT
└── schema_generation/            # Build-time scripts
    ├── generate_models.py        # datamodel-codegen wrapper
    └── validate_sync.py          # Check schema <-> model sync
```

---

## Implementation Spec

### Step 1: Enhanced JSON Schema

Create `schema.json` (enhanced from `ingestion_schema.json`):

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://code-graph-rag.io/schema/v2.json",
  "version": "2.0.0",
  "title": "Code-Graph RAG Ingestion Schema",
  
  "definitions": {
    "Entity": {
      "type": "object",
      "required": ["id", "name"],
      "additionalProperties": true,
      "properties": {
        "id": {
          "type": "string",
          "pattern": "^[a-zA-Z0-9_]+$",
          "description": "Unique identifier, URL-safe (alphanumeric and underscore only)"
        },
        "name": {
          "type": "string",
          "minLength": 1,
          "maxLength": 255,
          "description": "Human-readable name"
        },
        "type": {
          "type": "string",
          "default": "Entity",
          "description": "Primary type/label"
        },
        "labels": {
          "type": "array",
          "items": { "type": "string" },
          "default": [],
          "description": "Additional graph labels"
        },
        "operation": {
          "type": "string",
          "enum": ["add", "update", "delete"],
          "default": "add"
        },
        "last_updated": {
          "type": "string",
          "format": "date-time"
        },
        "properties": {
          "type": "object",
          "additionalProperties": true
        }
      }
    },
    
    "Relationship": {
      "type": "object",
      "required": ["source", "target", "relationship"],
      "properties": {
        "source": {
          "type": "string",
          "description": "Source entity ID (not name)"
        },
        "target": {
          "type": "string",
          "description": "Target entity ID (single, not comma-separated)"
        },
        "relationship": {
          "type": "string",
          "pattern": "^[A-Z][a-zA-Z0-9_]*$",
          "description": "Relationship type in PascalCase"
        },
        "operation": {
          "type": "string",
          "enum": ["add", "update", "delete"],
          "default": "add"
        },
        "last_updated": {
          "type": "string",
          "format": "date-time"
        },
        "confidence": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "default": 1.0
        },
        "explanation": {
          "type": ["string", "null"]
        },
        "isInferred": {
          "type": "boolean",
          "default": false
        },
        "properties": {
          "type": "object",
          "additionalProperties": true
        }
      }
    },
    
    "Metadata": {
      "type": "object",
      "required": ["dataset_id"],
      "properties": {
        "dataset_id": {
          "type": "string",
          "pattern": "^[a-zA-Z0-9_-]+$"
        },
        "source": { "type": ["string", "null"] },
        "created_at": { "type": "string", "format": "date-time" },
        "default_entity_labels": {
          "type": "array",
          "items": { "type": "string" }
        },
        "schema_version": {
          "type": "string",
          "const": "2.0.0"
        }
      }
    }
  },
  
  "type": "object",
  "required": ["entities"],
  "properties": {
    "metadata": { "$ref": "#/definitions/Metadata" },
    "operation": {
      "type": "string",
      "enum": ["add", "update", "delete"],
      "default": "add"
    },
    "batch_id": { "type": "string" },
    "last_updated": { "type": "string", "format": "date-time" },
    "entities": {
      "type": "array",
      "items": { "$ref": "#/definitions/Entity" },
      "minItems": 1
    },
    "relationships": {
      "type": "array",
      "items": { "$ref": "#/definitions/Relationship" }
    }
  }
}
```

---

### Step 2: Auto-Generated Pydantic Models

File: `codebase_rag/schemas/ingestion.py` (auto-generated)

**Note**: Entity `id` remains optional for backward compatibility during Phase 1-2. It becomes required in v2.0.

```python
# generated by datamodel-codegen
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, field_validator


class Metadata(BaseModel):
    dataset_id: str = Field(..., pattern=r'^[a-zA-Z0-9_-]+$')
    source: str | None = None
    created_at: datetime | None = None
    default_entity_labels: list[str] = Field(default_factory=list)
    schema_version: str | None = None  # Optional in Phase 1-2, required in v2.0


class Entity(BaseModel):
    id: str | None = None  # Optional for backward compatibility - auto-generated if not provided
    name: str = Field(..., min_length=1, max_length=255)
    type: str | None = None
    labels: list[str] = Field(default_factory=list)
    operation: str | None = None
    last_updated: datetime | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('Entity name cannot be empty or whitespace')
        return v.strip()


class Relationship(BaseModel):
    id: str | None = None
    source: str
    target: str
    relationship: str
    operation: str | None = None
    last_updated: datetime | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str | None = None
    is_inferred: bool | None = Field(default=None, alias="isInferred")
    properties: dict[str, Any] = Field(default_factory=dict)
    
    # Note: comma-separated target validation happens in business logic layer
    # not in Pydantic model to allow deprecation warnings


class IngestionPayload(BaseModel):
    metadata: Metadata | None = None
    operation: str | None = None
    batch_id: str | None = None
    last_updated: datetime | None = None
    entities: list[Entity] = Field(..., min_length=1)
    relationships: list[Relationship] = Field(default_factory=list)
```

---

### Step 3: Unified Validation Module

File: `codebase_rag/schemas/validation.py`

**Integration with existing `validate_json_input()`**: This module replaces the validation logic in `json_ingestion.py:286` while maintaining the same function signature for backward compatibility.

```python
from typing import Any
from pydantic import ValidationError as PydanticValidationError

from .ingestion import IngestionPayload


class ValidationResult:
    """Result of validating an ingestion payload.
    
    Maintains backward compatibility with existing tuple return format
    via the `to_tuple()` method.
    """
    def __init__(self):
        self.valid = False
        self.data: dict[str, Any] | None = None
        self.errors: list[dict[str, str]] = []
        self.warnings: list[str] = []
    
    def add_error(self, path: str, message: str, code: str = "validation_error"):
        self.errors.append({"path": path, "message": message, "code": code})
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def to_tuple(self) -> tuple[bool, dict[str, Any] | None, list[str]]:
        """Convert to legacy tuple format for backward compatibility.
        
        Returns:
            Tuple of (is_valid, normalized_data, error_messages)
        """
        error_messages = [f"{e['path']}: {e['message']}" for e in self.errors]
        return (self.valid, self.data, error_messages)


def validate_ingestion_payload(data: dict[str, Any], strict: bool = False) -> ValidationResult:
    """
    Two-phase validation:
    1. Pydantic validation (structure + types)
    2. Custom business logic validation
    
    Args:
        data: Raw JSON data to validate
        strict: If True, reject deprecated features (comma-separated targets, missing IDs)
    
    Returns:
        ValidationResult with errors, warnings, and normalized data
    """
    result = ValidationResult()
    
    # Phase 1: Pydantic validation
    try:
        payload = IngestionPayload.model_validate(data)
        result.data = payload.model_dump()
    except PydanticValidationError as e:
        for error in e.errors():
            result.add_error(
                path=".".join(str(x) for x in error["loc"]),
                message=error["msg"],
                code=error["type"]
            )
        return result
    
    # Phase 2: Business logic validation
    entity_ids = _collect_entity_ids(result, payload)
    _validate_references(result, payload, entity_ids)
    _validate_unique_ids(result, payload)
    _validate_deprecated_features(result, payload, strict)
    _validate_no_cycles(result, payload)
    
    result.valid = len(result.errors) == 0
    return result


def _collect_entity_ids(result: ValidationResult, payload: IngestionPayload) -> dict[str, str]:
    """Collect entity IDs, auto-generating if needed.
    
    Returns:
        Dict mapping entity names to their IDs (for reference resolution)
    """
    entity_map: dict[str, str] = {}
    for entity in payload.entities:
        entity_id = entity.id
        if not entity_id and entity.name:
            # Auto-generate ID from name (Phase 1-2 behavior)
            entity_id = _canonical_entity_id(entity.name)
            result.add_warning(
                f"Entity ID auto-generated from name '{entity.name}'. "
                "Explicit IDs will be required in v2.0."
            )
        entity_map[entity.name] = entity_id
    return entity_map


def _canonical_entity_id(name: str) -> str:
    """Generate canonical entity ID from name.

    Mirrors existing logic in json_ingestion.py:129.
    """
    import re
    return re.sub(r'[^a-zA-Z0-9_]+', '_', name.strip().lower())


def _validate_references(
    result: ValidationResult, 
    payload: IngestionPayload, 
    entity_ids: dict[str, str]
) -> None:
    """Ensure all relationship references exist in entities."""
    valid_ids = set(entity_ids.values()) | set(entity_ids.keys())
    
    for idx, rel in enumerate(payload.relationships):
        source = rel.source
        target = rel.target
        
        # Handle comma-separated targets (deprecated)
        if "," in target:
            targets = [t.strip() for t in target.split(",")]
        else:
            targets = [target]
        
        if source not in valid_ids:
            result.add_error(
                f"relationships[{idx}].source",
                f"Source entity '{source}' not found in entities",
                "reference_not_found"
            )
        
        for t in targets:
            if t not in valid_ids:
                result.add_error(
                    f"relationships[{idx}].target",
                    f"Target entity '{t}' not found in entities",
                    "reference_not_found"
                )


def _validate_unique_ids(result: ValidationResult, payload: IngestionPayload) -> None:
    """Ensure all entity IDs are unique."""
    seen_ids: set[str] = set()
    for entity in payload.entities:
        entity_id = entity.id or _canonical_entity_id(entity.name)
        if entity_id in seen_ids:
            result.add_error(
                f"entities[{entity_id}].id",
                f"Duplicate entity ID: {entity_id}",
                "duplicate_id"
            )
        seen_ids.add(entity_id)


def _validate_deprecated_features(
    result: ValidationResult, 
    payload: IngestionPayload, 
    strict: bool
) -> None:
    """Check for deprecated features.
    
    In strict mode, these become errors. Otherwise, they are warnings.
    """
    for idx, rel in enumerate(payload.relationships):
        if "," in rel.target:
            if strict:
                result.add_error(
                    f"relationships[{idx}].target",
                    f"Comma-separated targets are not allowed: '{rel.target}'. "
                    "Create multiple relationship objects instead.",
                    "comma_separated_target"
                )
            else:
                result.add_warning(
                    f"Comma-separated targets are deprecated at relationships[{idx}]. "
                    "Create multiple relationship objects instead. This will be an error in v2.0."
                )


def _validate_no_cycles(result: ValidationResult, payload: IngestionPayload) -> None:
    """Warn about potential circular dependencies."""
    # Build adjacency list
    graph: dict[str, list[str]] = {}
    for rel in payload.relationships:
        source = rel.source
        if source not in graph:
            graph[source] = []
        target = rel.target
        if "," in target:
            graph[source].extend(t.strip() for t in target.split(","))
        else:
            graph[source].append(target)
    
    # Simple cycle detection
    visited: set[str] = set()
    rec_stack: set[str] = set()
    
    def has_cycle(node: str) -> bool:
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    for entity in payload.entities:
        entity_id = entity.id or _canonical_entity_id(entity.name)
        if entity_id not in visited:
            if has_cycle(entity_id):
                result.add_warning(
                    f"Potential circular dependency detected starting from {entity_id}"
                )
```

---

### Step 4: Build Integration

**Note**: The following scripts must be created as part of Phase 1 implementation.

File: `scripts/generate_schemas.py`

```python
#!/usr/bin/env python3
"""Generate Pydantic models from JSON Schema."""

import subprocess
import sys
from pathlib import Path


def main() -> int:
    schema_path = Path("codebase_rag/schema.json")
    output_path = Path("codebase_rag/schemas/ingestion.py")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate models
    result = subprocess.run(
        [
            "datamodel-codegen",
            "--input", str(schema_path),
            "--output", str(output_path),
            "--input-file-type", "jsonschema",
            "--output-model-type", "pydantic_v2.BaseModel",
            "--use-union-operator",  # Use | for unions
            "--target-python-version", "3.11",
            "--disable-timestamp",  # Don't add generation timestamp
            "--use-standard-collections",  # Use list instead of List
            "--strict-nullable",  # Use | None for optional fields
            "--use-schema-description",  # Include descriptions
            "--use-field-description",  # Add Field descriptions
        ],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
        return 1
    
    print(f"Generated {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

File: `.pre-commit-config.yaml` addition:

```yaml
  - repo: local
    hooks:
      - id: generate-schemas
        name: Generate Pydantic schemas from JSON
        entry: python scripts/generate_schemas.py
        language: system
        files: ^codebase_rag/schema\.json$
        pass_filenames: false
      
      - id: check-schema-sync
        name: Check schema and models are in sync
        entry: python scripts/check_schema_sync.py
        language: system
        always_run: true
        pass_filenames: false
```

---

## Migration Guide

### From Current to v2.0

**Key Principle**: No breaking changes until v2.0. Phases 1-2 maintain full backward compatibility.

#### Phase 1: Compatibility (No Breakage)

1. Add `datamodel-codegen` to dev dependencies
2. Create auto-generated `schemas/ingestion.py` from JSON Schema
3. Add deprecation warnings (not errors) for:
   - Missing entity IDs (warn, auto-generate)
   - Comma-separated targets (warn, still work)
4. Add schema version field (optional)

#### Phase 2: Strict Mode Opt-In (No Breakage)

1. Add `strict` parameter to `validate_ingestion_payload()`
2. When `strict=True`:
   - Reject missing entity IDs
   - Reject comma-separated targets
3. Default `strict=False` maintains backward compatibility

#### Phase 3: v2.0 Breaking Changes (Major Version Bump)

1. Default `strict=True`
2. Entity IDs become required
3. Comma-separated targets rejected
4. Schema version field required

#### Deprecation Timeline (Revised)

| Version | Target Date | Changes |
|---------|-------------|---------|
| 1.0.x | Current | Baseline |
| 1.1.0 | +2-3 months | Add deprecation warnings, add datamodel-codegen |
| 1.2.0 | +4-5 months | Opt-in strict mode via parameter |
| 1.3.0 | +6-7 months | Deprecation warnings escalated |
| 2.0.0 | +8-10 months | Remove deprecated features, major version bump |

**Note**: Timeline extended to allow adequate user migration time. Breaking changes require a major version bump per semver.

---

## Testing Strategy

### Unit Tests

```python
# tests/schemas/test_validation.py
import pytest
from codebase_rag.schemas.validation import validate_ingestion_payload


def test_valid_payload():
    data = {
        "metadata": {"dataset_id": "test-dataset"},
        "entities": [
            {"id": "entity-1", "name": "Entity One"},
            {"id": "entity-2", "name": "Entity Two"}
        ],
        "relationships": [
            {"source": "entity-1", "target": "entity-2", "relationship": "DependsOn"}
        ]
    }
    result = validate_ingestion_payload(data)
    assert result.valid


def test_comma_separated_target_rejected():
    data = {
        "metadata": {"dataset_id": "test"},
        "entities": [
            {"id": "a", "name": "A"},
            {"id": "b", "name": "B"}
        ],
        "relationships": [
            {"source": "a", "target": "b, c", "relationship": "DependsOn"}
        ]
    }
    result = validate_ingestion_payload(data)
    assert not result.valid
    assert any("comma" in e["message"].lower() for e in result.errors)


def test_missing_entity_id():
    data = {
        "metadata": {"dataset_id": "test"},
        "entities": [{"name": "No ID Entity"}]  # Missing required 'id'
    }
    result = validate_ingestion_payload(data)
    assert not result.valid
```

### Integration Tests

```python
# tests/schemas/test_end_to_end.py
import json
import jsonschema


def test_generated_models_match_schema():
    """Ensure Pydantic models and JSON Schema stay in sync."""
    with open("codebase_rag/schema.json") as f:
        schema = json.load(f)
    
    # Generate sample valid payloads
    from hypothesis import given, strategies as st
    
    @given(...)
    def roundtrip(payload):
        # Pydantic -> dict -> JSON Schema validation
        from codebase_rag.schemas.ingestion import IngestionPayload
        
        model = IngestionPayload.model_validate(payload)
        data = model.model_dump()
        
        # Should pass JSON Schema validation
        jsonschema.validate(data, schema)
```

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| Schema files | 2 (JSON Schema + Pydantic) | 1 (JSON Schema) |
| Lines of validation code | ~500 | ~200 |
| Manual sync points | 2 | 0 (auto-generated) |
| Test coverage | ~60% | >80% |
| Validation error clarity | Poor | Structured with codes |
| Breaking changes | N/A | None until v2.0 |

---

*Spec Version: 1.3.0*
*Status: Ready for Implementation*
*Updated: 2026-04-26*

## Changelog

- **1.3.0**: Clarified schema location decision (move to `codebase_rag/schema.json` in Phase 1), added note about scripts to be created, fixed `any` → `Any` type hints
- **1.2.0**: Added deprecation timeline, fixed line number references
- **1.0.0**: Initial spec
