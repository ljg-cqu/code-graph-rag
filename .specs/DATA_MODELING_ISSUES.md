# Data Modeling Quality Assessment

## Executive Summary

After analyzing the code-graph-rag codebase, several data modeling issues have been identified that impact maintainability, consistency, and reliability. This document provides a comprehensive assessment with specific recommendations for optimal fixing.

---

## Critical Issues (P0)

### 1. Schema Duplication and Divergence Risk

**Issue**: The ingestion schema is defined in multiple places:
- `ingestion_schema.json` - JSON Schema (198 lines, source of truth)
- `schemas.py` - Pydantic models (JSONEntity, JSONRelationship, JSONMetadata)
- `json_ingestion.py` - Runtime validation logic

**Note**: `types_defs.py` (897 lines) contains separate `NODE_SCHEMAS` and `RELATIONSHIP_SCHEMAS` tuples for graph database schema. These define Memgraph node labels and relationship types - distinct from JSON ingestion schema but should be considered in any schema unification effort.

**Impact**: High risk of divergence when making changes. Changes to the schema must be synchronized across three locations.

**Evidence**:
```python
# schemas.py:102-109 - Pydantic models
class JSONEntity(BaseModel):
    id: str | None = None
    name: str
    type: str | None = None
    labels: list[str] | None = None
    ...

# ingestion_schema.json:49-96 - JSON Schema
"entities": {
  "items": {
    "properties": {
      "id": {"type": "string"},
      "name": {"type": "string"},
      ...
```

**Recommendation**: 
- Generate Pydantic models from JSON Schema using `datamodel-code-generator`
- Single source of truth: `ingestion_schema.json`
- Auto-generate `schemas.py` during build process

---

### 2. Relationship Target Normalization Violation

**Issue**: The `_split_relationship_targets` function supports comma-separated multiple targets in a single relationship:

```python
# json_ingestion.py:142-148
def _split_relationship_targets(target: str) -> list[str]:
    if "," not in target:
        return [target.strip()]
    targets = [part.strip() for part in target.split(",") if part.strip()]
    return targets or [target.strip()]
```

**Impact**: Violates First Normal Form (1NF). Relationships should be atomic. This design complicates querying, indexing, and maintaining referential integrity.

**Recommendation**:
- Deprecate comma-separated targets
- Normalize to one relationship per target
- Add validation to reject comma-separated targets in v2.0

---

### 3. Ambiguous Entity Reference Resolution

**Issue**: The `DatasetReferences` class and `_lookup_entity_reference` function maintain complex logic to handle ambiguous name references:

```python
# json_ingestion.py:102-106
@dataclass
class DatasetReferences:
    ids: dict[str, str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    ambiguous_names: set[str] = field(default_factory=set)
```

**Impact**: This is a workaround for allowing non-unique entity names within a dataset. Creates complexity in relationship resolution.

**Recommendation**:
- Enforce unique entity names within a dataset
- Require explicit IDs for all entities
- Remove ambiguous name resolution logic

---

## High Priority Issues (P1)

### 4. Inconsistent Null Handling

**Issue**: Mixed patterns for optional fields:
- Some use `str | None = None` (proper)
- Some use default factories for lists
- Some check for empty strings vs None inconsistently

**Evidence**:
```python
# schemas.py
labels: list[str] | None = None  # Can be None
last_updated: str | None = None  # Can be None

# json_ingestion.py:227-229
try:
    JSONEntity(**entity)
except Exception as exc:
    errors.append(f"Entity {index} validation failed: {exc}")
```

**Recommendation**:
- Standardize on: `field: type | None = None` for optionals
- Use Pydantic validators for normalization
- Never allow empty strings where None is expected

---

### 5. Schema Versioning Missing

**Issue**: No versioning mechanism for the ingestion schema. Changes to the schema could break existing data.

**Evidence**: No `$id` or version field in `ingestion_schema.json`.

**Recommendation**:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://code-graph-rag.io/schemas/ingestion/v1.json",
  "version": "1.0.0",
  ...
}
```

---

### 6. Type Coercion at Runtime

**Issue**: Excessive type coercion in `_normalize_entities` and `_normalize_relationships`:

```python
# json_ingestion.py:200-209
if entity.get("properties") is None:
    entity["properties"] = {}

if not entity.get("id") and entity.get("name"):
    entity["id"] = _canonical_entity_id(str(entity["name"]))
```

**Impact**: Silent data mutation makes debugging difficult. Violates principle of least surprise.

**Recommendation**:
- Strict validation: reject invalid data
- Separate normalization from validation
- Log all transformations

---

## Medium Priority Issues (P2)

### 7. Batch Operation Inheritance Complexity

**Issue**: Complex logic for inheriting batch-level operation and timestamps:

```python
# json_ingestion.py:150-156
def _batch_operation(data: dict[str, Any]) -> str:
    raw_operation = data.get("operation") or data.get("metadata", {}).get("operation")
    if raw_operation is None:
        return "add"
    return str(raw_operation).strip().lower() or "add"
```

**Impact**: Multiple sources of truth (top-level, metadata, entity-level) create confusion.

**Recommendation**:
- Single source: entity-level only
- Remove batch-level operation/timestamp overrides
- Simplify mental model

---

### 8. Embedding Cache Key Risk

**Issue**: Embedding cache uses raw text as key:

```python
# json_ingestion.py:680-695
texts = []
entity_ids = []
for entity in entities:
    ...
    texts.append(f"{name} - {description}")
    entity_ids.append(entity_id)
```

**Risk**: If embedding model changes, cache returns wrong embeddings.

**Recommendation**:
```python
cache_key = f"{embedding_model_name}:{embedding_version}:{hash(text)}"
```

---

### 9. Error Aggregation Pattern

**Issue**: Error collection uses mutable list passed through multiple layers:

```python
# json_ingestion.py:286-305
def validate_json_input(...) -> tuple[bool, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    ...
    errors.extend(_normalize_entities(normalized_data))
    errors.extend(_normalize_relationships(normalized_data))
    ...
```

**Recommendation**: Use a structured error type:
```python
@dataclass
class ValidationError:
    path: str  # JSON path to error
    code: str  # Machine-readable error code
    message: str  # Human-readable message
    severity: Literal["error", "warning"]
```

---

## Schema Architecture Clarification

The codebase maintains **two distinct schema systems** that serve different purposes:

| Schema System | Location | Purpose |
|--------------|----------|---------|
| **JSON Ingestion Schema** | `ingestion_schema.json`, `schemas.py` | Validates external JSON input format for custom data ingestion |
| **Graph Database Schema** | `types_defs.py` (NODE_SCHEMAS, RELATIONSHIP_SCHEMAS) | Defines Memgraph node labels, relationship types, and property schemas |

**Important**: These should remain separate but coordinated. The JSON ingestion schema handles input validation, while the graph schema defines the database structure. Any unification effort should:
1. Keep JSON ingestion schema as input validation layer
2. Keep graph schema as database definition layer
3. Ensure JSON entity types map correctly to graph node labels
4. Ensure JSON relationships map to valid RELATIONSHIP_SCHEMAS entries

---

## Design Specs for Optimal Fixing

### Spec 1: Unified Schema Architecture

```
ingestion_schema.json (source of truth)
    ↓
datamodel-codegen → schemas.py (auto-generated)
    ↓
Pydantic validation + jsonschema validation
```

**Implementation**:
```bash
# Add to build pipeline
datamodel-codegen --input ingestion_schema.json --output schemas.py
```

---

### Spec 2: Strict Entity Model

```python
class Entity(BaseModel):
    id: str  # REQUIRED - no auto-generation
    name: str  # REQUIRED - unique within dataset
    type: str = "Entity"
    labels: list[str] = Field(default_factory=list)
    operation: Literal["add", "update", "delete"] = "add"
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    properties: dict[str, Any] = Field(default_factory=dict)
    
    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Entity name cannot be empty")
        return v.strip()
```

---

### Spec 3: Normalized Relationship Model

```python
class Relationship(BaseModel):
    source: str  # Entity ID (not name, not comma-separated)
    target: str  # Entity ID (single target only)
    relationship: str  # Relationship type
    operation: Literal["add", "update", "delete"] = "add"
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    explanation: str | None = None
    is_inferred: bool = False
    properties: dict[str, Any] = Field(default_factory=dict)
```

---

### Spec 4: Versioned Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://code-graph-rag.io/schemas/ingestion/v2.json",
  "version": "2.0.0",
  "compatible_with": ["1.0.0"],
  "deprecated_features": [
    "comma_separated_relationship_targets",
    "auto_generated_entity_ids",
    "ambiguous_name_resolution"
  ]
}
```

---

## Migration Path

### Phase 1: Compatibility (Current → v1.1)
- Add deprecation warnings for comma-separated targets
- Log auto-generated IDs for tracking
- Add schema version field (optional)

### Phase 2: Validation (v1.1 → v1.2)
- Reject comma-separated targets (configurable)
- Require explicit IDs for new datasets
- Enforce unique names within dataset

### Phase 3: Cleanup (v1.2 → v2.0)
- Remove deprecated features
- Simplify reference resolution
- Auto-generated schemas from JSON Schema

---

## Testing Recommendations

1. **Property-Based Testing**: Use Hypothesis to generate valid/invalid entities
2. **Schema Compatibility Tests**: Verify forward/backward compatibility
3. **Fuzzing**: Test edge cases in relationship resolution
4. **Performance**: Benchmark with 10K+ entity datasets

---

## Files Requiring Updates

| File | Lines | Changes Required |
|------|-------|------------------|
| `ingestion_schema.json` | 198 | Add versioning, deprecate features |
| `schemas.py` | 191 | Auto-generate from schema |
| `json_ingestion.py` | ~1900 | Simplify validation, remove workarounds |
| `types_defs.py` | 897 | Contains graph schema (NODE_SCHEMAS, RELATIONSHIP_SCHEMAS) - coordinate but don't merge |
| `config.py` | ~200 | Add schema version settings |

---

*Generated by Code Graph Query Agent v3.0.0*
*Analysis Date: 2026-04-26*
*Updated: 2026-04-26 - Fixed line number references*
