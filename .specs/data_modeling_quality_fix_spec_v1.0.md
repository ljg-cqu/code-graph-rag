# Data Modeling Quality Fix Specification v1.0
## Metadata
| Field | Value |
|-------|-------|
| Spec ID | SPEC-DM-FIX-20240520 |
| Version | 1.0 |
| Status | Implementation Ready |
| Priority | Critical |
| Affected Components | `ingestion_schema.json`, `codebase_rag/schemas.py`, `codebase_rag/json_ingestion.py` |
| Related Issues | Schema inconsistency, broken delete functionality, duplicate code |
| Author | Codebase Audit Tool |
| Last Updated | 2024-05-20 |

---

## Executive Summary
This spec defines fixes for 3 critical data modeling quality issues identified during the codebase audit, plus 1 optional type safety enhancement. All fixes are fully backwards compatible with existing valid input following the official `ingestion_schema.json` specification, no breaking changes to public API or ingestion interfaces.

---

## Background & Root Cause Analysis
### Identified Issues
1. **Schema Inconsistency**: Mismatch between official ingestion schema (SSOT) and Pydantic validation models leading to valid input being rejected
2. **Broken Dataset Deletion**: Incorrect property reference in delete queries resulting in orphaned data that cannot be removed
3. **Duplicate Code**: Two identical implementations of relationship embedding generation function increasing maintenance overhead
4. **Optional Type Safety Gap**: Lack of Pydantic parsing after JSON schema validation leading to potential type errors during processing

---

## Detailed Fix Design
---

### Fix 1: Schema Consistency Correction
**Objective**: Align Pydantic models in `schemas.py` 100% with official `ingestion_schema.json`
#### Design Details
Update the following Pydantic models to match schema field structure exactly:
```python
# JSONEntity Model Update
class JSONEntity(BaseModel):
    id: str | None = None
    name: str  # Required top-level field per schema
    type: str | None = None
    labels: list[str] | None = None
    operation: Literal["add", "update", "delete"] | None = None
    last_updated: str | None = None  # ISO 8601 format
    properties: dict[str, Any] = Field(default_factory=dict)  # No required fields inside properties

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Entity name cannot be empty")
        return v


# JSONRelationship Model Update
class JSONRelationship(BaseModel):
    id: str | None = None
    source: str  # Required top-level source reference (name or ID)
    target: str  # Required top-level target reference (name or ID)
    relationship: str  # Required top-level relationship type
    operation: Literal["add", "update", "delete"] | None = None
    last_updated: str | None = None
    confidence: float | None = Field(None, ge=0, le=1)
    explanation: str | None = None
    isInferred: bool | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
```
#### Backwards Compatibility
- All existing valid JSON input following the public schema will pass validation
- No changes required for users of the ingestion API
---

### Fix 2: Dataset Deletion Function Repair
**Objective**: Fix the delete operation to actually remove all data for a given dataset ID
#### Design Details
Update the delete queries in `json_ingestion.py::delete_dataset()` to use the correct property name `dataset_id` instead of non-existent `source_dataset`:
```python
# Corrected relationship delete query
rel_query = """
MATCH ()-[r]->()
WHERE r.dataset_id = $dataset_id
DELETE r
RETURN count(r) as deleted
"""

# Corrected node delete query
node_query = """
MATCH (n)
WHERE n.dataset_id = $dataset_id
DELETE n
RETURN count(n) as deleted
"""
```
#### Additional Improvements
- Add explicit transaction wrapping for delete operations to ensure atomicity
- Add validation that dataset_id is not empty before executing delete
---

### Fix 3: Duplicate Code Removal
**Objective**: Eliminate redundant function implementation
#### Design Details
- Remove the first (older) implementation of `generate_embeddings_for_relationships` in `json_ingestion.py`
- Keep the second implementation that correctly handles the `explanation` field for relationship embedding generation
#### Impact
- No change to functionality, only reduces code maintenance overhead
---

### Fix 4 (Optional Recommended): Type Safety Enhancement
**Objective**: Add type safety after JSON schema validation
#### Design Details
Add Pydantic parsing step in `json_ingestion.py::validate_json_input()` after successful jsonschema validation:
```python
def validate_json_input(data: dict[str, Any]) -> tuple[bool, dict | None, list[str]]:
    errors = []
    try:
        jsonschema.validate(instance=data, schema=INGESTION_SCHEMA)
        # New step: Parse with Pydantic to enforce type safety
        if "entities" in data:
            for entity in data["entities"]:
                JSONEntity(**entity)
        if "relationships" in data:
            if isinstance(data["relationships"], list):
                for rel in data["relationships"]:
                    JSONRelationship(**rel)
            elif isinstance(data["relationships"], dict) and "relationships" in data["relationships"]:
                for rel in data["relationships"]["relationships"]:
                    JSONRelationship(**rel)
        return True, data, []
    except jsonschema.exceptions.ValidationError as e:
        errors.append(f"Schema validation failed: {str(e)}")
        return False, None, errors
    except ValidationError as e:
        errors.append(f"Type validation failed: {str(e)}")
        return False, None, errors
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, None, errors
```
---

## Acceptance Criteria
### Fix 1 Acceptance
1. Valid JSON input following `ingestion_schema.json` passes validation 100% of the time
2. Input missing required `name` field at entity top level is correctly rejected
3. Input with `name` at top level and optional `description` inside properties passes validation
4. Relationship input with `source`, `target`, `relationship` top level fields passes validation

### Fix 2 Acceptance
1. Running `delete_dataset("test_dataset")` removes all nodes and relationships with `dataset_id: test_dataset`
2. Running `delete_dataset("test_dataset")` removes all corresponding vector entries from vector store
3. Delete operation returns correct count of deleted nodes and relationships

### Fix 3 Acceptance
1. Only one implementation of `generate_embeddings_for_relationships` exists in codebase
2. Relationship embedding generation works exactly as before for all cases

### Fix 4 Acceptance
1. Type mismatches (e.g. string value for confidence score) are caught during validation instead of causing runtime errors later
2. Valid input continues to pass validation without changes

---

## Implementation Steps (Ordered)
1. Update `codebase_rag/schemas.py` with corrected `JSONEntity` and `JSONRelationship` models (Fix 1)
2. Add validation step in `validate_json_input()` function (Fix 4, optional)
3. Fix delete queries in `delete_dataset()` function in `json_ingestion.py` (Fix 2)
4. Remove duplicate `generate_embeddings_for_relationships` implementation (Fix 3)
5. Run all existing unit tests to verify no regression
6. Add new test cases for fixed functionality

---

## Test Plan
### Unit Tests
1. Test entity validation with valid and invalid input (top level name, missing name, description in properties)
2. Test relationship validation with valid and invalid field names
3. Test delete operation: create test dataset, delete it, verify no data remains
4. Test relationship embedding generation works correctly after duplicate removal
5. Test type validation catches mismatched data types (e.g. string for confidence score)

### Integration Tests
1. End-to-end ingestion test with valid JSON input following official schema
2. End-to-end delete test: ingest dataset, verify it exists, delete it, verify it's fully removed
3. Edge case test: ingest dataset with partial name matches for relationship resolution, delete it successfully

---

## Rollback Plan
1. Revert changes to `schemas.py` to restore original models
2. Revert changes to `delete_dataset()` function
3. Restore removed duplicate function implementation
4. Revert validation step additions to `validate_json_input()`
5. All rollback steps are fully backwards compatible with previous state

---

## Post-Implementation Validation
1. Run full test suite to confirm no regressions
2. Run manual ingestion test with sample valid JSON file
3. Run manual delete test to confirm data is fully removed
4. Verify no duplicate functions remain in codebase
5. Confirm type validation correctly catches type mismatches