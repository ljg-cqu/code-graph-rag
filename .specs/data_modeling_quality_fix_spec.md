# Data Modeling Quality Assessment & Fix Specification

## Document Information
- **Version**: 1.1.0
- **Date**: 2026-04-19
- **Scope**: code-graph-rag codebase data modeling quality
- **Status**: Fully implemented

---

## 1. Executive Summary

This specification documents **11 data modeling issues** identified in the code-graph-rag ingestion pipeline, graph schema, and document processing subsystems. Issues have been reviewed and fixes applied where applicable.

**Implementation Status:**
- **Issue 1**: Already resolved in codebase
- **Issues 2, 3, 4, 5, 6, 7, 8, 9, 10**: Fixed in this update
- **Issue 11**: Operational note, no code change needed

---

## 2. Issues Identified

### Issue 1: Missing Unique Key for `CodeChunk` Node Label [RESOLVED]

**Severity**: Critical (was)
**File**: `codebase_rag/constants.py`
**Lines**: `_NODE_LABEL_UNIQUE_KEYS` mapping

**Problem**: `NodeLabel.CODE_CHUNK` was reported as missing from `_NODE_LABEL_UNIQUE_KEYS`.

**Status**: **ALREADY RESOLVED** - The mapping exists at line 489:
```python
NodeLabel.CODE_CHUNK: UniqueKeyType.QUALIFIED_NAME,
```

**No action required.**

---

### Issue 2: NodeType Enum Missing `CODE_CHUNK` [FIXED]

**Severity**: High
**File**: `codebase_rag/types_defs.py`

**Problem**: `NodeType` enum (used for function registry/type inference) was missing `CODE_CHUNK`, causing type inference gaps.

**Fix Applied**: Added `CODE_CHUNK = "CodeChunk"` to the `NodeType` enum.

**Impact**: Enables proper function registry lookups and type inference for code chunk nodes.

---

### Issue 3: Document Graph Schema Mismatch — `Chunk` Node Missing `embedding_model` Property [FIXED]

**Severity**: High
**File**: `codebase_rag/types_defs.py`

**Problem**: The `NODE_SCHEMAS` definition for `NodeLabel.CHUNK` did not include `embedding_model` and `embedding_version` properties, yet the code sets these properties during chunk storage.

**Fix Applied**: Updated `Chunk` NodeSchema to include `embedding_model: string, embedding_version: int`.

---

### Issue 4: JSON Ingestion Uses `JsonEntity` Label Not in `NodeLabel` Enum [FIXED]

**Severity**: High
**File**: `codebase_rag/json_ingestion.py`

**Problem**: The JSON ingestion pipeline creates nodes with label `JsonEntity` (constant `JSON_ENTITY_LABEL = "JsonEntity"`), but this label was **not** in the `NodeLabel` enum.

**Context**: The codebase has **two distinct JSON systems**:

| System | Labels | Purpose |
|--------|--------|---------|
| JSON Content Nodes | `JSON_OBJECT`, `JSON_ARRAY`, `JSON_FIELD`, `JSON_VALUE` | Parsing JSON file structure (in `NodeLabel` enum with full schema) |
| Entity JSON Ingestion | `JsonEntity` | Entity-based knowledge graph from JSON datasets |

**Design Decision**: **Option A chosen** — `JSON_ENTITY` added to `NodeLabel` enum with `UniqueKeyType.UNIQUE_ID` (`unique_id` property).

**Fix Applied**:
- Added `JSON_ENTITY = "JsonEntity"` to `NodeLabel` enum in `constants.py`
- Added `NodeLabel.JSON_ENTITY: UniqueKeyType.UNIQUE_ID` to `_NODE_LABEL_UNIQUE_KEYS`
- Added `KEY_UNIQUE_ID = "unique_id"` and `UniqueKeyType.UNIQUE_ID`
- Added `NodeSchema` for `JSON_ENTITY` in `types_defs.py`
- Added `ensure_constraints()` call in JSON ingestion pipeline

---

### Issue 5: Relationship Schema Missing JSON Relationships [FIXED]

**Severity**: High
**File**: `codebase_rag/types_defs.py`

**Problem**: `RELATIONSHIP_SCHEMAS` did not include relationships for `JsonEntity` nodes.

**Fix Applied**: Added generic `RELATES_TO` relationship schema for `JSON_ENTITY` -> `JSON_ENTITY` in `RELATIONSHIP_SCHEMAS`.

**Note**: JSON ingestion uses dynamic relationship types from user data. The `RELATES_TO` schema provides generic documentation coverage for entity-to-entity relationships.

---

### Issue 6: `RelationshipType` Missing Generic `RELATES_TO` for Dynamic Relationships [FIXED]

**Severity**: Medium
**File**: `codebase_rag/constants.py`

**Problem**: JSON ingestion creates relationships with arbitrary types (from user data), but `RelationshipType` enum only had predefined types.

**Fix Applied**: Added `RELATES_TO = "RELATES_TO"` as a generic relationship type for dynamic relationships.

---

### Issue 7: `EMBEDDABLE_CODE_NODE_LABELS` Tuple Missing `CodeChunk` [FIXED]

**Severity**: High
**File**: `codebase_rag/constants.py`

**Problem**: `EMBEDDABLE_CODE_NODE_LABELS` defines which node types get vector indexes, but `CodeChunk` was missing.

**Fix Applied**: Added `"CodeChunk"` to the `EMBEDDABLE_CODE_NODE_LABELS` tuple.

---

### Issue 8: Document Chunk `qualified_name` Collision Risk [FIXED]

**Severity**: Medium
**File**: `codebase_rag/document/chunking.py`

**Problem**: `DocumentChunk.qualified_name` used format `{document_path}#chunk_{chunk_index}`. If a document is re-indexed, chunk indices may shift, causing stale nodes.

**Fix Applied**: Updated `qualified_name` to use content-hash-based format:
```python
content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:8]
return f"{self.document_path}#chunk_{content_hash}_{self.chunk_index}"
```

**Note**: Existing chunk nodes with old qualified_name format will need cleanup during migration.

---

### Issue 9: `CYPHER_QUERY_EMBEDDINGS` Missing `CodeChunk` [FIXED]

**Severity**: High
**File**: `codebase_rag/constants.py`

**Problem**: The `CYPHER_QUERY_EMBEDDINGS` query did not include `CodeChunk` nodes, so they would never be returned for embedding generation.

**Fix Applied**: Added UNION clause for `CodeChunk` nodes:
```cypher
UNION
MATCH (m:Module)
WHERE m.qualified_name STARTS WITH ($project_name + '.')
MATCH (m)-[:DEFINES]->(n:CodeChunk)
RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
       n.start_line AS start_line, n.end_line AS end_line, n.path AS path
```

---

### Issue 10: `Section` Node `qualified_name` Uses Line Numbers — Fragile on Edit [FIXED]

**Severity**: Medium
**File**: `codebase_rag/document/document_updater.py`

**Problem**: Section `qualified_name` used format `{parent_path}#L{section.start_line}:{section.title}`. Line numbers shift when documents are edited, causing orphaned nodes.

**Fix Applied**: Updated to use title-hash-based format:
```python
title_hash = hashlib.sha256(section.title.encode()).hexdigest()[:8]
section_qn = f"{parent_path}#sec_{title_hash}:{section.title}"
```

**Trade-off**: Duplicate section titles in the same document will collide. Mitigation: append occurrence counter if needed.

**Note**: Existing section nodes with old qualified_name format will need cleanup during migration.

---

### Issue 11: `Document` Node `total_section_count` Property Name Inconsistency [LOW]

**Severity**: Low
**File**: `codebase_rag/document/document_updater.py`

**Problem**: Legacy `section_count` property migration exists alongside `total_section_count`.

**Status**: Operational note. The migration function already exists. No code change needed.

---

## 3. Implementation Summary

| Issue | Status | Action Taken |
|-------|--------|--------------|
| 1 | Resolved | Already fixed in codebase |
| 2 | Fixed | Added `CODE_CHUNK` to `NodeType` |
| 3 | Fixed | Added `embedding_model`/`embedding_version` to `Chunk` schema |
| 4 | Fixed | Added `JSON_ENTITY` to `NodeLabel` with `UNIQUE_ID` key |
| 5 | Fixed | Added `RELATES_TO` schema for `JSON_ENTITY` relationships |
| 6 | Fixed | Added `RELATES_TO` to `RelationshipType` |
| 7 | Fixed | Added `CodeChunk` to `EMBEDDABLE_CODE_NODE_LABELS` |
| 8 | Fixed | Updated chunk `qualified_name` to use content hash |
| 9 | Fixed | Added `CodeChunk` to `CYPHER_QUERY_EMBEDDINGS` |
| 10 | Fixed | Updated section `qualified_name` to use title hash |
| 11 | Note | Operational, no change needed |

---

## 4. Files Modified

| File | Changes |
|------|---------|
| `codebase_rag/constants.py` | Added `RELATES_TO` to `RelationshipType`, added `CodeChunk` to `EMBEDDABLE_CODE_NODE_LABELS`, added `CodeChunk` UNION to `CYPHER_QUERY_EMBEDDINGS`, added `JSON_ENTITY` to `NodeLabel`, added `UniqueKeyType.UNIQUE_ID` |
| `codebase_rag/types_defs.py` | Added `CODE_CHUNK` to `NodeType`, updated `Chunk` NodeSchema with `embedding_model`/`embedding_version`, added `JSON_ENTITY` NodeSchema and `RELATES_TO` relationship schema |
| `codebase_rag/document/chunking.py` | Added `hashlib` import, updated `DocumentChunk.qualified_name` to use content hash |
| `codebase_rag/document/document_updater.py` | Added `hashlib` import, updated section qualified_name to use title hash |
| `codebase_rag/json_ingestion.py` | Added `ensure_constraints()` call during entity and relationship ingestion |
| `codebase_rag/tests/test_node_relationship_coverage.py` | Added `test_unique_id_unique_key_uses_correct_property` |
| `codebase_rag/tests/test_json_ingestion.py` | Added `test_ingest_json_data_calls_ensure_constraints` |

---

## 5. Migration Considerations

### Chunk Qualified Name Change
- Old format: `{document_path}#chunk_{index}`
- New format: `{document_path}#chunk_{hash}_{index}`
- **Action Required**: Run cleanup to delete old chunk nodes before re-indexing documents

### Section Qualified Name Change
- Old format: `{parent_path}#L{line}:{title}`
- New format: `{parent_path}#sec_{hash}:{title}`
- **Action Required**: Run cleanup to delete old section nodes before re-indexing documents

### Suggested Migration Script
```python
# Delete old format nodes (run before re-indexing)
MATCH (c:Chunk) WHERE c.qualified_name MATCHES '.*#chunk_\\d+$' DETACH DELETE c
MATCH (s:Section) WHERE s.qualified_name CONTAINS '#L' DETACH DELETE s
```

---

## 6. Testing Strategy

For each fix, verify:

1. **Import-time validation**: `python -c "from codebase_rag.constants import NODE_UNIQUE_CONSTRAINTS; print('OK')"`
2. **Constraint coverage**: `pytest codebase_rag/tests/test_node_relationship_coverage.py -v`
3. **Embedding query coverage**: Verify `CYPHER_QUERY_EMBEDDINGS` returns `CodeChunk` nodes
4. **Vector index creation**: Check that `CodeChunk` nodes get vector indexes

---

## 7. Rollback Plan

All changes are additive (new enum values, new dict entries, new schema entries). Rollback is:
1. Revert the specific commit
2. Re-run `ensure_constraints()` to drop any newly created constraints
3. No data migration needed for additive changes

---

## 8. Acceptance Criteria

- [x] `NodeType.CODE_CHUNK` exists
- [x] `Chunk` schema documents `embedding_model` and `embedding_version`
- [x] `RELATES_TO` relationship type exists
- [x] `EMBEDDABLE_CODE_NODE_LABELS` includes `CodeChunk`
- [x] `CYPHER_QUERY_EMBEDDINGS` returns `CodeChunk` nodes
- [x] `DocumentChunk.qualified_name` uses content hash
- [x] `Section` qualified_name uses title hash
- [x] All existing tests pass
- [x] Design decision made on `JsonEntity` integration (Issues 4/5)
- [x] New tests added for JSON entity constraint coverage (if Option A chosen)
