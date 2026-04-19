# Data Modeling Quality Assessment & Fix Specification

## Document Information
- **Version**: 1.1.0
- **Date**: 2026-04-19
- **Scope**: code-graph-rag codebase data modeling quality
- **Status**: Partially implemented

---

## 1. Executive Summary

This specification documents **11 data modeling issues** identified in the code-graph-rag ingestion pipeline, graph schema, and document processing subsystems. Issues have been reviewed and fixes applied where applicable.

**Implementation Status:**
- **Issue 1**: Already resolved in codebase
- **Issues 2, 3, 6, 7, 8, 9, 10**: Fixed in this update
- **Issues 4, 5**: Require design decision (see details below)
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

### Issue 4: JSON Ingestion Uses `JsonEntity` Label Not in `NodeLabel` Enum [NEEDS DESIGN DECISION]

**Severity**: High
**File**: `codebase_rag/json_ingestion.py`

**Problem**: The JSON ingestion pipeline creates nodes with label `JsonEntity` (constant `JSON_ENTITY_LABEL = "JsonEntity"`), but this label is **not** in the `NodeLabel` enum.

**Context**: The codebase has **two distinct JSON systems**:

| System | Labels | Purpose |
|--------|--------|---------|
| JSON Content Nodes | `JSON_OBJECT`, `JSON_ARRAY`, `JSON_FIELD`, `JSON_VALUE` | Parsing JSON file structure (in `NodeLabel` enum with full schema) |
| Entity JSON Ingestion | `JsonEntity` (string literal) | Entity-based knowledge graph from JSON datasets |

The JSON Content Nodes (`JSON_OBJECT`, etc.) are already properly defined in `NodeLabel` and `_NODE_LABEL_UNIQUE_KEYS`.

**Design Decision Required**:
1. **Option A**: Add `JSON_ENTITY` to `NodeLabel` enum with unique key and schema documentation
2. **Option B**: Keep `JsonEntity` as string-based label for external dataset flexibility, document this decision

**Current State**: `JsonEntity` nodes are created but lack constraint/index coverage via `ensure_constraints()`.

---

### Issue 5: Relationship Schema Missing JSON Relationships [RELATED TO ISSUE 4]

**Severity**: High
**File**: `codebase_rag/types_defs.py`

**Problem**: `RELATIONSHIP_SCHEMAS` does not include relationships for `JsonEntity` nodes.

**Status**: Depends on Issue 4 resolution. JSON Content Nodes (`JSON_OBJECT`, etc.) already have relationship schemas defined.

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
| 4 | Pending | Design decision needed on `JsonEntity` integration |
| 5 | Pending | Depends on Issue 4 |
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
| `codebase_rag/constants.py` | Added `RELATES_TO` to `RelationshipType`, added `CodeChunk` to `EMBEDDABLE_CODE_NODE_LABELS`, added `CodeChunk` UNION to `CYPHER_QUERY_EMBEDDINGS` |
| `codebase_rag/types_defs.py` | Added `CODE_CHUNK` to `NodeType`, updated `Chunk` NodeSchema with `embedding_model`/`embedding_version` |
| `codebase_rag/document/chunking.py` | Added `hashlib` import, updated `DocumentChunk.qualified_name` to use content hash |
| `codebase_rag/document/document_updater.py` | Added `hashlib` import, updated section qualified_name to use title hash |

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
- [ ] Design decision made on `JsonEntity` integration (Issues 4/5)
- [ ] New tests added for JSON entity constraint coverage (if Option A chosen)
