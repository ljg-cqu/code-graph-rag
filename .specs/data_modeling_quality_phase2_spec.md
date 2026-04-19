# Data Modeling Quality Issues: Phase 2 Design Specification

## Document Information
- **Version**: 1.0.0
- **Date**: 2026-04-19
- **Scope**: code-graph-rag graph data quality, ingestion pipeline improvements
- **Status**: Design Phase

---

## 1. Executive Summary

This specification addresses **7 data modeling quality issues** identified through graph database inspection after codebase ingestion. These issues affect query accuracy, semantic search coverage, and graph traversal patterns.

**Issue Summary:**
| # | Issue | Severity | Nodes Affected |
|---|-------|----------|----------------|
| 1 | Builtin Function nodes orphaned (no parent) | High | 48 |
| 2 | Test nodes with minimal properties | Medium | 2 |
| 3 | JSON content nodes missing `name` property | Medium | 5,759 |
| 4 | Functions without embeddings | Medium | 130 |
| 5 | Self-referential CALLS relationships | Low | 86 |
| 6 | External module paths in `path` property | High | 151 |
| 7 | Missing embeddings for Modules | Low | 687 |

---

## 2. Issue Analysis

### Issue 1: Orphaned Builtin Function Nodes

**Severity**: High
**Files**: `codebase_rag/parsers/call_processor.py`, `codebase_rag/constants.py`

**Problem**:
Builtin functions are created as `Function` nodes with:
- `qualified_name`: `builtin.python.{name}` (e.g., `builtin.python.dict`)
- `name`: `{name}` (e.g., `dict`)
- `is_builtin`: `True`
- `path`: `NULL`

These nodes have **no DEFINES relationship** to any parent node, making them orphaned.

**Root Cause**:
The `_ensure_builtin_node()` method in `call_processor.py:385-393` creates builtin nodes without establishing a parent relationship:

```python
def _ensure_builtin_node(self, callee_type: str, callee_qn: str) -> None:
    self.ingestor.ensure_node(
        callee_type,
        {
            cs.KEY_QUALIFIED_NAME: callee_qn,
            cs.KEY_NAME: callee_qn.split(cs.SEPARATOR_DOT)[-1],
            cs.KEY_IS_BUILTIN: True,
        },
    )
```

**Impact**:
- Queries like `MATCH (n:Function) WHERE NOT (n)<-[:DEFINES]-()` return builtin nodes
- Builtin nodes appear as "orphans" in structural integrity checks
- Semantic search may have inconsistent coverage

**Proposed Solution**:
Create a virtual `Builtin` container node that serves as the parent for all builtin functions.

```cypher
CREATE (b:Module:Builtins {
    qualified_name: 'builtin',
    name: '__builtins__',
    is_virtual: true
})
```

Then establish DEFINES relationships:
```cypher
MATCH (f:Function) WHERE f.qualified_name STARTS WITH 'builtin.'
MATCH (b:Module {qualified_name: 'builtin'})
MERGE (b)-[:DEFINES]->(f)
```

**Implementation Changes**:

1. Add `BUILTIN_MODULE_QN = "builtin"` constant
2. Modify `_ensure_builtin_node()` to:
   - Ensure `Builtins` module node exists
   - Create DEFINES relationship from Builtins to builtin function

---

### Issue 2: Test Nodes with Minimal Properties

**Severity**: Medium
**Affected Nodes**: 2

**Problem**:
Two `Test` nodes exist with only these properties:
- `community_id`: -1
- `id`: 1 or 2
- `pagerank_score`: 3.829517243491605e-05

Missing: `name`, `qualified_name`, `path`

**Root Cause**:
Unknown. These appear to be artifact nodes from graph algorithm runs or test data.

**Proposed Solution**:
1. Add validation to reject nodes without `name` property
2. Add cleanup query to remove incomplete `Test` nodes:

```cypher
MATCH (t:Test) WHERE t.name IS NULL DETACH DELETE t
```

---

### Issue 3: JSON Content Nodes Missing `name` Property

**Severity**: Medium
**Files**: `codebase_rag/parsers/json_content_processor.py`
**Affected Nodes**:
- JsonField: 2,592
- JsonValue: 2,482
- JsonObject: 515
- JsonArray: 170

**Problem**:
JSON content nodes have `qualified_name` but no `name` property. This violates the node schema expectation that all nodes should have a `name` property for display purposes.

**Current Node Schemas** (from `types_defs.py`):
```python
NodeSchema(NodeLabel.JSON_OBJECT, "{qualified_name: string, path: string, depth: int}")
NodeSchema(NodeLabel.JSON_ARRAY, "{qualified_name: string, path: string, depth: int, length: int}")
NodeSchema(NodeLabel.JSON_FIELD, "{qualified_name: string, path: string, key: string, value: string, value_type: string, depth: int}")
NodeSchema(NodeLabel.JSON_VALUE, "{qualified_name: string, path: string, value: string, value_type: string, depth: int}")
```

**Root Cause**:
`json_content_processor.py` creates nodes with `qualified_name`, `path`, and JSON-specific properties but omits `name`:

```python
self.ingestor.ensure_node_batch(
    cs.NodeLabel.JSON_OBJECT,
    {
        cs.KEY_QUALIFIED_NAME: object_qn,
        cs.KEY_PATH: file_path,
        cs.KEY_JSON_DEPTH: depth,
    },
)
```

**Proposed Solution**:
Add `name` property to JSON content nodes by deriving it from the JSON key or qualified name:

| Node Type | `name` Value | Source |
|-----------|--------------|--------|
| JsonObject | Last segment of `qualified_name` | e.g., `_object` or `s_xxx` (base64 encoded key) |
| JsonArray | Last segment of `qualified_name` | e.g., `_array` or `s_xxx` |
| JsonField | `{key}` | The actual JSON key from `cs.KEY_JSON_KEY` |
| JsonValue | Truncated value | First 50 chars of `value` property |

**Implementation Changes**:

1. Update `NODE_SCHEMAS` in `types_defs.py` to include `name`:
```python
NodeSchema(NodeLabel.JSON_OBJECT, "{qualified_name: string, name: string, path: string, depth: int}")
NodeSchema(NodeLabel.JSON_ARRAY, "{qualified_name: string, name: string, path: string, depth: int, length: int}")
NodeSchema(NodeLabel.JSON_FIELD, "{qualified_name: string, name: string, path: string, key: string, value: string, value_type: string, depth: int}")
NodeSchema(NodeLabel.JSON_VALUE, "{qualified_name: string, name: string, path: string, value: string, value_type: string, depth: int}")
```

2. Modify `_ingest_object_node()`:
```python
self.ingestor.ensure_node_batch(
    cs.NodeLabel.JSON_OBJECT,
    {
        cs.KEY_QUALIFIED_NAME: object_qn,
        cs.KEY_NAME: object_qn.split(".")[-1],  # Add name
        cs.KEY_PATH: file_path,
        cs.KEY_JSON_DEPTH: depth,
    },
)
```

3. Modify `_ingest_array_node()`:
```python
self.ingestor.ensure_node_batch(
    cs.NodeLabel.JSON_ARRAY,
    {
        cs.KEY_QUALIFIED_NAME: array_qn,
        cs.KEY_NAME: array_qn.split(".")[-1],  # Add name
        cs.KEY_PATH: file_path,
        cs.KEY_JSON_DEPTH: depth,
        cs.KEY_JSON_LENGTH: len(arr),
    },
)
```

4. Modify `_ingest_scalar_node()`:
```python
value_str, value_type = self._classify_value(value)
name = value_str[:50] + "..." if len(value_str) > 50 else value_str
self.ingestor.ensure_node_batch(
    cs.NodeLabel.JSON_VALUE,
    {
        cs.KEY_QUALIFIED_NAME: value_qn,
        cs.KEY_NAME: name,  # Add truncated value as name
        cs.KEY_PATH: file_path,
        cs.KEY_JSON_VALUE: value_str,
        cs.KEY_JSON_VALUE_TYPE: value_type,
        cs.KEY_JSON_DEPTH: depth,
    },
)
```

5. For `JSON_FIELD`, the `key` property already exists, so use it as `name`:
```python
self.ingestor.ensure_node_batch(
    cs.NodeLabel.JSON_FIELD,
    {
        cs.KEY_QUALIFIED_NAME: field_qn,
        cs.KEY_NAME: raw_key,  # Use key as name
        cs.KEY_PATH: file_path,
        cs.KEY_JSON_KEY: raw_key,
        # ... other properties
    },
)
```

---

### Issue 4: Functions Without Embeddings

**Severity**: Medium
**Affected Nodes**: 130 Function nodes

**Problem**:
130 Function nodes have no `embedding` property while all Method and Class nodes do.

**Analysis**:
This is likely due to:
1. Functions added during incremental updates
2. Functions that failed embedding generation
3. Builtin functions (which shouldn't have embeddings)

**Proposed Solution**:
1. Run a backfill query to identify and queue functions for embedding
2. Add logging when embedding generation fails
3. Consider excluding builtin functions from embedding requirement

**Backfill Query**:
```cypher
MATCH (f:Function)
WHERE f.embedding IS NULL AND NOT f.qualified_name STARTS WITH 'builtin.'
RETURN f.qualified_name
```

---

### Issue 5: Self-Referential CALLS Relationships

**Severity**: Low (Informational)
**Affected Relationships**: 86 (59 Method, 27 Function)

**Problem**:
Some nodes have CALLS relationships to themselves:
```cypher
MATCH (n)-[r:CALLS]->(n) RETURN labels(n)[0], count(*)
```

**Analysis**:
This is **expected behavior** for recursive functions. Examples:
- Recursive Fibonacci
- Tree traversal functions
- Mutual recursion (though these create cycles, not self-loops)

**Proposed Solution**:
No action needed. Add documentation note that self-loops are valid for recursive functions.

---

### Issue 6: External Module Paths in `path` Property

**Severity**: High
**Files**: `codebase_rag/parsers/import_processor.py`
**Affected Nodes**: 151 Module nodes

**Problem**:
151 Module nodes have `path` values like:
- `mgclient`
- `collections.abc.Sequence`
- `typing.Any`
- `loguru.logger`

These are **import paths**, not file paths. The `path` property should contain actual file system paths.

**Root Cause**:
`_ensure_external_module_node()` in `import_processor.py:254-269` sets `path` to the import path:

```python
def _ensure_external_module_node(self, module_path: str, full_name: str) -> None:
    if not self.ingestor or not module_path:
        return
    if cs.SEPARATOR_DOUBLE_COLON in module_path:
        name = module_path.rsplit(cs.SEPARATOR_DOUBLE_COLON, 1)[-1]
    else:
        name = module_path.rsplit(cs.SEPARATOR_DOT, 1)[-1]
    self.ingestor.ensure_node_batch(
        cs.NodeLabel.MODULE,
        {
            cs.KEY_NAME: name,
            cs.KEY_QUALIFIED_NAME: module_path,
            cs.KEY_PATH: full_name,  # BUG: This is the import path, not file path!
            cs.KEY_IS_EXTERNAL: True,
        },
    )
```

Note: `KEY_IS_EXTERNAL` already exists in `constants.py:242`.

**Proposed Solution**:
1. Use a separate `import_path` property for import paths
2. Keep `path` reserved for file system paths (NULL for external modules)
3. `is_external` is already set correctly

**Schema Changes**:

| Property | Current | Proposed |
|----------|---------|----------|
| `path` | Import path (e.g., `typing.Any`) | NULL for external modules |
| `import_path` | N/A | Import path (e.g., `typing.Any`) |
| `is_external` | True | True (unchanged) |

**Implementation Changes**:

1. Add constant in `constants.py`:
```python
KEY_IMPORT_PATH = "import_path"
```

2. Modify `_ensure_external_module_node()`:
```python
def _ensure_external_module_node(self, module_path: str, full_name: str) -> None:
    if not self.ingestor or not module_path:
        return
    if cs.SEPARATOR_DOUBLE_COLON in module_path:
        name = module_path.rsplit(cs.SEPARATOR_DOUBLE_COLON, 1)[-1]
    else:
        name = module_path.rsplit(cs.SEPARATOR_DOT, 1)[-1]
    self.ingestor.ensure_node_batch(
        cs.NodeLabel.MODULE,
        {
            cs.KEY_NAME: name,
            cs.KEY_QUALIFIED_NAME: module_path,
            cs.KEY_PATH: None,  # No file path for external modules
            cs.KEY_IMPORT_PATH: full_name,  # Store import path separately
            cs.KEY_IS_EXTERNAL: True,
        },
    )
```

---

### Issue 7: Missing Embeddings for Modules

**Severity**: Low
**Affected Nodes**: 687 Module nodes

**Problem**:
All 687 Module nodes have 0 embeddings.

**Analysis**:
This is **by design**. Modules are container nodes that group functions, classes, and other definitions. They don't have code content to embed.

**Proposed Solution**:
Consider adding module-level embeddings for use cases like:
- Finding similar modules
- Module recommendation

**Optional Enhancement**:
Generate module embeddings from aggregated function/class signatures:

```python
module_content = "\n".join([
    f"def {f.name}({f.parameters}): ..." 
    for f in module.functions
] + [
    f"class {c.name}: ..."
    for c in module.classes
])
module.embedding = embed(module_content)
```

---

## 3. Implementation Plan

### Phase 1: Critical Fixes (Issues 1, 6)

1. **Builtin Container Node**
   - Add `BUILTIN_MODULE_QN` constant
   - Modify `_ensure_builtin_node()` to create parent relationship
   - Add migration to fix existing orphaned builtin nodes

2. **External Module Path Fix**
   - Add `import_path` and `is_external` properties
   - Update import processor
   - Add migration to fix existing external modules

### Phase 2: Data Quality (Issues 2, 3, 4)

1. **JSON Node Names**
   - Update `json_content_processor.py` to set `name` property
   - Add migration to backfill existing nodes

2. **Test Node Cleanup**
   - Add validation in node creation
   - Run cleanup query

3. **Embedding Backfill**
   - Create backfill utility
   - Add logging for failed embeddings

### Phase 3: Documentation (Issues 5, 7)

1. Document self-loops as valid for recursive functions
2. Consider optional module embeddings feature

---

## 4. Migration Scripts

### Migration 1: Fix Orphaned Builtin Functions

```cypher
// Create Builtins module if not exists
MERGE (b:Module:Builtins {
    qualified_name: 'builtin',
    name: '__builtins__',
    is_virtual: true
})

// Create DEFINES relationships
MATCH (f:Function)
WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-()
MATCH (b:Module {qualified_name: 'builtin'})
MERGE (b)-[:DEFINES]->(f)
```

### Migration 2: Fix External Module Paths

```cypher
// For modules with is_external=true, move path to import_path
MATCH (m:Module)
WHERE m.is_external = true AND m.path IS NOT NULL
SET m.import_path = m.path,
    m.path = null
```

**Note**: This migration relies on `is_external` property already being set correctly.

### Migration 3: Add Names to JSON Nodes

```cypher
// JsonObject nodes - derive from qualified_name
MATCH (n:JsonObject)
WHERE n.name IS NULL
SET n.name = split(n.qualified_name, '.')[-1]

// JsonArray nodes - derive from qualified_name
MATCH (n:JsonArray)
WHERE n.name IS NULL
SET n.name = split(n.qualified_name, '.')[-1]

// JsonField nodes - use key property (already exists)
MATCH (n:JsonField)
WHERE n.name IS NULL AND n.key IS NOT NULL
SET n.name = n.key

// JsonValue nodes - use truncated value
MATCH (n:JsonValue)
WHERE n.name IS NULL AND n.value IS NOT NULL
WITH n, toString(n.value) AS val
SET n.name = CASE
    WHEN size(val) > 50 THEN left(val, 50) + '...'
    ELSE val
END
```

### Migration 4: Cleanup Incomplete Test Nodes

```cypher
MATCH (t:Test)
WHERE t.name IS NULL AND t.qualified_name IS NULL
DETACH DELETE t
```

---

## 5. Acceptance Criteria

- [ ] Builtin functions have DEFINES relationship to `builtin` module
- [ ] All JSON content nodes have `name` property
- [ ] External modules use `import_path` instead of `path`
- [ ] No orphaned Function nodes except valid edge cases
- [ ] All Test nodes have `name` and `qualified_name`
- [ ] Migration scripts tested and documented
- [ ] Tests pass for all modified code paths

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Migration breaks existing queries | Medium | High | Run in dry-run mode first |
| External module detection false positives | Low | Medium | Add comprehensive patterns |
| Embedding backfill causes performance issues | Medium | Low | Batch processing with rate limits |

---

## 7. Related Documents

- `.specs/data_modeling_quality_fix_spec.md` - Phase 1 fixes
- `codebase_rag/constants.py` - Node labels and relationship types
- `codebase_rag/types_defs.py` - Node and relationship schemas
- `MEMORY.md` - Project architecture overview
