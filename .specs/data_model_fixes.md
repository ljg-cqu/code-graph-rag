# Data Model Fix Design Specification
## Version: 1.1
## Status: Implementation-ready
## Alignment: Compatible with existing code-graph-rag codebase and migration framework
## Updated: 2026-04-20

---

## 1. Identified Data Modeling Issues
Found from existing migration logic and schema analysis:

| Issue ID | Description | Severity | Affected Entities | Status |
|----------|-------------|----------|-------------------|--------|
| DM-001 | Orphaned builtin functions have no parent DEFINES relationship | Medium | Function nodes with qualified_name starting with 'builtin.' | Implemented |
| DM-002 | External modules store file path in `path` property instead of dedicated `import_path` | Medium | Module nodes with is_external=true | Implemented |
| DM-003 | JSON entity nodes missing human-readable `name` property | Low | JsonObject, JsonArray, JsonField, JsonValue nodes | Implemented |
| DM-004 | Method nodes missing `is_exported` boolean flag | Medium | Method nodes | Implemented |
| DM-005 | JsonEntity nodes use `labels` property name which conflicts with graph node labels | Medium | JsonEntity nodes | Implemented |
| DM-006 | Missing uniqueness constraints on `qualified_name` property for entity nodes | High | All code entity nodes | **Requires Implementation** |
| DM-007 | Relationship flush failures during ingestion | High | All relationship types | **Active - Confirmed 2026-04-20** |
| DM-008 | mgclient.Column exception during quality validation | Critical | Quality checker | **Active - Confirmed 2026-04-20** |

### DM-007: Relationship Flush Failures (Active - confirmed 2026-04-20)
**Evidence from logs (2026-04-20):**
```
Flushed 1000 relationships (910 successful, 90 failed)
Flushed 1000 relationships (916 successful, 84 failed)
Flushed 1000 relationships (913 successful, 87 failed)
Flushed 756 relationships (694 successful, 62 failed)
Flushed 1000 relationships (945 successful, 55 failed)
Flushed 1000 relationships (931 successful, 69 failed)
Flushed 1000 relationships (998 successful, 2 failed)
Flushed 1000 relationships (990 successful, 10 failed)
Flushed 1000 relationships (958 successful, 42 failed)
Flushed 260 relationships (257 successful, 3 failed)
Flushed 1000 relationships (950 successful, 50 failed)
Flushed 369 relationships (355 successful, 14 failed)
Flushed 63 relationships (63 successful, 0 failed)
Flushed 1000 relationships (934 successful, 66 failed)
```

**Impact:** Data loss - relationships are not being persisted to the database
**Root Cause:** Unknown - likely related to constraint violations or missing target nodes
**Investigation needed:** Add detailed error logging to identify which relationships fail and why
**Status:** Confirmed still occurring with varying failure rates (2-95 per 1000 relationships)

### DM-008: mgclient.Column Exception in Quality Validation (Active - confirmed 2026-04-20)
**Evidence from logs (2026-04-20 09:22:29):**
```
2026-04-20 09:22:29.206 | WARNING | codebase_rag.tools.health_checker:validate_ingestion_quality:980
  - Quality validation error: <class 'mgclient.Column'> returned a result with an exception set
2026-04-20 09:22:29.207 | INFO | codebase_rag.graph_updater:run:358
  - Ingestion quality validation completed: 0/1 checks passed
2026-04-20 09:22:29.207 | WARNING | codebase_rag.graph_updater:run:364
  - Quality check failed: Ingestion validation failed - Validation could not complete due to an error
```

**Impact:** Quality validation completely fails - ingestion reports 0/1 checks passed
**Root Cause:** mgclient cursor handling issue - the cursor returns a result with an exception set
**Status:** Active issue - see `.specs/mgclient_cursor_error_handling_spec.md` for fix design
**Note:** Error originates in `health_checker.py:validate_ingestion_quality()` at line 980

### DM-009: Call Resolution Failures (Observation - from logs 2026-04-20)
**Evidence from logs (many instances):**
```
Could not resolve call: pytest.fixture
Could not resolve call: MagicMock
Could not resolve call: create_query_tool
Could not resolve call: self.code_graph.fetch_all
```

**Impact:** CALLS relationships not created for unresolved calls
**Severity:** Low-Medium - many are expected (test fixtures, external libraries, dynamic attributes)
**Note:** These are likely expected for MagicMock, pytest fixtures, and dynamic method calls

**Note:** DM-007 (Cardinality Enforcement) was considered but **removed** because Memgraph does not support existence constraints on relationship patterns. Application-level validation should be used instead (see Section 2.7).

---

## 2. Fix Design Specifications

### 2.1 DM-001: Orphaned Builtin Functions Fix
**Implementation Status:** Complete in `codebase_rag/migrations/data_model_migrations.py`

**Implementation:**
1. Virtual builtin module creation: `MERGE (b:Module {qualified_name: 'builtin', name: '__builtins__', is_virtual: true})`
2. Create DEFINES relationships between virtual builtin module and all orphaned builtin functions

**Validation Check:**
```cypher
MATCH (f:Function) WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-() RETURN count(f) = 0
```

### 2.2 DM-002: External Module Path Fix
**Implementation Status:** Complete in `codebase_rag/migrations/data_model_migrations.py`

**Implementation:**
1. Move `path` property value to `import_path` for all external modules
2. Set `path = null` for external modules

**Validation Check:**
```cypher
MATCH (m:Module) WHERE m.is_external = true AND m.path IS NOT NULL RETURN count(m) = 0
```

### 2.3 DM-003: JSON Node Name Property Fix
**Implementation Status:** Complete in `codebase_rag/migrations/data_model_migrations.py`

**Implementation:**
1. JsonObject/JsonArray: Extract name from last segment of qualified_name
2. JsonField: Use existing `key` property as name
3. JsonValue: Truncate value string to max 50 characters for name

**Validation Check:**
```cypher
MATCH (n) WHERE (n:JsonObject OR n:JsonArray OR n:JsonField OR n:JsonValue) AND n.name IS NULL RETURN count(n) = 0
```

### 2.4 DM-004: Method is_exported Flag Fix
**Implementation Status:** Complete in `codebase_rag/migrations/data_model_migrations.py`

**Implementation:**
1. Set default `is_exported = false` for all Method nodes missing this property

**Validation Check:**
```cypher
MATCH (m:Method) WHERE m.is_exported IS NULL RETURN count(m) = 0
```

### 2.5 DM-005: JsonEntity Labels Property Rename Fix
**Implementation Status:** Complete in `codebase_rag/migrations/data_model_migrations.py`

**Implementation:**
1. Rename `labels` property to `entity_labels` on all JsonEntity nodes
2. Remove old `labels` property

**Validation Check:**
```cypher
MATCH (n:JsonEntity) WHERE n.labels IS NOT NULL RETURN count(n) = 0
```

### 2.6 DM-006: Uniqueness Constraints Add
**Implementation Status:** Requires implementation

**Context:**
The codebase already has:
- `_NODE_LABEL_UNIQUE_KEYS` mapping in `constants.py` defining unique key types per label
- `NODE_UNIQUE_CONSTRAINTS` dict derived from above
- `ensure_constraints()` in `GraphService` using `build_constraint_query()`
- `build_constraint_query()` at `cypher_queries.py:244-245` using Memgraph syntax

**Memgraph Constraint Syntax:**
```cypher
CREATE CONSTRAINT ON (n:Label) ASSERT n.property IS UNIQUE;
```

**Implementation Options:**

**Option A (Recommended):** Use existing infrastructure
- Constraints are already created via `GraphService.ensure_constraints()` during ingestion
- No migration needed - constraints are created automatically on first run
- Ensure all ingestion paths call `ensure_constraints()`

**Option B:** Add migration function for one-time constraint creation
```python
def _migrate_create_constraints(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Create uniqueness constraints for all entity labels."""
    from codebase_rag.constants import NODE_UNIQUE_CONSTRAINTS
    from codebase_rag.cypher_queries import build_constraint_query
    
    count = 0
    for label, prop in NODE_UNIQUE_CONSTRAINTS.items():
        if dry_run:
            logger.info(f"Would create constraint on {label}.{prop}")
            count += 1
        else:
            try:
                cursor.execute(build_constraint_query(label, prop))
                count += 1
            except Exception as e:
                logger.warning(f"Constraint {label}.{prop} may already exist: {e}")
    return count
```

**Validation Check:**
```cypher
CALL schema.constraints() YIELD label, property
RETURN count(DISTINCT label) >= expected_count
```

### 2.7 DM-007: Cardinality Enforcement (Removed)
**Original Proposal:**
Add existence constraints for mandatory relationships:
```cypher
-- NOT SUPPORTED BY MEMGRAPH
CREATE CONSTRAINT FOR (f:Function) REQUIRE EXISTS ((f)<-[:DEFINES]-(:Module))
```

**Reason for Removal:**
Memgraph does not support existence constraints on relationship patterns. Only uniqueness constraints on node properties are supported.

**Alternative Approach - Application-Level Validation:**
Add validation in the ingestion pipeline to ensure:
1. Every Function has a parent Module (via DEFINES relationship)
2. Every Method has a parent Class (via DEFINES relationship)
3. Every CALLS relationship has a valid target

**Implementation Location:** `GraphService.ensure_node_batch()` or pre-ingestion validation hook.

---

## 3. Implementation Roadmap

### Step 1: Verify existing migrations work
```bash
# Run migrations in dry-run mode
python -c "from codebase_rag.migrations.data_model_migrations import run_migrations; print(run_migrations(dry_run=True))"
```

### Step 2: Add DM-006 constraint migration (if Option B chosen)
Extend `run_migrations()` with `_migrate_create_constraints()` function.

### Step 3: Validate all fixes pass validation checks
Run validation Cypher queries against the database.

### Step 4: Add pre-ingestion validation for cardinality
Add validation hooks in ingestion pipeline to prevent:
- Functions without parent modules
- Methods without parent classes
- Orphaned nodes in general

---

## 4. Compatibility Guarantees
1. All changes are non-breaking: existing code that reads `qualified_name`, `import_path`, `entity_labels` will continue to work
2. No data loss: all migrations preserve existing data, only add/rename properties
3. Idempotent: migrations can be run multiple times without side effects
4. Backward compatible: migrations check for existing state before making changes

---

## 5. Test Coverage

| Migration Function | Unit Tests | Status |
|-------------------|------------|--------|
| `_migrate_orphaned_builtins` | `TestMigrateOrphanedBuiltins` | Complete |
| `_migrate_external_module_paths` | `TestMigrateExternalModulePaths` | Complete |
| `_migrate_json_node_names` | `TestMigrateJsonNodeNames` | Complete |
| `_migrate_method_is_exported` | - | **Needs Tests** |
| `_migrate_json_entity_labels` | - | **Needs Tests** |

### New Issues Requiring Investigation (from 2026-04-20 logs)

| Issue | Investigation Needed | Priority | Status |
|-------|---------------------|----------|--------|
| DM-007 - Relationship flush failures | Add error logging to `flush_relationships()` to identify failure patterns | High | **Confirmed Active - 2-95 failures per 1000** |
| DM-008 - mgclient.Column exception | Implement cursor error handling per `mgclient_cursor_error_handling_spec.md` | Critical | **Confirmed Active - quality check 0/1 passed** |

### Successful Operations (2026-04-20)

| Operation | Result | Notes |
|-----------|--------|-------|
| Code ingestion | Completed with warnings | 9062 functions/methods found, 8289 embeddings generated, PageRank and community detection completed |
| Document ingestion | Completed successfully | 57 documents indexed, all relationships flushed successfully (1000/1000) |
| Post-ingestion algorithms | Completed | PageRank updated 9670 nodes, Louvain community detection assigned communities |

### Observations

1. **Relationship flush failures are inconsistent** - Failure rates vary from 2 to 95 per 1000 relationships, suggesting non-deterministic issues (possibly race conditions or constraint violations)
2. **Document graph ingestion is healthier** - Document ingestion shows 100% success rates on relationship flushes
3. **Quality validation completely fails** - The mgclient.Column exception prevents any quality checks from running, masking potential data quality issues

---

## 6. Related Specifications
- `.specs/data-model-quality-fixes.md` - Additional data quality fixes
- `.specs/data_modeling_migration_spec.md` - Original migration framework design
