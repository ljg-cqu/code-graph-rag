# Data Modeling Migration Specification

## Document Information
- **Version**: 1.0.1
- **Date**: 2026-04-20
- **Scope**: code-graph-rag graph data migration for existing nodes
- **Status**: Design Phase

---

## 1. Executive Summary

This specification addresses **data migration needs** identified through graph database inspection. While the code fixes have been applied in recent commits, existing ingested data requires migration to align with the new schema expectations.

**Issue Summary:**
| # | Issue | Severity | Nodes Affected | Code Status |
|---|-------|----------|----------------|-------------|
| 1 | Orphaned builtin function nodes | High | 48 | Fixed in code, needs migration |
| 2 | External modules with `path` set | High | 107 | Fixed in code, needs migration |
| 3 | JSON nodes missing `name` property | Medium | 5,759 | Fixed in code, needs migration |
| 4 | Test nodes with minimal properties | Medium | 2 | Needs cleanup |
| 5 | Internal modules missing `is_external` flag | Low | Optional enhancement |

---

## 2. Issue Analysis

### Issue 1: Orphaned Builtin Function Nodes

**Severity**: High
**Files**: `codebase_rag/parsers/call_processor.py`
**Affected Nodes**: 48 Function nodes

**Current State**:
```cypher
MATCH (f:Function)
WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-()
RETURN count(f)  // Returns 48
```

**Root Cause**:
The `_ensure_builtin_node()` method in `call_processor.py:394-429` now creates:
1. A `Module` node with `qualified_name: 'builtin'` and `is_virtual: true`
2. A `DEFINES` relationship from the builtin module to each builtin function

However, existing builtin functions were created before this fix.

**Proposed Migration**:
```cypher
// Create builtin module if not exists
MERGE (b:Module {
    qualified_name: 'builtin',
    name: '__builtins__',
    is_virtual: true
})

// Create DEFINES relationships for orphaned builtin functions
MATCH (f:Function)
WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-()
MATCH (b:Module {qualified_name: 'builtin'})
MERGE (b)-[:DEFINES]->(f)
```

---

### Issue 2: External Modules with `path` Property Set

**Severity**: High
**Files**: `codebase_rag/parsers/import_processor.py`
**Affected Nodes**: 107 Module nodes

**Current State**:
```cypher
MATCH (m:Module)
WHERE m.is_external = true AND m.path IS NOT NULL
RETURN count(m)  // Returns 107
```

**Root Cause**:
The `_ensure_external_module_node()` method in `import_processor.py:254-270` now correctly sets:
- `path: None` (no file system path for external modules)
- `import_path: full_name` (the import path like `typing.Any`)
- `is_external: True`

However, existing external modules still have `path` set to the import path.

**Proposed Migration**:
```cypher
// Move path to import_path for external modules
MATCH (m:Module)
WHERE m.is_external = true AND m.path IS NOT NULL
SET m.import_path = m.path,
    m.path = null
```

---

### Issue 3: JSON Content Nodes Missing `name` Property

**Severity**: Medium
**Files**: `codebase_rag/parsers/json_content_processor.py`
**Affected Nodes**: 5,759 total
- JsonField: 2,592
- JsonValue: 2,482
- JsonObject: 515
- JsonArray: 170

**Current State**:
```cypher
MATCH (n)
WHERE any(label IN labels(n) WHERE label IN ['JsonObject', 'JsonArray', 'JsonField', 'JsonValue'])
AND n.name IS NULL
RETURN labels(n)[0] AS label, count(n) AS count
```

**Root Cause**:
The `json_content_processor.py` now correctly sets `name` property for all JSON node types:
- `JsonObject`: Last segment of `qualified_name` (line 75)
- `JsonArray`: Last segment of `qualified_name` (line 128)
- `JsonField`: The raw key (line 90)
- `JsonValue`: Truncated value (lines 203-205)

However, existing JSON nodes were created before this fix.

**Proposed Migration**:
```cypher
// JsonObject: derive name from qualified_name
MATCH (n:JsonObject)
WHERE n.name IS NULL
SET n.name = split(n.qualified_name, '.')[-1]

// JsonArray: derive name from qualified_name
MATCH (n:JsonArray)
WHERE n.name IS NULL
SET n.name = split(n.qualified_name, '.')[-1]

// JsonField: use key property as name
MATCH (n:JsonField)
WHERE n.name IS NULL AND n.key IS NOT NULL
SET n.name = n.key

// JsonValue: use truncated value as name
MATCH (n:JsonValue)
WHERE n.name IS NULL AND n.value IS NOT NULL
WITH n, toString(n.value) AS val
SET n.name = CASE
    WHEN size(val) > 50 THEN left(val, 50) + '...'
    ELSE val
END
```

---

### Issue 4: Test Nodes with Minimal Properties

**Severity**: Medium
**Affected Nodes**: 2 Test nodes

**Current State**:
```cypher
MATCH (t:Test)
WHERE t.name IS NULL OR t.qualified_name IS NULL
RETURN count(t)  // Returns 2
```

**Root Cause**:
Two `Test` nodes exist with only these properties:
- `community_id`: -1
- `id`: 1 or 2
- `pagerank_score`: 3.829517243491605e-05

Missing: `name`, `qualified_name`, `path`

These appear to be artifact nodes from graph algorithm runs. `:Test` is not a first-class node label in the application schema (`constants.py` does not define `NodeLabel.TEST`).

**Proposed Migration**:
```cypher
// Delete incomplete Test nodes
MATCH (t:Test)
WHERE t.name IS NULL AND t.qualified_name IS NULL
DETACH DELETE t
```

**Note**: These nodes are disconnected from the graph and serve no purpose.

---

### Issue 5: Internal Modules Missing `is_external` Flag

**Severity**: Low (Optional Enhancement)
**Files**: `codebase_rag/graph_updater.py`, `codebase_rag/parsers/import_processor.py`
**Affected Nodes**: 536 Module nodes

**Current State**:
```cypher
MATCH (m:Module)
WHERE m.is_external IS NULL
RETURN count(m)  // Returns 536
```

**Root Cause**:
Internal modules are created without setting `is_external` property. The schema expects `is_external: bool | null`, and queries assume NULL = internal.

**Design Decision**:
This is **informational only**. The current behavior is correct:
- External modules: `is_external = true`
- Internal modules: `is_external = NULL` (implicit false)

**Optional Enhancement**:
For explicit clarity, internal modules could have `is_external = false`:

```cypher
// Set is_external = false for internal modules
MATCH (m:Module)
WHERE m.is_external IS NULL
SET m.is_external = false
```

**Trade-offs**:
| Approach | Pros | Cons |
|----------|------|------|
| NULL (current) | No migration needed | Requires `IS NULL OR = false` in queries |
| Explicit false | Clearer data model | Migration required, no functional benefit |

**Recommendation**: Keep current behavior (NULL = internal). Update query patterns to handle NULL correctly.

---

## 3. Implementation Plan

### Phase 1: Create Migration Package

Create `codebase_rag/migrations/__init__.py` (empty) and `codebase_rag/migrations/data_model_migrations.py`:

```python
"""Data model migration utilities for graph database."""
from __future__ import annotations

from loguru import logger
import mgclient

from .. import logs as ls
from ..config import settings


def run_migrations(dry_run: bool = True) -> dict[str, int]:
    """Run all data model migrations.

    Args:
        dry_run: If True, only report what would be changed.

    Returns:
        Dict mapping migration name to affected node count.
    """
    results = {}
    conn = mgclient.connect(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
    )
    cursor = conn.cursor()

    try:
        results["orphaned_builtins"] = _migrate_orphaned_builtins(cursor, dry_run)
        results["external_module_paths"] = _migrate_external_module_paths(cursor, dry_run)
        results["json_node_names"] = _migrate_json_node_names(cursor, dry_run)
        results["incomplete_test_nodes"] = _cleanup_incomplete_test_nodes(cursor, dry_run)
    finally:
        cursor.close()
        conn.close()

    return results


def _migrate_orphaned_builtins(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Create DEFINES relationships for orphaned builtin functions."""
    cursor.execute("""
        MATCH (f:Function)
        WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-()
        RETURN count(f)
    """)
    count = cursor.fetchone()[0]

    if count == 0:
        return 0

    if dry_run:
        logger.info(ls.MIGRATION_DRY_RUN_BUILTINS.format(count=count))
        return count

    cursor.execute("""
        MERGE (b:Module {
            qualified_name: 'builtin',
            name: '__builtins__',
            is_virtual: true
        })
    """)

    cursor.execute("""
        MATCH (f:Function)
        WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-()
        MATCH (b:Module {qualified_name: 'builtin'})
        MERGE (b)-[:DEFINES]->(f)
    """)

    logger.info(ls.MIGRATION_BUILTINS_DONE.format(count=count))
    return count


def _migrate_external_module_paths(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Move path to import_path for external modules."""
    cursor.execute("""
        MATCH (m:Module)
        WHERE m.is_external = true AND m.path IS NOT NULL
        RETURN count(m)
    """)
    count = cursor.fetchone()[0]

    if count == 0:
        return 0

    if dry_run:
        logger.info(ls.MIGRATION_DRY_RUN_EXTERNAL_PATHS.format(count=count))
        return count

    cursor.execute("""
        MATCH (m:Module)
        WHERE m.is_external = true AND m.path IS NOT NULL
        SET m.import_path = m.path,
            m.path = null
    """)

    logger.info(ls.MIGRATION_EXTERNAL_PATHS_DONE.format(count=count))
    return count


def _migrate_json_node_names(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Add name property to JSON nodes."""
    total = 0

    cursor.execute("MATCH (n:JsonObject) WHERE n.name IS NULL RETURN count(n)")
    json_object_count = cursor.fetchone()[0]
    total += json_object_count

    cursor.execute("MATCH (n:JsonArray) WHERE n.name IS NULL RETURN count(n)")
    json_array_count = cursor.fetchone()[0]
    total += json_array_count

    cursor.execute("MATCH (n:JsonField) WHERE n.name IS NULL RETURN count(n)")
    json_field_count = cursor.fetchone()[0]
    total += json_field_count

    cursor.execute("MATCH (n:JsonValue) WHERE n.name IS NULL RETURN count(n)")
    json_value_count = cursor.fetchone()[0]
    total += json_value_count

    if total == 0:
        return 0

    if dry_run:
        logger.info(ls.MIGRATION_DRY_RUN_JSON_NAMES.format(count=total))
        return total

    cursor.execute("""
        MATCH (n:JsonObject) WHERE n.name IS NULL
        SET n.name = split(n.qualified_name, '.')[-1]
    """)

    cursor.execute("""
        MATCH (n:JsonArray) WHERE n.name IS NULL
        SET n.name = split(n.qualified_name, '.')[-1]
    """)

    cursor.execute("""
        MATCH (n:JsonField) WHERE n.name IS NULL AND n.key IS NOT NULL
        SET n.name = n.key
    """)

    cursor.execute("""
        MATCH (n:JsonValue) WHERE n.name IS NULL AND n.value IS NOT NULL
        WITH n, toString(n.value) AS val
        SET n.name = CASE
            WHEN size(val) > 50 THEN left(val, 50) + '...'
            ELSE val
        END
    """)

    logger.info(ls.MIGRATION_JSON_NAMES_DONE.format(count=total))
    return total


def _cleanup_incomplete_test_nodes(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Delete incomplete Test nodes."""
    cursor.execute("""
        MATCH (t:Test)
        WHERE t.name IS NULL AND t.qualified_name IS NULL
        RETURN count(t)
    """)
    count = cursor.fetchone()[0]

    if count == 0:
        return 0

    if dry_run:
        logger.info(ls.MIGRATION_DRY_RUN_TEST_NODES.format(count=count))
        return count

    cursor.execute("""
        MATCH (t:Test)
        WHERE t.name IS NULL AND t.qualified_name IS NULL
        DETACH DELETE t
    """)

    logger.info(ls.MIGRATION_TEST_NODES_DONE.format(count=count))
    return count
```

**Note**: Memgraph uses autocommit mode; explicit `conn.commit()` is not required and should not be used with `mgclient`.

---

### Phase 2: Add CLI Command

Add the following constants to `codebase_rag/cli_help.py`:

```python
class CLICommandName(StrEnum):
    # ... existing entries ...
    MIGRATE_DATA = "migrate-data"

CMD_MIGRATE_DATA = "Run data model migrations for existing graph data"
```

Add to `codebase_rag/cli.py`:

```python
@app.command(name=ch.CLICommandName.MIGRATE_DATA, help=ch.CMD_MIGRATE_DATA)
def migrate_data(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Report changes without applying them. Use --no-dry-run to execute.",
    ),
) -> None:
    """Run data model migrations for existing graph data.

    Defaults to dry-run mode for safety. Pass --no-dry-run to apply changes.
    """
    from .migrations.data_model_migrations import run_migrations

    results = run_migrations(dry_run=dry_run)

    if dry_run:
        app_context.console.print("[yellow]Dry run mode - no changes made[/yellow]")

    app_context.console.print("\nMigration Results:")
    for name, count in results.items():
        status = "would migrate" if dry_run else "migrated"
        app_context.console.print(f"  {name}: {count} nodes {status}")
```

---

### Phase 3: Add Health Check

Add the following constants to `codebase_rag/constants.py`:

```python
HEALTH_CHECK_MIGRATION_PASS = "Data migrations applied"
HEALTH_CHECK_MIGRATION_NEEDED = "Data migrations needed"
HEALTH_CHECK_MIGRATION_PASS_MSG = "All data model migrations are up to date"
HEALTH_CHECK_MIGRATION_NEEDED_MSG = "Data migrations needed: {issues}"
HEALTH_CHECK_MIGRATION_ERROR_MSG = "Run 'cgr migrate-data --no-dry-run' to apply migrations"
```

Add to `codebase_rag/tools/health_checker.py`:

```python
def check_data_migrations_needed(self) -> HealthCheckResult:
    """Check if data migrations are needed."""
    conn = None
    cursor = None
    issues = []

    try:
        conn = mgclient.connect(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
        )
        cursor = conn.cursor()

        cursor.execute("""
            MATCH (f:Function)
            WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-()
            RETURN count(f)
        """)
        orphaned_builtins = cursor.fetchone()[0]
        if orphaned_builtins > 0:
            issues.append(f"{orphaned_builtins} orphaned builtin functions")

        cursor.execute("""
            MATCH (m:Module)
            WHERE m.is_external = true AND m.path IS NOT NULL
            RETURN count(m)
        """)
        external_with_path = cursor.fetchone()[0]
        if external_with_path > 0:
            issues.append(f"{external_with_path} external modules with path set")

        cursor.execute("""
            MATCH (n)
            WHERE any(label IN labels(n) WHERE label IN ['JsonObject', 'JsonArray', 'JsonField', 'JsonValue'])
            AND n.name IS NULL
            RETURN count(n)
        """)
        json_missing_name = cursor.fetchone()[0]
        if json_missing_name > 0:
            issues.append(f"{json_missing_name} JSON nodes missing name")

        cursor.execute("""
            MATCH (t:Test)
            WHERE t.name IS NULL AND t.qualified_name IS NULL
            RETURN count(t)
        """)
        incomplete_tests = cursor.fetchone()[0]
        if incomplete_tests > 0:
            issues.append(f"{incomplete_tests} incomplete Test nodes")

        if not issues:
            return HealthCheckResult(
                name=cs.HEALTH_CHECK_MIGRATION_PASS,
                passed=True,
                message=cs.HEALTH_CHECK_MIGRATION_PASS_MSG,
            )

        return HealthCheckResult(
            name=cs.HEALTH_CHECK_MIGRATION_NEEDED,
            passed=False,
            message=cs.HEALTH_CHECK_MIGRATION_NEEDED_MSG.format(issues="; ".join(issues)),
            error=cs.HEALTH_CHECK_MIGRATION_ERROR_MSG,
        )

    except Exception as e:
        return HealthCheckResult(
            name=cs.HEALTH_CHECK_MIGRATION_NEEDED,
            passed=False,
            message=cs.HEALTH_CHECK_MIGRATION_NEEDED,
            error=str(e),
        )
    finally:
        if cursor is not None:
            try:
                HealthChecker._consume_all_results(cursor)
                cursor.close()
            except Exception as e:
                logger.debug(f"Failed to close Memgraph cursor: {e}")
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.debug(f"Failed to close Memgraph connection: {e}")
```

Update `HealthChecker.run_all_checks()` to include the new check:

```python
def run_all_checks(self) -> list[HealthCheckResult]:
    self.results = []
    self.results.append(self.check_docker())
    self.results.append(self.check_memgraph_connection())
    self.results.extend(self.check_api_keys())
    for tool_name, cmd in cs.HEALTH_CHECK_EXTERNAL_TOOLS:
        self.results.append(self.check_external_tool(tool_name, cmd))
    self.results.append(self.check_disconnected_nodes())
    self.results.append(self.check_required_properties())
    self.results.append(self.check_embedding_correlation())
    self.results.append(self.check_vector_indexes())
    self.results.append(self.check_vector_search())
    self.results.append(self.check_file_layer())
    self.results.append(self.check_large_document_chunk_coverage())
    self.results.append(self.check_log_directory())
    self.results.append(self.check_data_migrations_needed())  # NEW
    sample_json_path = Path(cs.HEALTH_CHECK_JSON_SAMPLE_FILE)
    if sample_json_path.exists():
        self.results.append(self.check_json_ingestion_schema(str(sample_json_path)))
    return self.results
```

---

### Phase 4: Add Log Templates

Add to `codebase_rag/logs.py`:

```python
MIGRATION_DRY_RUN_BUILTINS = "[DRY RUN] Would migrate {count} orphaned builtin functions"
MIGRATION_BUILTINS_DONE = "Migrated {count} orphaned builtin functions"
MIGRATION_DRY_RUN_EXTERNAL_PATHS = "[DRY RUN] Would migrate {count} external module paths"
MIGRATION_EXTERNAL_PATHS_DONE = "Migrated {count} external module paths"
MIGRATION_DRY_RUN_JSON_NAMES = "[DRY RUN] Would migrate {count} JSON node names"
MIGRATION_JSON_NAMES_DONE = "Migrated {count} JSON node names"
MIGRATION_DRY_RUN_TEST_NODES = "[DRY RUN] Would delete {count} incomplete Test nodes"
MIGRATION_TEST_NODES_DONE = "Deleted {count} incomplete Test nodes"
```

---

## 4. Migration Execution Plan

### Step 1: Dry Run (Report Only)
```bash
cgr migrate-data --dry-run
```

### Step 2: Review Output
```
Migration Results:
  orphaned_builtins: 48 nodes would migrate
  external_module_paths: 107 nodes would migrate
  json_node_names: 5759 nodes would migrate
  incomplete_test_nodes: 2 nodes would delete
```

### Step 3: Apply Migrations
```bash
cgr migrate-data --no-dry-run
```

### Step 4: Verify
```bash
cgr doctor
```

---

## 5. Acceptance Criteria

- [ ] Migration package created (`codebase_rag/migrations/__init__.py` and `data_model_migrations.py`)
- [ ] Log templates added to `logs.py`
- [ ] CLI command added using `CLICommandName.MIGRATE_DATA` constant
- [ ] Health check added with proper connection cleanup (`_consume_all_results`)
- [ ] `run_all_checks()` invokes the new migration health check
- [ ] All 48 orphaned builtin functions have DEFINES relationship
- [ ] All 107 external modules have `path: NULL` and `import_path` set
- [ ] All 5,759 JSON nodes have `name` property
- [ ] All 2 incomplete Test nodes deleted
- [ ] Tests pass for migration functions
- [ ] Documentation updated in MEMORY.md

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Migration creates duplicate relationships | Low | Medium | Use MERGE instead of CREATE |
| JSON value truncation loses data | Low | Low | Original value preserved in `value` property |
| Migration fails mid-way | Low | Medium | Each migration is idempotent; safe to re-run |

---

## 7. Related Documents

- `.specs/data_modeling_quality_fix_spec.md` - Phase 1 schema fixes
- `.specs/data_modeling_quality_phase2_spec.md` - Phase 2 issue analysis
- `codebase_rag/parsers/call_processor.py` - Builtin node creation
- `codebase_rag/parsers/import_processor.py` - External module creation
- `codebase_rag/parsers/json_content_processor.py` - JSON node creation
