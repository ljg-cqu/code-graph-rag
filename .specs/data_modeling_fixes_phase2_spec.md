# Data Modeling Fixes - Phase 2 Design Specification

## Overview

This document addresses remaining data modeling issues discovered after the initial Phase 1 fixes were applied. The Phase 1 fixes (deferred relationship creation, cursor exception handling) were correct but incomplete.

## Issues Identified

### Issue 1: Automatic Relationship Flush on Buffer Size (NEW)

**Symptoms:**
Relationship flush failures still occurring despite Phase 1 fixes:
```
WARNING  | Relationship flush failures for IMPORTS: 102/299 failed (Module -> Module)
WARNING  | Relationship flush failures for INHERITS: 9/11 failed (Class -> Class)
```

**Root Cause:**
The `ensure_relationship_batch()` method in `graph_service.py` (lines 954-957) automatically flushes both nodes AND relationships when the relationship buffer reaches `batch_size` (default 1000):

```python
# graph_service.py:954-957
if self._rel_count >= self.batch_size:
    logger.debug(ls.MG_REL_BUFFER_FLUSH, size=self.batch_size)
    self.flush_nodes()
    self.flush_relationships()  # <-- THIS IS THE PROBLEM
```

**Why This Breaks Deferred Relationship Creation:**
1. Workers process files and add relationships to the buffer
2. When relationship count reaches 1000, automatic flush triggers
3. But target nodes (referenced by MATCH clause) may not exist yet
4. The Cypher `MATCH (a:Module {qualified_name: row.from_val})` fails
5. Relationship creation fails silently (logged but not retried)

**Timeline:**
```
File 1 processing -> Adds 500 relationships
File 2 processing -> Adds 500 relationships -> Buffer reaches 1000
                      -> Automatic flush triggered
                      -> But File 3's nodes haven't been created yet
                      -> Relationships referencing File 3 nodes FAIL
```

**Evidence:**
The log shows relationship flushes happening during processing:
```
12:30:57.145 | flush_nodes:1081 - Flushed 683 of 683 buffered nodes
12:30:57.146 | flush_relationships:1181 - Parallel flushing 12 relationship groups
12:30:57.186 | WARNING - Relationship flush failures for IMPORTS: 102/299 failed
```

### Issue 2: Health Checker Row Index Access (PARTIALLY FIXED)

**Symptoms:**
```
WARNING  | Quality validation query failed: Memgraph query failed:
<class 'mgclient.Column'> returned a result with an exception set
Query: MATCH (n:File) RETURN count(n) AS count
```

**Root Cause:**
The `_fetch_single_int()` method in `health_checker.py` (line 46) accesses `row[0]` directly:
```python
# health_checker.py:46
return int(row[0])  # <-- Can fail if Column has exception set
```

The Phase 1 fix added exception handling to `_cursor_to_results()` but this code path doesn't use that method. It uses `cursor.fetchone()` directly and indexes the result.

### Issue 3: Node Buffer Automatic Flush (CORRECT BEHAVIOR)

**Note:** The automatic flush in `ensure_node()` (lines 895-897) is correct:
```python
if len(self.node_buffer) >= self.batch_size:
    logger.debug(ls.MG_NODE_BUFFER_FLUSH, size=self.batch_size)
    self.flush_nodes()  # This is correct - only flushes nodes
```

This should be kept as-is. Only flush nodes, never flush relationships automatically.

## Proposed Solutions

### Solution 1: Remove Automatic Relationship Flush (CRITICAL)

**File:** `codebase_rag/services/graph_service.py`
**Lines:** 954-957

**Change:** Remove `flush_relationships()` from the automatic flush trigger.

**Before:**
```python
if self._rel_count >= self.batch_size:
    logger.debug(ls.MG_REL_BUFFER_FLUSH, size=self.batch_size)
    self.flush_nodes()
    self.flush_relationships()
```

**After:**
```python
if self._rel_count >= self.batch_size:
    logger.debug(ls.MG_REL_BUFFER_FLUSH, size=self.batch_size)
    self.flush_nodes()
    # Note: Relationships are flushed at the end of ingestion via flush_all()
    # to ensure all target nodes exist before relationships are created.
```

**Rationale:**
- Relationships must only be flushed after ALL nodes are created
- The `run()` method already calls `flush_all()` at the correct time
- Periodic node flushes are fine, but relationship flushes must be deferred

### Solution 2: Add Exception Handling to Health Checker

**File:** `codebase_rag/tools/health_checker.py`
**Lines:** 39-48

**Change:** Add try/except wrapper around row access.

**Before:**
```python
@staticmethod
def _fetch_single_int(
    cursor: mgclient.Cursor,
    query: str,
    params: dict[str, object] | None = None,
) -> int:
    try:
        cursor.execute(query, params)
        row = cursor.fetchone()
        HealthChecker._consume_all_results(cursor)
        if row is None:
            return 0
        return int(row[0])
    except Exception as e:
        raise QueryExecutionError(query, params, e) from e
```

**After:**
```python
@staticmethod
def _fetch_single_int(
    cursor: mgclient.Cursor,
    query: str,
    params: dict[str, object] | None = None,
) -> int:
    try:
        cursor.execute(query, params)
        row = cursor.fetchone()
        HealthChecker._consume_all_results(cursor)
        if row is None:
            return 0
        # Handle potential exception on Column object
        try:
            return int(row[0])
        except Exception as col_error:
            # mgclient.Column may have an exception set
            # Try to extract value using alternative method
            if hasattr(row, '__iter__'):
                values = list(row)
                if values:
                    return int(values[0])
            raise QueryExecutionError(
                query, params,
                RuntimeError(f"Failed to extract column value: {col_error}")
            ) from col_error
    except QueryExecutionError:
        raise
    except Exception as e:
        raise QueryExecutionError(query, params, e) from e
```

**Rationale:**
- The mgclient library's Column objects can have exceptions set
- Direct indexing `row[0]` may fail with confusing error messages
- Alternative extraction via `list(row)` may work when indexing fails

### Solution 3: Add Memory Monitoring for Large Ingestions (OPTIONAL)

**File:** `codebase_rag/services/graph_service.py`

**Change:** Add memory warning when relationship buffer grows large.

```python
def ensure_relationship_batch(self, ...):
    # ... existing code ...
    self._rel_count += 1

    # Add memory warning (not automatic flush)
    if self._rel_count > 0 and self._rel_count % 10000 == 0:
        import sys
        buffer_size_mb = sys.getsizeof(self._rel_groups) / (1024 * 1024)
        logger.debug(
            f"Relationship buffer size: {self._rel_count} relationships, "
            f"~{buffer_size_mb:.1f}MB in memory"
        )
```

**Rationale:**
- Large codebases (>100k files) may accumulate many relationships
- Provide visibility into memory usage without triggering premature flushes
- Helps identify if memory becomes a bottleneck

## Implementation Plan

### Phase 1: Critical Fix (Immediate)

1. **Modify `ensure_relationship_batch()`** in `graph_service.py`
   - Remove `flush_relationships()` call from automatic flush trigger
   - Add comment explaining why relationships are deferred

2. **Improve `_fetch_single_int()`** in `health_checker.py`
   - Add try/except around `row[0]` access
   - Add fallback extraction method

### Phase 2: Validation

1. Run full ingestion on test codebase
2. Verify no relationship flush failures in logs
3. Verify quality validation completes successfully
4. Check relationship counts match expected values

### Phase 3: Monitoring (Optional)

1. Add relationship buffer size logging
2. Monitor memory usage during large ingestions
3. Consider adding config option for max buffer size warnings

## Testing Strategy

### Unit Tests

```python
def test_ensure_relationship_batch_does_not_flush_relationships():
    """Verify ensure_relationship_batch does not auto-flush relationships."""
    ingestor = MemgraphIngestor(batch_size=10)

    # Add nodes first
    for i in range(5):
        ingestor.ensure_node("Function", {"qualified_name": f"func_{i}"})

    # Add relationships to trigger batch_size (10)
    for i in range(10):
        ingestor.ensure_relationship_batch(
            ("Function", "qualified_name", "func_0"),
            "CALLS",
            ("Function", "qualified_name", "func_1"),
        )

    # Verify nodes were flushed (buffer reached batch_size)
    # But relationships should still be in buffer
    assert ingestor._rel_count == 10
    assert len(ingestor.node_buffer) == 0  # Nodes were flushed

def test_fetch_single_int_handles_column_exception():
    """Verify _fetch_single_int handles mgclient.Column exceptions."""
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = MagicMock()

    # Simulate Column exception on indexing
    def raise_on_index(index):
        raise RuntimeError("<class 'mgclient.Column'> returned a result with an exception set")

    mock_cursor.fetchone.return_value.__getitem__ = raise_on_index
    # But iteration works
    mock_cursor.fetchone.return_value.__iter__ = lambda self: iter([42])

    result = HealthChecker._fetch_single_int(mock_cursor, "MATCH (n) RETURN 1")
    assert result == 42
```

### Integration Tests

1. **Test large codebase ingestion (1000+ files)**
   - Verify no relationship failures in logs
   - Verify all INHERITS relationships created
   - Verify all IMPORTS relationships created

2. **Test cross-file relationship creation**
   - Create test codebase with classes inheriting from other files
   - Verify 100% success rate for INHERITS relationships

3. **Test quality validation**
   - Run validate_ingestion_quality() after ingestion
   - Verify no mgclient.Column errors

## Success Criteria

1. **Relationship flush failure rate**: 0% (down from 28.5%)
2. **Quality validation**: Completes without errors
3. **Memory usage**: No significant increase (<5% higher peak)
4. **Ingestion time**: No significant impact (<5% slower)
5. **All existing tests pass**

## Rollback Plan

If issues arise:
1. Revert `ensure_relationship_batch()` change
2. Relationships will flush automatically again (may have failures)
3. System will still be functional, just with partial data

## Appendix: Root Cause Analysis

### Why Phase 1 Fixes Were Incomplete

| Phase 1 Fix | What Was Fixed | What Was Missed |
|-------------|----------------|-----------------|
| Workers don't process calls | Removed call processing from workers | Didn't address automatic flush trigger |
| Periodic flush only nodes | Changed `flush_all()` to `flush_nodes()` in `_process_files()` | Didn't change `ensure_relationship_batch()` |
| Cursor exception handling | Added try/except in `_cursor_to_results()` | Didn't fix `_fetch_single_int()` |

### Relationship Failure Pattern

```
Time    Event                               Node Buffer   Rel Buffer   DB State
------  ---------------------------------   -----------   ----------   --------
T0      Start processing                    0             0            Empty
T1      Process File 1 (adds nodes+rels)    300           100          Empty
T2      Process File 2 (adds nodes+rels)    600           200          Empty
T3      Process File 3 (adds nodes+rels)    900           300          Empty
T4      Process File 4 (adds nodes+rels)    1200          400          Empty
T5      Node buffer reaches batch_size      0             400          1200 nodes
T6      Process File 5 (adds nodes+rels)    300           500          1200 nodes
...     ...                                 ...           ...          ...
T20     Rel buffer reaches batch_size       500           1000         6000 nodes
T21     AUTO FLUSH TRIGGERS (BUG!)          0             0            6500 nodes
        - flush_nodes() -> OK               (flushed)     (still 1000)
        - flush_relationships() -> FAILS!                 (flushed)    (missing targets)
        - File 20's nodes not in DB yet!
        - Relationships to File 20 nodes fail
```

The fix ensures relationships are never auto-flushed, only flushed at the end when all nodes exist.
