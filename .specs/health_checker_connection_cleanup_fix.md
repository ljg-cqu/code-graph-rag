# Health Checker Connection Cleanup Fix

## Problem Statement

The `validate_ingestion_quality` method in `health_checker.py` logs a warning when closing the Memgraph connection after an exception during query execution:

```
WARNING | codebase_rag.tools.health_checker:validate_ingestion_quality:881 - Failed to close Memgraph connection: cannot close connection during execution of a query
```

This happens because the mgclient driver prevents closing connections while a query is still executing. The method then reports "0/1 checks passed" even though the actual issue is just connection cleanup, not data quality.

## Root Cause Analysis

1. **Incomplete Query Consumption**: When an exception occurs mid-query, the cursor may have pending results that haven't been consumed via `fetchall()`.

2. **Driver Constraint**: mgclient enforces that all query results must be consumed before closing a connection to prevent resource leaks on the server side.

3. **Error Attribution**: The method treats the connection cleanup failure as a validation failure, returning `HEALTH_CHECK_INGESTION_VALIDATION_FAILED` with the error message "Validation could not complete due to an error".

## Impact

- **False Negatives**: Quality checks report failure even when ingestion was successful
- **User Confusion**: Warning messages suggest a problem when there isn't one
- **Monitoring Noise**: Log-based alerting may trigger false alarms

## Proposed Solution

### 1. Add Result Consumption Helper

Add a helper method to safely consume all pending results before closing:

```python
def _consume_all_results(self, cursor: mgclient.Cursor) -> None:
    """Consume all pending results to allow safe connection close."""
    try:
        while cursor.fetchone() is not None:
            pass
    except Exception:
        pass  # Ignore errors during cleanup
```

### 2. Update validate_ingestion_quality Cleanup

Modify the finally block to consume results before closing:

```python
finally:
    if cursor is not None:
        try:
            # Consume any pending results before closing
            self._consume_all_results(cursor)
            cursor.close()
        except Exception as e:
            logger.debug(f"Failed to close Memgraph cursor: {e}")
    if conn is not None:
        try:
            conn.close()
        except Exception as e:
            logger.debug(f"Failed to close Memgraph connection: {e}")
```

### 3. Separate Validation Errors from Cleanup Errors

Ensure that cleanup errors don't affect the validation results:

```python
# In the except block, capture the actual validation error
except Exception as e:
    validation_error = str(e)
    results.append(
        HealthCheckResult(
            name=cs.HEALTH_CHECK_INGESTION_VALIDATION_FAILED,
            passed=False,
            message=cs.HEALTH_CHECK_INGESTION_VALIDATION_ERROR_MSG,
            error=validation_error,
        )
    )
finally:
    # Cleanup should not affect results
    ...
```

### 4. Use Debug Level for Cleanup Warnings

Change the log level from WARNING to DEBUG for cleanup failures, since they don't affect functionality:

```python
logger.debug(f"Failed to close Memgraph connection: {e}")
```

## Implementation Checklist

- [x] Add `_consume_all_results` helper method to HealthChecker class
- [x] Update `validate_ingestion_quality` finally block
- [x] Change log level from WARNING to DEBUG for cleanup messages
- [x] Add unit test for connection cleanup during exception
- [x] Verify no regression in existing tests

## Alternative Approaches

### A. Context Manager Pattern

Use a context manager for connection handling:

```python
@contextmanager
def _get_connection(self) -> Generator[tuple[mgclient.Connection, mgclient.Cursor], None, None]:
    conn = None
    cursor = None
    try:
        conn = mgclient.connect(host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT)
        cursor = conn.cursor()
        yield conn, cursor
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass
```

**Pros**: Reusable, consistent cleanup across all methods
**Cons**: More refactoring required

### B. Connection Pool Usage

Use the existing connection pool from `connection_pool.py`:

```python
from ..services.connection_pool import get_connection_pool

pool = get_connection_pool()
with pool.get_connection() as (conn, cursor):
    ...
```

**Pros**: Centralized connection management, better for concurrent access
**Cons**: Adds dependency on connection pool module

## Recommended Approach

Implement **Option 1 (Result Consumption Helper)** as it:
- Requires minimal code changes
- Maintains backward compatibility
- Addresses the root cause directly
- Can be applied consistently across all health check methods

## Testing Strategy

1. **Unit Test**: Mock cursor to raise exception mid-query, verify cleanup succeeds
2. **Integration Test**: Run quality checks after successful ingestion, verify no warnings
3. **Edge Case Test**: Verify behavior when Memgraph is unresponsive

## Files to Modify

- `codebase_rag/tools/health_checker.py`: Add helper method, update cleanup logic
- `codebase_rag/tests/test_health_checker.py`: Add tests for cleanup during exception
