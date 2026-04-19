# Quality Check Error Visibility Fix

## Document Information
- **Version**: 1.1.0
- **Date**: 2026-04-20
- **Scope**: code-graph-rag ingestion quality checks, error logging
- **Status**: Implemented

---
## Implementation Summary

**Implemented in**: Commits `0ed01d3` and uncommitted changes in working tree.

| # | Issue | Status | Implementation |
|---|-------|--------|----------------|
| 1 | Quality check error details not visible | ✅ Fixed | Error details now logged at WARNING level inline |
| 2 | Exception stack trace not logged | ✅ Fixed | `LOG_QUALITY_CHECK_STACKTRACES` config option added |
| 3 | Misleading "0/1 checks passed" message | ⏸️ Deferred | Low priority, not implemented |

## 1. Executive Summary

This specification addresses the **lack of visibility** into ingestion quality check failures. When validation encounters an exception, users see "Validation could not complete due to an error" but the actual error details are hidden in the `error` field of the result and not logged at WARNING level.

**Issue Summary:**
| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | Quality check error details not visible | Medium | Users cannot diagnose failures |
| 2 | Exception stack trace not logged | Medium | Debugging requires code modification |
| 3 | Misleading "0/1 checks passed" message | Low | Implies single check when multiple expected |

---

## 2. Issue Analysis

### Issue 1: Quality Check Error Details Not Visible

**Severity**: Medium
**File**: `codebase_rag/graph_updater.py:361-365`
**Priority**: P2

**Problem**:
When `validate_ingestion_quality()` catches an exception, it returns a single failed result with the error message. However, the log output only shows:

```
WARNING | Quality check failed: Ingestion validation failed - Validation could not complete due to an error
```

The actual exception message (which contains the root cause) is in the `error` field but only logged at DEBUG level:

```python
for fail in failed:
    logger.warning(f"Quality check failed: {fail.name} - {fail.message}")
    if fail.error:
        logger.debug(f"Error details: {fail.error}")  # Not visible!
```

**Observed Behavior**:
```
2026-04-19 23:27:10.054 | INFO     | Ingestion quality validation completed: 0/1 checks passed
2026-04-19 23:27:10.055 | WARNING  | Quality check failed: Ingestion validation failed - Validation could not complete due to an error
```

**Expected Behavior**:
```
2026-04-19 23:27:10.054 | INFO     | Ingestion quality validation completed: 0/1 checks passed
2026-04-19 23:27:10.055 | WARNING  | Quality check failed: Ingestion validation failed - Validation could not complete due to an error: [actual error message]
```

---

### Issue 2: Exception Stack Trace Not Logged

**Severity**: Medium
**File**: `codebase_rag/tools/health_checker.py:889-902`

**Problem**:
When an exception is caught, only `str(e)` is captured in the `error` field. The full stack trace is not logged, making it difficult to diagnose complex issues.

**Current Code**:
```python
except Exception as e:
    results.append(
        HealthCheckResult(
            name=cs.HEALTH_CHECK_INGESTION_VALIDATION_FAILED,
            passed=False,
            message=cs.HEALTH_CHECK_INGESTION_VALIDATION_ERROR_MSG,
            error=str(e),
        )
    )
```

---

### Issue 3: Misleading "0/1 checks passed" Message

**Severity**: Low
**File**: `codebase_rag/graph_updater.py:356-360`

**Problem**:
When validation fails early due to an exception, only ONE result is returned (the failure), leading to "0/1 checks passed". This is misleading because normally 5+ checks would be run.

**Expected Checks** (from `validate_ingestion_quality`):
1. Node count check
2. Edge count check
3. Missing embeddings check
4. Embedding dimension check
5. Duplicate nodes check

When an exception occurs, users see "0/1 checks passed" instead of "0/5 checks passed" or "Validation failed before completing all checks".

---

## 3. Proposed Solution

### Solution 1: Log Error Details at WARNING Level

Change the log level for error details from DEBUG to WARNING when quality checks fail:

```python
# In graph_updater.py
for fail in failed:
    logger.warning(f"Quality check failed: {fail.name} - {fail.message}")
    if fail.error:
        logger.warning(f"Error details: {fail.error}")  # Changed from DEBUG
```

### Solution 2: Add Stack Trace Logging Option

Add a configuration option to enable full stack trace logging:

```python
# In config.py
LOG_QUALITY_CHECK_STACKTRACES: bool = False
"""Whether to log full stack traces for quality check failures."""
```

```python
# In health_checker.py
import traceback

except Exception as e:
    error_detail = str(e)
    if settings.LOG_QUALITY_CHECK_STACKTRACES:
        error_detail = f"{e}\n{traceback.format_exc()}"
    
    results.append(
        HealthCheckResult(
            name=cs.HEALTH_CHECK_INGESTION_VALIDATION_FAILED,
            passed=False,
            message=cs.HEALTH_CHECK_INGESTION_VALIDATION_ERROR_MSG,
            error=error_detail,
        )
    )
```

### Solution 3: Improve "Checks Passed" Message

Add context when validation fails early:

```python
# In graph_updater.py
if passed < total:
    failed_count = total - passed
    early_failure = any(
        f.name == cs.HEALTH_CHECK_INGESTION_VALIDATION_FAILED 
        for f in failed
    )
    if early_failure:
        logger.warning(
            f"Quality validation encountered an error before completing all checks "
            f"({passed}/{total} completed)"
        )
```

---

## 4. Implementation Plan

### Phase 1: Immediate Fix (High Value)

1. Change log level from DEBUG to WARNING for error details
2. Add error message to the WARNING log line

### Phase 2: Enhanced Debugging (Optional)

1. Add `LOG_QUALITY_CHECK_STACKTRACES` configuration option
2. Log full stack trace when enabled

### Phase 3: Message Clarity (Low Priority)

1. Detect early failures and adjust message
2. Add documentation for quality check troubleshooting

---

## 5. Code Changes

### File: `codebase_rag/graph_updater.py`

```python
# Before (line 361-365)
failed = [res for res in validation_results if not res.passed]
for fail in failed:
    logger.warning(f"Quality check failed: {fail.name} - {fail.message}")
    if fail.error:
        logger.debug(f"Error details: {fail.error}")

# After
failed = [res for res in validation_results if not res.passed]
for fail in failed:
    error_msg = f" - {fail.error}" if fail.error else ""
    logger.warning(f"Quality check failed: {fail.name} - {fail.message}{error_msg}")
```

### File: `codebase_rag/config.py`

```python
# Add new setting
LOG_QUALITY_CHECK_STACKTRACES: bool = False
"""Whether to log full stack traces for quality check failures. 
Enable for debugging complex issues."""
```

### File: `codebase_rag/tools/health_checker.py`

```python
# Add import at top
import traceback

# Modify exception handler (line 887-895)
except Exception as e:
    error_detail = str(e)
    if settings.LOG_QUALITY_CHECK_STACKTRACES:
        error_detail = f"{e}\n{traceback.format_exc()}"
    logger.warning(f"Quality validation error: {error_detail}")  # Add visible log
    results.append(
        HealthCheckResult(
            name=cs.HEALTH_CHECK_INGESTION_VALIDATION_FAILED,
            passed=False,
            message=cs.HEALTH_CHECK_INGESTION_VALIDATION_ERROR_MSG,
            error=error_detail,
        )
    )
```

---

## 6. Acceptance Criteria

- [ ] Error details are visible in WARNING log when quality checks fail
- [ ] Full stack trace logging available via configuration option
- [ ] Error message appears in the same log line as the failure message
- [ ] No breaking changes to existing log format
- [ ] Tests verify error visibility

---

## 7. Testing Strategy

### Unit Tests

1. Test that error details appear in WARNING log
2. Test stack trace logging with configuration option
3. Test message formatting with various error scenarios

### Integration Tests

1. Trigger a validation error and verify log output
2. Verify error message helps diagnose the root cause

---

## 8. Related Documents

- `.specs/health_checker_connection_cleanup_fix.md` - Related fix for connection cleanup
- `codebase_rag/tools/health_checker.py` - Quality check implementation
- `codebase_rag/graph_updater.py` - Quality check invocation

---

## 9. Root Cause Investigation Guide

When quality checks fail with "Validation could not complete", check for:

1. **Memgraph Connection Issues**: Verify Memgraph is running on the configured port
2. **Vector Index Issues**: Check if vector indexes are properly created
3. **Query Syntax Errors**: Verify Cypher query syntax in health checker
4. **Missing Properties**: Ensure required node properties exist
5. **Memory Issues**: Check for out-of-memory errors during large queries

Common fixes:
- Restart Memgraph if connection fails
- Run `cgr start --clean` to recreate indexes
- Check Memgraph logs for query errors
