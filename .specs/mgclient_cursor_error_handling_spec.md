# Mgclient Cursor Error Handling Fix — Design Specification

## Document Information
- **Version**: 1.1.0
- **Date**: 2026-04-20
- **Status**: Implementation-Ready
- **Depends On**: `.specs/quality_check_error_visibility_spec.md`

---

## 1. Problem Statement

The quality validation in `health_checker.py` fails with an opaque error message:

```
WARNING | codebase_rag.tools.health_checker:validate_ingestion_quality:980 - Quality validation error: <class 'mgclient.Column'> returned a result with an exception set
INFO  | codebase_rag.graph_updater:run:358 - Ingestion quality validation completed: 0/1 checks passed
```

This error occurs when `cursor.fetchone()` is called on a cursor that has an error state from a failed Memgraph query. The error message is not actionable and doesn't indicate which query failed or why.

---

## 2. Root Cause Analysis

### 2.1 Mgclient Cursor State Error

**Location:** `codebase_rag/tools/health_checker.py:33-44`

The `_fetch_single_int` helper method doesn't properly handle the case where a Memgraph query fails server-side:

```python
@staticmethod
def _fetch_single_int(
    cursor: mgclient.Cursor,
    query: str,
    params: dict[str, object] | None = None,
) -> int:
    cursor.execute(query, params)
    row = cursor.fetchone()  # FAILS if query had server-side error
    HealthChecker._consume_all_results(cursor)
    if row is None:
        return 0
    return int(row[0])
```

**Problem:** When `cursor.execute()` succeeds but Memgraph returns an error in the result set, `cursor.fetchone()` raises:
```
<class 'mgclient.Column'> returned a result with an exception set
```

This is an internal mgclient error indicating the result set is in an error state.

### 2.2 Potential Query Failure Causes

The queries in `validate_ingestion_quality` that could trigger server-side errors:

1. **`size(n.embedding)` on invalid property** (line 920):
   ```cypher
   RETURN size(n.{embedding_property}) AS dim
   ```
   If `embedding` is stored as a string or binary instead of a list, `size()` may fail.

2. **`ANY(label IN labels(n) WHERE label IN $embedded_labels)`** (line 918):
   The parameter binding might fail if `embedded_labels` has unexpected format.

3. **Duplicate node query** (line 951-956):
   ```cypher
   MATCH (n:{node_label})
   WITH n.path AS path, count(n) AS cnt
   WHERE cnt > 1
   RETURN count(path) AS duplicate_count
   ```
   If `node_label` contains special characters or doesn't exist.

### 2.3 Error Propagation Gap

The current error handling catches the exception but loses context:

```python
except Exception as e:
    error_detail = str(e)  # Only gets "<class 'mgclient.Column'>..."
    if settings.LOG_QUALITY_CHECK_STACKTRACES:
        error_detail = f"{e}\n{traceback.format_exc()}"
    logger.warning(f"Quality validation error: {error_detail}")
```

The actual query that failed is not logged, making diagnosis impossible.

### 2.4 Additional Affected Code Path

The inline query at lines 915-948 also executes queries directly (not via `_fetch_single_int`) and needs the same protection:

```python
# 4. Check invalid embeddings dimension
if missing_embeddings_count < embedded_node_count:
    cursor.execute(
        f"""
        MATCH (n)
        WHERE ANY(label IN labels(n) WHERE label IN $embedded_labels)
          AND n.{embedding_property} IS NOT NULL
        RETURN size(n.{embedding_property}) AS dim
        LIMIT 1
    """,
        {"embedded_labels": embedded_labels},
    )
    result = cursor.fetchone()  # <-- Also needs error wrapping
```

---

## 3. Proposed Solution

### 3.1 Add QueryExecutionError Exception Class

Create a custom exception in `exceptions.py` that captures query context:

```python
class QueryExecutionError(Exception):
    """Raised when a Memgraph query fails with context."""

    def __init__(
        self,
        query: str,
        params: dict | None,
        original_error: Exception
    ):
        self.query = query
        self.params = params
        self.original_error = original_error

        # Truncate query for readability
        query_preview = query[:200] + "..." if len(query) > 200 else query
        super().__init__(
            f"Memgraph query failed: {original_error}\n"
            f"Query: {query_preview}\n"
            f"Params: {params}"
        )
```

### 3.2 Wrap Query Execution with Query Context

Update `_fetch_single_int` to wrap errors with query context:

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
        # Wrap error with query context for debugging
        raise QueryExecutionError(query, params, e) from e
```

### 3.3 Wrap Inline Query Execution

Extract the inline embedding dimension check into a helper method `_fetch_embedding_dim`:

```python
def _fetch_embedding_dim(
    self,
    cursor: mgclient.Cursor,
    embedded_labels: list[str],
    embedding_property: str,
) -> int | None:
    """Fetch embedding dimension with error handling."""
    query = f"""
        MATCH (n)
        WHERE ANY(label IN labels(n) WHERE label IN $embedded_labels)
          AND n.{embedding_property} IS NOT NULL
        RETURN size(n.{embedding_property}) AS dim
        LIMIT 1
    """
    params = {"embedded_labels": embedded_labels}

    try:
        cursor.execute(query, params)
        result = cursor.fetchone()
        HealthChecker._consume_all_results(cursor)
        return result[0] if result else None
    except Exception as e:
        raise QueryExecutionError(query, params, e) from e
```

Then replace lines 915-927 in `validate_ingestion_quality` with:

```python
# 4. Check invalid embeddings dimension
if missing_embeddings_count < embedded_node_count:
    actual_dim = self._fetch_embedding_dim(
        cursor, embedded_labels, embedding_property
    )
    if actual_dim is not None:
        dim_passed = actual_dim == vector_dim
        results.append(
            HealthCheckResult(
                name=cs.HEALTH_CHECK_EMBEDDING_DIMENSION,
                passed=dim_passed,
                message=(
                    cs.HEALTH_CHECK_EMBEDDING_DIMENSION_OK_MSG.format(
                        dim=actual_dim
                    )
                    if dim_passed
                    else cs.HEALTH_CHECK_EMBEDDING_DIMENSION_MISMATCH_MSG.format(
                        actual=actual_dim, expected=vector_dim
                    )
                ),
                error=None
                if dim_passed
                else f"Expected dimension {vector_dim}, got {actual_dim}",
            )
        )
```

### 3.4 Handle QueryExecutionError in Exception Handler

The existing exception handler at lines 975-988 catches `Exception`, so it will automatically catch `QueryExecutionError`. The error message will include query context via `str(e)`.

---

## 4. Implementation Plan

### Phase 1: Immediate Fix

1. Add `QueryExecutionError` exception class to `codebase_rag/exceptions.py`
2. Import `QueryExecutionError` in `codebase_rag/tools/health_checker.py`
3. Wrap `_fetch_single_int` with try/except that adds query context
4. Extract inline embedding dimension query to `_fetch_embedding_dim` method with error wrapping

---

## 5. Code Changes

### File: `codebase_rag/exceptions.py`

Add after the `EmbeddingDimensionMismatchError` alias (around line 408):

```python
class QueryExecutionError(Exception):
    """Raised when a Memgraph query fails with context."""

    def __init__(
        self,
        query: str,
        params: dict | None,
        original_error: Exception
    ):
        self.query = query
        self.params = params
        self.original_error = original_error

        # Truncate query for readability
        query_preview = query[:200] + "..." if len(query) > 200 else query
        super().__init__(
            f"Memgraph query failed: {original_error}\n"
            f"Query: {query_preview}\n"
            f"Params: {params}"
        )
```

### File: `codebase_rag/tools/health_checker.py`

**Import the exception:**

```python
from ..exceptions import QueryExecutionError
```

**Update `_fetch_single_int`:**

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

**Add `_fetch_embedding_dim` method** (after `_fetch_single_int`):

```python
def _fetch_embedding_dim(
    self,
    cursor: mgclient.Cursor,
    embedded_labels: list[str],
    embedding_property: str,
) -> int | None:
    """Fetch embedding dimension with error handling."""
    query = f"""
        MATCH (n)
        WHERE ANY(label IN labels(n) WHERE label IN $embedded_labels)
          AND n.{embedding_property} IS NOT NULL
        RETURN size(n.{embedding_property}) AS dim
        LIMIT 1
    """
    params = {"embedded_labels": embedded_labels}

    try:
        cursor.execute(query, params)
        result = cursor.fetchone()
        HealthChecker._consume_all_results(cursor)
        return result[0] if result else None
    except Exception as e:
        raise QueryExecutionError(query, params, e) from e
```

**Update `validate_ingestion_quality` embedding dimension check** (replace lines 913-948):

```python
            # 4. Check invalid embeddings dimension
            if missing_embeddings_count < embedded_node_count:
                actual_dim = self._fetch_embedding_dim(
                    cursor, embedded_labels, embedding_property
                )
                if actual_dim is not None:
                    dim_passed = actual_dim == vector_dim
                    results.append(
                        HealthCheckResult(
                            name=cs.HEALTH_CHECK_EMBEDDING_DIMENSION,
                            passed=dim_passed,
                            message=(
                                cs.HEALTH_CHECK_EMBEDDING_DIMENSION_OK_MSG.format(
                                    dim=actual_dim
                                )
                                if dim_passed
                                else cs.HEALTH_CHECK_EMBEDDING_DIMENSION_MISMATCH_MSG.format(
                                    actual=actual_dim, expected=vector_dim
                                )
                            ),
                            error=None
                            if dim_passed
                            else f"Expected dimension {vector_dim}, got {actual_dim}",
                        )
                    )
```

**No changes needed** to the exception handler at lines 975-988 - it already catches `Exception` and logs error details.

---

## 6. Testing Strategy

### Unit Tests

Add to `codebase_rag/tests/test_health_checker.py` (create if doesn't exist):

```python
import pytest
from unittest.mock import MagicMock, Mock
import mgclient

from codebase_rag.tools.health_checker import HealthChecker
from codebase_rag.exceptions import QueryExecutionError


class TestQueryExecutionError:
    """Tests for QueryExecutionError exception."""

    def test_error_captures_query_and_params(self):
        """Test that QueryExecutionError captures query context."""
        original_error = ValueError("Original error")
        query = "MATCH (n) RETURN count(n)"
        params = {"key": "value"}

        error = QueryExecutionError(query, params, original_error)

        assert error.query == query
        assert error.params == params
        assert error.original_error is original_error
        assert "MATCH (n)" in str(error)
        assert "Original error" in str(error)

    def test_error_truncates_long_query(self):
        """Test that QueryExecutionError truncates long queries."""
        original_error = ValueError("Original error")
        query = "MATCH (n) RETURN n" + " " * 300
        params = None

        error = QueryExecutionError(query, params, original_error)

        assert "..." in str(error)
        assert len(str(error).split("Query: ")[1].split("\n")[0]) <= 210


class TestFetchSingleInt:
    """Tests for _fetch_single_int error handling."""

    def test_fetch_single_int_wraps_errors(self):
        """Test that _fetch_single_int wraps errors with QueryExecutionError."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = Exception(
            "<class 'mgclient.Column'> returned a result with an exception set"
        )

        with pytest.raises(QueryExecutionError) as exc_info:
            HealthChecker._fetch_single_int(
                mock_cursor,
                "MATCH (n) RETURN count(n)",
                {"param": "value"}
            )

        assert "MATCH (n)" in str(exc_info.value)
        assert "param" in str(exc_info.value)


class TestFetchEmbeddingDim:
    """Tests for _fetch_embedding_dim error handling."""

    def test_fetch_embedding_dim_wraps_errors(self):
        """Test that _fetch_embedding_dim wraps errors with QueryExecutionError."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = Exception(
            "<class 'mgclient.Column'> returned a result with an exception set"
        )

        checker = HealthChecker()

        with pytest.raises(QueryExecutionError) as exc_info:
            checker._fetch_embedding_dim(
                mock_cursor,
                ["Function", "Method"],
                "embedding"
            )

        assert "size(n.embedding)" in str(exc_info.value)
        assert "embedded_labels" in str(exc_info.value)
```

### Integration Tests

1. Simulate Memgraph query failure and verify error message includes query context
2. Test with invalid embedding property type
3. Test with invalid label parameters

### Manual Testing

1. Run `cgr start --index-all` and verify error messages are actionable
2. Check logs show which query failed when `LOG_QUALITY_CHECK_STACKTRACES=true`

---

## 7. Acceptance Criteria

- [x] `QueryExecutionError` exception class exists in `exceptions.py`
- [x] `_fetch_single_int` wraps errors with query context
- [x] `_fetch_embedding_dim` helper method exists and wraps errors
- [x] Error messages include the failed query when `fetchone()` fails
- [x] Error messages are actionable (show query and params)
- [x] Stack trace logging includes query execution context
- [x] No regression in successful query execution
- [x] Tests verify error context is captured

---

## 8. Related Documents

- `.specs/quality_check_error_visibility_spec.md` - Error logging visibility (depends on)
- `.specs/health_checker_connection_cleanup_fix.md` - Connection cleanup
- `.specs/query_mechanism_cypher_fix_spec.md` - Memgraph syntax fixes

---

## 9. Investigation Notes

### Potential Root Causes to Investigate

1. **Embedding property type mismatch**: The `size(n.embedding)` query might fail if embeddings are stored as binary strings instead of lists.

2. **Label parameter binding**: The `$embedded_labels` parameter might have issues with the `ANY(label IN labels(n) WHERE label IN $embedded_labels)` syntax.

3. **Memgraph version compatibility**: Some Cypher functions might behave differently across Memgraph versions.

### Debugging Steps

1. Enable `LOG_QUALITY_CHECK_STACKTRACES=true` in config
2. Run the failing query directly in Memgraph console
3. Check embedding property type with:
   ```cypher
   MATCH (n:Function)
   WHERE n.embedding IS NOT NULL
   RETURN n.embedding, typeof(n.embedding)
   LIMIT 1
   ```

---

## 10. Changelog

### 1.1.0 (2026-04-20)
- Moved `QueryExecutionError` to `exceptions.py` for consistency with codebase patterns
- Removed Phase 2 and 3 (over-engineering) - Phase 1 is sufficient
- Added `_fetch_embedding_dim` helper to handle inline query at lines 915-948
- Updated acceptance criteria to be complete and consistent
- Updated all code examples to be implementation-ready

### 1.0.0 (Initial)
- Initial specification
