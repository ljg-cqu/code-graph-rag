# Ingestion Hanging Fix Design Specification
## Revision: 1.3 | Date: 2026-04-17 | Status: Implementation Ready

---

## 1. Overview

This document analyzes the root cause of the ingestion process hanging during the `flush_all()` phase and provides implementation-ready fixes. The issue manifests when the codebase indexer completes node flushing but appears to get stuck before completing relationship flushing or the final flush completion log.

---

## 2. Problem Description

### 2.1 Symptoms
When running `cgr start --repo-path <path> --index-all --ingest-json --yolo`, the ingestion process hangs with the following last log output:

```
2026-04-17 09:47:50.750 | INFO | codebase_rag.services.graph_service:flush_nodes:858 - Parallel flushing 2 label groups with 4 workers
2026-04-17 09:47:50.755 | INFO | codebase_rag.services.graph_service:flush_nodes:891 - Flushed 4 of 4 buffered nodes.
```

**Missing logs that should appear:**
- `MG_PARALLEL_FLUSH_RELS` or relationship flush completion from `flush_relationships()`
- `MG_FLUSH_COMPLETE` from `flush_all()`
- `MG_DISCONNECTED` from `__exit__`

### 2.2 Impact
- Ingestion process appears frozen indefinitely
- No data is committed to Memgraph (flush never completes)
- User must manually kill the process (Ctrl+C)
- Partial ingestion state may leave hash cache in inconsistent state

---

## 3. Root Cause Analysis

### 3.1 Primary Root Cause: Connection Timeout Calculated But Never Applied

**File:** `codebase_rag/services/graph_service.py`, method `_create_connection()` (lines 388-461)

**Problem:** The `timeout` variable is calculated but **NEVER USED** in the `mgclient.connect()` call:

```python
def _create_connection(self) -> mgclient.Connection:
    # Get appropriate timeout based on connection type
    timeout = self._connection_timeout
    if timeout is None:
        # Determine based on port
        if self._port == settings.DOC_MEMGRAPH_PORT:
            timeout = settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT  # 600s
        elif self._port == settings.JSON_MEMGRAPH_PORT:
            timeout = settings.JSON_MEMGRAPH_CONNECTION_TIMEOUT
        else:
            timeout = settings.MEMGRAPH_CONNECTION_TIMEOUT  # 600s default

    # Create connection (timeout is configured server-side via Docker)
    # ^^^ THIS COMMENT IS INCORRECT - timeout is NOT applied anywhere!
    if self._username is not None:
        conn = mgclient.connect(
            host=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
        )
    else:
        conn = mgclient.connect(
            host=self._host, port=self._port
        )
    # timeout variable is NEVER passed to mgclient.connect()
```

**Why this causes hanging:** 
- `mgclient.connect()` does **NOT** accept a timeout parameter
- If Memgraph becomes unresponsive or the connection is slow, `mgclient.connect()` will block indefinitely
- This happens specifically in `_flush_rel_group_with_own_conn()` which creates NEW connections for each relationship group

### 3.2 Execution Flow When Hanging Occurs

```
flush_all()
  └─ flush_nodes()  ✅ Completes successfully
  └─ flush_relationships()  ⛔ HANGS HERE
       └─ if self._executor and len(self._rel_groups) > 1:
            └─ futures = {
                 self._executor.submit(
                   self._flush_rel_group_with_own_conn, pattern, params_list
                 ): pattern
                 for pattern, params_list in self._rel_groups.items()
               }
            └─ for future in as_completed(futures):  ⛔ WAITS FOREVER
                 └─ attempted, successful = future.result()  ⛔ BLOCKS HERE
                      └─ _flush_rel_group_with_own_conn()
                           └─ conn = self._create_connection()  ⛔ mgclient.connect() blocks
```

### 3.3 Secondary Root Causes

#### 3.3.1 No Socket-Level Timeout
After connection is established, TCP keepalive is configured but no socket read/write timeout is set. Long-running queries can block indefinitely.

#### 3.3.2 Thread Pool Executor Blocks Indefinitely
In `__exit__()`:
```python
if self._executor:
    self._executor.shutdown(wait=True)  # BLOCKS if workers are stuck
```
If any worker thread is blocked waiting on `mgclient.connect()`, shutdown will block forever.

#### 3.3.3 Missing Timeout on Batch Queries
Methods `_execute_batch_on()` and `_execute_batch_with_return_on()` have retry logic but no execution timeout. If a `MATCH...MERGE` query is expensive or Memgraph is under resource pressure, it can hang.

---

## 4. Fixing Approach

### 4.1 Fix 1: Add Threading-Based Connection Timeout Wrapper (CRITICAL)

**Priority:** P0 - Must implement  
**Risk:** Low - Backward compatible, with known race condition on connection leak (documented below)  
**Files:** `codebase_rag/services/graph_service.py`

#### 4.1.1 Extract Timeout Resolution Helper

The timeout resolution logic is currently duplicated between `_create_connection()` (where it computes `timeout` but never uses it) and the proposed `_create_connection_with_timeout()`. Extract it into a single helper to eliminate duplication and ensure single-source-of-truth:

```python
def _get_connection_timeout(self) -> int:
    """Resolve the effective connection timeout based on port/config."""
    if self._connection_timeout is not None:
        return self._connection_timeout
    if self._port == settings.DOC_MEMGRAPH_PORT:
        return settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT
    elif self._port == settings.JSON_MEMGRAPH_PORT:
        return settings.JSON_MEMGRAPH_CONNECTION_TIMEOUT
    return settings.MEMGRAPH_CONNECTION_TIMEOUT
```

**Update `_create_connection()` to use the helper** (this also removes the misleading comment):

```python
def _create_connection(self) -> mgclient.Connection:
    """Create a new Memgraph connection with timeout configuration.

    Sets up TCP keepalive and socket timeout to prevent connection drops
    and indefinite blocking during long-running operations.
    """
    timeout = self._get_connection_timeout()  # <-- CHANGED: use helper

    # Create connection (mgclient.connect() does not accept a timeout parameter;
    # timeout is enforced via socket.settimeout() below and threading wrapper
    # in _create_connection_with_timeout() for callers that need it)
    if self._username is not None:
        conn = mgclient.connect(
            host=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
        )
    else:
        conn = mgclient.connect(host=self._host, port=self._port)
    conn.autocommit = True

    # Set socket timeout to prevent indefinite blocking on I/O operations
    # (see Fix 2 for details)
    try:
        if hasattr(conn, "socket") or hasattr(conn, "_socket"):
            sock = getattr(conn, "socket", getattr(conn, "_socket", None))
            if sock:
                sock.settimeout(timeout)
    except (OSError, AttributeError) as e:
        logger.warning(
            f"Could not set socket timeout for {self._host}:{self._port}: {e}"
        )

    # Configure TCP keepalive for long-running connections ... (existing code unchanged)

    return conn
```

> **Note:** The `_create_connection()` update above incorporates Fix 2 (socket timeout) directly,
> since both fixes modify the same method. This avoids conflicting patches and ensures the
> `timeout` variable — now resolved via `_get_connection_timeout()` — is actually consumed.

#### 4.1.2 Add Connection Timeout Wrapper

Add a new method that wraps `_create_connection()` with a threading-based timeout:

```python
def _create_connection_with_timeout(self) -> mgclient.Connection:
    """Create connection with timeout enforcement.

    Uses threading to enforce connection timeout since mgclient.connect()
    doesn't support a timeout parameter natively. If the connection attempt
    times out, any connection object that may have been created by the
    background thread is closed to prevent connection leaks.
    """
    timeout = self._get_connection_timeout()

    result: list[mgclient.Connection | None] = [None]
    error: list[Exception | None] = [None]

    def connect_worker():
        try:
            result[0] = self._create_connection()
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=connect_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Connection attempt timed out. The daemon thread may still complete
        # the connection in the background — attempt to close any connection
        # that was created to prevent a connection leak. There is an inherent
        # race condition between this check and the thread writing result[0],
        # so this is best-effort cleanup rather than guaranteed.
        if result[0] is not None:
            try:
                result[0].close()
            except Exception:
                pass
        raise TimeoutError(
            f"Connection to Memgraph at {self._host}:{self._port} "
            f"timed out after {timeout}s. Check if Memgraph is running "
            f"and accessible."
        )

    if error[0] is not None:
        raise error[0]

    if result[0] is None:
        raise ConnectionError("Failed to create Memgraph connection")

    return result[0]
```

#### 4.1.3 Update Call Sites

All call sites that create standalone connections (not the shared `self.conn`) must
switch from `_create_connection()` to `_create_connection_with_timeout()`, and use
defensive `try/except` around `conn.close()` consistently — when a connection was
established via timeout-enforced creation, `close()` can also fail if the connection
is in a degraded state.

**Note on `__enter__()` and `_ensure_connection()`:** These methods also call
`_create_connection()` and could block indefinitely. However, they create the **shared
connection** (`self.conn`) used for all subsequent operations, not a per-task connection.
Applying `_create_connection_with_timeout()` there would wrap the initial connection in
a threading timeout, which is appropriate — if Memgraph is unavailable at startup, the
process should fail fast rather than hang silently. For `_ensure_connection()`, the
same applies — a mid-operation reconnect that blocks forever would stall all subsequent
queries. Both should be updated to use `_create_connection_with_timeout()`.

**Note on `_reset_shared_connection()`:** This method also calls `_create_connection()`,
but it's used within `_execute_query()` retry logic which already has its own retry loop.
Using `_create_connection_with_timeout()` there adds an additional safety layer — if
a reconnect attempt blocks, the retry loop would never get a chance to retry, so the
timeout wrapper prevents the retry mechanism from becoming a hang mechanism.

1. In `_flush_rel_group_with_own_conn()`:
```python
def _flush_rel_group_with_own_conn(
    self,
    pattern: tuple[str, str, str, str, str],
    params_list: list[RelBatchRow],
) -> tuple[int, int]:
    conn = self._create_connection_with_timeout()  # <-- CHANGED
    try:
        return self._flush_rel_pattern_group(pattern, params_list, conn=conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass  # Best-effort close — connection may be degraded after timeout
```

2. In `_flush_node_group_with_own_conn()`:
```python
def _flush_node_group_with_own_conn(
    self,
    label: str,
    props_list: list[dict[str, PropertyValue]],
) -> tuple[int, int]:
    conn = self._create_connection_with_timeout()  # <-- CHANGED
    try:
        return self._flush_node_label_group(label, props_list, conn=conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass  # Best-effort close — consistent with _flush_rel_group_with_own_conn
```

3. In `__enter__()` — change initial connection creation:
```python
def __enter__(self) -> MemgraphIngestor:
    logger.info(ls.MG_CONNECTING.format(host=self._host, port=self._port))
    self.conn = self._create_connection_with_timeout()  # <-- CHANGED
    self._executor = ThreadPoolExecutor(max_workers=settings.FLUSH_THREAD_POOL_SIZE)
    # ... rest unchanged ...
```

4. In `_ensure_connection()` — change reconnect creation:
```python
def _ensure_connection(self) -> None:
    if not self.conn or not self._check_connection_health():
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.conn = self._create_connection_with_timeout()  # <-- CHANGED
```

5. In `_reset_shared_connection()` — change reconnect creation:
```python
def _reset_shared_connection(self) -> None:
    current_conn = self.conn
    self.conn = None
    if current_conn is not None:
        try:
            current_conn.close()
        except Exception:
            pass
    self.conn = self._create_connection_with_timeout()  # <-- CHANGED
```

### 4.2 Fix 2: Add Socket-Level Timeout After Connection

**Priority:** P0 - Must implement  
**Risk:** Low - Adds socket timeout (uses `logger.warning` for failures, deliberately higher than the `logger.debug` used for keepalive failures, since socket timeout directly prevents the hanging bug)  
**Files:** `codebase_rag/services/graph_service.py`

> **Note:** This fix is already incorporated into the `_create_connection()` rewrite in §4.1.1.
> The `sock.settimeout(timeout)` call and its surrounding try-except block appear there
> directly after `conn.autocommit = True`, replacing the previously unused `timeout` variable
> with a concrete, applied socket timeout. No additional code changes are needed beyond §4.1.1.

**What this fix changes in the original `_create_connection()`:**

| Before | After |
|--------|-------|
| `timeout` computed but never used | `timeout` from `_get_connection_timeout()` applied via `sock.settimeout(timeout)` |
| No socket timeout — I/O can block indefinitely | Socket timeout set — I/O operations fail with `socket.timeout` after `timeout` seconds |
| TCP keepalive failures logged at `logger.debug` | Socket timeout failures logged at `logger.warning` (higher severity — directly prevents the bug) |
| Misleading comment: "timeout is configured server-side via Docker" | Accurate comment: "mgclient.connect() does not accept a timeout parameter; timeout is enforced via socket.settimeout()" |

**Log-level rationale:** The existing TCP keepalive code uses `logger.debug` for failures because keepalive is a passive optimization — the connection still works without it. Socket timeout is an **active guard** against the hanging bug, so failures to set it should be visible at `WARNING` level to alert operators that the primary fix may not be in effect.

### 4.3 Fix 3: Add Query Execution Timeout (Both Variants)

**Priority:** P1 - Should implement  
**Risk:** Low - Adds query timeout  
**Files:** `codebase_rag/services/graph_service.py`

**Why both variants are needed:** The original spec only covered `_execute_batch_on()` (used by node flushing in `_flush_node_label_group()`), but the **primary hang scenario** involves `_execute_batch_with_return_on()` (used by relationship flushing in `_flush_rel_pattern_group()`). Leaving the return-value variant unprotected means the actual hang path remains unguarded. Both must be covered.

#### 4.3.1 Helper: Socket Timeout Context Manager

To avoid duplicating the socket timeout save/set/restore logic across two wrapper methods, extract it into a reusable context manager:

```python
@contextmanager
def _socket_timeout_context(
    self,
    conn: mgclient.Connection,
    timeout: int,
) -> Generator[None, None, None]:
    """Temporarily set socket timeout on a connection, restoring it on exit.
    
    Uses the same socket access pattern as existing keepalive/timeout code
    in _create_connection() for consistency.
    """
    original_timeout = None
    try:
        if hasattr(conn, "socket") or hasattr(conn, "_socket"):
            sock = getattr(conn, "socket", getattr(conn, "_socket", None))
            if sock:
                original_timeout = sock.gettimeout()
                sock.settimeout(timeout)
    except (OSError, AttributeError):
        pass
    try:
        yield
    finally:
        if original_timeout is not None:
            try:
                if hasattr(conn, "socket") or hasattr(conn, "_socket"):
                    sock = getattr(conn, "socket", getattr(conn, "_socket", None))
                    if sock:
                        sock.settimeout(original_timeout)
            except (OSError, AttributeError):
                pass
```

#### 4.3.2 Timeout Wrapper for Node Flushing (no return value)

```python
def _execute_batch_with_timeout(
    self,
    conn: mgclient.Connection,
    query: str,
    params_list: Sequence[BatchParams],
    timeout: int | None = None,
) -> None:
    """Execute batch query with timeout enforcement (no return value)."""
    if timeout is None:
        timeout = settings.MEMGRAPH_QUERY_TIMEOUT
    with self._socket_timeout_context(conn, timeout):
        self._execute_batch_on(conn, query, params_list)
```

#### 4.3.3 Timeout Wrapper for Relationship Flushing (with return value)

```python
def _execute_batch_with_return_with_timeout(
    self,
    conn: mgclient.Connection,
    query: str,
    params_list: Sequence[BatchParams],
    timeout: int | None = None,
) -> list[ResultRow]:
    """Execute batch query with timeout enforcement (returns results).
    
    This is the variant used by relationship flushing via
    _flush_rel_pattern_group(), which is the primary hang scenario.
    """
    if timeout is None:
        timeout = settings.MEMGRAPH_QUERY_TIMEOUT
    with self._socket_timeout_context(conn, timeout):
        return self._execute_batch_with_return_on(conn, query, params_list)
```

#### 4.3.4 Call-Site Integration

**In `_flush_node_label_group()` — replace `_execute_batch_on` call:**

Current code:
```python
self._execute_batch_on(target_conn, query, batch_rows)
```

Replace with:
```python
self._execute_batch_with_timeout(target_conn, query, batch_rows)
```

**In `_flush_rel_pattern_group()` — replace `_execute_batch_with_return_on` call:**

Current code:
```python
results = self._execute_batch_with_return_on(target_conn, query, params_list)
```

Replace with:
```python
results = self._execute_batch_with_return_with_timeout(target_conn, query, params_list)
```

### 4.4 Fix 4: Graceful Executor Shutdown

**Priority:** P1 - Should implement  
**Risk:** Medium - Changes shutdown semantics (trade-off documented below)  
**Files:** `codebase_rag/services/graph_service.py`

**Implementation:**

Update `__exit__()` to handle stuck workers gracefully:

```python
def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: types.TracebackType | None,
) -> None:
    try:
        if exc_type:
            logger.exception(ls.MG_EXCEPTION.format(error=exc_val))
            try:
                self.flush_all()
            except Exception as flush_err:
                logger.error(ls.MG_FLUSH_ERROR.format(error=flush_err))
        else:
            self.flush_all()
    finally:
        if self._executor:
            # Shutdown without waiting to allow forced exit.
            # cancel_futures=True (Python ≥3.9) prevents waiting for stuck workers.
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            logger.info(ls.MG_DISCONNECTED)
```

**Trade-off — `shutdown(wait=False, cancel_futures=True)` vs `shutdown(wait=True)`:**

On **normal exit**, `flush_all()` completes all futures before reaching the `finally` block,
so `shutdown(wait=False, cancel_futures=True)` behaves identically to `shutdown(wait=True)` —
there are no pending futures to cancel.

On **exceptional exit** (e.g., `TimeoutError` from Fix 1, or `KeyboardInterrupt`), some
futures may still be running. The original `shutdown(wait=True)` blocks indefinitely if
any worker thread is stuck on `mgclient.connect()`. The new `shutdown(wait=False, cancel_futures=True)`
immediately cancels pending futures and returns, allowing the process to exit cleanly.

**Accepted trade-off:** Cancelling futures on exceptional exit means some in-flight
relationship writes may be abandoned (partial writes). This is preferable to an
indefinitely hanging process, since:
- The `flush_all()` best-effort attempt in the `try` block already committed as much
  data as possible before the exception.
- Partial writes in Memgraph are atomically committed per-UNWIND-batch (each batch
  is an atomic transaction due to `autocommit=True`), so there is no half-written
  relationship — either a full batch succeeded or it didn't. The data is either
  fully committed or fully absent.
- A hanging process leaves the user with zero recourse other than `kill -9`.

### 4.5 Fix 5: Unconditional Progress Logging for Relationship Flush

**Priority:** P2 - Nice to have  
**Risk:** Very Low - Changes logging behavior only  
**Files:** `codebase_rag/services/graph_service.py`

**Current behavior:** The `MG_PARALLEL_FLUSH_RELS` log already exists in `flush_relationships()`, but it is **conditional** — it only appears when `self._executor and len(self._rel_groups) > 1` (parallel mode with multiple groups). For small flushes with ≤1 relationship group, or when no executor is available, this log is suppressed. Additionally, the early-return case (`self._rel_count == 0`) produces no log at all, making it difficult to distinguish between "no relationships" and "hanging before the log was emitted."

**Change:** Make the `MG_PARALLEL_FLUSH_RELS` log unconditional (always logged regardless of executor/group count), and add a `logger.debug()` message for the early-return case:

```python
def flush_relationships(self) -> None:
    if not self._rel_count:
        logger.debug("No relationships to flush, skipping")
        return
    
    # Always log relationship flush start (previously conditional on executor + >1 groups)
    logger.info(
        ls.MG_PARALLEL_FLUSH_RELS.format(
            count=len(self._rel_groups),
            workers=settings.FLUSH_THREAD_POOL_SIZE,
        )
    )
    
    # ... rest of the method unchanged ...
```

**Why this helps debugging:** The symptom report (§2.1) notes that `MG_PARALLEL_FLUSH_RELS` was missing from logs. With the current conditional logging, its absence could mean either (a) the method is hanging before the log line, or (b) the condition wasn't met. Making it unconditional eliminates ambiguity — if the log doesn't appear, the method either wasn't called or hung before reaching it.

---

## 5. Implementation Priority

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| P0 | Fix 1: Connection timeout wrapper (helper + wrapper + call sites) | 50 min | Prevents indefinite hangs on connection |
| P0 | Fix 2: Socket-level timeout (incorporated into §4.1.1) | 0 min* | Prevents I/O operation hangs — already included in Fix 1 |
| P1 | Fix 3: Query execution timeout (context manager + both variants + call sites) | 45 min | Prevents query hangs on both node and relationship flush |
| P1 | Fix 4: Graceful executor shutdown | 20 min | Allows clean process exit on exceptional termination |
| P2 | Fix 5: Unconditional progress logging | 10 min | Eliminates ambiguity in hang diagnosis |

**Total estimated effort:** ~2 hours  
\* Fix 2 requires no separate implementation time since it was incorporated into the `_create_connection()` rewrite in §4.1.1.

---

## 6. Validation Criteria

### 6.1 Test Scenarios

1. **Normal Ingestion:**
   - Run `cgr start --repo-path <path> --index-all`
   - Verify completion with all logs including `MG_FLUSH_COMPLETE` and `MG_DISCONNECTED`

2. **Memgraph Unavailable:**
   - Stop Memgraph service
   - Run `cgr start --repo-path <path> --index-all`
   - Verify process exits with `TimeoutError` within `MEMGRAPH_CONNECTION_TIMEOUT` seconds (default: 600s)
   - Verify error message: "Connection to Memgraph at localhost:7687 timed out after 600s..."

3. **Interrupted Ingestion:**
   - Start ingestion, then Ctrl+C during flush
   - Verify process exits cleanly within 5 seconds
   - Verify no zombie threads or connection leaks

4. **Large Relationship Flush:**
   - Index a codebase with many function calls
   - Verify relationship flush completes without hanging
   - Monitor Memgraph resource usage during flush

### 6.2 Success Metrics

1. Ingestion completes within expected time (no hanging)
2. Process exits with clear timeout error if Memgraph is unresponsive
3. No hanging observed in 10 consecutive test runs
4. Clean process exit on interrupt within 5 seconds

---

## 7. Configuration Recommendations

Current relevant settings in `codebase_rag/config.py`:

```python
MEMGRAPH_CONNECTION_TIMEOUT: int = Field(default=600, gt=0)
DOC_MEMGRAPH_CONNECTION_TIMEOUT: int = Field(default=600, gt=0)
JSON_MEMGRAPH_CONNECTION_TIMEOUT: int = Field(default=600, gt=0)
FLUSH_THREAD_POOL_SIZE: int = Field(default=4, gt=0)
```

**Recommended additions:**

```python
# Query execution timeout (separate from connection timeout)
MEMGRAPH_QUERY_TIMEOUT: int = Field(default=120, gt=0)
```

> **Note on `MEMGRAPH_EXECUTOR_SHUTDOWN_TIMEOUT`:** The original spec proposed this setting,
> but Fix 4's implementation uses `shutdown(wait=False, cancel_futures=True)` which does not
> reference any configurable timeout — it immediately returns without waiting. If a future
> revision adopts a timed-wait approach (try `shutdown(wait=True)` for N seconds, then force),
> this setting would become relevant and should be added at that time. For now, it is omitted
> to avoid an unused configuration field.

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Socket timeout breaks existing connections | Low | Medium | Wrap in try-except, fallback gracefully |
| False positives (valid operations timeout) | Medium | Low | Conservative defaults (600s connection / 120s query), allow override via env vars |
| Connection leak on timeout (Fix 1 race condition) | Low | Medium | Best-effort cleanup: close `result[0]` if set after `thread.join()` timeout; daemon thread ensures eventual GC; inherent race is documented and accepted |
| Partial data loss on exceptional exit (Fix 4) | Medium | Low | Memgraph UNWIND batches are atomic transactions — no half-written data; best-effort `flush_all()` in `try` block maximizes committed data before cancellation |
| Thread wrapper adds overhead | Low | Low | Only used for connection creation (not per-query); `threading.Thread` is lightweight |
| `_get_connection_timeout()` not in `__slots__` | None | None | Method, not instance attribute — `__slots__` only restricts instance attributes, not methods |

---

## 9. Backward Compatibility

All fixes are **backward compatible**:
- No API changes to public interfaces
- No configuration file format changes
- New timeouts use existing defaults
- Socket timeout setting is optional (graceful fallback)

---

## 10. Related Code References

### Files to Modify:
- `codebase_rag/services/graph_service.py` - Main fixes
- `codebase_rag/config.py` - Optional new settings

### Key Methods:
- `_get_connection_timeout()` - New helper, single source of truth for timeout resolution (§4.1.1)
- `_create_connection()` - Rewrite: use helper, add socket timeout, fix misleading comment (§4.1.1 + §4.2)
- `_create_connection_with_timeout()` - New method, threading-based connection timeout wrapper (§4.1.2)
- `_socket_timeout_context()` - New context manager for per-operation socket timeout (§4.3.1)
- `_execute_batch_with_timeout()` - New method, timeout wrapper for node batch queries (§4.3.2)
- `_execute_batch_with_return_with_timeout()` - New method, timeout wrapper for relationship batch queries (§4.3.3)
- `_flush_node_label_group()` - Call-site: switch `_execute_batch_on` → `_execute_batch_with_timeout` (§4.3.4)
- `_flush_rel_pattern_group()` - Call-site: switch `_execute_batch_with_return_on` → `_execute_batch_with_return_with_timeout` (§4.3.4)
- `_flush_rel_group_with_own_conn()` - Call-site: switch `_create_connection` → `_create_connection_with_timeout`; defensive `conn.close()` (§4.1.3)
- `_flush_node_group_with_own_conn()` - Call-site: switch `_create_connection` → `_create_connection_with_timeout`; defensive `conn.close()` (§4.1.3)
- `__enter__()` - Call-site: switch `_create_connection` → `_create_connection_with_timeout` for initial connection (§4.1.3)
- `_ensure_connection()` - Call-site: switch `_create_connection` → `_create_connection_with_timeout` for reconnect (§4.1.3)
- `_reset_shared_connection()` - Call-site: switch `_create_connection` → `_create_connection_with_timeout` for retry reconnect (§4.1.3)
- `__exit__()` - Change `shutdown(wait=True)` → `shutdown(wait=False, cancel_futures=True)` (§4.4)
- `flush_relationships()` - Make `MG_PARALLEL_FLUSH_RELS` log unconditional; add early-return debug log (§4.5)

### Settings Used:
- `MEMGRAPH_CONNECTION_TIMEOUT` (default: 600s) — existing, used by `_get_connection_timeout()`
- `DOC_MEMGRAPH_CONNECTION_TIMEOUT` (default: 600s) — existing, used by `_get_connection_timeout()`
- `JSON_MEMGRAPH_CONNECTION_TIMEOUT` (default: 600s) — existing, used by `_get_connection_timeout()`
- `FLUSH_THREAD_POOL_SIZE` (default: 4) — existing, referenced in unconditional log
- `MEMGRAPH_QUERY_TIMEOUT` (default: 120s) — **new**, used by `_execute_batch_with_timeout()` and `_execute_batch_with_return_with_timeout()`

### New `__slots__` Impact:
No changes required. All additions are methods (not instance attributes), and `__slots__` only restricts instance attributes. The `_connection_timeout` slot already exists for the existing constructor parameter.

---

## 11. External Dependencies

- **mgclient** (Memgraph Python client): Does not support timeout parameter in `connect()`
- **Python threading**: Used for timeout wrapper
- **Python socket**: Used for socket-level timeout

---

## 12. Future Improvements

1. **Connection Pooling:** Replace per-flush connection creation with a connection pool
2. **Circuit Breaker:** Implement circuit breaker pattern for Memgraph failures
3. **Health Check Integration:** Add periodic health checks during long operations
4. **Adaptive Timeout:** Dynamically adjust timeouts based on operation size
5. **Query Optimization:** Optimize `MATCH...MERGE` queries for large batches
