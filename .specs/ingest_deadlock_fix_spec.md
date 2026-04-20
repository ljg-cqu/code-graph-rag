# Ingest Deadlock Fix Design Specification

## Overview

This document outlines the design for fixing critical deadlock issues identified during document graph ingestion. The system experiences complete process freeze where all threads become unresponsive, requiring manual intervention.

## Issues Identified

### Issue 1: Process Freeze During Document Ingestion

**Symptoms:**
- Process becomes completely unresponsive (all threads in `S` sleeping state)
- Last log activity: "Flushed 684 of 684 buffered nodes" at 13:45:29
- No subsequent log output for 30+ minutes
- Process must be killed with SIGKILL
- All 14 threads waiting on futexes (`futex_wait_queue`)

**Observed State:**
```
PID 1820800: cgr - State: S (sleeping) - futex_wait_queue
Threads: 14 (all in futex_wait_queue)
voluntary_ctxt_switches: 4799
nonvoluntary_ctxt_switches: 1943
```

**Deadlock Location:**
- Document 13/42 (LANGUAGE_RECOMMENDATIONS.md) being processed
- After successful `flush_nodes()` with 684 nodes flushed
- Before next log entry (likely during `flush_relationships()` or document completion)

**Root Cause Analysis:**

The deadlock occurs in the parallel flush mechanism. The sequence of events leading to deadlock:

1. **ThreadPoolExecutor.submit()** creates futures for parallel node/relationship flushing
2. **as_completed()** waits for all futures to complete
3. Worker threads attempt to create database connections via `_create_connection_with_timeout()`
4. Connection creation uses `threading.Thread` internally (for timeout enforcement)
5. Nested thread creation + futex synchronization creates circular wait condition

**Code Path Analysis:**

```python
# graph_service.py
futures = {
    self._executor.submit(
        self._flush_node_group_with_own_conn, label, props_list
    ): label
    for label, props_list in nodes_by_label.items()
}
for future in as_completed(futures):  # <-- Main thread blocks here
    label = futures[future]
    try:
        flushed, skipped = future.result()  # <-- Waits for worker
```

Worker thread execution:
```python
def _flush_node_group_with_own_conn(self, label, props_list):
    conn = self._create_connection_with_timeout()  # <-- Creates thread internally
    try:
        return self._flush_node_label_group(label, props_list, conn=conn)
    finally:
        conn.close()
```

Connection creation with timeout (before fix):
```python
def _create_connection_with_timeout(self):
    timeout = self._get_connection_timeout()
    result: list[mgclient.Connection | None] = [None]
    error: list[Exception | None] = [None]

    def connect_worker():
        try:
            result[0] = self._create_connection_internal()
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=connect_worker)
    thread.start()
    thread.join(timeout=timeout)  # <-- Worker thread waits here
    # ...
```

**Deadlock Scenario:**
1. Main thread holds GIL while iterating `as_completed()`
2. Worker threads spawn child threads for connection timeout enforcement
3. Child threads compete for GIL with parent workers
4. If Memgraph is slow/unresponsive, all threads block on I/O or locks
5. Thread.join(timeout) in connection creation doesn't release resources properly
6. Futex synchronization between parent/child threads creates circular wait

### Issue 2: Lack of Flush Operation Timeout

**Symptoms:**
- No timeout on flush operations
- No heartbeat/keepalive mechanism
- No automatic recovery from stuck states
- Logs provide no visibility into internal thread states

**Current Behavior:**
- `flush_nodes()` can block indefinitely on `as_completed()`
- `flush_relationships()` can block indefinitely on `as_completed()`
- No mechanism to detect or break deadlocks

### Issue 3: Resource Leak on Exception

**Symptoms:**
- ThreadPoolExecutor not shut down on exception
- Database connections left open
- Memory growth over multiple ingestion runs

**Code Path:**
```python
for future in as_completed(futures):
    label = futures[future]
    try:
        flushed, skipped = future.result()
    except Exception as e:
        if first_error is None:
            first_error = e
# Executor only shut down in __exit__()
```

### Issue 4: Document Updater Async Bug

**Symptoms:**
- `DocumentGraphUpdater.run_async()` creates two `MemgraphIngestor` instances
- The outer ingestor is never used for document processing
- Duplicate constraint/index setup calls

**Code Path:**
```python
# document_updater.py:run_async()
async with MemgraphIngestor(...) as ingestor:      # outer ingestor
    ...
    try:
        async with MemgraphIngestor(...) as ingestor:  # inner ingestor (shadows outer)
            ...
```

## Proposed Solutions

### Solution 1: Eliminate Nested Thread Creation (Implemented)

**Change:** Modify `_create_connection_with_timeout()` to avoid spawning child threads when called from ThreadPoolExecutor worker threads.

**Rationale:**
- The deadlock is caused specifically by worker threads spawning child threads
- The main thread (and process workers) can safely use the threaded timeout
- A socket pre-check provides bounded timeout without threads
- Minimal code change with maximum safety impact

**Implementation Details:**

```python
def _create_connection_with_timeout(self) -> mgclient.Connection:
    """Create connection with timeout enforcement.

    Worker threads inside ThreadPoolExecutor must not spawn child threads,
    as nested thread creation causes futex deadlocks on Linux. When called
    from the main thread (or a process worker), the original threaded
    timeout is preserved. When called from a worker thread, a socket
    pre-check is used instead.
    """
    timeout = self._get_connection_timeout()

    if threading.current_thread() is not threading.main_thread():
        # Worker thread: use socket pre-check to avoid nested threads
        try:
            with socket.create_connection(
                (self._host, self._port),
                timeout=min(timeout, 30.0),
            ):
                pass
        except socket.timeout as e:
            raise TimeoutError(
                f"Connection to Memgraph at {self._host}:{self._port} "
                f"timed out after {timeout}s. Check if Memgraph is running "
                f"and accessible."
            ) from e
        except OSError as e:
            raise ConnectionError(
                f"Cannot connect to Memgraph at {self._host}:{self._port}: {e}"
            ) from e
        return self._create_connection()

    # Main thread: safe to use threaded timeout
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
    # ... (existing cleanup and return logic)
```

**File Changes:**
- `codebase_rag/services/graph_service.py`:
  - Modify `_create_connection_with_timeout()` to branch on main thread vs worker thread

**Trade-offs:**
- ✅ Eliminates deadlock root cause with minimal changes
- ✅ Preserves threaded timeout for main thread (existing behavior)
- ✅ Double TCP handshake in worker threads (acceptable cost)
- ❌ Slightly higher connection latency in worker threads (~1 extra TCP handshake)

### Solution 2: Flush Operation Timeout and Progress Monitoring (Implemented)

**Change:** Replace `as_completed()` with `wait(..., timeout=...)` in `flush_nodes()` and `flush_relationships()`.

**Rationale:**
- Provides bounded wait time for flush operations
- Progress warnings every N seconds improve observability
- Cancellation of pending futures on timeout
- Aligns with existing codebase patterns

**Implementation Details:**

```python
def flush_nodes(self) -> None:
    # ... existing setup ...
    if self._executor and len(nodes_by_label) > 1:
        futures = {
            self._executor.submit(
                self._flush_node_group_with_own_conn, label, props_list
            ): label
            for label, props_list in nodes_by_label.items()
        }
        pending = set(futures.keys())
        start_time = time.monotonic()

        while pending:
            elapsed = time.monotonic() - start_time
            remaining = settings.FLUSH_OPERATION_TIMEOUT - elapsed
            if remaining <= 0:
                for future in pending:
                    future.cancel()
                raise FlushTimeoutError(
                    f"Node flush timed out after {settings.FLUSH_OPERATION_TIMEOUT}s "
                    f"({len(pending)} of {len(futures)} groups still pending)"
                )

            done, pending = wait(
                pending,
                timeout=min(remaining, settings.FLUSH_PROGRESS_LOG_INTERVAL),
                return_when=FIRST_COMPLETED,
            )

            if not done:
                logger.warning(
                    f"No flush progress for {settings.FLUSH_PROGRESS_LOG_INTERVAL}s, "
                    f"{len(pending)} pending"
                )
                continue

            for future in done:
                label = futures[future]
                try:
                    flushed, skipped = future.result()
                    flushed_total += flushed
                    skipped_total += skipped
                except Exception as e:
                    logger.error(f"Flush failed for {label}: {e}")
                    if first_error is None:
                        first_error = e
    # ... existing sequential fallback and cleanup ...
```

**File Changes:**
- `codebase_rag/services/graph_service.py`:
  - Modify `flush_nodes()` with timeout and deadlock detection
  - Modify `flush_relationships()` with timeout and deadlock detection
  - Import `wait` and `FIRST_COMPLETED` from `concurrent.futures`

**Trade-offs:**
- ✅ Bounded flush operations (no indefinite blocking)
- ✅ Regular progress warnings
- ✅ Minimal changes to existing architecture
- ❌ `future.cancel()` only affects pending tasks, not running ones

### Solution 3: Asyncio-based Concurrent Flushing (Future)

**Change:** Replace ThreadPoolExecutor with asyncio-based concurrent execution

**Rationale:**
- Asyncio provides better cancellation semantics
- No nested thread creation (connection timeout via asyncio.wait_for)
- Proper resource cleanup on timeout/cancellation
- Native support for timeout on gather operations

**Status:** Not implemented in Phase 1. The codebase already has async bridges (`run_async()`, MCP tools), but a full migration would require maintaining dual sync/async APIs or major refactoring.

**File Changes (if implemented):**
- `codebase_rag/services/graph_service.py`:
  - Add `flush_nodes_async()` method
  - Add `flush_relationships_async()` method
  - Add `flush_all_async()` method

### Solution 4: Circuit Breaker Pattern for Flush Operations (Future)

**Change:** Implement circuit breaker to fail fast when system is unhealthy

**Rationale:**
- Prevents cascade failures
- Fast fail when system unhealthy

**Status:** Not implemented in Phase 1. The codebase already has a simple failure counter circuit breaker in `retrieval/query_orchestrator.py`. A more robust implementation could be added later.

### Solution 5: Observability and Debugging Improvements (Partially Implemented)

**Change:** Add structured flush operation metrics and progress logging

**Implementation Details:**

Flush progress logging is now integrated into the `wait()` loop in `flush_nodes()` and `flush_relationships()`.

**File Changes:**
- `codebase_rag/services/graph_service.py`:
  - Progress warnings during long-running flushes
- `codebase_rag/logs.py`:
  - `MG_FLUSH_TIMEOUT`: timeout message template
  - `MG_FLUSH_NO_PROGRESS`: progress warning template

## Implementation Plan

### Phase 1: Immediate Fix (Critical) — COMPLETED

**Goal:** Eliminate nested thread creation and add flush operation timeouts

1. **Fix `_create_connection_with_timeout()`**
   - File: `codebase_rag/services/graph_service.py`
   - Change: Branch on `threading.current_thread() is threading.main_thread()`
   - Worker threads: use `socket.create_connection()` pre-check
   - Main thread: preserve existing threaded timeout

2. **Add Flush Operation Timeout**
   - File: `codebase_rag/services/graph_service.py`
   - Lines: `flush_nodes()`, `flush_relationships()`
   - Changes:
     - Replace `as_completed()` with `wait(..., timeout=...)`
     - Add overall operation timeout via `settings.FLUSH_OPERATION_TIMEOUT`
     - Add progress logging every `settings.FLUSH_PROGRESS_LOG_INTERVAL` seconds
     - Cancel pending futures on timeout
     - Raise `FlushTimeoutError`

3. **Add Configuration Options**
   - File: `codebase_rag/config.py`
   - Add:
     ```python
     FLUSH_OPERATION_TIMEOUT: int = Field(default=60, gt=0)
     FLUSH_PROGRESS_LOG_INTERVAL: float = Field(default=5.0, gt=0)
     ```

4. **Add Exception Classes**
   - File: `codebase_rag/exceptions.py`
   - Add:
     ```python
     class CodebaseRAGError(Exception):
         pass

     class FlushTimeoutError(CodebaseRAGError):
         def __init__(self, message: str) -> None:
             super().__init__(message)
     ```

5. **Fix Document Updater Async Bug**
   - File: `codebase_rag/document/document_updater.py`
   - Remove duplicate nested `async with MemgraphIngestor(...)` in `run_async()`

6. **Add Log Templates**
   - File: `codebase_rag/logs.py`
   - Add `MG_FLUSH_TIMEOUT` and `MG_FLUSH_NO_PROGRESS`

**Expected Outcome:**
- Worker threads no longer spawn child threads → deadlock eliminated
- Flush operations timeout after 60s instead of blocking forever
- Clear error messages indicate timeout vs actual failure
- Process can be restarted without manual intervention

### Phase 2: Robustness Improvements (Future)

**Goal:** Add circuit breaker and better error recovery

1. **Implement Circuit Breaker**
   - New file or extend `retrieval/query_orchestrator.py`
   - Integrate into `graph_service.py`

2. **Improve Connection Handling**
   - Add connection validation before use
   - Implement connection retry with exponential backoff
   - Add connection health checks

3. **Add Graceful Degradation**
   - If parallel flush fails, fall back to sequential
   - If connection pool exhausted, queue and retry
   - Document partial failures without stopping entire ingestion

### Phase 3: Asyncio Migration (Optional)

**Goal:** Long-term architecture improvement

1. **Implement async flush methods**
   - Create async versions of all flush methods
   - Migrate document updater to use async
   - Migrate graph updater to use async

2. **Benefits:**
   - Better resource utilization
   - Easier cancellation
   - More predictable concurrency

## Testing Strategy

### Unit Tests

```python
def test_flush_nodes_timeout():
    """Verify flush_nodes raises timeout after configured duration."""
    ingestor = MemgraphIngestor(...)
    ingestor.conn = MagicMock()

    # Add many nodes to trigger parallel flush
    for i in range(1000):
        ingestor.ensure_node_batch("TestNode", {"id": i})

    # Mock slow worker
    with mock.patch.object(
        ingestor, '_flush_node_group_with_own_conn', side_effect=lambda *a, **k: time.sleep(100)
    ):
        with pytest.raises(FlushTimeoutError):
            ingestor.flush_nodes()

def test_worker_thread_uses_socket_precheck():
    """Verify worker threads use socket pre-check, not nested threads."""
    ingestor = MemgraphIngestor(host="localhost", port=7687)

    def check_in_worker():
        with mock.patch("socket.create_connection") as mock_connect:
            mock_connect.return_value.__enter__ = mock.Mock(return_value=(None, None))
            mock_connect.return_value.__exit__ = mock.Mock(return_value=False)
            with mock.patch.object(ingestor, "_create_connection") as mock_create:
                ingestor._create_connection_with_timeout()
                mock_connect.assert_called_once()
                mock_create.assert_called_once()

    # Run in a worker thread (not main thread)
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(check_in_worker).result()

def test_main_thread_uses_threaded_timeout():
    """Verify main thread preserves threaded timeout behavior."""
    ingestor = MemgraphIngestor(host="localhost", port=7687)

    with mock.patch("threading.Thread") as mock_thread:
        mock_thread.return_value.start = mock.Mock()
        mock_thread.return_value.join = mock.Mock()
        mock_thread.return_value.is_alive.return_value = True
        with pytest.raises(TimeoutError):
            ingestor._create_connection_with_timeout()
        mock_thread.assert_called_once()
```

### Integration Tests

```python
def test_document_ingestion_no_deadlock():
    """Verify document ingestion completes without deadlock."""
    updater = DocumentGraphUpdater(...)
    docs = [create_test_document() for _ in range(10)]
    updater.run()
    # Should complete without hanging
```

## Migration Notes

No database migration required. These are code-only fixes.

**Configuration Changes:**
Add to `.env` or environment:
```bash
# New configuration options
FLUSH_OPERATION_TIMEOUT=60
FLUSH_PROGRESS_LOG_INTERVAL=5.0
```

## Success Criteria

1. **No Process Freezes:** Ingestion completes or fails with clear error within 2x expected time
2. **Timeout Works:** Flush operations timeout after `FLUSH_OPERATION_TIMEOUT` seconds
3. **Clear Errors:** Timeout errors include operation name and duration
4. **Recovery Possible:** Process can be restarted without manual intervention (no zombie locks)
5. **Observability:** Logs show flush progress at regular intervals
6. **No Nested Threads:** Worker threads never spawn child threads for connection creation

## Performance Impact

| Metric | Before | After (Phase 1) |
|--------|--------|-----------------|
| Flush timeout | None | 60s |
| Deadlock detection | None | Progress warnings every 5s |
| Connection creation (main thread) | Per worker with threaded timeout | Preserved |
| Connection creation (worker thread) | Per worker with threaded timeout | Socket pre-check + connect |
| Recovery time | Manual | Automatic |

## Appendix: Log Analysis Summary

**Deadlock Signature:**
```
2026-04-20 13:45:29.457 | INFO | flush_nodes:1096 - Flushed 684 of 684 buffered nodes.
[NO FURTHER OUTPUT]
```

**Thread States at Deadlock:**
```
Main thread: futex_wait_queue (waiting on as_completed)
Workers 1-4: futex_wait_queue (waiting on thread.join in connection creation)
```

**Root Cause Confirmation:**
- Document 13/42 processing
- 684 nodes flushed successfully
- All threads in futex wait state
- No database errors logged
- Connection timeout = 300s (document graph default)
- Process would have unblocked after 5 minutes (connection timeout) but futex state suggests nested threading issue

This confirms the deadlock is in thread synchronization, not database I/O.
