# Resource Management and Graceful Shutdown Design Specification

## Executive Summary

This document outlines critical resource management vulnerabilities and graceful shutdown deficiencies in the current codebase implementation. The analysis reveals multiple potential resource leaks, inadequate thread lifecycle management, insufficient signal handling, and race conditions in concurrent operations. This specification provides implementation-ready solutions to ensure 100% health lifetime guarantees with proper resource cleanup and graceful termination.

## Critical Issues Identified

### 1. Thread/Process Resource Leaks

**Issue**: The `SubAgentOrchestrator` creates `ThreadPoolExecutor` instances but lacks proper shutdown mechanisms in error scenarios.

- **Location**: `./codebase_rag/orchestrator/subagent_orchestrator.py`
- **Risk**: Thread pool executors may not be properly shutdown during exceptions or forced termination
- **Impact**: Accumulating zombie threads consuming system resources

**Current Code Problem**:
```python
def execute_tasks(self, subtasks, ...):
    try:
        with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
            # ... execution logic
    finally:
        self.running = False
        # Missing explicit executor shutdown in error paths
```

### 2. Memgraph Connection Management Deficiencies

**Issue**: Multiple connection creation patterns without centralized connection pooling and cleanup.

- **Location**: `./codebase_rag/services/graph_service.py`, `./codebase_rag/orchestrator/subagent_orchestrator.py`
- **Risk**: Connection exhaustion under high load, orphaned connections during failures
- **Impact**: Database connection pool saturation, memory leaks

**Current Strengths** (already implemented):
- `MemgraphIngestor._socket_timeout_context()` - configurable socket timeouts
- `_execute_batch_with_timeout()` - batch query timeouts with retry logic
- `_check_connection_health()` - proactive connection health monitoring
- `_ensure_connection()` - automatic reconnection on unhealthy connections
- TCP keepalive configuration for long-running connections

**Remaining Problems**:
- `ReadOnlySubAgent` creates individual `MemgraphIngestor` instances per worker (line 103-120 in `subagent_orchestrator.py`)
- Each sub-agent worker maintains its own connection instead of sharing a pool
- Connection resources are not reused between parallel workers

### 3. Fragmented Signal Handling for Graceful Shutdown

**Issue**: Signal handlers are installed in multiple locations without centralized coordination.

- **Location**: `./codebase_rag/main.py:1406-1450`, `./codebase_rag/orchestrator/subagent_orchestrator.py:313-317`
- **Risk**: Inconsistent cleanup, handler conflicts, orphaned resources
- **Impact**: Some resources may not be cleaned up during shutdown

**Current Implementation** (already exists):
- `main.py:1441-1442` - Registers SIGINT/SIGTERM handlers with double-interrupt support
- `subagent_orchestrator.py:313-317` - Registers SIGINT/SIGTERM/SIGHUP/SIGQUIT handlers
- `realtime_updater.py:326-327` - Handles KeyboardInterrupt for watcher cleanup

**Current Limitations**:
- Multiple components register their own signal handlers independently
- No centralized coordination of cleanup order
- Potential conflicts when multiple handlers respond to same signal
- Real-time file watchers rely on KeyboardInterrupt only (no SIGTERM handling)

### 4. Race Conditions in Parallel Execution

**Issue**: Shared state modification without proper synchronization in concurrent environments.

- **Location**: `./codebase_rag/orchestrator/subagent_orchestrator.py`, `./codebase_rag/orchestrator/result_aggregator.py`
- **Risk**: Data corruption, inconsistent state, deadlocks
- **Impact**: Unpredictable behavior, failed operations, corrupted graph data

**Current Strengths** (already implemented):
- `ResultAggregator` (lines 15-68) uses `threading.Lock()` for thread-safe `add_result()`, `add_error()`, and metadata operations
- `MemgraphIngestor._conn_lock` protects cursor access
- Debounce handlers in `realtime_updater.py` use `threading.Lock()` for pending events

**Remaining Problems**:
- `SubAgentOrchestrator.running` and `_shutdown_called` (lines 301-302) are plain booleans, not atomic
- No compare-and-set operation for safe state transitions
- Potential race between shutdown signal and worker state check

### 5. Missing Resource Cleanup in Exception Paths

**Issue**: Exception handling often bypasses normal resource cleanup procedures.

- **Location**: Throughout the codebase, particularly in `graph_updater.py` and `main.py`
- **Risk**: Resources remain allocated after errors occur
- **Impact**: Memory leaks, file descriptor exhaustion, database connection leaks

## Design Solutions

### 1. Comprehensive Thread/Process Lifecycle Management

#### Solution A: Enhanced ThreadPoolExecutor Management

Implement a `ManagedThreadPoolExecutor` wrapper that ensures proper cleanup:

```python
class ManagedThreadPoolExecutor:
    def __init__(self, max_workers, thread_name_prefix=""):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, 
            thread_name_prefix=thread_name_prefix
        )
        self._shutdown = False
        self._lock = threading.Lock()
    
    def submit(self, fn, *args, **kwargs):
        with self._lock:
            if self._shutdown:
                raise RuntimeError("Executor is shutdown")
            return self.executor.submit(fn, *args, **kwargs)
    
    def shutdown(self, wait=True, cancel_futures=True):
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            self.executor.shutdown(wait=wait, cancel_futures=cancel_futures)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True, cancel_futures=True)
```

#### Solution B: Sub-Agent Orchestrator Cleanup Enhancement

**Current Implementation** (already uses context manager):
The existing `execute_tasks()` method (line 433) already uses `with ThreadPoolExecutor()` which guarantees cleanup.

**Proposed Enhancement**: Replace with `ManagedThreadPoolExecutor` for explicit shutdown control:

```python
def execute_tasks(self, subtasks, result_aggregator=None, retry_attempts=None, dry_run=False):
    # ... existing initialization logic ...
    
    self.running = True
    start_time = time.time()
    
    try:
        with ManagedThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix=f"subagent-{id(self)}",
            shutdown_timeout=30.0
        ) as executor:
            # ... existing execution logic with proper exception handling ...
            pass
    except Exception as e:
        logger.error(f"Parallel execution failed: {e}")
        raise
    finally:
        self.running = False
        total_time = time.time() - start_time
        result_aggregator.set_total_execution_time(total_time)
        logger.info(f"Parallel execution completed in {total_time:.2f}s")
        
        # Ensure all workers are properly shutdown
        for worker in self.workers:
            worker.shutdown()
```

### 2. Centralized Connection Pool Management

**Note**: The existing `MemgraphIngestor` already has robust single-connection management including:
- Socket timeouts and TCP keepalive
- Connection health monitoring with `_check_connection_health()`
- Automatic reconnection via `_ensure_connection()`
- Retry logic for transient errors

The proposed enhancement adds **connection pooling** for parallel workers to share connections efficiently.

#### Solution A: Connection Pool Factory Pattern

Create a centralized connection pool manager:

```python
class MemgraphConnectionPool:
    _pools = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_pool(cls, host, port, username=None, password=None, max_connections=10):
        key = f"{host}:{port}:{username or 'default'}"
        with cls._lock:
            if key not in cls._pools:
                cls._pools[key] = cls(host, port, username, password, max_connections)
            return cls._pools[key]
    
    def __init__(self, host, port, username, password, max_connections):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.max_connections = max_connections
        self._connections = queue.Queue(maxsize=max_connections)
        self._active_connections = 0
        self._lock = threading.Lock()
    
    def get_connection(self, timeout=30):
        try:
            return self._connections.get(timeout=timeout)
        except queue.Empty:
            with self._lock:
                if self._active_connections < self.max_connections:
                    conn = self._create_connection()
                    self._active_connections += 1
                    return conn
                else:
                    raise ConnectionError("Connection pool exhausted")
    
    def return_connection(self, conn):
        try:
            self._connections.put_nowait(conn)
        except queue.Full:
            conn.close()
            with self._lock:
                self._active_connections -= 1
    
    def _create_connection(self):
        # Delegate to MemgraphIngestor's proven connection creation logic
        from codebase_rag.services.graph_service import MemgraphIngestor
        temp_ingestor = MemgraphIngestor.__new__(MemgraphIngestor)
        temp_ingestor._host = self.host
        temp_ingestor._port = self.port
        temp_ingestor._username = self.username
        temp_ingestor._password = self.password
        return temp_ingestor._create_connection_with_timeout()
    
    def close_all(self):
        with self._lock:
            while not self._connections.empty():
                try:
                    conn = self._connections.get_nowait()
                    conn.close()
                except queue.Empty:
                    break
            self._active_connections = 0
```

#### Solution B: PooledMemgraphProxy for Sub-Agents

Create a lightweight proxy that borrows connections from the pool:

```python
class PooledMemgraphProxy:
    """Lightweight proxy that borrows connections from pool for query execution.
    
    Provides the same query interface as MemgraphIngestor but uses pooled
    connections instead of creating new ones per instance.
    """
    def __init__(self, pool: MemgraphConnectionPool):
        self._pool = pool
    
    @contextmanager
    def _get_cursor(self):
        conn = self._pool.get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
        finally:
            cursor.close()
            self._pool.return_connection(conn)
    
    def fetch_all(self, query: str, params: dict | None = None) -> list:
        with self._get_cursor() as cursor:
            cursor.execute(query, params or {})
            if not cursor.description:
                return []
            column_names = [desc.name for desc in cursor.description]
            return [dict(zip(column_names, row)) for row in cursor.fetchall()]
    
    def execute_write(self, query: str, params: dict | None = None) -> None:
        with self._get_cursor() as cursor:
            cursor.execute(query, params or {})
```

#### Solution C: Update ReadOnlySubAgent to Use Connection Pool

Modify `ReadOnlySubAgent._initialize()` to use pooled connections:

```python
def _initialize(self):
    if self.agent is not None:
        return
    
    try:
        # Get connection pools instead of creating individual MemgraphIngestor instances
        self.code_graph_pool = get_connection_pool(
            settings.MEMGRAPH_HOST,
            settings.MEMGRAPH_PORT,
            settings.MEMGRAPH_USERNAME,
            settings.MEMGRAPH_PASSWORD,
            max_connections=5  # Limit per sub-agent type
        )
        self.code_graph = PooledMemgraphProxy(self.code_graph_pool)
        
        if self.enable_document_graph:
            self.doc_graph_pool = get_connection_pool(
                settings.DOC_MEMGRAPH_HOST,
                settings.DOC_MEMGRAPH_PORT,
                settings.DOC_MEMGRAPH_USERNAME,
                settings.DOC_MEMGRAPH_PASSWORD,
                max_connections=3
            )
            self.doc_graph = PooledMemgraphProxy(self.doc_graph_pool)
            self.query_router = QueryRouter(
                code_graph=self.code_graph,
                doc_graph=self.doc_graph,
            )
            self.query_router.current_mode = self.query_mode
        
        # ... rest of initialization ...
    except Exception:
        self.shutdown()
        raise
```

### 3. Enhanced Graceful Shutdown Mechanism

**Migration Note**: The codebase currently has signal handlers registered in multiple locations:
- `main.py:1441-1442` - Main application SIGINT/SIGTERM handlers
- `subagent_orchestrator.py:313-317` - Orchestrator SIGINT/SIGTERM/SIGHUP/SIGQUIT handlers

The proposed `ShutdownManager` consolidates these into a single coordinator, with components registering cleanup handlers instead of signal handlers.

#### Solution A: Global Shutdown Manager

Implement a global shutdown coordinator that replaces multiple signal handler registrations:

```python
class ShutdownManager:
    def __init__(self):
        self._shutdown_handlers = []
        self._shutdown_lock = threading.Lock()
        self._is_shutting_down = False
        
    def register_handler(self, handler, priority: int = 0):
        """Register a cleanup handler with optional priority.
        
        Higher priority handlers run first. Use priority for critical cleanup.
        """
        with self._shutdown_lock:
            if not self._is_shutting_down:
                self._shutdown_handlers.append((priority, handler))
                self._shutdown_handlers.sort(key=lambda x: -x[0])  # Higher first
    
    def initiate_shutdown(self, signum=None, frame=None):
        with self._shutdown_lock:
            if self._is_shutting_down:
                # Second signal - force exit
                logger.warning("Forced shutdown - second signal received")
                sys.exit(1)
            self._is_shutting_down = True
            
        logger.info(f"Initiating graceful shutdown (signal {signum})...")
        
        # Execute handlers in priority order
        for priority, handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Shutdown handler failed: {e}")
        
        logger.info("Graceful shutdown completed")
        sys.exit(0)

# Global instance
shutdown_manager = ShutdownManager()

# Single point of signal registration (replaces multiple registrations)
for sig in [signal.SIGINT, signal.SIGTERM]:
    signal.signal(sig, shutdown_manager.initiate_shutdown)
```

#### Solution B: Migration Path for Existing Components

**Step 1: Update main.py** - Replace direct signal handler registration:

```python
# BEFORE (main.py:1441-1442):
loop.add_signal_handler(signal.SIGINT, _handle_interrupt)
loop.add_signal_handler(signal.SIGTERM, _handle_interrupt)

# AFTER:
from codebase_rag.utils.shutdown_manager import shutdown_manager

# Register cleanup handler instead of signal handler
shutdown_manager.register_handler(
    lambda: _cancel_current_task(), 
    priority=10
)
```

**Step 2: Update SubAgentOrchestrator** - Remove signal registration, add cleanup handler:

```python
# BEFORE (subagent_orchestrator.py:313-317):
for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT]:
    try:
        signal.signal(sig, self._handle_shutdown)
    except ValueError:
        pass

# AFTER:
from codebase_rag.utils.shutdown_manager import shutdown_manager

def __init__(self, ...):
    # ... existing initialization ...
    # Register cleanup handler instead of signal handler
    shutdown_manager.register_handler(self.shutdown, priority=5)
```

#### Solution C: Component Registration Examples

```python
# In MemgraphIngestor.__enter__
def __enter__(self):
    # ... existing connection logic ...
    shutdown_manager.register_handler(self._cleanup_on_shutdown, priority=20)
    return self

def _cleanup_on_shutdown(self):
    if self._executor:
        self._executor.shutdown(wait=False, cancel_futures=True)
    if self.conn:
        try:
            self.conn.close()
        except Exception:
            pass

# In UnifiedWatcherManager (realtime_updater.py)
def start(self) -> None:
    self.observer = Observer()
    self.observer.schedule(self.event_handler, str(self.repo_path), recursive=True)
    self.observer.start()
    shutdown_manager.register_handler(self.stop, priority=5)

def stop(self) -> None:
    if self.observer:
        self.observer.stop()
```

### 4. Race Condition Prevention

**Note**: The existing `ResultAggregator` class (`orchestrator/result_aggregator.py`) already implements thread-safe operations using `threading.Lock()`. The solutions below focus on the remaining atomic state management issues.

#### Solution A: Atomic State Management

Replace simple boolean flags with atomic operations:

```python
import threading

class AtomicBoolean:
    """Thread-safe boolean with compare-and-set for safe state transitions."""
    def __init__(self, initial_value: bool = False):
        self._value = initial_value
        self._lock = threading.Lock()
    
    def get(self) -> bool:
        with self._lock:
            return self._value
    
    def set(self, value: bool) -> bool:
        """Set new value, return old value."""
        with self._lock:
            old_value = self._value
            self._value = value
            return old_value
    
    def compare_and_set(self, expected: bool, new_value: bool) -> bool:
        """Atomically set value if current matches expected. Returns success."""
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False
    
    def get_and_set(self, new_value: bool) -> bool:
        """Atomically set value and return old value."""
        with self._lock:
            old_value = self._value
            self._value = new_value
            return old_value

# Usage in SubAgentOrchestrator
class SubAgentOrchestrator:
    def __init__(self, ...):
        # Replace plain booleans with atomic operations
        self.running = AtomicBoolean(False)
        self._shutdown_called = AtomicBoolean(False)
```

#### Solution B: Update SubAgentOrchestrator State Management

Replace plain boolean flags with atomic operations in `subagent_orchestrator.py`:

```python
# BEFORE (lines 301-302):
self.running = False
self._shutdown_called = False

# AFTER:
from codebase_rag.utils.atomic import AtomicBoolean

def __init__(self, ...):
    self.running = AtomicBoolean(False)
    self._shutdown_called = AtomicBoolean(False)

def execute_tasks(self, ...):
    if self._shutdown_called.get():
        raise RuntimeError("Orchestrator is shutting down")
    
    self.running.set(True)
    # ... execution logic ...

def _handle_shutdown(self, signum, frame):
    """Handle shutdown signals to gracefully terminate all workers."""
    logger.warning(f"Received signal {signum}, initiating graceful shutdown")
    # Use compare-and-set to ensure only one shutdown initiation
    if self._shutdown_called.compare_and_set(False, True):
        self.running.set(False)

def shutdown(self):
    """Shutdown the orchestrator and cleanup all resources."""
    if self._shutdown_called.compare_and_set(False, True):
        logger.info("Shutting down sub-agent orchestrator")
        self.running.set(False)
        # ... cleanup logic ...
```
```

### 5. Comprehensive Exception Handling with Resource Cleanup

#### Solution A: Context Manager for Resource Tracking

Implement a resource tracking context manager:

```python
class ResourceTracker:
    def __init__(self):
        self._resources = []
        self._lock = threading.Lock()
    
    def track(self, resource, cleanup_func):
        with self._lock:
            self._resources.append((resource, cleanup_func))
    
    def cleanup_all(self):
        with self._lock:
            resources_to_cleanup = list(reversed(self._resources))
            self._resources.clear()
        
        for resource, cleanup_func in resources_to_cleanup:
            try:
                cleanup_func(resource)
            except Exception as e:
                logger.warning(f"Failed to cleanup resource {resource}: {e}")

@contextmanager
def tracked_resources():
    tracker = ResourceTracker()
    try:
        yield tracker
    finally:
        tracker.cleanup_all()
```

#### Solution B: Apply Resource Tracking in Critical Sections

```python
def execute_tasks(self, subtasks, ...):
    with tracked_resources() as tracker:
        # Track thread pool
        executor = ManagedThreadPoolExecutor(max_workers=self.worker_count)
        tracker.track(executor, lambda x: x.shutdown(wait=False, cancel_futures=True))
        
        # Track workers
        for worker in self.workers:
            tracker.track(worker, lambda w: w.shutdown())
        
        # ... execution logic ...
```

## Implementation Roadmap

### Phase 1: Critical Fixes (High Priority)

1. **Implement ManagedThreadPoolExecutor** - Replace all ThreadPoolExecutor instances
2. **Enhance Signal Handling** - Add comprehensive shutdown manager
3. **Fix Connection Management** - Implement connection pooling for Memgraph
4. **Add Atomic State Management** - Replace boolean flags with atomic operations

### Phase 2: Robustness Improvements (Medium Priority)

1. **Implement Resource Tracking** - Add context managers for resource cleanup
2. **Thread-Safe Result Aggregation** - Ensure safe concurrent result collection
3. **Enhanced Exception Handling** - Comprehensive cleanup in all exception paths
4. **Connection Health Monitoring** - Add proactive connection validation

### Phase 3: Optimization and Monitoring (Low Priority)

1. **Performance Monitoring** - Add metrics for resource usage and cleanup
2. **Deadlock Detection** - Implement timeout-based deadlock prevention
3. **Graceful Degradation** - Add fallback mechanisms for resource exhaustion
4. **Comprehensive Testing** - Add integration tests for shutdown scenarios

## Validation Criteria

### Resource Leak Prevention
- [ ] Zero memory growth during sustained parallel operations
- [ ] All threads properly terminated after task completion
- [ ] Database connections returned to pool or closed appropriately
- [ ] File descriptors properly closed after operations

**Already Verified** (existing implementations):
- [x] ThreadPoolExecutor cleanup via context manager (`subagent_orchestrator.py:433`)
- [x] Connection health monitoring and reconnection (`graph_service.py:305-339`)
- [x] Socket timeout and TCP keepalive (`graph_service.py:447-532`)

### Graceful Shutdown
- [ ] All active operations cancelled cleanly on SIGINT/SIGTERM
- [ ] Database transactions rolled back or committed appropriately
- [ ] Background services (real-time watchers) stopped gracefully
- [ ] Exit within configured timeout limits

**Already Verified** (existing implementations):
- [x] Signal handlers registered for main application (`main.py:1441-1442`)
- [x] Double-interrupt handling for forced exit (`main.py:1417-1425`)
- [x] Orchestrator shutdown signal handling (`subagent_orchestrator.py:313-317`)

### Concurrency Safety
- [ ] No race conditions in shared state modifications
- [ ] Consistent results across multiple parallel executions
- [ ] Proper isolation between concurrent sub-agents
- [ ] Deadlock-free operation under high concurrency

**Already Verified** (existing implementations):
- [x] Thread-safe result aggregation (`result_aggregator.py:15-68`)
- [x] Connection lock for cursor access (`graph_service.py:130`)

### Error Resilience
- [ ] Resources cleaned up properly in all exception scenarios
- [ ] System remains stable after repeated error conditions
- [ ] Recovery from transient failures without resource accumulation
- [ ] Clear error messages with proper context for debugging

**Already Verified** (existing implementations):
- [x] Retry logic for transient Memgraph errors (`graph_service.py:341-435`)
- [x] Exception handling with flush in `__exit__` (`graph_service.py:237-266`)

## Backward Compatibility

All proposed changes maintain backward compatibility with existing APIs and configuration. The enhancements are implemented as internal improvements that don't affect the public interface or user experience.

## Performance Impact

The proposed solutions have minimal performance impact:
- Connection pooling reduces connection overhead
- Managed thread pools provide better resource utilization
- Atomic operations have negligible overhead compared to current locking
- Resource tracking adds minimal overhead during normal operation

The benefits of improved stability and resource management far outweigh any minor performance costs.

## Conclusion

This design specification addresses all identified resource management and graceful shutdown issues in the codebase. By implementing these solutions, the system will achieve 100% health lifetime guarantees with proper resource cleanup, graceful termination, and robust concurrent operation management.