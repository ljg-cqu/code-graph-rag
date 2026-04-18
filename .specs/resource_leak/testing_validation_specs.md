# Testing and Validation Specification for Resource Management

## Overview

This document outlines comprehensive testing strategies and validation criteria to ensure that all resource management and graceful shutdown improvements are properly implemented and effective. The tests cover unit, integration, and system-level scenarios to verify 100% health lifetime guarantees.

## Test Categories

### 1. Resource Leak Detection Tests

#### Memory Leak Tests
**Objective**: Verify no memory accumulation during sustained operations

**Test Cases**:
1. **Long-Running Parallel Execution**
   - Execute 1000+ parallel sub-agent tasks over 30 minutes
   - Monitor memory usage with `psutil`
   - Assert memory growth < 5% of initial usage
   - Verify all threads properly terminated

2. **Connection Pool Stress Test**
   - Create 100 concurrent Memgraph connections
   - Execute queries with varying timeouts
   - Verify connection count returns to baseline after completion
   - Test pool exhaustion handling

3. **Exception Path Resource Cleanup**
   - Force exceptions during parallel execution
   - Verify all resources (connections, threads, file handles) are cleaned up
   - Test nested exception scenarios

**Implementation**:
```python
# test_resource_leaks.py
import psutil
import gc
import threading
import time
from codebase_rag.orchestrator.subagent_orchestrator import SubAgentOrchestrator

def test_memory_leak_during_parallel_execution():
    """Test for memory leaks during sustained parallel operations.
    
    Note: Python's GC is non-deterministic. We use:
    - gc.collect() to encourage collection
    - Settling period for finalizers to run
    - 10% threshold to account for GC overhead
    - Trend analysis across batches rather than absolute limits
    """
    process = psutil.Process()
    
    # Warm up and establish true baseline
    gc.collect()
    time.sleep(0.5)
    gc.collect()  # Second pass for cyclic references
    time.sleep(0.5)
    initial_memory = process.memory_info().rss
    
    orchestrator = SubAgentOrchestrator(worker_count=10)
    subtasks = [{"id": i, "prompt": f"Analyze file {i}"} for i in range(100)]
    
    memory_samples = []
    
    # Execute multiple batches and track memory trend
    for batch in range(10):
        result = orchestrator.execute_tasks(subtasks[:10])
        assert result is not None
        
        # Force garbage collection and settling
        gc.collect()
        time.sleep(0.3)
        
        current_memory = process.memory_info().rss
        memory_samples.append(current_memory)
    
    orchestrator.shutdown()
    
    # Final cleanup and measurement
    gc.collect()
    time.sleep(1.0)
    final_memory = process.memory_info().rss
    
    # Check final memory growth (allow 10% for GC overhead)
    memory_growth_pct = ((final_memory - initial_memory) / initial_memory) * 100
    assert memory_growth_pct < 10.0, f"Memory growth {memory_growth_pct:.2f}% exceeds threshold"
    
    # Check for memory growth trend (should not be monotonically increasing)
    # This catches leaks that GC hasn't cleaned up yet
    if len(memory_samples) >= 3:
        early_avg = sum(memory_samples[:3]) / 3
        late_avg = sum(memory_samples[-3:]) / 3
        growth_ratio = late_avg / early_avg
        assert growth_ratio < 1.2, f"Memory trend shows leak: {early_avg:.0f} -> {late_avg:.0f} bytes"
```

#### Thread Leak Tests
**Objective**: Ensure all spawned threads are properly managed and terminated

**Test Cases**:
1. **Thread Count Verification**
   - Monitor active thread count before/after parallel execution
   - Verify thread count returns to baseline
   - Test forced termination scenarios

2. **ThreadPoolExecutor Cleanup**
   - Verify executor shutdown in normal and exception paths
   - Test timeout-based shutdown behavior
   - Validate cancel_futures functionality

**Implementation**:
```python
def test_thread_leak_prevention():
    """Verify no thread leaks during parallel execution.
    
    Note: Thread termination is not instantaneous. We:
    - Allow adequate settling time (2 seconds)
    - Accept small variance for background threads
    - Focus on net growth, not absolute counts
    """
    initial_threads = threading.active_count()
    
    orchestrator = SubAgentOrchestrator(worker_count=5)
    subtasks = [{"id": i, "prompt": f"Task {i}"} for i in range(20)]
    
    try:
        result = orchestrator.execute_tasks(subtasks)
        assert result is not None
    finally:
        orchestrator.shutdown()
    
    # Allow time for threads to terminate (daemon threads may linger briefly)
    time.sleep(2)
    
    # Force check that threads are actually terminating
    for _ in range(3):
        current_threads = threading.active_count()
        if current_threads <= initial_threads:
            break
        time.sleep(0.5)
    
    final_threads = threading.active_count()
    
    # Thread count should return to baseline or very close
    # Allow +1 for any lingering background threads (e.g., from libraries)
    assert final_threads <= initial_threads + 1, \
        f"Thread leak detected: {initial_threads} -> {final_threads}. " \
        f"Threads: {[t.name for t in threading.enumerate()]}"
```

### 2. Graceful Shutdown Tests

#### Signal Handling Tests
**Objective**: Verify proper response to shutdown signals

**Test Cases**:
1. **SIGINT During Active Execution**
   - Start parallel execution
   - Send SIGINT signal
   - Verify graceful cleanup and exit
   - Ensure no orphaned processes

2. **SIGTERM Handling**
   - Similar to SIGINT but for production environments
   - Verify database transaction rollback
   - Test with multiple concurrent operations

3. **Double Signal Handling**
   - Send SIGINT twice rapidly
   - First should initiate graceful shutdown
   - Second should force immediate exit
   - Verify no resource corruption

**Implementation**:
```python
# test_graceful_shutdown.py
import os
import signal
import subprocess
import time
import pytest

@pytest.mark.parametrize("signal_type", [signal.SIGINT, signal.SIGTERM])
def test_signal_handling(signal_type):
    """Test graceful shutdown on signal receipt."""
    # Start the application as subprocess
    proc = subprocess.Popen(
        ["python", "-m", "codebase_rag.cli", "start", "--repo-path", ".", "--no-confirm"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for startup
    time.sleep(5)
    
    # Send signal
    proc.send_signal(signal_type)
    
    # Wait for graceful shutdown
    try:
        stdout, stderr = proc.communicate(timeout=30)
        assert proc.returncode == 0 or proc.returncode == -signal_type
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("Application did not shutdown gracefully within timeout")
```

#### Resource Cleanup Tests
**Objective**: Ensure all resources are properly cleaned up during shutdown

**Test Cases**:
1. **Database Connection Cleanup**
   - Verify all Memgraph connections closed on shutdown
   - Test with active transactions
   - Validate connection pool cleanup

2. **File Handle Cleanup**
   - Monitor open file descriptors
   - Verify cleanup of temporary files
   - Test with large file operations

3. **Network Resource Cleanup**
   - Verify socket cleanup
   - Test with network timeouts
   - Validate connection pool cleanup

**Implementation**:
```python
def test_database_connection_cleanup():
    """Verify all database connections are closed during shutdown."""
    from codebase_rag.services.graph_service import MemgraphIngestor
    
    initial_connections = get_active_db_connections()
    
    with MemgraphIngestor(host="localhost", port=7687, batch_size=100) as ingestor:
        # Perform some operations
        ingestor.ensure_node("Test", {"name": "test"})
        ingestor.flush_all()
        
        active_connections = get_active_db_connections()
        assert active_connections > initial_connections
    
    # After context manager exit
    final_connections = get_active_db_connections()
    assert final_connections <= initial_connections + 1  # Allow for monitoring connections
```

### 3. Concurrency Safety Tests

#### Race Condition Tests
**Objective**: Detect and prevent race conditions in concurrent operations

**Test Cases**:
1. **Shared State Modification**
   - Multiple threads modifying shared result aggregator
   - Verify data consistency
   - Test with high contention scenarios

2. **Atomic Operation Verification**
   - Test compare-and-set operations under load
   - Verify thread safety of boolean flags
   - Validate state consistency

3. **Deadlock Detection**
   - Create potential deadlock scenarios
   - Verify timeout-based deadlock prevention
   - Test lock ordering

**Implementation**:
```python
# test_concurrency_safety.py
import threading
import time
from codebase_rag.orchestrator.result_aggregator import ThreadSafeResultAggregator

def test_race_condition_in_result_aggregation():
    """Test thread safety of result aggregation."""
    aggregator = ThreadSafeResultAggregator()
    num_threads = 50
    results_per_thread = 20
    
    def worker(thread_id):
        for i in range(results_per_thread):
            aggregator.add_result(
                {"id": f"{thread_id}-{i}"}, 
                f"result-{thread_id}-{i}",
                execution_time=0.1
            )
    
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    results = aggregator.get_results()
    expected_count = num_threads * results_per_thread
    assert len(results) == expected_count
    
    # Verify no duplicate IDs
    ids = [r['subtask']['id'] for r in results]
    assert len(set(ids)) == len(ids), "Duplicate task IDs found"
```

#### Thread Safety Tests
**Objective**: Ensure all shared components are thread-safe

**Test Cases**:
1. **Connection Pool Thread Safety**
   - Multiple threads accessing connection pool simultaneously
   - Verify no connection corruption
   - Test pool exhaustion scenarios

2. **Atomic Flag Thread Safety**
   - Concurrent access to atomic boolean flags
   - Verify correct state transitions
   - Test compare-and-set operations

3. **Shutdown Manager Thread Safety**
   - Multiple threads registering shutdown handlers
   - Verify handler execution order
   - Test concurrent shutdown initiation

### 4. Error Resilience Tests

#### Exception Handling Tests
**Objective**: Verify proper resource cleanup in all exception scenarios

**Test Cases**:
1. **Database Connection Failures**
   - Simulate database connection drops
   - Verify connection pool recovery
   - Test retry logic

2. **Network Timeout Handling**
   - Force network timeouts during operations
   - Verify proper cleanup and retry
   - Test timeout escalation

3. **Memory Exhaustion Scenarios**
   - Simulate low-memory conditions
   - Verify graceful degradation
   - Test resource cleanup under pressure

**Implementation**:
```python
def test_exception_handling_with_resource_cleanup():
    """Test resource cleanup during exceptions."""
    from unittest.mock import patch
    
    orchestrator = SubAgentOrchestrator(worker_count=3)
    
    # Mock a failure during execution
    with patch.object(orchestrator, '_execute_subtask', side_effect=RuntimeError("Simulated failure")):
        subtasks = [{"id": i, "prompt": f"Task {i}"} for i in range(5)]
        
        try:
            orchestrator.execute_tasks(subtasks)
            pytest.fail("Expected exception was not raised")
        except RuntimeError:
            pass  # Expected
        
        # Verify resources are still properly cleaned up
        assert orchestrator.running.get() == False
        assert len(orchestrator.workers) == 3  # Workers should still exist but be clean
        
        # Shutdown should work normally
        orchestrator.shutdown()
```

### 5. Performance and Scalability Tests

#### Load Testing
**Objective**: Verify system stability under high load

**Test Cases**:
1. **High Concurrency Load**
   - Execute 100+ parallel workers simultaneously
   - Monitor system resource usage
   - Verify graceful degradation

2. **Long-Running Operations**
   - Sustained operations over hours
   - Monitor for resource accumulation
   - Test periodic cleanup mechanisms

3. **Scalability Testing**
   - Vary worker counts from 1 to maximum
   - Measure performance vs. resource usage
   - Identify optimal configurations

**Implementation**:
```python
def test_high_concurrency_stability():
    """Test system stability under high concurrency."""
    max_workers = 30
    subtasks = [{"id": i, "prompt": f"Analyze component {i}"} for i in range(200)]
    
    orchestrator = SubAgentOrchestrator(worker_count=max_workers)
    
    start_time = time.time()
    result = orchestrator.execute_tasks(subtasks)
    end_time = time.time()
    
    execution_time = end_time - start_time
    throughput = len(subtasks) / execution_time
    
    # Verify reasonable performance
    assert throughput > 1.0, f"Throughput too low: {throughput:.2f} tasks/second"
    
    # Verify no errors
    assert len(result.get_errors()) == 0
    
    orchestrator.shutdown()
```

## Validation Metrics

### Resource Usage Metrics
- **Memory Growth**: < 10% over sustained operations (allows for Python GC behavior)
- **Thread Count**: Returns to baseline within 2 seconds after completion
- **Connection Count**: Properly managed by connection pools
- **File Descriptors**: No accumulation during operations

**Note on Memory Testing**: Python's garbage collector does not guarantee immediate collection. Tests should:
1. Call `gc.collect()` before measurements
2. Allow a brief settling period (1-2 seconds)
3. Use percentage thresholds that account for GC non-determinism
4. Run multiple iterations to establish baseline trends

### Shutdown Performance Metrics
- **Graceful Shutdown Time**: < 30 seconds for normal operations
- **Forced Shutdown Time**: < 5 seconds for emergency scenarios
- **Resource Cleanup Completeness**: 100% of allocated resources freed

### Concurrency Safety Metrics
- **Race Condition Detection**: Zero race conditions in test scenarios
- **Deadlock Prevention**: No deadlocks detected in stress tests
- **Data Consistency**: 100% consistency in shared state modifications

### Error Resilience Metrics
- **Exception Recovery**: 100% resource cleanup in exception paths
- **Retry Success Rate**: > 95% for transient failures
- **Graceful Degradation**: System remains operational under stress

## Test Environment Requirements

### Hardware Requirements
- **Memory**: Minimum 8GB RAM for comprehensive testing
- **CPU**: Multi-core processor (4+ cores recommended)
- **Storage**: SSD storage for I/O intensive tests

### Software Requirements
- **Python**: 3.8+ with required dependencies
- **Memgraph**: Running instances for code, document, and JSON graphs
- **Testing Tools**: pytest, psutil, pytest-timeout, coverage

### Test Data Requirements
- **Sample Codebases**: Various sizes (small, medium, large)
- **Document Collections**: Mixed formats (PDF, Markdown, etc.)
- **JSON Datasets**: Structured data for ingestion testing

## Continuous Integration Integration

### CI Pipeline Integration
1. **Unit Tests**: Run on every commit
2. **Integration Tests**: Run on pull requests
3. **System Tests**: Run nightly with full resource monitoring
4. **Performance Tests**: Run weekly with benchmarking

### Automated Monitoring
1. **Resource Leak Detection**: Automated memory/thread monitoring
2. **Shutdown Validation**: Automated signal handling tests
3. **Concurrency Testing**: Automated race condition detection
4. **Performance Regression**: Automated performance benchmarking

## Conclusion

This comprehensive testing specification ensures that all resource management and graceful shutdown improvements are thoroughly validated. The tests cover all critical scenarios including normal operation, error conditions, high concurrency, and system stress. By implementing these tests, we can guarantee 100% health lifetime for the application with proper resource management and graceful shutdown capabilities.