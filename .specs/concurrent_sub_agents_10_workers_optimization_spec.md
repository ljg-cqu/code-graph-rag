# Concurrent Sub-Agents Optimization Spec for 10+ Parallel Workers
## Status: Draft | Last Updated: 2024
## Target: Fix existing implementation issues and optimize for 10+ parallel sub-agent workers

---
## Identified Issues in Current Implementation
### 1. Critical Thread Safety Risks
- **ResultAggregator has no locks**: Concurrent writes from 10+ workers will cause race conditions, lost results/errors, corrupted metadata, and crashes
- **Shared RateLimiter has no thread safety**: Incorrect rate calculation with concurrent access will lead to API throttling or bans
- **Agent pool modifications are not atomic**: Concurrent pop/append operations on the agent pool list can cause lost worker instances

### 2. Performance Limitations at Scale
- **ThreadPoolExecutor exclusive use**: Poor performance for CPU-heavy sub-agent pre/post processing due to GIL limitations, only optimal for pure I/O bound workloads
- **No connection pooling across workers**: Each worker creates independent HTTP connections for API calls, leading to too many open connections, high latency, and TCP overhead at 10+ worker scale
- **Static worker count**: No automatic adjustment based on system load, API rate limits, or task queue backpressure
- **No per-worker performance metrics**: No way to identify slow workers, stuck tasks, or bottlenecks when running at scale

### 3. Reliability & Stability Issues
- **No agent state reset between tasks**: Reused agents retain mutable state from previous tasks leading to incorrect results and cross-task contamination
- **No resource usage checks**: Spawning 10+ workers without checking available CPU/memory can lead to OOM crashes, especially when running local models
- **No worker crash isolation**: Hard crashes in individual worker threads can take down the entire orchestrator process
- **Unbounded task queue**: No limit on pending tasks leads to memory bloat when processing thousands of tasks with 10 workers
- **No deadlock detection**: Stuck tasks holding workers indefinitely reduce effective concurrency

---
## Optimized Design Specification
### 1. Thread Safety Fixes (Critical Priority)
| Component | Fix |
|-----------|-----|
| ResultAggregator | Add `threading.Lock()` around all write operations (`add_result`, `add_error`, metadata updates) |
| RateLimiter | Replace existing implementation with a thread-safe rate limiter that uses atomic counters or locks for concurrent access |
| Agent Pool | Replace list-based pool with a `queue.Queue` for thread-safe atomic get/put operations |

### 2. Performance Optimizations for 10+ Workers
#### 2.1 Executor Strategy
- Add support for configurable executor type:
  - Default: `ThreadPoolExecutor` for I/O bound LLM API workloads (optimal for 10+ workers)
  - Optional `ProcessPoolExecutor` for CPU-heavy workloads (local embeddings, code parsing)
  - Add configuration parameter `CGR_SUBAGENT_EXECUTOR_TYPE = "thread" / "process"`

#### 2.2 Connection & Resource Pooling
- Implement shared HTTP connection pool for all workers using `httpx.Client` with connection limits
- Add maximum concurrent connection limit configuration: `CGR_SUBAGENT_MAX_CONCURRENT_CONNECTIONS = 20` (2x worker count for headroom)
- Reuse API client instances across workers instead of creating new clients per task/worker

#### 2.3 Dynamic Auto-Scaling
- Add automatic worker count adjustment based on:
  1. Current system CPU/memory usage (keep total usage below 80% by default)
  2. API rate limit feedback (reduce workers if 429 rate limit errors are detected)
  3. Task queue backlog (add workers up to max limit if queue is growing)
- Add configuration parameters:
  - `CGR_AUTO_SCALE_INTERVAL = 10` (check scaling every 10s)
  - `CGR_AUTO_SCALE_CPU_THRESHOLD = 80` (max CPU usage percentage)
  - `CGR_AUTO_SCALE_MEMORY_THRESHOLD = 80` (max memory usage percentage)

### 3. Reliability Improvements
#### 3.1 Agent Lifecycle Management
- Add mandatory `reset()` method to sub-agent interface, called automatically before returning an agent to the pool
- Add optional `cleanup()` method called on worker shutdown to release resources
- Add configuration option `CGR_SUBAGENT_REUSE_MAX = 100` (recycle agents after 100 tasks to prevent memory leaks)

#### 3.2 Worker Isolation & Resilience
- Add per-worker exception handling to isolate crashes, prevent single worker failures from taking down the entire orchestrator
- Add deadlock detection: monitor task execution time, terminate and restart workers that exceed 2x the configured timeout
- Add pending task queue limit: `CGR_SUBAGENT_MAX_PENDING_TASKS = 1000` (reject new tasks if queue exceeds limit to prevent OOM)

#### 3.3 Resource Protection
- Add pre-execution resource check: calculate required memory per worker, verify total required memory for requested workers is available before spawning
- Add minimum required resource configuration: `CGR_SUBAGENT_MIN_MEMORY_PER_WORKER_MB = 256` (adjust based on model size for local deployments)

### 4. Observability & Metrics
- Add per-worker metrics tracking:
  - Tasks completed per worker
  - Average execution time per worker
  - Error rate per worker
- Add real-time metrics endpoint for monitoring when running as a service
- Add structured logging for all worker events with worker ID, task ID, and timing information

---
## Implementation Priority
1. **Critical (Must do before running 10 workers)**: Thread safety fixes for ResultAggregator, RateLimiter, and Agent Pool
2. **High**: Connection pooling, agent state reset, resource checks
3. **Medium**: Dynamic auto-scaling, worker isolation, metrics
4. **Low**: ProcessPoolExecutor support, deadlock detection

---
## Expected Performance Improvements for 10 Workers
- 30-50% reduction in API call latency due to connection pooling
- 0% race condition related errors/crashes
- 20% better throughput for CPU-heavy workloads with ProcessPoolExecutor
- Automatic prevention of OOM crashes and API throttling
