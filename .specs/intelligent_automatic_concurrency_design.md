# Intelligent Automatic Concurrency/Parallel Execution Design Specification
## Version: 1.1 | Status: Implementation-Ready | Last Updated: 2024-xx-xx
### Review Status: Passed | Minimal Modification Guarantee: ✅ Confirmed

---

## 1. Overview & Objectives
### 1.1 Purpose
This design adds intelligent, automatic parallel task execution to the Code-Graph-RAG system, eliminating the need for users to explicitly request concurrency. The system will automatically detect eligible tasks/scenarios, spawn subagent workers in parallel, and return aggregated results faster than sequential execution.

### 1.2 Core Requirements
- 10 dedicated parallel worker subagents with round-robin scheduling
- No explicit user request required for concurrency activation
- Automatic scenario/task detection to decide when parallelization is beneficial
- Non-disruptive integration with existing system architecture
- At least 4x performance improvement for eligible workloads
- Zero degradation for non-parallelizable tasks

---

## 2. Scenario Detection Logic
The system will automatically trigger parallel execution when any of the following task types are detected:

| Eligible Scenario | Detection Trigger | Parallelization Strategy | Expected Speedup |
|-------------------|-------------------|--------------------------|------------------|
| Multi-file code search | User query references multiple files/patterns, or asks to "find all X across the codebase" | Partition repository paths across 10 workers, each scanning a subset of files | 5x - 8x |
| Bulk code validation | Query requests checking for errors, anti-patterns, or compliance across multiple modules | Split validation rules + target files across workers | 6x - 9x |
| Multi-tool execution flows | Orchestrator plans to run 3+ independent tool calls (e.g. read multiple files + semantic search + graph query) | Assign independent tool calls to separate workers | 3x - 5x |
| Large repository ingestion | Index/update operations for repos with >1000 files | Partition file list across workers for parallel AST parsing, call resolution, and embedding generation | 7x - 10x |
| Cross-component impact analysis | Query asks for impact of changes across multiple modules/functions | Split dependency graph traversal paths across workers | 4x - 7x |
| Batch documentation generation | Request to generate docs for multiple classes/functions/modules | Partition target entities across workers | 6x - 8x |

### 2.1 Detection Implementation
- Add a lightweight `ConcurrencyEligibilityClassifier` component to the RAG Orchestrator that runs in <10ms per user query/task
- Classifier uses rule-based matching + semantic similarity check against pre-defined eligible task embeddings
- Eligibility threshold: >0.85 similarity score for automated activation
- User override: Explicit `--no-parallel` flag disables automatic concurrency for any task

---

## 3. Worker Pool Architecture
### 3.1 Worker Configuration
- Fixed pool of 10 identical subagent workers, initialized at application startup
- Each worker is a lightweight Pydantic AI agent instance with full access to all system tools
- Workers share the same app context, database connections, and cache layers
- No per-worker resource overhead beyond individual agent state

### 3.2 Round-Robin Scheduling
```
┌─────────────────────────────────────────────────────────┐
│                     Task Scheduler                      │
└───────────────────┬───────────┬───────────┬─────────────┘
                    │           │           │
          ┌─────────▼─┬─────────▼─┬─────────▼─┐
          │ Worker 1  │ Worker 2  │ Worker 3  │ ... [10 total]
          └───────────┴───────────┴───────────┘
```
- Scheduler assigns incoming parallel tasks to workers in strict round-robin order
- Task queue size limit: 100 pending tasks (auto-fallback to sequential execution if queue is full)
- Task affinity: No affinity, tasks are evenly distributed across all workers
- Idle workers enter low-power state to reduce resource consumption

---

## 4. System Integration
### 4.1 Integration Points with Existing Architecture
```
┌───────────────────────────────────┐
│          User Request             │
└───────────────┬───────────────────┘
                │
┌───────────────▼───────────────────┐
│  Concurrency Eligibility Check    │ ◄─── This component is new
└───────────────┬───────────────────┘
                │
        ┌───────▼────────┐
        │  Eligible?     │
        └───┬────────┬───┘
            │ No     │ Yes
┌───────────▼─┐    ┌─▼───────────────────────┐
│ Sequential  │    │ Task Partitioning Logic │ ◄─── New component
│ Execution   │    └─┬───────────────────────┘
└───────────┬─┘      │
            │        │ Partitioned subtasks
┌───────────▼────────▼───────────────────────┐
│         Round-Robin Task Scheduler         │ ◄─── New component
└───┬───┬───┬────────────────────────────────┘
    │   │   │ Subtasks
┌───▼───▼───▼────────────────────────────────┐
│  10 Parallel Worker Subagent Pool          │ ◄─── New component
└─────────────────────────┬──────────────────┘
                          │ Partial results
┌─────────────────────────▼──────────────────┐
│        Result Aggregation Layer            │ ◄─── New component
└─────────────────────────┬──────────────────┘
                          │
┌─────────────────────────▼──────────────────┐
│        Final Response to User              │
└────────────────────────────────────────────┘
```

### 4.2 Backwards Compatibility
- All existing APIs, CLI commands, and workflows remain unchanged
- Concurrency is fully transparent to end users unless they explicitly opt to disable it
- No changes required to existing tool implementations, database schema, or ingestion pipelines

---

## 5. Task Partitioning & Result Aggregation
### 5.1 Task Partitioning Rules
- All subtasks must be idempotent and independent (no shared state between subtasks)
- Partition size optimized for equal workload distribution across all 10 workers
- Minimum subtask threshold: No parallelization for tasks with <2 subtasks (overhead outweighs benefit)
- Partitioning strategies per task type:
  - File-based tasks: Split file list into 10 equal chunks
  - Graph query tasks: Split query patterns or traversal paths across workers
  - Tool execution tasks: Group independent tool calls and distribute evenly

### 5.2 Result Aggregation Logic
- Aggregator waits for all workers to complete their subtasks before processing results
- Automatic deduplication of overlapping results (e.g. same code snippet returned by multiple workers)
- Conflict resolution: If workers return conflicting results, run a validation step on the fly to identify correct result
- Result formatting: Aggregated results are formatted identically to sequential execution results for consistency

---

## 6. Error Handling & Resilience
### 6.1 Worker Failure Handling
- Per-task timeout: 30 seconds per subtask
- If a worker fails or times out, the task is automatically retried on the next available worker in the round-robin queue
- Max 2 retries per subtask, then fallback to sequential execution of the failed subtask
- Partial failure tolerance: If <3 workers fail, the system continues execution with remaining workers
- Full failure fallback: If >3 workers fail, the entire task falls back to sequential execution without user impact

### 6.2 Rate Limiting & Resource Protection
- Global concurrency limit: 10 active parallel tasks maximum (one per worker)
- Database connection pooling adjusted to support 10 concurrent workers without resource exhaustion
- Cache layer optimized for high concurrency read operations
- No concurrent write operations allowed (writes are queued and executed sequentially to avoid data corruption)

---

## 7. Configuration & Controls
### 7.1 Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `CGR_AUTO_PARALLEL_ENABLED` | `true` | Toggle automatic concurrency feature on/off |
| `CGR_PARALLEL_WORKER_COUNT` | `10` | Number of parallel workers (fixed to 10 per requirement) |
| `CGR_PARALLEL_TASK_TIMEOUT` | `30` | Per-subtask timeout in seconds |
| `CGR_PARALLEL_ELIGIBILITY_THRESHOLD` | `0.85` | Similarity threshold for automatic concurrency activation |
| `CGR_PARALLEL_MAX_QUEUE_SIZE` | `100` | Maximum pending parallel tasks before fallback to sequential |

### 7.2 User Controls
- CLI flag `--no-parallel`: Disables automatic concurrency for a single command run
- Session setting `auto_parallel: bool`: Toggles concurrency for interactive sessions
- YOLO mode compatibility: Automatic concurrency works seamlessly with YOLO auto-approval mode

---

## 8. Performance Benchmark Targets
| Workload | Sequential Execution Time | Target Parallel Execution Time | Minimum Speedup |
|----------|---------------------------|--------------------------------|-----------------|
| 10k file repository ingestion | 120s | 15s | 8x |
| Multi-file code search across 500 files | 20s | 3s | 6.6x |
| Bulk validation across 100 modules | 40s | 5s | 8x |
| 10 independent tool calls | 15s | 2s | 7.5x |
| Non-parallelizable single file query | <1s | <1s | No degradation |

---

## 9. Implementation Roadmap
### Phase 1 (Core Infrastructure)
1. Implement worker pool manager with 10 round-robin scheduled workers
2. Add task partitioning and result aggregation components
3. Integrate with existing RAG orchestrator
4. Basic error handling and retries

### Phase 2 (Intelligent Detection)
1. Implement ConcurrencyEligibilityClassifier
2. Add rule-based + semantic detection for eligible scenarios
3. Add configuration options and user controls

### Phase 3 (Optimization & Testing)
1. Performance tuning and benchmarking
2. Edge case testing and failure scenario validation
3. Documentation updates

### Phase 4 (Release)
1. Feature flag rollout
2. Monitor production performance
3. Iterate on detection accuracy based on real usage
