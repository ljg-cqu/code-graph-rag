# Parallel Sub-Agent/Worker Orchestration Design Specification

## 1. Problem Statement
The current implementation's parallel execution mechanism is non-functional:
- Fake parallel execution: Reported results show unrealistic execution time (0.2s for 10 subtasks across different LLM models) with no actual concurrent processing
- No LLM awareness: The orchestrator LLM cannot identify tasks that can benefit from parallel divide-and-conquer processing
- No user request handling: Explicit user requests for parallel execution are not acted upon
- No true concurrency: Sub-agent/worker tasks run sequentially instead of in parallel

## 2. Core Requirements
### 2.1 Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Support explicit user requests for parallel execution: When user asks for parallel processing, enable it if the task is suitable | High |
| FR2 | Intelligent parallelism detection: Orchestrator LLM automatically identifies tasks that can be split into independent subtasks (e.g., processing multiple files, multi-domain analysis, independent research tasks) | High |
| FR3 | True parallel execution: Run subtasks concurrently using isolated sub-agent/worker instances, with actual time reduction proportional to parallelism level | High |
| FR4 | Subtask isolation: Each worker/sub-agent has independent context, tool access, and LLM calls to avoid cross-contamination | High |
| FR5 | Result aggregation & conflict resolution: Automatically merge results from multiple subtasks, detect and resolve conflicts (majority vote for overlapping results, flag unresolved conflicts for user review) | High |
| FR6 | Parallelism controls: Allow users to set max parallel workers, subtask timeouts, and enable/disable automatic parallelism | Medium |
| FR7 | Telemetry & reporting: Provide accurate parallel execution metrics (total time, per-subtask time, success/failure rate, conflict count) | Medium |

### 2.2 Non-Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| NFR1 | Performance: For parallelizable tasks with N subtasks, total execution time should be ≤ (sequential time / N) * 1.2 (accounting for orchestration overhead) | High |
| NFR2 | Reliability: Failed subtasks are retried up to 2 times before marking as failed; partial results are returned even if some subtasks fail | High |
| NFR3 | Cost efficiency: Parallelism is only enabled when the benefit (time saved) outweighs additional LLM/cost overhead | Medium |
| NFR4 | Backward compatibility: Existing sequential execution workflows continue to work unchanged | High |

## 3. Architecture Design
### 3.1 Component Overview
```
┌───────────────────────────────────────────────────────────┐
│                     User Input / Prompt                   │
└─────────────────────────────┬─────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                Orchestrator Agent Layer                   │
│  1. Check for explicit parallel request in user prompt    │
│  2. Analyze task to detect if it can be split into        │
│     independent subtasks                                  │
│  3. Split task into bounded subtasks with clear scopes    │
└─────────────────────────────┬─────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                   Worker Pool Manager                     │
│  1. Enforce max parallel worker limits                    │
│  2. Assign subtasks to available workers                  │
│  3. Track subtask execution status and timeouts           │
└───────────┬───────────────────┬───────────────────┬───────┘
            │                   │                   │
┌───────────▼───────┐ ┌─────────▼─────────┐ ┌───────▼───────────┐
│  Sub-Agent Worker │ │ Sub-Agent Worker  │ │ Sub-Agent Worker  │
│  (Isolated)       │ │ (Isolated)        │ │ (Isolated)        │
│  - Own LLM call   │ │ - Own LLM call    │ │ - Own LLM call    │
│  - Scoped tool access │ - Scoped tool access │ - Scoped tool access │
│  - Independent context │ - Independent context │ - Independent context │
└───────────┬───────┘ └─────────┬─────────┘ └───────┬───────────┘
            │                   │                   │
└───────────┴───────────────────┴───────────────────┴───────┐
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                Result Aggregator Layer                    │
│  1. Collect all subtask results                           │
│  2. Detect overlapping content and conflicts              │
│  3. Resolve conflicts (majority vote, semantic comparison)│
│  4. Merge results into a single coherent response         │
│  5. Generate execution summary with metrics               │
└─────────────────────────────┬─────────────────────────────┘
                              │
┌─────────────────────────────▼─────────────────────────────┐
│                      Final Response                       │
└───────────────────────────────────────────────────────────┘
```

### 3.2 Key Component Details
#### 3.2.1 Orchestrator Agent Enhancements
- Updated system prompt: Add instructions to identify parallelizable tasks, split them into independent subtasks, and only use parallelism when subtasks have no dependencies
- New tool: `split_task_into_subtasks` that accepts a task description and returns a list of independent, actionable subtasks
- Parallelism guardrails: Prevent parallel execution for tasks with sequential dependencies, small tasks where overhead outweighs benefit, or tasks requiring stateful execution

#### 3.2.2 Worker Pool Implementation
- Use `asyncio.gather` for concurrent execution of async sub-agent tasks
- Configurable max workers (default: 5, configurable via `MAX_PARALLEL_WORKERS` environment variable)
- Per-subtask timeout (default: 120s, configurable per task)
- Automatic retry for failed subtasks (up to 2 retries)

#### 3.2.3 Result Aggregation & Conflict Resolution
- Semantic comparison of overlapping subtask results to detect conflicts
- Conflict resolution rules:
  1. If 2+ workers return the same result for overlapping scope, use that result
  2. If results conflict and no majority, include all conflicting versions with clear labeling
  3. Flag unresolved conflicts in the final response for user review
- Structured result merging: Combine subtask outputs into a logical flow matching the original task scope

## 4. Implementation Plan
### 4.1 Phase 1: Core Parallel Execution (High Priority)
1. Add parallel execution configuration parameters to settings:
   - `MAX_PARALLEL_WORKERS`: Default 5
   - `PARALLEL_SUBTASK_TIMEOUT`: Default 120s
   - `ENABLE_AUTO_PARALLELISM`: Default true
2. Enhance orchestrator agent system prompt with parallelism detection and task splitting instructions
3. Implement isolated sub-agent worker class that executes individual subtasks
4. Implement worker pool manager to handle concurrent subtask execution
5. Implement basic result aggregator that combines subtask outputs

### 4.2 Phase 2: Conflict Resolution & Telemetry (Medium Priority)
1. Add semantic conflict detection for overlapping subtask results
2. Implement majority vote conflict resolution logic
3. Add parallel execution metrics collection and reporting
4. Add user controls to adjust parallelism settings per request

### 4.3 Phase 3: Advanced Features (Low Priority)
1. Add cost-benefit analysis to automatically decide if parallelism is worth the additional cost
2. Add dynamic worker scaling based on subtask count and complexity
3. Add support for nested parallelism (subtasks that can be split further into sub-subtasks)

## 5. Testing Plan
### 5.1 Unit Tests
- Test orchestrator correctly identifies parallelizable vs non-parallelizable tasks
- Test task splitting produces independent, non-overlapping subtasks
- Test worker pool correctly executes tasks in parallel and enforces max worker limits
- Test result aggregation correctly merges non-conflicting results
- Test conflict resolution correctly handles majority and unresolved conflicts

### 5.2 Integration Tests
- Test end-to-end parallel execution for file processing tasks (process multiple files in parallel)
- Test explicit user parallel requests are honored
- Test sequential execution continues to work for non-parallelizable tasks
- Test failed subtask retry and partial result handling

### 5.3 Performance Tests
- Verify execution time for parallel tasks is reduced proportional to number of workers
- Measure orchestration overhead to ensure it stays below 20% of total execution time

## 6. Success Metrics
1. **Correctness**: Parallel execution produces same or better quality results as sequential execution for parallelizable tasks
2. **Performance**: Parallel execution reduces total time by ≥ 60% for tasks with 5 independent subtasks
3. **Detection Accuracy**: Orchestrator correctly identifies ≥ 90% of parallelizable tasks
4. **User Satisfaction**: Explicit user parallel requests are honored in 100% of suitable cases
5. **Reliability**: ≤ 1% of parallel executions fail due to orchestration/worker errors
