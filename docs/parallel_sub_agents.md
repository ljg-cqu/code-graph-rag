# Parallel Sub-Agents

## Overview
Parallel sub-agents provide a production-ready read-only execution path for multi-file analysis tasks. The runtime now previews subtasks before enabling parallel execution, blocks write-like prompts from parallel execution, scopes file-based splits to the paths mentioned in the prompt when possible, and aggregates per-worker metadata for debugging.

Sequential orchestration remains the fallback for unsupported, unsafe, or underspecified work.

## Runtime Flow
1. The interactive runtime applies CLI overrides such as `--parallel-workers`, `--no-parallel`, `--parallel-dry-run`, and `--scheduling-strategy`.
2. The runtime detects write intent before parallel eligibility is evaluated.
3. If auto-splitting is enabled, it previews a scope-aware file split to estimate safe subtask count.
4. The eligibility classifier receives the prompt, the preview subtask count, and the write-safety signal.
5. Parallel execution is used only when the task is read-only and at least two safe subtasks remain.
6. Results are aggregated with execution telemetry, worker/model metadata, and unresolved conflict reporting.

## Safety Rules
- Parallel execution is read-only in v1.
- Write-like requests do not auto-run in parallel, even if the prompt explicitly asks for parallelism.
- File-based splits respect prompt scope when paths or folders are mentioned.
- If fewer than two safe subtasks remain after filtering, the runtime falls back to sequential execution.
- `--parallel-dry-run` returns an execution plan instead of fake findings.

## CLI Usage
```bash
# Let the runtime auto-detect safe read-only work and preview subtasks
cgr start --parallel-workers 8 --auto-split

# Force sequential execution for this run
cgr start --no-parallel

# Inspect the parallel execution plan without making worker LLM calls
cgr start --parallel-dry-run --parallel-workers 6 --auto-split

# Use round-robin worker scheduling
cgr start --scheduling-strategy round-robin --parallel-workers 12
```

## Programmatic Usage
```python
from codebase_rag.orchestrator import ResultAggregator, SubAgentOrchestrator, TaskSplitter

splitter = TaskSplitter(repo_path="/path/to/repo")
subtasks = splitter.split_task("Review files in src for error handling issues")

orchestrator = SubAgentOrchestrator(
    worker_count=8,
    scheduling_strategy="round-robin",
    repo_path="/path/to/repo",
)

aggregator = orchestrator.execute_tasks(subtasks)
report = aggregator.consolidate(output_format="json")
orchestrator.shutdown()
```

## Worker Behavior
Parallel workers use a real read-only tool stack rather than placeholder sleeps. Each worker can investigate with the same evidence sources used by the main read path:
- code graph queries
- code snippet retrieval
- file reads
- directory listing
- semantic search and source lookup
- document analysis when document graph support is enabled

Worker metadata includes:
- `worker_id`
- `provider`
- `model_id`
- `retry_count`
- execution status and duration in aggregated output

## Configuration
The runtime contract is defined by the existing `CGR_*` settings in `AppConfig`.

| Setting | Default | Purpose |
|--------|---------|---------|
| `CGR_DEFAULT_PARALLEL_WORKERS` | `20` | Default worker count when no override is provided |
| `CGR_MAX_PARALLEL_WORKERS` | `30` | Upper bound for auto-scaled or user-requested workers |
| `CGR_ALLOW_DYNAMIC_MAX_OVERRIDE` | `True` | Allow runtime logic to exceed the soft max when explicitly supported |
| `CGR_AUTO_SCALE_WORKERS` | `True` | Scale worker count down to match safe subtask count |
| `CGR_SUBAGENT_TIMEOUT` | `300` | Timeout per subtask |
| `CGR_SUBAGENT_RETRY_ATTEMPTS` | `2` | Retry count for failed subtasks |
| `CGR_SUBAGENT_ALLOW_WRITE` | `False` | Worker write toggle; write-like tasks remain sequential in v1 |
| `CGR_AUTO_PARALLEL_ENABLED` | `True` | Master switch for automatic parallel activation |
| `CGR_PARALLEL_ELIGIBILITY_THRESHOLD` | `0.7` | Confidence threshold for LLM-based eligibility |
| `CGR_AUTO_SPLIT_ENABLED` | `True` | Enable automatic subtask preview and generation |
| `CGR_PARALLEL_MAX_QUEUE_SIZE` | `100` | Maximum queued subtask count before fallback |
| `CGR_WORKER_LLMS` | empty | Optional dedicated worker LLM list |
| `CGR_WORKER_LLM_ASSIGNMENT_STRATEGY` | `round-robin` | Worker model assignment strategy |

## Output Contract
The consolidated parallel result includes:
- total subtask count
- completed and failed counts
- total execution time
- unresolved conflict count
- worker count
- scheduling strategy
- dry-run flag

JSON output also includes per-result worker metadata and unresolved conflicts for downstream debugging.

## Current Scope
Included in v1:
- scope-aware file splitting
- read-only worker execution
- FIFO and round-robin scheduling
- queue limit enforcement
- retry and timeout handling
- dry-run plan output

Not included in v1:
- write-task parallelism
- semantic merge or majority-vote conflict resolution
- nested parallelism
- node/query/manual split strategies
- distributed execution