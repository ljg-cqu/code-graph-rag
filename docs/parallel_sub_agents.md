# Parallel Sub-Agents Feature Documentation

## Overview
The parallel sub-agents feature enables concurrent execution of independent tasks across multiple sub-agent workers, significantly improving performance for bulk operations like code review, test generation, and multi-file analysis.

## Key Features
- **Zero breaking changes**: All parallel functionality is opt-in, existing workflows continue to work unchanged
- **Multiple scheduling strategies**: FIFO (default) and round-robin for even load distribution
- **Dynamic worker adjustment**: Add/remove workers mid-execution
- **Auto-scaling**: Automatically reduces worker count to match number of subtasks to avoid resource waste
- **Task splitting**: Automatically split parallelizable tasks into independent subtasks (file-based strategy implemented in MVP)
- **Result aggregation & deduplication**: Consolidate results from all workers and remove duplicate findings
- **Configurable safety limits**: Max worker limit, per-subtask timeout, retry attempts, and read-only mode by default
- **Natural language support**: Automatically detect parallel execution requests from user prompts (Phase 2)

## Usage

### CLI Usage
```bash
# Enable parallel mode with 15 workers (5 base + 10 additional) for code review
cgr start --parallel-workers 15 --auto-split --prompt "Review all Python files for security vulnerabilities"

# Use round-robin scheduling for even load distribution
cgr start --parallel-workers 10 --auto-split --prompt "Generate unit tests for all functions"

# Disable parallel execution (default behavior)
cgr start --no-parallel
```

### Programmatic Usage

#### Basic Parallel Code Review
```python
from codebase_rag.orchestrator import TaskSplitter, SubAgentOrchestrator, ResultAggregator

# Split task into file-based subtasks
splitter = TaskSplitter()
prompt = "Review this file for security vulnerabilities"
subtasks = splitter.split_task(prompt)

# Initialize orchestrator with round-robin scheduling
orchestrator = SubAgentOrchestrator(
    worker_count=15,
    scheduling_strategy="round-robin"
)

# Execute tasks in parallel
aggregator = orchestrator.execute_tasks(subtasks)

# Print consolidated markdown report
print(aggregator.consolidate(output_format="markdown"))

orchestrator.shutdown()
```

#### Dynamic Worker Adjustment
```python
# Start with 5 workers
orchestrator = SubAgentOrchestrator(worker_count=5)

# Add 10 additional workers mid-execution
orchestrator.adjust_worker_count(adjustment=10) # Now 15 workers total

# Remove 3 workers
orchestrator.adjust_worker_count(adjustment=-3) # Now 12 workers total
```

## Configuration Options
All options can be set via environment variables or in the AppConfig:

| Option | Default | Description |
|--------|---------|-------------|
| `CGR_MAX_PARALLEL_WORKERS` | 20 | Maximum allowed concurrent sub-agents |
| `CGR_DEFAULT_PARALLEL_WORKERS` | 5 | Default worker count if not specified |
| `CGR_ALLOW_DYNAMIC_MAX_OVERRIDE` | True | Allow exceeding max worker limit with user approval |
| `CGR_AUTO_SCALE_WORKERS` | True | Auto-scale worker count to match subtask count |
| `CGR_SUBAGENT_TIMEOUT` | 300 | Timeout in seconds per sub-agent task |
| `CGR_SUBAGENT_ALLOW_WRITE` | False | Allow sub-agents to perform write operations |
| `CGR_AUTO_SPLIT_ENABLED` | True | Enable automatic task splitting |
| `CGR_SUBAGENT_RETRY_ATTEMPTS` | 2 | Number of retries for failed tasks |

## Scheduling Strategies
1. **FIFO (First-In-First-Out)**: Default strategy. Assigns tasks to the first available worker. Best for general purpose use.
2. **Round-Robin**: Cycles through all available workers evenly distributing tasks. Best for homogeneous workloads to avoid stragglers and optimize resource utilization.

## Roadmap
### Phase 1 (MVP - Completed)
- ✅ File-based task splitting
- ✅ FIFO and round-robin scheduling
- ✅ Result aggregation and deduplication
- ✅ CLI parallel flags
- ✅ Dynamic worker adjustment
- ✅ Configuration options
- ✅ Unit tests
- ✅ Documentation and examples

### Phase 2 (Upcoming)
- Node-based task splitting (split by function/class)
- Query-based task splitting
- Manual task splitting support
- Natural language concurrency detection
- Real-time progress UI
- Structured JSON output with metrics

### Phase 3 (Future)
- Async sub-agent execution
- Sub-agent LLM configuration overrides
- Persistent sub-agent sessions
- Distributed execution across multiple machines

## Security Best Practices
1. **Read-only by default**: Sub-agents have only read access unless explicitly enabled with `--subagent-allow-write`
2. **No shared mutable state**: All sub-agents have isolated session state
3. **Parent approval flow**: All write operations are routed to the parent agent for user approval
4. **Rate limiting**: Built-in per-worker rate limiting to avoid LLM API limits
5. **Resource limits**: Configurable timeouts and worker limits prevent resource exhaustion
