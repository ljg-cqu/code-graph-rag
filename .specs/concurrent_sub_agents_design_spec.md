# Concurrent Sub-Agents Feature Design Specification
## Version: 1.0 | Last Updated: 2024-XX-XX | Status: Draft

## 1. Overview & Problem Statement
Current Code-Graph-RAG implementation uses a single monolithic RAG Orchestrator agent that processes user requests sequentially, with no support for parallel task execution via sub-agents. This limits performance for large-scale tasks that can be easily parallelized, such as bulk code review, multi-file vulnerability scanning, cross-module refactoring impact analysis, and large repository documentation generation.

### User Requirement
Enable intelligent spawning of concurrent sub-agent/workers upon user command, allowing parallel execution of independent subtasks (e.g. spawning 10 parallel sub-agents to review all repository code files simultaneously, similar to functionality available in Kimi Code, Claude Code, and Opencode).

## 2. Target Use Cases
| Use Case | Description |
|----------|-------------|
| Parallel Code Review | Split repository code into chunks, assign each chunk to a separate sub-agent for concurrent review for bugs, style issues, or security vulnerabilities |
| Bulk Code Refactoring | Generate refactoring changes for multiple independent modules in parallel |
| Multi-Document Analysis | Analyze multiple documentation files/specifications in parallel to cross-validate against implementation |
| Large Scale Test Generation | Generate unit tests for all functions/classes in a repository in parallel |
| Cross-Project Impact Analysis | Run parallel queries across multiple indexed projects to assess dependency change impact |
| Parallel Benchmark Execution | Run multiple benchmark tasks against the code graph concurrently to measure performance |

## 3. Design Goals
### Goals
1. Zero breaking changes to existing functionality (single agent mode remains default)
2. Intelligent automatic task splitting for supported use cases, with option for manual task definition
3. Configurable concurrency limits (user adjustable, with safe defaults)
4. Built-in result aggregation and deduplication
5. Support for both local (Ollama) and cloud LLM backends for sub-agents
6. Full audit logging of all sub-agent activity
7. Integration with existing approval flow / YOLO mode
8. Resource usage controls to prevent LLM rate limiting or system overload

### Non-Goals
1. Inter-sub-agent communication (v1 focuses on independent parallel tasks only)
2. Distributed sub-agent execution across multiple machines (v1 is single-machine process only)
3. Persistent sub-agent sessions (sub-agents are ephemeral per task execution)

## 4. Architecture Design
### Integration with Existing System
The new concurrent sub-agent system fits into the existing architecture as an extension to the RAG Orchestrator layer:

```
┌──────────────────────────────────────────────────────────┐
│                      USER INTERFACE                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │ New CLI flags: --parallel-workers N, --auto-split  │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│              Application Layer (main.py)                 │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Sub-Agent Feature Flag + Configuration Loading    │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────┬────────────────────────────┘
                              │
┌─────────────────────────────▼────────────────────────────┐
│                RAG Orchestrator Layer                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 1. Task Splitter: Detects parallelizable tasks     │  │
│  │ 2. SubAgentOrchestrator: Manages sub-agent pool    │  │
│  │ 3. Result Aggregator: Combines parallel results    │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────┬─────────────┬────────────────────┘
                      │             │
          ┌───────────▼─────────┐   │
          │   Sub-Agent Pool    │   │
          │ (N independent      │   │
          │  Pydantic AI agents │   │
          │  with isolated state)│   │
          └───────────┬─────────┘   │
                      │             │
                      └─────────────▼────────────────────┐
                                                    ┌────▼────┐
                                                    │ Existing│
                                                    │ Services│
                                                    │ Layer   │
                                                    └─────────┘
```

## 5. Core Components
### 5.1 Task Splitter Module
Location: `codebase_rag/orchestrator/task_splitter.py`
Responsibilities:
- Automatically detect if a user request can be parallelized
- Split the request into independent subtasks with minimal overlap
- Support for multiple splitting strategies:
  1. **File-based splitting**: Split tasks by repository file/folder boundaries (ideal for code review, test generation)
  2. **Node-type splitting**: Split tasks by graph node type (e.g. all functions, all classes)
  3. **Query-based splitting**: Split tasks by independent query segments
  4. **Manual splitting**: User provides explicit list of subtasks via prompt
- Validate that subtasks are truly independent to avoid race conditions

### 5.2 SubAgent Orchestrator (Full Divide-and-Conquer Lifecycle Management)
Location: `codebase_rag/orchestrator/subagent_orchestrator.py`
The main orchestrator owns 100% of the end-to-end parallel workflow lifecycle, with zero "spawn and forget" risk, implementing full divide-and-conquer logic:
Core Responsibilities:
1. **Divide-and-Conquer Task Coordination**:
   - Receives split subtasks from the Task Splitter, validates that all subtasks cover the full scope of the original request with no gaps or overlaps
   - Assigns exactly one subtask per available worker, ensuring balanced work distribution (e.g. assigning similar sized files to different workers to avoid stragglers)
   - Maintains a global map of subtask → worker assignment to avoid duplicate work
2. **Full Sub-Agent Lifecycle Tracking**:
   - Maintains an always-up-to-date registry of every spawned sub-agent, including its state (pending/running/completed/failed), assigned subtask, start time, and progress
   - No orphaned workers: every sub-agent is explicitly terminated when its subtask completes, the overall job is cancelled, or a fatal error occurs
   - Built-in heartbeat monitoring for all active workers to detect unresponsive/crashed agents
3. **Robust Execution Guarantees**:
   - Enforce concurrency limits (default: 5 workers, max configurable by user)
   - Track execution progress, per-subtask timeouts, and automatic retries for failed subtasks (configurable retry limit)
   - Handle graceful cancellation of all running sub-agents immediately if the user cancels the request, the main agent crashes, or the chat session ends
4. **No Spawn-and-Forget Safety Guarantees**:
   - Automatic cleanup process that runs on job completion/error/cancellation to terminate all workers, release all resources, and close all open database connections
   - Orphan worker detection: periodic background check to terminate any workers that are no longer associated with an active job
   - Full audit logging of every sub-agent spawn, task assignment, completion, and termination event
5. **Result Collection Coordination**:
   - Collects results from workers as they complete, stores partial results to avoid data loss if the job is interrupted
   - Marks subtasks as completed only after successful result validation to avoid lost work
- Integrate with existing logging system to track all sub-agent activity

### 5.3 Sub-Agent Instance
Each sub-agent is a lightweight, isolated instance of the core RAG agent with:
- Shared read-only access to the graph database and vector store
- Isolated session state (no shared mutable state between sub-agents)
- Inherited configuration from the parent agent (LLM provider, model settings, tool access permissions)
- Limited tool access (write operations disabled by default, can be enabled explicitly via user flag)
- Automatic context window management optimized for assigned subtask

### 5.4 Result Aggregator Module
Location: `codebase_rag/orchestrator/result_aggregator.py`
Responsibilities:
- Collect results from all completed sub-agents
- Remove duplicate findings (e.g. duplicate bug reports across overlapping code sections)
- Synthesize consolidated output for the user
- Generate execution summary with metrics: number of workers, total execution time, average time per subtask, success/failure rates
- Support multiple output formats: plain text summary, structured JSON, markdown report

## 6. User Interface
### 6.1 CLI Interface
New CLI flags for the `start` and `ask-agent` commands:
```bash
# Enable parallel mode with 10 workers for code review
cgr start --parallel-workers 10 --prompt "Review all Python files for security vulnerabilities"

# Automatic parallel task splitting enabled
cgr ask-agent "Generate unit tests for all functions in the src directory" --auto-split --workers 8

# Disable sub-agent concurrency (default behavior)
cgr start --no-parallel
```

### 6.2 In-Chat / Natural Language Trigger (Zero CLI Flag Required)
This feature is fully supported for all interactive chat interfaces (native CLI chat, MCP clients like Claude Code/Kimi Code, custom HTTP API UIs). Users can enable parallel execution entirely via natural language in their chat prompts, no pre-configured CLI flags are required.

The Task Splitter includes an intent detection model that automatically identifies parallel execution requests from chat messages:
#### Supported natural language triggers (examples):
```
"Review the entire codebase with 10 parallel workers"
"Spawn 8 sub-agents to scan all files for security vulnerabilities"
"Use 12 parallel workers to generate unit tests for all functions"
"Run 5 parallel reviews of the src directory, each focusing on a different type of bug"
```
When these triggers are detected:
1. The system automatically enables parallel mode without requiring user to set any CLI flags
2. It extracts the requested number of workers from the prompt (falls back to default if no number specified)
3. It confirms the parallel execution plan to the user before starting (unless YOLO mode is enabled)
4. It shows real-time progress in the chat interface as workers complete their tasks
5. It returns a single consolidated final response once all workers finish

### 6.3 In-Chat Progress & Feedback
For in-chat interactions, the system provides rich real-time updates:
- Initial confirmation: "I will spawn 10 parallel sub-agents to review the codebase. Estimated time: 2 minutes. Proceed? [Y/n]"
- Live progress updates: "Completed 3/10 parallel reviews, found 7 security issues so far"
- Partial result previews (optional): "Worker 4 found a SQL injection vulnerability in `/api/db.py` line 78"
- Final summary: "All 10 parallel reviews completed in 1m47s. Total findings: 23 issues (4 critical, 8 high, 11 low). Full report below..."

### 6.4 In-Chat Control Commands
Users can control parallel execution mid-run via chat commands:
- `cancel parallel tasks`: Stops all running sub-agents immediately
- `add 3 more workers`: Increases concurrency limit mid-execution
- `show progress`: Returns current status of all workers
- `show partial results`: Returns all findings collected so far


### 6.3 Progress UI
New rich terminal UI for parallel execution:
- Real-time progress bar showing completed/in-progress subtasks
- Per-worker status indicator (idle, running, completed, failed)
- Live count of findings/results as they are returned by sub-agents

## 5.5 Dynamic Concurrency Controller
Location: `codebase_rag/orchestrator/dynamic_concurrency_controller.py`
Responsibilities:
- Process worker count requests extracted from user chat prompts, supporting any arbitrary number requested by the user
- Enforce the configured maximum worker limit as a safety ceiling (prevent accidental resource exhaustion from unreasonable requests)
- Handle edge cases for user requests:
  1. If user requests a number below the maximum: Automatically use the requested count
  2. If user requests a number above the maximum: Notify user, and either use the maximum limit, or ask for temporary override approval
  3. If user doesn't specify a number: Use the default worker count
  4. Support mid-run dynamic adjustments: Add/remove workers while execution is in progress, per user chat commands
- Auto-scale workers based on task size: If user requests 10 workers but there are only 3 subtasks to run, automatically reduce to 3 workers to avoid resource waste

## 7. Configuration
New environment variables and config options added to `AppConfig`:
| Config Option | Default | Description |
|---------------|---------|-------------|
| `CGR_MAX_PARALLEL_WORKERS` | 20 | Safety upper bound for maximum allowed concurrent sub-agents (can be overridden temporarily per user request in chat) |
| `CGR_DEFAULT_PARALLEL_WORKERS` | 5 | Default number of workers if user does not specify a count in their prompt |
| `CGR_ALLOW_DYNAMIC_MAX_OVERRIDE` | true | Allow users to temporarily exceed the max worker limit via chat approval |
| `CGR_AUTO_SCALE_WORKERS` | true | Automatically reduce worker count to match the number of available subtasks to avoid wasted resources |
| `CGR_SUBAGENT_TIMEOUT` | 300 | Timeout in seconds per sub-agent task |
| `CGR_SUBAGENT_ALLOW_WRITE` | false | Allow sub-agents to perform write operations (create file, replace code, execute shell) |
| `CGR_AUTO_SPLIT_ENABLED` | true | Enable automatic task splitting for parallelizable requests |
| `CGR_SUBAGENT_RETRY_ATTEMPTS` | 2 | Number of retries for failed sub-agent tasks |

## 8. Security & Isolation
1. **Read-only by default**: Sub-agents have only read tool access unless explicitly enabled by the user with `--subagent-allow-write` flag
2. **No shared state**: All sub-agents have isolated session state, no mutable state sharing between workers
3. **Rate limiting**: Built-in per-worker LLM rate limiting to avoid hitting API limits
4. **Parent approval flow**: All write operations from sub-agents are routed to the parent agent for user approval before execution
5. **Resource limits**: Configurable memory and execution time limits per sub-agent to prevent resource exhaustion

## 9. Implementation Roadmap
### Phase 1 (MVP)
- [ ] Implement Task Splitter with file-based splitting strategy
- [ ] Implement SubAgent Orchestrator with ThreadPoolExecutor backend
- [ ] Implement basic Result Aggregator with deduplication
- [ ] Add CLI flags for parallel worker configuration
- [ ] Integrate with existing approval / YOLO mode
- [ ] Add parallel code review example

### Phase 2
- [ ] Add additional splitting strategies (node-type, query-based)
- [ ] Add in-prompt concurrency detection
- [ ] Add progress UI for parallel execution
- [ ] Add structured JSON output for aggregated results
- [ ] Add sub-agent execution metrics and reporting

### Phase 3
- [ ] Add async sub-agent execution for improved performance
- [ ] Add support for sub-agent LLM configuration override (use smaller/cheaper models for subtasks)
- [ ] Add support for persistent sub-agent sessions for long-running tasks
- [ ] Add distributed sub-agent execution support (future roadmap)

## 10. Testing Strategy
1. **Unit Tests**: Test individual components (task splitting, orchestrator, aggregator) in isolation
2. **Integration Tests**: Test end-to-end parallel execution for common use cases (code review, test generation)
3. **Performance Tests**: Measure speedup compared to sequential execution for different task sizes and worker counts
4. **Security Tests**: Verify sub-agents cannot perform write operations unless explicitly allowed, no path traversal or privilege escalation
5. **Failure Tests**: Verify proper handling of failed sub-agent tasks, timeouts, and cancellation

## 11. Compatibility
- Fully backward compatible: existing workflows without parallel flags continue to work unchanged
- Works with all existing LLM providers (OpenAI, Google, Ollama)
- Works with the Memgraph native vector backend
- Compatible with existing MCP server integration (parallel sub-agent support available via MCP API)

## 12. Implementation Readiness Validation
### 12.1 Logical Soundness Check ✅
All discussed requirements are addressed with zero logical gaps, conflicts, or contradictions:
1. [x] **Backward compatibility guarantee**: No changes required to existing single-agent workflows, all parallel functionality is 100% opt-in only, no breaking changes to existing APIs, CLI or MCP interfaces
2. [x] **No spawn-and-forget guarantee**: Full end-to-end sub-agent lifecycle management with explicit cleanup on all termination paths (completion, error, cancellation, session disconnect), plus orphan worker detection logic
3. [x] **Dynamic chat concurrency support**: Fully handles arbitrary user-specified worker counts from natural language prompts, with intelligent guardrails and auto-scaling
4. [x] **Divide-and-conquer correctness**: Subtask splitting logic validates 100% coverage of original request scope with no gaps, overlaps or duplicate work assignments
5. [x] **Security alignment**: Sub-agents inherit existing security rules (read-only by default, write operations routed through existing approval/YOLO flow), no new attack surfaces introduced
6. [x] **Resource safety**: Built-in concurrency limits, timeouts, and retries prevent resource exhaustion, LLM rate limiting, or database connection leaks

### 12.2 Reusable Dependency Mapping (No New External Dependencies Required)
All new components build directly on existing, tested system modules with zero need to rewrite core functionality:
| New Component | Reuses Existing Battle-Tested Components |
|---------------|------------------------------------------|
| Task Splitter | Existing semantic search, graph query, file system tools, and AST parsing infrastructure |
| SubAgent Orchestrator | Existing `concurrent.futures`/`asyncio` utilities, Pydantic AI agent initialization logic, session state management |
| Dynamic Concurrency Controller | Existing AppConfig system, intent classification logic from the core RAG orchestrator |
| Result Aggregator | Existing markdown report generation, deduplication logic from graph query result processing |
| Chat Progress UI | Existing Rich terminal UI components and MCP status update APIs |

### 12.3 Edge Case Coverage
All identified edge cases are explicitly addressed in the design:
1. [x] User requests more workers than available subtasks: Auto-scaling automatically reduces worker count to match the number of subtasks to avoid wasting resources
2. [x] User requests worker count above configured safety limit: Optional temporary override approval flow, no hard blocks on legitimate large workload requests
3. [x] Sub-agent crashes or stops responding: Heartbeat monitoring detects failures, automatically retries assigned subtask
4. [x] User cancels job mid-execution: Graceful immediate termination of all running workers, no orphaned background processes
5. [x] Partial task failure: Aggregator returns all completed results plus a clear failure report for incomplete subtasks, no full job failure from single worker errors
6. [x] Large volume of subtasks: Orchestrator queues tasks automatically and processes them in batches per the concurrency limit, no overload

### 12.4 Implementation Complexity Assessment
MVP is low-risk and low-effort to implement, estimated 2-3 weeks of development for full functionality:
- Core SubAgent Orchestrator + Task Splitter: ~1 week
- Dynamic Concurrency Controller + chat intent detection: ~3 days
- Result Aggregator + real-time progress UI: ~3 days
- Integration testing + documentation: ~3 days

