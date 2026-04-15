# Parallel Sub-Agent/Worker Orchestration Design Specification

## 1. Objective
Make the existing parallel orchestration stack production-ready.

This repository already contains the core modules for parallel execution:
- `codebase_rag/orchestrator/concurrency_eligibility_classifier.py`
- `codebase_rag/orchestrator/task_splitter.py`
- `codebase_rag/orchestrator/subagent_orchestrator.py`
- `codebase_rag/orchestrator/result_aggregator.py`
- integration in `codebase_rag/main.py`
- CLI surface in `codebase_rag/cli.py`

The correct implementation target is therefore not a greenfield design. The target is to harden the current path so that parallel execution is safe, scoped, observable, and functionally equivalent to the read-only parts of the main orchestration flow.

## 2. Verified Current-State Baseline
| Area | Existing Code | Current Behavior | Notes |
|------|---------------|------------------|-------|
| Eligibility detection | `ConcurrencyEligibilityClassifier` | Async classifier with explicit parallel/sequential overrides and write-operation blocking support | `main.py` does not currently pass write/scope context into it |
| Task splitting | `TaskSplitter` | File-based splitting exists; node/query/manual splitting are placeholders | Current file strategy walks all code files in the repo |
| Worker execution | `SubAgentOrchestrator` | Uses `ThreadPoolExecutor`, retries, timeouts, worker LLM assignment, dry-run mode | Default worker implementation is still a placeholder |
| Aggregation | `ResultAggregator` | Tracks results/errors, deduplicates, applies simple conflict resolution, formats markdown/json/text | Does not implement semantic merge or majority voting |
| Runtime integration | `main.py` interactive loop | Auto-checks for parallel eligibility, splits into subtasks, runs orchestrator, appends consolidated result to prompt context | Uses the existing stack but with incomplete safety/wiring |
| Configuration | `AppConfig` | Has concrete `CGR_*` settings for worker count, retries, timeouts, auto-parallelism, worker LLMs | These names must remain the contract |
| CLI surface | `cli.py` | Exposes `--parallel-workers`, `--auto-split`, `--no-parallel`, `--parallel-dry-run`, `--scheduling-strategy` | These flags are not fully wired into runtime behavior |

## 3. Confirmed Gaps To Close
### 3.1 Functional Gaps
1. **Worker behavior is not production-ready**
  The default worker returned by `SubAgentOrchestrator._default_agent_factory()` is `SimpleSubAgent`, which only sleeps briefly and returns a placeholder string. Parallel execution is concurrent, but the worker output is not equivalent to a real read-only sub-agent investigation.

2. **Eligibility is evaluated without enough context**
  `main.py` currently calls `ConcurrencyEligibilityClassifier.is_eligible()` with only the raw prompt. It does not pass:
  - `has_write_operations`
  - a preview `subtask_count`
  - any CLI override state such as `--no-parallel`

3. **Splitting is too broad for real user intent**
  `TaskSplitter._split_by_file()` currently enumerates all code files under the repo path. That is acceptable for a bulk scan MVP, but it is not implementation-ready for prompts that target:
  - a specific folder
  - a user-specified file list
  - a filtered subset inferred from prompt scope

4. **CLI and config switches are partially dead**
  The repo exposes parallel execution flags and config values, but not all of them currently influence the interactive execution path. In particular, the spec must assume these are wired explicitly rather than parsed only.

5. **Worker-count rules are inconsistent across code and docs**
  The current codebase mixes multiple assumptions:
  - `AppConfig` defaults: `CGR_DEFAULT_PARALLEL_WORKERS=20`, `CGR_MAX_PARALLEL_WORKERS=30`
  - `DynamicConcurrencyController`: hard-coded permanent/burst worker architecture of `10 + 10`
  - docs/specs elsewhere mention `5`, `10`, and `20`

6. **Aggregation behavior is overspecified in the current document**
  The current implementation supports deduplication and priority-based conflict selection by target path/entity. It does not yet support semantic overlap detection or majority-vote resolution across equivalent scopes.

7. **Docs and tests have drifted from the live implementation**
  The shipped test and documentation set still assumes older defaults and, in some cases, a synchronous classifier API. The production-ready scope must include alignment of tests and docs with the actual runtime contract.

### 3.2 Non-Goals For This Iteration
- No parallel execution for write or destructive tasks
- No `asyncio.gather` rewrite of the worker executor in v1
- No new `split_task_into_subtasks` tool; use the existing orchestrator classes directly
- No nested parallelism
- No semantic-majority conflict resolution in v1
- No distributed or multi-machine execution
- No node/query/manual splitting beyond explicit MVP extensions listed below

## 4. Design Principles
1. **Backward compatible by default**
  Sequential execution must remain the fallback for unsupported, unsafe, or underspecified tasks.

2. **Read-only parity first**
  The first production-ready target is a read-only sub-agent path that can inspect code, docs, graph results, and search outputs using the same evidence standards as the main orchestrator.

3. **Scope before speed**
  The system must first establish task scope and safety before enabling parallelism.

4. **Single source of truth for configuration**
  `AppConfig` owns the runtime contract. Worker-count logic in controllers, docs, tests, and specs must derive from that contract rather than introducing parallel constants in multiple places.

5. **Observable behavior**
  Parallel execution must emit enough metadata to debug incorrect splits, slow workers, timeouts, and fallback decisions.

## 5. Target Runtime Flow
```
User request
  -> CLI/runtime overrides applied
  -> detect write intent and explicit parallel/sequential intent
  -> preview split (when auto-split is enabled) to estimate safe subtask count
  -> eligibility classifier receives prompt + subtask_count + has_write_operations
  -> if safe and beneficial: execute parallel subtasks with real read-only workers
  -> aggregate results + telemetry
  -> append structured parallel findings to the main response path
  -> otherwise fall back to sequential orchestration
```

## 6. Required Implementation Changes
### 6.1 `codebase_rag/main.py`
Update the interactive execution path so that it:
1. Honors runtime overrides from CLI flags and/or temporary session settings.
2. Computes `has_write_operations` before calling the classifier.
3. Performs a preview split before the classifier decision when auto-splitting is enabled.
4. Passes `subtask_count` and `has_write_operations` into `ConcurrencyEligibilityClassifier.is_eligible()`.
5. Honors explicit sequential overrides such as `--no-parallel`.
6. Honors dry-run mode and scheduling strategy.
7. Uses parallel execution only when there are at least 2 valid subtasks after scope-aware filtering.

### 6.2 `codebase_rag/cli.py`
Wire the existing flags into runtime behavior rather than leaving them at parse time only:
- `--parallel-workers`
- `--auto-split`
- `--no-parallel`
- `--parallel-dry-run`
- `--scheduling-strategy`

If a flag intentionally remains unsupported in interactive mode, remove it from the CLI surface in the same change set rather than keeping a misleading option.

### 6.3 `codebase_rag/orchestrator/subagent_orchestrator.py`
Replace the placeholder worker implementation with a real read-only sub-agent wrapper.

Implementation requirements:
1. The worker must use an actual agent/tool stack, not a sleep-based placeholder.
2. The worker should reuse existing model/provider infrastructure from `codebase_rag/services/llm.py` and existing tool definitions where possible.
3. Worker permissions must remain read-only by default.
4. Dry-run mode must not claim fake investigative results; it should emit execution-plan metadata only.
5. The orchestrator must record per-subtask metadata at minimum:
  - `worker_id`
  - assigned model/provider
  - retry count
  - execution time
  - final status
6. The executor model remains `ThreadPoolExecutor` in this iteration.
7. `CGR_PARALLEL_MAX_QUEUE_SIZE` must be enforced or removed from config if not needed.

### 6.4 `codebase_rag/orchestrator/task_splitter.py`
Keep file-based splitting as the MVP, but make it scope-aware.

Required behavior:
1. Respect explicit path/file hints from the user prompt when present.
2. Limit subtasks to the relevant subset of files instead of always scanning the full repo.
3. Preserve prompt-injection safeguards around literal file paths.
4. Continue to support `extract_worker_llms_from_prompt()`.
5. Treat node/query/manual splitting as out of scope unless implemented in the same PR.

### 6.5 `codebase_rag/orchestrator/concurrency_eligibility_classifier.py`
Retain the async classifier design, but make its contract explicit:
1. It is evaluated only after write-intent detection and optional preview split.
2. Explicit user requests for parallel execution do not override write-safety blocks.
3. It may auto-approve only read-only tasks with at least 2 safe subtasks.
4. Task type labels returned by the classifier must match what tests expect, or tests must be updated to the real contract.

### 6.6 `codebase_rag/orchestrator/dynamic_concurrency_controller.py`
Reconcile worker-count logic with `AppConfig`.

Required outcome:
1. No contradictory hard-coded worker ceilings.
2. Effective worker count is derived from:
  - explicit user override, if safe
  - configured defaults/maxima
  - CPU guardrails
  - subtask count
3. If permanent/burst pools are retained, their sizes must be derived from config or clearly documented as internal implementation details.

### 6.7 `codebase_rag/orchestrator/result_aggregator.py`
Align the implementation contract with the actual merge behavior.

v1 requirements:
1. Deduplicate identical results when enabled.
2. Group results by target scope (path/entity).
3. Surface unresolved conflicts instead of pretending they were fully resolved.
4. Include structured metadata in JSON output for downstream debugging.

Explicitly out of scope for v1:
- semantic conflict detection
- majority voting across overlapping scopes
- automatic rewrite/merge of contradictory textual findings

### 6.8 Documentation And Tests
Update the shipped docs and tests in the same implementation series.

At minimum align:
- `README.md`
- `docs/parallel_sub_agents.md`
- `codebase_rag/tests/test_orchestrator_parallel.py`

## 7. Runtime Configuration Contract
The spec must use the existing `CGR_*` names from `AppConfig`.

| Setting | Current Default In Code | Purpose |
|---------|-------------------------|---------|
| `CGR_DEFAULT_PARALLEL_WORKERS` | `20` | Default worker count when no override is provided |
| `CGR_MAX_PARALLEL_WORKERS` | `30` | Upper bound for user-requested or auto-scaled workers |
| `CGR_ALLOW_DYNAMIC_MAX_OVERRIDE` | `True` | Allows runtime logic to exceed soft limits when explicitly supported |
| `CGR_AUTO_SCALE_WORKERS` | `True` | Allows worker count reduction based on subtask count |
| `CGR_SUBAGENT_TIMEOUT` | `300` | Timeout per subtask |
| `CGR_SUBAGENT_RETRY_ATTEMPTS` | `2` | Retry count for failed subtasks |
| `CGR_SUBAGENT_ALLOW_WRITE` | `False` | Read-only default for workers |
| `CGR_AUTO_PARALLEL_ENABLED` | `True` | Master switch for auto-parallel activation |
| `CGR_PARALLEL_ELIGIBILITY_THRESHOLD` | `0.7` | Confidence threshold for LLM-based eligibility |
| `CGR_AUTO_SPLIT_ENABLED` | `True` | Enables automatic subtask generation |
| `CGR_PARALLEL_MAX_QUEUE_SIZE` | `100` | Maximum queued subtask count if enforced |
| `CGR_WORKER_LLMS` | empty | Optional dedicated worker LLM list |
| `CGR_WORKER_LLM_ASSIGNMENT_STRATEGY` | `round-robin` | Worker LLM assignment strategy |

Any implementation or documentation that cites other setting names such as `MAX_PARALLEL_WORKERS`, `PARALLEL_SUBTASK_TIMEOUT`, or `ENABLE_AUTO_PARALLELISM` is out of date for this repository.

## 8. Acceptance Criteria
### 8.1 Functional Acceptance
1. A safe read-only multi-file task can execute through real sub-agents in parallel.
2. A write task never auto-runs in parallel, even if the user explicitly asks for parallelism.
3. A path-scoped request only spawns subtasks for files inside the requested scope.
4. `--no-parallel` forces sequential behavior.
5. `--parallel-workers`, `--auto-split`, `--parallel-dry-run`, and `--scheduling-strategy` measurably affect runtime behavior.
6. Worker outputs are evidence-based and no longer produced by the placeholder `SimpleSubAgent` implementation.
7. Parallel execution falls back to sequential when fewer than 2 valid subtasks remain after filtering.

### 8.2 Observability Acceptance
1. Consolidated results include total execution time, completed/failed counts, and unresolved conflict counts.
2. JSON output includes per-subtask worker/model metadata.
3. Logs make it clear why parallel execution was activated, skipped, or downgraded to sequential.

### 8.3 Compatibility Acceptance
1. Existing sequential workflows continue to work unchanged.
2. Existing worker LLM configuration support remains intact.
3. Existing retry and timeout behavior remains supported.

## 9. Testing Plan
### 9.1 Unit Tests
- classifier respects explicit parallel/sequential intent plus write-safety gating
- preview split produces bounded, scope-aware file subtasks
- orchestrator retries timed-out or transiently failed tasks
- dry-run mode returns plan metadata, not fake findings
- aggregator emits unresolved conflicts without claiming semantic resolution
- controller derives effective worker counts from config plus subtask count

### 9.2 Integration Tests
- end-to-end read-only parallel analysis over a scoped set of files
- explicit user parallel request is honored for safe tasks
- explicit sequential override disables parallel path
- write-like prompt falls back to sequential path
- CLI flags affect the interactive/runtime execution path

### 9.3 Performance Tests
Use deterministic mock workers for concurrency verification.

Required checks:
1. Parallel execution is materially faster than sequential execution for mocked independent subtasks.
2. Orchestration overhead stays bounded in mock-based tests.

Do not set hard latency ratios against real LLM calls, because network/provider variance makes those assertions unstable.

## 10. Delivery Phases
### Phase 1: Correctness And Safety
1. Replace placeholder worker execution.
2. Pass write/scope context into eligibility detection.
3. Enforce sequential fallback for unsafe or underspecified tasks.

### Phase 2: Scope And Runtime Wiring
1. Make file splitting scope-aware.
2. Wire CLI/runtime overrides into execution.
3. Reconcile worker-count logic with config.

### Phase 3: Observability And Alignment
1. Extend per-subtask metadata.
2. Align aggregator contract with implementation.
3. Update docs and tests to the final behavior.

## 11. Success Definition
This work is complete when the repository has one coherent story for parallel execution:
- the spec matches the code
- the code matches the CLI and docs
- the runtime uses real sub-agents instead of placeholders
- unsafe tasks stay sequential
- parallel tasks are actually scoped, observable, and test-covered
