# Design Spec: Query Flow and Data Modeling Fixes

**Spec ID:** 26-04-20-202201_query-flow-data-modeling-fix_f8a2
**Target System:** Code-Graph-RAG Orchestrator and Query Router
**Generated:** 2026-04-20T21:30:00Z
**Priority:** P0 (Critical) — Query routing and subtask execution failures
**Status:** 🔄 Ready for Implementation

---

## 1. Executive Summary

Analysis of the logs reveals critical issues in the query flow and data modeling that cause complete parallel execution failure (9/9 subtasks failed with 404 errors) and inefficient query routing. The root causes are:

1. **Query Mode Mismatch**: System defaults to `CODE_ONLY` mode for document-only repositories
2. **Subtask Over-Generation**: Task splitter creates file-based subtasks for ALL files (including markdown, JSON) instead of respecting file type semantics
3. **Worker LLM Configuration Gap**: Single LLM config creates false parallelism with 1 worker doing all work
4. **Error Classification Deficiency**: 404 errors from model unavailability aren't distinguished from other 404s

---

## 2. Issues Inventory

### 2.1 P0 — Query Mode Auto-Detection Failure

| Field | Value |
|-------|-------|
| **ID** | BUG-QF-001 |
| **Severity** | Blocking |
| **Category** | Query Routing / Data Modeling |
| **Symptom** | Log shows `Query Mode: code_only` but repository has 0 code files and 6 document files. User's question "Why categorical think important?" is clearly document-focused but routed to code graph. |
| **Root Cause** | `main.py` initializes with `QueryMode.CODE_ONLY` default (line 1761, 2763, 2940). No auto-detection based on repository content. `QueryRouter` has 5 modes but no mode selection logic based on graph statistics. |
| **Evidence** | Log: `Found 0 code files... Document graph (if enabled) may still contain data`; `Query Mode: code_only`; All 9 subtasks failed on code graph queries. |
| **Impact** | 100% query failure on document-only repositories; unnecessary subtask generation; wasted API calls. |

**Fix:**
```python
# File: codebase_rag/main.py
# Add repository content analysis for default query mode selection

def _determine_default_query_mode(
    code_graph: MemgraphIngestor | None,
    doc_graph: MemgraphIngestor | None,
) -> QueryMode:
    """Determine default query mode based on repository content statistics.
    
    Priority:
    1. If only document graph has data -> DOCUMENT_ONLY
    2. If only code graph has data -> CODE_ONLY
    3. If both have data -> BOTH_MERGED
    4. If neither has data -> CODE_ONLY (fallback)
    """
    code_count = 0
    doc_count = 0
    
    if code_graph:
        try:
            result = code_graph.fetch_all(
                "MATCH (n) WHERE labels(n)[0] IN ['Function', 'Class', 'Method'] RETURN count(n) as count"
            )
            code_count = result[0].get("count", 0) if result else 0
        except Exception:
            pass
    
    if doc_graph:
        try:
            result = doc_graph.fetch_all(
                "MATCH (d:Document) RETURN count(d) as count"
            )
            doc_count = result[0].get("count", 0) if result else 0
        except Exception:
            pass
    
    if code_count == 0 and doc_count > 0:
        logger.info(f"Auto-selecting DOCUMENT_ONLY mode (docs: {doc_count}, code: {code_count})")
        return QueryMode.DOCUMENT_ONLY
    elif code_count > 0 and doc_count == 0:
        logger.info(f"Auto-selecting CODE_ONLY mode (code: {code_count}, docs: {doc_count})")
        return QueryMode.CODE_ONLY
    elif code_count > 0 and doc_count > 0:
        logger.info(f"Auto-selecting BOTH_MERGED mode (code: {code_count}, docs: {doc_count})")
        return QueryMode.BOTH_MERGED
    else:
        logger.info("No graph data found, defaulting to CODE_ONLY")
        return QueryMode.CODE_ONLY
```

---

### 2.2 P0 — Task Splitter Creates Subtasks for Non-Code Files

| Field | Value |
|-------|-------|
| **ID** | BUG-QF-002 |
| **Severity** | Blocking |
| **Category** | Task Splitting / Data Modeling |
| **Symptom** | Log shows `Split task into 9 file-based subtasks` for a document query. The 9 files include: 5 markdown files, 1 JSON file, 1 hash cache file, and 2 temp directories. |
| **Root Cause** | `TaskSplitter._collect_scoped_files()` uses `get_all_code_files()` which returns ALL files (line 238), not just code files. No filtering by file type relevance to the query mode. |
| **Evidence** | Log: `Using all 9 code files (no scope paths or type hints found)` — but these are not code files. |
| **Impact** | Wasteful subtask generation; parallel execution overhead for files that can't be analyzed by code-focused subagents. |

**Fix:**
```python
# File: codebase_rag/orchestrator/task_splitter.py
# Add query-mode-aware file filtering

from codebase_rag.shared.query_router import QueryMode

class TaskSplitter:
    def __init__(self, repo_path: str | None = None, query_mode: QueryMode = QueryMode.CODE_ONLY):
        self.repo_path = Path(repo_path or settings.TARGET_REPO_PATH).resolve()
        self.query_mode = query_mode
    
    def _collect_scoped_files(self, prompt: str) -> list[Path]:
        """Enhanced file collection with query-mode-aware filtering."""
        # ... existing scope path and type hint logic ...
        
        # Strategy 3: Fallback to relevant files based on query mode
        all_files = get_all_code_files(self.repo_path)
        
        # Filter files based on query mode
        if self.query_mode == QueryMode.CODE_ONLY:
            # Only code files
            relevant_files = [f for f in all_files if self._is_code_file(f)]
        elif self.query_mode == QueryMode.DOCUMENT_ONLY:
            # Only document files
            relevant_files = [f for f in all_files if self._is_document_file(f)]
        else:
            # BOTH_MERGED or validation modes - include both
            relevant_files = all_files
        
        if not relevant_files:
            logger.warning(
                f"No relevant files found for query mode {self.query_mode}. "
                f"Consider switching query mode or indexing the repository."
            )
        
        logger.info(f"Using {len(relevant_files)} relevant files for {self.query_mode}")
        return relevant_files
    
    def _is_code_file(self, path: Path) -> bool:
        """Check if file is a code file based on extension."""
        code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.h', '.hpp',
            '.go', '.rs', '.cs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.c', '.lua', '.sol', '.vy'
        }
        return path.suffix.lower() in code_extensions
    
    def _is_document_file(self, path: Path) -> bool:
        """Check if file is a document file based on extension."""
        doc_extensions = {'.md', '.rst', '.txt', '.pdf', '.docx'}
        return path.suffix.lower() in doc_extensions
```

---

### 2.3 P0 — SubAgentOrchestrator False Parallelism

| Field | Value |
|-------|-------|
| **ID** | BUG-QF-003 |
| **Severity** | High |
| **Category** | Concurrency / Data Modeling |
| **Symptom** | Log shows `Using 5 workers` but then `Sub-agent pool initialized with 1 worker LLMs`. All 9 subtasks are processed sequentially by 1 worker. |
| **Root Cause** | `SubAgentOrchestrator.initialize_agents()` reduces worker count to 1 when only 1 LLM config is available (lines 369-374). This is correct behavior but the UX is confusing — user thinks they're getting parallelism. |
| **Evidence** | Log: `Only 1 LLM config available. Reducing worker_count from 5 to 1 to avoid false parallelism contention.` |
| **Impact** | User confusion; wasted overhead from thread pool setup; 9 sequential API calls instead of parallel. |

**Fix:**
```python
# File: codebase_rag/orchestrator/subagent_orchestrator.py
# In execute_tasks method, add early exit for non-parallel scenarios

def execute_tasks(
    self,
    subtasks: list[dict[str, Any]],
    result_aggregator: ResultAggregator | None = None,
    retry_attempts: int | None = None,
    dry_run: bool = False,
) -> ResultAggregator:
    """Execute subtasks with early sequential fallback for single-worker scenarios."""
    
    # Initialize agents first to determine actual worker count
    self.initialize_agents()
    
    # If only 1 worker available, skip thread pool overhead and execute sequentially
    if self.worker_count == 1 and not dry_run:
        logger.info(
            f"Single worker mode: executing {len(subtasks)} subtasks sequentially "
            f"(avoiding thread pool overhead)"
        )
        return self._execute_sequentially(subtasks, result_aggregator, retry_attempts)
    
    # Continue with parallel execution for multi-worker scenarios
    # ... existing parallel execution code ...

def _execute_sequentially(
    self,
    subtasks: list[dict[str, Any]],
    result_aggregator: ResultAggregator | None = None,
    retry_attempts: int | None = None,
) -> ResultAggregator:
    """Execute subtasks sequentially when only 1 worker is available."""
    retry_attempts = retry_attempts or settings.CGR_SUBAGENT_RETRY_ATTEMPTS
    result_aggregator = result_aggregator or ResultAggregator()
    result_aggregator.set_total_subtasks(len(subtasks))
    
    worker = self.workers[0] if self.workers else None
    if not worker:
        raise RuntimeError("No workers available for sequential execution")
    
    start_time = time.time()
    
    for subtask in subtasks:
        if self._shutdown_called.get():
            break
        
        execution_start = time.time()
        try:
            result, exec_time = worker.execute(subtask)
            result_aggregator.add_result(
                subtask,
                result,
                execution_time=time.time() - execution_start,
                status="completed",
                worker_metadata={
                    "worker_id": worker.worker_id,
                    "execution_time": exec_time,
                    "status": "completed",
                },
            )
        except Exception as e:
            result_aggregator.add_error(
                subtask,
                str(e),
                execution_time=time.time() - execution_start,
                worker_metadata={"worker_id": worker.worker_id, "status": "failed"},
            )
    
    result_aggregator.set_total_execution_time(time.time() - start_time)
    return result_aggregator
```

---

### 2.4 P1 — Error Classification Lacks Model Unavailability Distinction

| Field | Value |
|-------|-------|
| **ID** | BUG-QF-004 |
| **Severity** | High |
| **Category** | Error Handling / Data Modeling |
| **Symptom** | All 9 subtasks fail with identical 404 errors: `{'message': 'The requested resource was not found', 'type': 'resource_not_found_error'}`. Error classification treats these as `model_unavailable` but logs don't indicate if this is a model ID issue, endpoint issue, or auth issue. |
| **Root Cause** | `_classify_error()` in subagent_orchestrator.py (line 589-602) groups all 404s as `model_unavailable`. But 404s can mean: unknown model ID, invalid endpoint URL, or resource temporarily unavailable. These need different handling. |
| **Evidence** | Log: `Subtask subtask_0 failed: status_code: 404, model_name: k2.6-code-preview` — but no detail on what resource was not found. |
| **Impact** | Can't distinguish between configuration errors (should fail fast) and transient issues (should retry). |

**Fix:**
```python
# File: codebase_rag/orchestrator/subagent_orchestrator.py
# Enhance error classification with detailed error types

from enum import StrEnum

class ErrorType(StrEnum):
    """Detailed error classification for subtask failures."""
    MODEL_NOT_FOUND = "model_not_found"  # Unknown model ID
    ENDPOINT_NOT_FOUND = "endpoint_not_found"  # Invalid API endpoint
    RESOURCE_UNAVAILABLE = "resource_unavailable"  # Temporarily unavailable
    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    NETWORK_ERROR = "network_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"

def _classify_error(self, error_msg: str, status_code: int | None = None) -> ErrorType:
    """Classify error with detailed type for appropriate handling."""
    error_lower = error_msg.lower()
    
    # Check for specific 404 variants
    if status_code == 404 or "404" in error_msg:
        if "model" in error_lower and any(x in error_lower for x in ["not found", "unknown", "invalid"]):
            return ErrorType.MODEL_NOT_FOUND
        elif "endpoint" in error_lower or "url" in error_lower:
            return ErrorType.ENDPOINT_NOT_FOUND
        else:
            return ErrorType.RESOURCE_UNAVAILABLE
    
    if "rate_limit" in error_lower or "429" in error_msg:
        return ErrorType.RATE_LIMIT
    if any(x in error_msg for x in ["401", "403", "auth", "unauthorized"]):
        return ErrorType.AUTH_ERROR
    if any(x in error_lower for x in ["connection", "network", "dns"]):
        return ErrorType.NETWORK_ERROR
    if "timeout" in error_lower:
        return ErrorType.TIMEOUT
    
    return ErrorType.UNKNOWN

def _should_retry(self, error_type: ErrorType, retry_count: int, max_retries: int) -> bool:
    """Determine if error type supports retry."""
    # Never retry these - they're configuration errors
    if error_type in (ErrorType.MODEL_NOT_FOUND, ErrorType.ENDPOINT_NOT_FOUND, ErrorType.AUTH_ERROR):
        return False
    
    # Always retry these if under limit
    if error_type in (ErrorType.RATE_LIMIT, ErrorType.NETWORK_ERROR, ErrorType.TIMEOUT):
        return retry_count < max_retries
    
    # Conditionally retry resource unavailable
    if error_type == ErrorType.RESOURCE_UNAVAILABLE:
        return retry_count < max_retries // 2  # Fewer retries for 404s
    
    return False
```

---

### 2.5 P1 — Concurrency Eligibility Ignores Query Mode

| Field | Value |
|-------|-------|
| **ID** | BUG-QF-005 |
| **Severity** | Medium |
| **Category** | Query Flow / Data Modeling |
| **Symptom** | Log shows `Task eligible for parallel execution: llm_rejected (confidence: 0.95)` but the task is a simple document question that should use DOCUMENT_ONLY mode with direct semantic search, not parallel file analysis. |
| **Root Cause** | `ConcurrencyEligibilityClassifier` doesn't consider query mode. Document-only queries for conceptual questions shouldn't be parallelized — they should use vector search. |
| **Evidence** | User asked "Why categorical think important?" — a conceptual question. System classified it as eligible for parallel execution with 9 file subtasks, but this is the wrong approach entirely. |
| **Impact** | Wrong execution strategy for document queries; unnecessary API costs; poor user experience. |

**Fix:**
```python
# File: codebase_rag/orchestrator/concurrency_eligibility_classifier.py
# Add query mode awareness to eligibility check

from codebase_rag.shared.query_router import QueryMode

async def is_eligible(
    self,
    prompt: str,
    subtask_count: int | None = None,
    has_write_operations: bool = False,
    query_mode: QueryMode = QueryMode.CODE_ONLY,
) -> tuple[bool, str, float]:
    """Determine eligibility with query mode awareness."""
    
    # DOCUMENT_ONLY mode for conceptual questions should NOT use parallel file analysis
    if query_mode == QueryMode.DOCUMENT_ONLY:
        # Check if this is a conceptual question (not asking about specific files)
        if self._is_conceptual_question(prompt):
            logger.info(
                "Task not eligible for parallel execution: DOCUMENT_ONLY mode "
                "with conceptual question - use semantic search instead"
            )
            return False, "document_conceptual_query", 0.0
    
    # ... rest of existing eligibility checks ...

def _is_conceptual_question(self, prompt: str) -> bool:
    """Detect if question is conceptual rather than file-specific."""
    conceptual_patterns = [
        r"\b(what is|what are|why|how to|explain|describe|what does)\b",
        r"\b(importance of|benefits of|purpose of|meaning of)\b",
        r"\b(categorical thinking|concept|theory|framework|methodology)\b",
    ]
    
    # Check for file-specific references
    file_patterns = [
        r"\b(file|function|class|method)\s+\w+",
        r"\b(in|from)\s+[\w/]+\.(py|js|ts|java|cpp)\b",
        r"```[\w/]+```",  # Code blocks with paths
    ]
    
    has_conceptual = any(re.search(p, prompt, re.I) for p in conceptual_patterns)
    has_file_ref = any(re.search(p, prompt, re.I) for p in file_patterns)
    
    return has_conceptual and not has_file_ref
```

---

### 2.6 P2 — QueryRouter Lacks Mode-Specific Optimization Hints

| Field | Value |
|-------|-------|
| **ID** | BUG-QF-006 |
| **Severity** | Low |
| **Category** | Query Router / Data Modeling |
| **Symptom** | QueryRouter executes the same query path regardless of graph content. No optimization hints are exposed to callers about which graph has more relevant data. |
| **Root Cause** | `QueryResponse` doesn't include metadata about result quality or source graph statistics. Callers can't make informed decisions about mode switching. |
| **Impact** | Suboptimal query routing; can't auto-switch modes based on results. |

**Fix:**
```python
# File: codebase_rag/shared/query_router.py
# Add query statistics to response

@dataclass
class QueryResponse:
    """Query response with clear source attribution and statistics."""
    
    answer: str
    sources: list[Source]
    mode: QueryMode
    validation_report: ValidationReport | None = None
    warnings: list[str] = field(default_factory=list)
    
    # NEW: Query execution statistics
    stats: QueryStats = field(default_factory=lambda: QueryStats())

def _query_code_only(self, request: QueryRequest) -> QueryResponse:
    """Query code graph with statistics."""
    start_time = time.time()
    
    # ... existing query logic ...
    
    response = QueryResponse(
        answer="...",
        sources=sources,
        mode=request.mode,
        warnings=warnings,
        stats=QueryStats(
            execution_time_ms=(time.time() - start_time) * 1000,
            nodes_queried=len(sources),
            graph_type="code",
            result_confidence=self._calculate_confidence(sources),
        ),
    )
    return response
```

---

## 3. Implementation Plan

### Phase 1: Query Mode Auto-Detection (P0)

| Step | Action | File | Validation |
|------|--------|------|------------|
| 1.1 | Add `_determine_default_query_mode()` | `main.py` | Correct mode selected for doc-only repo |
| 1.2 | Update `_initialize_services_and_agent()` to use auto-detection | `main.py` | Mode passed to agent init |
| 1.3 | Update interactive loop to respect auto-detected mode | `main.py` | User sees correct mode in table |

### Phase 2: Task Splitter Query Mode Awareness (P0)

| Step | Action | File | Validation |
|------|--------|------|------------|
| 2.1 | Add query_mode parameter to TaskSplitter | `task_splitter.py` | Constructor accepts mode |
| 2.2 | Add file type filtering methods | `task_splitter.py` | `_is_code_file()`, `_is_document_file()` |
| 2.3 | Update `_collect_scoped_files()` with mode filtering | `task_splitter.py` | Only relevant files returned |
| 2.4 | Update all TaskSplitter instantiations | `main.py`, others | Mode passed to splitter |

### Phase 3: Sequential Fallback for Single Worker (P0)

| Step | Action | File | Validation |
|------|--------|------|------------|
| 3.1 | Add `_execute_sequentially()` method | `subagent_orchestrator.py` | Sequential execution works |
| 3.2 | Add early check in `execute_tasks()` | `subagent_orchestrator.py` | Single worker skips thread pool |
| 3.3 | Update logging for clarity | `subagent_orchestrator.py` | Clear "sequential mode" message |

### Phase 4: Enhanced Error Classification (P1)

| Step | Action | File | Validation |
|------|--------|------|------------|
| 4.1 | Add `ErrorType` enum | `subagent_orchestrator.py` | Enum values defined |
| 4.2 | Update `_classify_error()` with detailed types | `subagent_orchestrator.py` | Correct type returned for each error |
| 4.3 | Add `_should_retry()` method | `subagent_orchestrator.py` | Retry logic uses new types |
| 4.4 | Update retry logic in `execute_tasks()` | `subagent_orchestrator.py` | Model errors fail fast, network errors retry |

### Phase 5: Concurrency Eligibility Query Mode Awareness (P1)

| Step | Action | File | Validation |
|------|--------|------|------------|
| 5.1 | Add query_mode parameter to `is_eligible()` | `concurrency_eligibility_classifier.py` | Parameter accepted |
| 5.2 | Add `_is_conceptual_question()` helper | `concurrency_eligibility_classifier.py` | Correctly identifies conceptual questions |
| 5.3 | Add DOCUMENT_ONLY conceptual check | `concurrency_eligibility_classifier.py` | Document queries skip parallelization |
| 5.4 | Update call sites to pass query_mode | `main.py` | Mode passed from context |

---

## 4. Validation Criteria

| Criterion | Check | Pass Condition |
|-----------|-------|----------------|
| Auto mode selection | Start with doc-only repo | Defaults to DOCUMENT_ONLY |
| Task splitter filtering | Split in DOCUMENT_ONLY mode | Only .md, .txt files returned |
| Single worker efficiency | Run with 1 LLM config | Sequential execution, no thread pool |
| Error classification | Trigger model not found | ErrorType.MODEL_NOT_FOUND returned |
| Conceptual question handling | Ask "What is X?" in doc mode | Not eligible for parallel execution |

---

## 5. Affected File Inventory

| File | Type | Issue IDs | Action |
|------|------|-----------|--------|
| `codebase_rag/main.py` | Code | QF-001, QF-002, QF-005 | Add mode auto-detection, update splitter init |
| `codebase_rag/orchestrator/task_splitter.py` | Code | QF-002 | Add mode-aware file filtering |
| `codebase_rag/orchestrator/subagent_orchestrator.py` | Code | QF-003, QF-004 | Sequential fallback, error classification |
| `codebase_rag/orchestrator/concurrency_eligibility_classifier.py` | Code | QF-005 | Query mode awareness |
| `codebase_rag/shared/query_router.py` | Code | QF-006 | Add query statistics (optional) |

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Mode auto-detection latency | Medium | Low | Cache graph statistics; async detection |
| File filtering too aggressive | Low | Medium | Keep fallback to all files if filter returns empty |
| Sequential execution slower | Low | Low | Only affects single-LLM configs; parallel was fake anyway |
| Error classification false positives | Medium | Low | Default to unknown type for ambiguous errors |

---

## 7. Testing Strategy

| Test Case | Setup | Expected Behavior |
|-----------|-------|-------------------|
| Doc-only repository | 0 code files, 5 .md files | Auto-selects DOCUMENT_ONLY, 0 subtasks generated |
| Code-only repository | 10 .py files, 0 docs | Auto-selects CODE_ONLY, 10 subtasks generated |
| Mixed repository | 5 .py, 5 .md | Auto-selects BOTH_MERGED |
| Single LLM config | CGR_WORKER_LLMS unset | Sequential execution, no thread pool |
| Invalid model ID | ORCHESTRATOR_MODEL=invalid | Fail fast with MODEL_NOT_FOUND |
| Conceptual doc question | "What is X?" in DOCUMENT_ONLY | Not eligible for parallel execution |

---

*Spec generated from log analysis and codebase review | 2026-04-20T21:30:00Z*
