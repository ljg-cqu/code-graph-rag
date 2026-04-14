# Parallel Task Orchestration System v2 Design Specification
**Version**: 1.0
**Date**: 2024-04-14
**Status**: Final
**Author**: Code Graph RAG Team
**Depends On**: [001-context-window-management-system.md](./001-context-window-management-system.md)

## Revision History
| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2024-04-14 | Initial complete design | Team |

## 1. Purpose
This document specifies the design for version 2 of the Parallel Task Orchestration System, which addresses critical limitations in the current implementation:
1. Fix low-confidence parallel task execution (current threshold of 0.8 is rarely met, leading to underutilization of parallel processing)
2. Improve task splitting accuracy to eliminate overlapping or missing subtasks
3. Add dynamic worker scaling to optimize resource usage based on workload size
4. Implement better result aggregation and deduplication to reduce redundant work
5. Add support for per-subtask model selection to optimize cost/performance tradeoffs

This design builds directly on the Context Window Management System from Specification 001, leveraging per-model context window sizes to optimize subtask batching.

## 2. Scope
### 2.1 In Scope
- Enhanced parallel task eligibility detection with improved confidence calibration
- Dynamic worker scaling that adjusts parallelism based on subtask count and available resources
- Smart task splitting that uses code graph structure to create non-overlapping, atomic subtasks
- Hierarchical result aggregation with deduplication and conflict resolution
- Per-subtask model selection support to use smaller/cheaper models for simple tasks
- Configurable parallel execution parameters via environment variables
- Complete integration with existing context compression system
- Detailed metrics and logging for parallel execution performance

### 2.2 Out of Scope
- Distributed execution across multiple nodes (single process only)
- Custom task splitting rules defined by users
- Support for long-running asynchronous tasks
- Checkpointing and resuming interrupted parallel execution runs

## 3. Requirements
### 3.1 Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | System shall automatically detect parallel-eligible tasks with minimum 70% confidence threshold (down from 80%) | Critical |
| FR2 | System shall calibrate confidence scores dynamically based on past execution success rates | High |
| FR3 | Task splitter shall generate non-overlapping, atomic subtasks using code graph structure information | High |
| FR4 | System shall dynamically scale worker count between 1 and `CGR_MAX_PARALLEL_WORKERS` based on subtask count | High |
| FR5 | System shall support per-subtask model selection to use smaller/cheaper models for simple tasks (e.g. filtering, sorting) | Medium |
| FR6 | Result aggregator shall deduplicate overlapping results and resolve conflicts between subtask outputs | High |
| FR7 | Users shall be able to configure all parallel execution parameters via environment variables | High |
| FR8 | System shall collect and log detailed metrics for parallel execution: total time, subtask count, success rate, confidence scores | Medium |
| FR9 | System shall fall back to sequential execution automatically if parallel processing fails or confidence is too low | Critical |

### 3.2 Non-Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| NFR1 | Parallel execution shall provide minimum 2x speedup over sequential execution for eligible tasks | Critical |
| NFR2 | Task splitting shall add <100ms overhead to request processing | High |
| NFR3 | Result aggregation shall add <200ms overhead per 10 subtasks | Medium |
| NFR4 | All changes shall be backwards compatible (existing parallel execution configuration continues to work) | Critical |
| NFR5 | System shall handle up to 50 subtasks in a single parallel run without performance degradation | Medium |

## 4. System Architecture
### 4.1 Component Overview
The v2 Parallel Task Orchestration System consists of 5 new/updated components:

```
┌─────────────────────────┐
│  Eligibility Classifier │
│  (confidence calibration,│
│   task detection)        │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│      Task Splitter      │
│  (code graph-aware task │
│   decomposition)         │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│  Dynamic Worker Pool     │
│  (scales workers based  │
│   on workload)           │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│  Sub-Agent Orchestrator  │
│  (per-subtask model      │
│   selection, execution)  │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│  Result Aggregator      │
│  (deduplication, conflict│
│   resolution, merging)   │
└─────────────────────────┘
```

### 4.2 Workflow
1. **Eligibility Check**: Classifier analyzes incoming request, calculates confidence score for parallel execution
2. **Task Splitting**: If eligible, request is split into atomic subtasks using code graph structure
3. **Worker Pool Scaling**: Number of workers is dynamically adjusted based on subtask count and system load
4. **Subtask Execution**: Each subtask is assigned to a worker, with optional smaller model for simple tasks
5. **Result Aggregation**: Results from all subtasks are merged, deduplicated, and conflicts are resolved
6. **Final Response**: Aggregated result is returned to the user, with performance metrics logged

## 5. Implementation Details
### 5.1 Eligibility Classifier Enhancements
**File**: `codebase_rag/orchestrator/concurrency_classifier.py`
1. Lower default confidence threshold from 0.8 to 0.7:
   ```python
   DEFAULT_CONFIDENCE_THRESHOLD = 0.7
   ```
2. Add dynamic confidence calibration:
   - Maintain a running success rate for each task type
   - Adjust threshold automatically if success rate for a task type is above 90% (lower threshold) or below 60% (higher threshold)
3. Add additional eligibility signals:
   - Request length (longer requests are more eligible for parallelization)
   - Number of distinct code entities referenced in the request
   - Presence of keywords indicating multi-part requests ("list all", "find all", "compare", "analyze")

### 5.2 Code Graph-Aware Task Splitting
**File**: `codebase_rag/orchestrator/task_splitter.py`
1. Integrate with code graph to generate optimal subtasks:
   - Split requests based on code structure boundaries (modules, classes, functions)
   - Ensure each subtask maps to a distinct part of the codebase to avoid overlap
   - Add subtask metadata including target code entities, expected output type, and complexity score
2. New splitting algorithms:
   - **Entity-based splitting**: Split requests by referenced code entities (e.g. one subtask per class)
   - **Operation-based splitting**: Split requests by operation type (e.g. one subtask for finding issues, one for generating fixes)
   - **Size-based splitting**: Split large requests into chunks that fit within the selected model's context window

### 5.3 Dynamic Worker Pool
**File**: `codebase_rag/orchestrator/worker_pool.py`
1. Implement dynamic scaling logic:
   - Minimum workers: 1
   - Maximum workers: `CGR_MAX_PARALLEL_WORKERS` (default: 20)
   - Scaling formula: `worker_count = min(max(ceil(subtask_count / 2), 1), max_workers)`
   - Adjust for available CPU cores: never exceed number of physical cores - 1
2. Add worker reuse to reduce initialization overhead
3. Implement worker timeout per subtask (default: 300s, configurable via `CGR_SUBTASK_TIMEOUT`)

### 5.4 Per-Subtask Model Selection
**File**: `codebase_rag/orchestrator/sub_agent_orchestrator.py`
1. Add model routing logic based on subtask complexity:
   - **Simple tasks** (filtering, sorting, counting): Use smaller/cheaper model if available
   - **Complex tasks** (code generation, analysis, reasoning): Use main orchestrator model
2. Add configuration for model assignment:
   ```env
   # Simple task model override
   SIMPLE_TASK_MODEL=openai:gpt-4o-mini
   # Complex task model defaults to orchestrator model
   ```
3. Integrate with Context Window Management System to use correct context window for each subtask model

### 5.5 Hierarchical Result Aggregation
**File**: `codebase_rag/orchestrator/result_aggregator.py`
1. Implement 3-stage aggregation pipeline:
   1. **Deduplication**: Remove identical results from different subtasks
   2. **Conflict Resolution**: Resolve conflicting information using confidence scores and source priority
   3. **Synthesis**: Merge deduplicated results into a single coherent response
2. Add conflict resolution rules:
   - Higher confidence results take precedence over lower confidence
   - Results from more authoritative sources (code > docs > LLM generation) take precedence
   - Flag unresolved conflicts for the main agent to resolve

### 5.6 Configuration Parameters
Add new environment variables to `codebase_rag/config.py` and `.env.example`:
| Variable Name | Default | Description |
|---------------|---------|-------------|
| `CGR_PARALLEL_ELIGIBILITY_THRESHOLD` | 0.7 | Minimum confidence score to enable parallel execution |
| `CGR_AUTO_SCALE_WORKERS` | true | Enable dynamic worker scaling |
| `CGR_SIMPLE_TASK_MODEL` | (use orchestrator model) | Model to use for simple subtasks |
| `CGR_SUBTASK_TIMEOUT` | 300 | Timeout per subtask in seconds |
| `CGR_AGGREGATION_DEDUPLICATION_ENABLED` | true | Enable result deduplication |
| `CGR_PARALLEL_METRICS_ENABLED` | true | Enable parallel execution metrics logging |

## 6. API Specification
### 6.1 ConcurrencyClassifier.is_eligible()
```python
def is_eligible(self, request: str, context: dict = None) -> tuple[bool, str, float]:
    """
    Check if a request is eligible for parallel execution.
    
    Args:
        request: User request string
        context: Optional context dictionary with code graph information
        
    Returns:
        Tuple of (eligible: bool, task_type: str, confidence: float)
    """
```

### 6.2 TaskSplitter.split_task()
```python
def split_task(
    self,
    request: str,
    max_subtasks: int = 20,
    context_window: int = 256000
) -> list[SubTask]:
    """
    Split a request into atomic subtasks.
    
    Args:
        request: User request string
        max_subtasks: Maximum number of subtasks to generate
        context_window: Maximum context window size for each subtask
        
    Returns:
        List of SubTask objects containing:
        - id: Unique subtask ID
        - query: Subtask query string
        - target_entities: List of code entities to process
        - complexity: Complexity score (1-5)
        - expected_output_type: Expected output format
    """
```

### 6.3 WorkerPool.execute()
```python
async def execute(
    self,
    subtasks: list[SubTask],
    model_override: str = None,
    timeout: int = 300
) -> list[SubTaskResult]:
    """
    Execute a list of subtasks in parallel.
    
    Args:
        subtasks: List of SubTask objects to execute
        model_override: Optional model override for all subtasks
        timeout: Timeout per subtask in seconds
        
    Returns:
        List of SubTaskResult objects containing:
        - subtask_id: ID of the subtask
        - success: Boolean indicating success
        - result: Result data if successful
        - error: Error message if failed
        - execution_time: Time taken to execute subtask
    """
```

### 6.4 ResultAggregator.aggregate()
```python
def aggregate(
    self,
    results: list[SubTaskResult],
    original_request: str
) -> AggregatedResult:
    """
    Aggregate subtask results into a single coherent response.
    
    Args:
        results: List of SubTaskResult objects
        original_request: Original user request
        
    Returns:
        AggregatedResult object containing:
        - response: Merged response string
        - conflicts: List of unresolved conflicts (if any)
        - metrics: Aggregation performance metrics
    """
```

## 7. Data Structures
### 7.1 SubTask Object
```python
@dataclass
class SubTask:
    id: str
    query: str
    target_entities: list[str]
    complexity: int  # 1 (simple) - 5 (complex)
    expected_output_type: str
    context_requirements: dict[str, Any]
```

### 7.2 SubTaskResult Object
```python
@dataclass
class SubTaskResult:
    subtask_id: str
    success: bool
    result: Any | None = None
    error: str | None = None
    execution_time: float = 0.0
    confidence_score: float = 1.0
    source_type: str = "llm"
```

### 7.3 AggregatedResult Object
```python
@dataclass
class AggregatedResult:
    response: str
    conflicts: list[Conflict] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    subtask_count: int = 0
    success_count: int = 0
    total_execution_time: float = 0.0
```

## 8. Testing Plan
### 8.1 Unit Tests
1. **Eligibility Classifier Tests**:
   - Test confidence thresholding works correctly
   - Test dynamic calibration adjusts thresholds based on success rates
   - Test eligibility detection for common request types
2. **Task Splitter Tests**:
   - Test split tasks are non-overlapping and atomic
   - Test split results respect context window limits
   - Test code graph integration produces optimal splits
3. **Worker Pool Tests**:
   - Test dynamic scaling adjusts worker count correctly based on subtask count
   - Test worker timeout works correctly
   - Test error handling for failed subtasks
4. **Result Aggregator Tests**:
   - Test deduplication removes overlapping results
   - Test conflict resolution works according to priority rules
   - Test merged responses are coherent and complete

### 8.2 Integration Tests
1. End-to-end parallel execution test for eligible tasks
2. Performance test to verify minimum 2x speedup over sequential execution
3. Fallback test to verify sequential execution works when parallel eligibility is low
4. Model selection test to verify simple tasks use the configured smaller model

### 8.3 Edge Case Tests
1. Test with maximum subtask count (50) to verify scalability
2. Test with conflicting subtask results to verify conflict resolution
3. Test with partial subtask failures to verify graceful degradation
4. Test with very simple requests to verify they fall back to sequential execution

## 9. Migration Guide
This release is fully backwards compatible:
- Existing parallel execution configuration continues to work without modification
- Default threshold is lowered from 0.8 to 0.7 to enable more parallel execution by default
- Users who want to keep the old 0.8 threshold can set:
  ```env
  CGR_PARALLEL_ELIGIBILITY_THRESHOLD=0.8
  ```
- All new features are disabled by default unless explicitly configured

## 10. Documentation Updates
1. Update `.env.example` with all new parallel execution configuration options
2. Add parallel execution performance section to main README
3. Add troubleshooting guide for common parallel execution issues
4. Update API documentation for all new orchestration components

## 11. Future Enhancements
1. Add distributed execution support across multiple nodes
2. Add user-defined custom task splitting rules
3. Add checkpointing and resume support for long-running parallel tasks
4. Add adaptive model selection that learns optimal model assignments based on past performance
5. Add support for heterogeneous worker pools with specialized workers for different task types
