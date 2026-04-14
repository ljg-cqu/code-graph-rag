# Unified Hybrid Retrieval System Design Specification
**Version**: 1.0
**Date**: 2024-04-14
**Status**: Final
**Author**: Code Graph RAG Team
**Depends On**: 
- [001-context-window-management-system.md](./001-context-window-management-system.md)
- [002-parallel-task-orchestration-system-v2.md](./002-parallel-task-orchestration-system-v2.md)

## Revision History
| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2024-04-14 | Initial complete design | Team |

## 1. Purpose
This document specifies the design for the Unified Hybrid Retrieval System, which unifies the three currently separate search capabilities in Code Graph RAG:
1. Semantic vector search (embedding-based similarity matching)
2. Graph traversal search (structure-based queries like callers, dependencies, inheritance)
3. Keyword/regex search (exact string matching across code and documents)

The system addresses critical limitations in the current implementation:
- Siloed search workflows requiring users to know which search type to use
- No unified result ranking across different search types
- Redundant code paths for similar retrieval operations
- No cross-signal optimization to improve result relevance
- Inconsistent result formats across search methods

This design builds on both preceding specifications: it uses the Parallel Task Orchestration System v2 to run multiple search methods in parallel, and leverages the Context Window Management System to optimize result batching for LLM context limits.

## 2. Scope
### 2.1 In Scope
- Single unified API entry point for all search queries
- Hybrid result fusion that combines scores from all search types using Reciprocal Rank Fusion (RRF)
- Intelligent query routing that automatically selects optimal search methods based on query intent
- Configurable signal weighting for tuning search result relevance
- Support for both code and document graph search
- Deduplication of identical results across different search types
- Backwards compatibility with all existing search APIs
- Query preprocessing including entity detection, keyword extraction, and intent classification
- Result formatting that fits within configured context window limits

### 2.2 Out of Scope
- Custom machine learning reranking models (uses rule-based and embedding-based ranking only)
- Distributed search across multiple instances
- Cross-repository search (limited to single active repository)
- Real-time index updates (uses existing batch indexing workflow)
- User-specific search personalization

## 3. Requirements
### 3.1 Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | System shall provide a single unified API for all search queries that automatically selects optimal search methods | Critical |
| FR2 | System shall combine results from multiple search types using Reciprocal Rank Fusion (RRF) algorithm | High |
| FR3 | Users shall be able to configure signal weights for each search type via environment variables | High |
| FR4 | System shall automatically detect query intent to route to appropriate search methods (e.g. "callers of X" → graph search, "how does auth work" → semantic + document search) | High |
| FR5 | System shall deduplicate identical results from different search types | High |
| FR6 | System shall support searching across both code and document graphs simultaneously | Medium |
| FR7 | Result formatting shall automatically truncate and prioritize results to fit within the configured context window limit | High |
| FR8 | All existing search APIs (semantic search, graph query, keyword search) shall continue to work without modification | Critical |
| FR9 | System shall provide detailed search performance metrics (latency per search type, result count per source, relevance scores) | Medium |

### 3.2 Non-Functional Requirements
| ID | Requirement | Priority |
|----|-------------|----------|
| NFR1 | System shall achieve minimum 30% improvement in retrieval relevance measured against the existing benchmark suite | Critical |
| NFR2 | Unified search shall add <200ms overhead compared to running single search types | High |
| NFR3 | System shall handle up to 100 results per query while maintaining performance | Medium |
| NFR4 | Search latency shall be <500ms for 95% of queries | High |
| NFR5 | All changes shall be fully backwards compatible with no breaking API changes | Critical |

## 4. System Architecture
### 4.1 Component Overview
The Unified Hybrid Retrieval System consists of 5 core components:

```
┌─────────────────────────┐
│   Query Preprocessor    │
│  (normalization, entity │
│   detection, intent     │
│   classification)       │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│     Query Router        │
│  (selects optimal search│
│   methods based on intent│
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│  Hybrid Search Executor │
│  (runs selected searches│
│   in parallel using v2  │
│   orchestration system) │
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│ Result Fusion & Ranker  │
│  (RRF scoring, deduplication,│
│   context-aware trimming)│
└─────────────────────────┘
              │
              ▼
┌─────────────────────────┐
│  Configuration Manager  │
│  (manages signal weights,│
│   RRF parameters, limits)│
└─────────────────────────┘
```

### 4.2 End-to-End Workflow
1. **Query Preprocessing**: User query is normalized, code entities are detected, and intent is classified
2. **Query Routing**: Based on intent classification, router selects which search methods to run (semantic, graph, keyword, or combination)
3. **Parallel Execution**: Selected search methods are executed in parallel using the v2 Parallel Task Orchestration System
4. **Result Fusion**: Results from all search methods are combined using RRF, deduplicated, and ranked
5. **Context Trimming**: Ranked results are trimmed and formatted to fit within the configured context window limit
6. **Response**: Unified, ranked results are returned to the user with performance metrics

## 5. Implementation Details
### 5.1 Query Preprocessor
**File**: `codebase_rag/retrieval/query_preprocessor.py`
1. Normalization steps:
   - Lowercase conversion, extra whitespace removal, special character handling
   - Stopword removal for English queries
   - Code entity extraction (detect class names, function names, file paths using code graph index)
2. Intent classification:
   - Rule-based classifier for common query patterns:
     - **Graph intent**: Keywords like "callers", "callees", "dependencies", "inherits", "uses", "children", "parents"
     - **Semantic intent**: Keywords like "how does", "explain", "what is", "find", "search for"
     - **Keyword intent**: Quoted strings, regex patterns, exact match keywords
     - **Hybrid intent**: Queries containing multiple pattern types
   - Confidence scoring for each intent type to support multi-method search

### 5.2 Query Router
**File**: `codebase_rag/retrieval/query_router.py`
1. Routing logic:
   - For single intent queries: Route only to the matching search method
   - For hybrid intent queries: Route to all applicable search methods
   - Fallback: Run all search methods if intent classification confidence is low (<0.6)
2. Configurable routing rules via environment variables
3. Support for explicit search method override via API parameter

### 5.3 Hybrid Search Executor
**File**: `codebase_rag/retrieval/search_executor.py`
1. Integrates with Parallel Task Orchestration System v2 to run multiple search methods in parallel
2. Wraps existing search implementations:
   - Semantic search: Uses existing semantic search component
   - Graph search: Uses existing Cypher query generation and execution
   - Keyword search: Uses existing ripgrep-based keyword search component
3. Standardizes result format across all search types with consistent metadata:
   - Result ID, content, source file, start line, end line, score, source type (semantic/graph/keyword)
4. Handles per-search-method timeouts and graceful degradation if a search method fails

### 5.4 Result Fusion & Ranker
**File**: `codebase_rag/retrieval/result_fuser.py`
1. Implements Reciprocal Rank Fusion (RRF) algorithm for combining results:
   ```
   RRF Score(doc) = sum(1 / (k + rank(doc, method)) for each method)
   where k = 60 (default configurable constant)
   ```
2. Optional weighted RRF support for tuning signal importance:
   ```
   Weighted RRF Score(doc) = sum(weight[method] / (k + rank(doc, method)) for each method)
   ```
3. Deduplication logic:
   - Identical results (same file + line range) are merged, highest score is retained
   - Overlapping results are merged into a single result with expanded line range
4. Context trimming:
   - Sorts results by RRF score descending
   - Truncates results to fit within configured context window limit
   - Preserves highest priority results when truncating
   - Adds summary of truncated results if needed

### 5.5 Configuration Parameters
Add new environment variables to `codebase_rag/config.py` and `.env.example`:
| Variable Name | Default | Description |
|---------------|---------|-------------|
| `RETRIEVAL_SEMANTIC_WEIGHT` | 1.0 | Weight for semantic search results in RRF |
| `RETRIEVAL_GRAPH_WEIGHT` | 1.5 | Weight for graph search results in RRF |
| `RETRIEVAL_KEYWORD_WEIGHT` | 2.0 | Weight for keyword search results in RRF |
| `RETRIEVAL_RRF_K_CONSTANT` | 60 | K constant for RRF algorithm |
| `RETRIEVAL_MAX_RESULTS` | 50 | Maximum number of results to return |
| `RETRIEVAL_AUTO_ROUTING_ENABLED` | true | Enable automatic intent-based routing |
| `RETRIEVAL_FUSION_ENABLED` | true | Enable result fusion (disable for raw multi-source results) |
| `RETRIEVAL_DEDUPLICATION_ENABLED` | true | Enable result deduplication |
| `RETRIEVAL_INTENT_CONFIDENCE_THRESHOLD` | 0.6 | Minimum confidence to use intent-based routing |

## 6. API Specification
### 6.1 Unified Search API
```python
async def unified_search(
    query: str,
    search_methods: list[SearchMethod] | None = None,
    max_results: int | None = None,
    include_docs: bool = False,
    context_window_size: int | None = None
) -> UnifiedSearchResponse:
    """
    Unified search API that automatically selects optimal search methods and returns ranked results.
    
    Args:
        query: User search query string
        search_methods: Optional explicit list of search methods to use (overrides auto-routing)
        max_results: Optional override for maximum number of results to return
        include_docs: Include document graph results in addition to code results
        context_window_size: Optional override for context window size to use for result trimming
        
    Returns:
        UnifiedSearchResponse object containing:
        - results: List of ranked SearchResult objects
        - search_methods_used: List of search methods that were executed
        - metrics: Search performance metrics
        - truncated: Boolean indicating if results were truncated to fit context window
    """
```

### 6.2 SearchMethod Enum
```python
class SearchMethod(StrEnum):
    SEMANTIC = "semantic"
    GRAPH = "graph"
    KEYWORD = "keyword"
```

### 6.3 SearchResult Object
```python
@dataclass
class SearchResult:
    id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    score: float  # RRF combined score 0.0-1.0
    source_type: SearchMethod
    entity_type: str | None = None  # class/function/module/document etc.
    entity_name: str | None = None
    semantic_similarity: float | None = None
    graph_relevance: float | None = None
    keyword_match_score: float | None = None
```

### 6.4 UnifiedSearchResponse Object
```python
@dataclass
class UnifiedSearchResponse:
    results: list[SearchResult]
    search_methods_used: list[SearchMethod]
    metrics: dict[str, Any]
    truncated: bool = False
    truncated_count: int = 0
    total_result_count: int = 0
```

## 7. Integration Points
### 7.1 Integration with Parallel Orchestration v2
- The Hybrid Search Executor uses the existing Worker Pool to run multiple search methods in parallel
- Each search method is executed as a separate subtask
- Timeouts and error handling are managed by the existing orchestration system

### 7.2 Integration with Context Window Management
- Result Fusion & Ranker uses the configured context window size to automatically trim results
- Trimming logic prioritizes higher scoring results to maximize relevance within context limits
- Supports dynamic context window sizes for model override scenarios

### 7.3 Backwards Compatibility
- All existing search APIs remain unchanged and continue to work
- Existing users can opt in to the new unified API gradually
- No changes required to existing indexing workflows

## 8. Testing Plan
### 8.1 Unit Tests
1. **Query Preprocessor Tests**:
   - Test entity detection correctly identifies code entities (classes, functions, file paths)
   - Test intent classification correctly identifies graph, semantic, keyword, and hybrid intents
   - Test query normalization handles special characters and formatting correctly
2. **Query Router Tests**:
   - Test routing correctly selects appropriate search methods for different intent types
   - Test explicit method override works correctly
   - Test fallback to multi-method search when confidence is low
3. **Result Fusion & Ranker Tests**:
   - Test RRF algorithm correctly combines results from multiple search types
   - Test weighted RRF correctly applies configured signal weights
   - Test deduplication correctly merges identical and overlapping results
   - Test context trimming correctly fits results within specified context window limit
4. **Search Executor Tests**:
   - Test parallel execution of multiple search methods works correctly
   - Test standardization of result formats across search types
   - Test graceful degradation when individual search methods fail

### 8.2 Integration Tests
1. End-to-end unified search test for common query types
2. Relevance benchmark test to verify 30% improvement over existing single search methods
3. Performance test to verify <200ms overhead compared to single search methods
4. Backwards compatibility test to verify all existing search APIs continue to work

### 8.3 Edge Case Tests
1. Test with queries that have ambiguous intent to verify fallback behavior
2. Test with large result sets to verify deduplication and context trimming work correctly
3. Test with partial search method failures to verify graceful degradation
4. Test with custom signal weight configurations to verify ranking adjusts correctly

## 9. Migration Guide
This release is fully backwards compatible with no breaking changes:
- All existing search APIs continue to work without modification
- New unified search API is optional and can be adopted gradually
- Default configuration values are optimized for general use cases
- Users can customize signal weights and other parameters via environment variables as needed

## 10. Documentation Updates
1. Update `.env.example` with all new retrieval configuration options
2. Add unified search API documentation to the main README
3. Add retrieval tuning guide for optimizing search relevance for specific codebases
4. Add benchmark results comparing hybrid search to single search methods
5. Update existing search API documentation to note the new unified API option

## 11. Future Enhancements
1. Add custom machine learning reranking model for further relevance improvements
2. Add user feedback learning loop to adjust signal weights based on user interactions
3. Add cross-repository search support for multi-repository workspaces
4. Add real-time incremental index updates to support dynamic codebases
5. Add support for structured filter queries (e.g. "search in files modified in last 7 days")
6. Add multi-lingual search support for non-English codebases and documentation
