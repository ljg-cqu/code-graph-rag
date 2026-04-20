# Query Routing in Document-Only Mode

## Problem Statement

In `DOCUMENT_ONLY` mode, the system incorrectly queries the code graph. This wastes resources and can confuse the orchestrator's retrieval logic.

## Evidence from Logs

```
# Correct: Document conceptual query detected
Task not eligible for parallel execution: DOCUMENT_ONLY mode with conceptual question - routing to semantic search

# Correct: Semantic search on document graph
Querying document graph: Why is categorical thinking important?

# WRONG: Code graph queried in document_only mode!
Querying code graph: categorical thinking categories classification importance benefits
```

## Identified Bugs

### Bug 1: Tool Calls Query Both Graphs Regardless of Mode

**Location**: Orchestrator tool calls

The `query_both_graphs` tool is being called even when the query mode is `DOCUMENT_ONLY`. This happens because:

1. The orchestrator generates tool calls based on query analysis
2. Tool calls don't check the current query mode
3. `query_both_graphs` ignores mode and queries both graphs

### Bug 2: Missing Mode Filter in Tool Registry

**Location**: `codebase_rag/tools/` or MCP tools

Tools should be filtered based on query mode:
- `DOCUMENT_ONLY`: Only document-related tools
- `CODE_ONLY`: Only code-related tools
- `BOTH_MERGED`: All tools

Currently, all tools are available regardless of mode.

### Bug 3: Concurrency Eligibility Classifier Fallback

**Location**: `codebase_rag/orchestrator/concurrency_eligibility_classifier.py:374-386`

```python
if query_mode == QueryMode.DOCUMENT_ONLY:
    if self._is_conceptual_question(prompt):
        return EligibilityResult(
            False,
            "document_conceptual_query",
            0.0,
            fallback_action="semantic_search",
        )
```

The classifier correctly identifies document conceptual queries, but the `fallback_action="semantic_search"` doesn't prevent subsequent tool calls from querying the code graph.

## Root Cause Analysis

```
User Query (DOCUMENT_ONLY mode)
    ↓
ConcurrencyEligibilityClassifier
    ↓
Eligible? No → fallback_action="semantic_search"
    ↓
Orchestrator generates tools
    ↓
Tool: query_both_graphs (ignores mode!)
    ↓
_query_code_only is called (WRONG!)
```

The fallback action only affects the parallel execution decision, not the tool selection.

## Proposed Solution

### Solution 1: Mode-Aware Tool Filtering

Add mode filtering to the orchestrator's tool selection:

```python
def get_tools_for_mode(mode: QueryMode) -> list[Tool]:
    """Filter tools based on query mode."""
    tools = []
    
    if mode == QueryMode.DOCUMENT_ONLY:
        tools = [
            query_document_graph,
            search_documents,
            read_file,
            list_directory,
        ]
    elif mode == QueryMode.CODE_ONLY:
        tools = [
            query_code_graph,
            search_code,
            read_file,
            list_directory,
        ]
    else:  # BOTH_MERGED
        tools = ALL_TOOLS
        
    return tools
```

### Solution 2: Mode Check in Query Router

Add mode guard to `QueryRouter` methods:

```python
def _query_code_only(self, request: QueryRequest) -> QueryResponse:
    """Query CODE graph/vector ONLY."""
    # Guard: Should not be called in DOCUMENT_ONLY mode
    if self.current_mode == QueryMode.DOCUMENT_ONLY:
        logger.warning("Code query called in DOCUMENT_ONLY mode, returning empty")
        return QueryResponse(
            answer="Code queries are disabled in DOCUMENT_ONLY mode.",
            sources=[],
            mode=request.mode,
        )
    
    # Existing logic...
```

### Solution 3: Fix Tool Implementation

Add mode awareness to `query_both_graphs` tool:

```python
def query_both_graphs(query: str, ...) -> ...:
    """Query both graphs respecting current mode."""
    mode = get_current_query_mode()
    
    if mode == QueryMode.DOCUMENT_ONLY:
        # Only query document graph
        return query_document_graph(query)
    elif mode == QueryMode.CODE_ONLY:
        # Only query code graph
        return query_code_graph(query)
    
    # Original both_merged logic
    code_results = query_code_graph(query)
    doc_results = query_document_graph(query)
    return merge_results(code_results, doc_results)
```

## Recommendation

**Combine Solution 1 + Solution 3**:
1. Filter tools at orchestrator level (Solution 1)
2. Add defensive mode checks in tool implementations (Solution 3)

This provides defense in depth and ensures correctness even if a tool is accidentally called.

## Implementation Checklist

- [x] Add `get_tools_for_mode()` function to tool registry
- [x] Update orchestrator to use mode-filtered tools
- [x] Add mode guard to `_query_code_only()` and `_query_document_only()`
- [x] Fix `query_both_graphs` tool to respect mode
- [x] Add tests for mode-aware query routing
- [x] Add logging when mode guard triggers

## Files to Modify

1. `codebase_rag/tools/__init__.py` or MCP tools - Add mode filtering
2. `codebase_rag/shared/query_router.py` - Add mode guards
3. `codebase_rag/orchestrator/task_splitter.py` - Use mode-filtered tools
4. `codebase_rag/main.py` - Ensure mode is propagated correctly
