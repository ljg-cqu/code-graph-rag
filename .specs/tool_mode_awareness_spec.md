# Tool Mode Awareness Fix

## Problem Statement

The `query_both_graphs` tool ignores the current query mode and always queries both graphs with `QueryMode.BOTH_MERGED`, even when the system is in `DOCUMENT_ONLY` mode.

## Evidence from Logs

```
# System correctly in DOCUMENT_ONLY mode
Auto-selecting DOCUMENT_ONLY mode (docs: 6, code: 0)

# But query_both_graphs tool is called with BOTH_MERGED mode
Querying code graph: categorical thinking categories classification importance benefits
```

## Root Cause Analysis

**Location**: `codebase_rag/tools/document_query.py:119-123`

```python
request = QueryRequest(
    question=natural_language_query,
    mode=QueryMode.BOTH_MERGED,  # HARDCODED! Ignores current mode
    top_k=top_k,
)
```

The tool always uses `BOTH_MERGED` mode regardless of the global query mode.

## Call Flow

```
User Query: "Why categorical thinking is important?"
    ↓
ConcurrencyEligibilityClassifier
    ↓
Returns fallback_action="semantic_search" (DOCUMENT_ONLY mode)
    ↓
Orchestrator generates tools
    ↓
Tool: query_both_graphs(natural_language_query)
    ↓
Creates QueryRequest with mode=BOTH_MERGED (WRONG!)
    ↓
_query_code_only is called
```

## Proposed Solution

### Solution 1: Pass Mode to Tool

Add mode parameter to tool signature:

```python
async def query_both_graphs(
    natural_language_query: str,
    top_k: int = 5,
    mode: str = "auto",  # "auto", "code_only", "document_only", "both_merged"
) -> str:
    """Query graphs respecting the specified mode."""
    # Resolve mode
    if mode == "auto":
        actual_mode = get_current_query_mode()
    else:
        actual_mode = QueryMode(mode)
    
    request = QueryRequest(
        question=natural_language_query,
        mode=actual_mode,
        top_k=top_k,
    )
    ...
```

### Solution 2: Use Shared Mode Context

Create a global query mode context that tools can access:

```python
# In shared module
class QueryModeContext:
    _current_mode: QueryMode = QueryMode.CODE_ONLY
    
    @classmethod
    def get_mode(cls) -> QueryMode:
        return cls._current_mode
    
    @classmethod
    def set_mode(cls, mode: QueryMode) -> None:
        cls._current_mode = mode

# In tool
async def query_both_graphs(...) -> str:
    mode = QueryModeContext.get_mode()
    request = QueryRequest(
        question=natural_language_query,
        mode=mode,
        ...
    )
```

### Solution 3: Filter Tools by Mode (Recommended)

Don't expose `query_both_graphs` tool in `DOCUMENT_ONLY` mode:

```python
def get_tools_for_mode(mode: QueryMode) -> list[Tool]:
    if mode == QueryMode.DOCUMENT_ONLY:
        return [
            query_document_graph_tool,
            # query_both_graphs_tool NOT included
        ]
    elif mode == QueryMode.CODE_ONLY:
        return [
            query_code_graph_tool,
            # query_both_graphs_tool NOT included
        ]
    else:  # BOTH_MERGED
        return [
            query_code_graph_tool,
            query_document_graph_tool,
            query_both_graphs_tool,
        ]
```

## Recommendation

**Implement Solution 3** as the primary fix, with Solution 1 as a defensive backup:
1. Filter tools at registration time based on mode
2. Add mode parameter to `query_both_graphs` for edge cases

This ensures:
- Correct tools are available for each mode
- Defensive coding prevents accidental misuse
- Clear separation of concerns

## Implementation Checklist

- [x] Add `get_tools_for_mode()` function to tool registry
- [x] Update main.py to use mode-filtered tools
- [x] Add mode parameter to `query_both_graphs` tool
- [x] Add tests for mode-aware tool filtering
- [ ] Document mode-specific tool availability

## Files to Modify

1. `codebase_rag/tools/__init__.py` - Add tool filtering
2. `codebase_rag/tools/document_query.py` - Add mode parameter
3. `codebase_rag/main.py` - Use filtered tools
4. `codebase_rag/shared/query_router.py` - Export QueryModeContext if using Solution 2
