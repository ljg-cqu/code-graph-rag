# Test Failure Fix Design Spec

## Summary

Fixed all test failures in `test_mcp_update_and_search.py`:
1. ✅ Incorrect enum usage in `mcp/tools.py`
2. ✅ Outdated test architecture (semantic search refactored)
3. ✅ Cancellation state corruption

## Issues Fixed

### Issue 1: Wrong Enum Usage in MCP Tools ✅ FIXED

**Location:** `codebase_rag/mcp/tools.py:548-552`

**Problem:** Using `MCPToolName.QUESTION` and `MCPToolName.TOP_K` as schema property keys.

**Fix Applied:**
```python
# Changed from:
cs.MCPToolName.QUESTION → cs.MCPParamName.QUESTION
cs.MCPToolName.TOP_K → cs.MCPParamName.TOP_K
```

---

### Issue 2: Outdated Test Architecture ✅ FIXED

**Location:** `codebase_rag/tests/test_mcp_update_and_search.py`

**Problem:** Tests expected `has_semantic_dependencies()` to exist but architecture was refactored.

**Architecture Change:**
- **Before:** Semantic search conditionally registered based on `has_semantic_dependencies()`
- **After:** Semantic search ALWAYS registered with fallback chain (HybridRetriever → Direct vector → Keyword)

**Fix Applied:**
- Removed test `test_semantic_search_not_registered_without_deps` (no longer applicable)
- Simplified `test_semantic_search_registered_with_deps` → `test_semantic_search_always_registered`
- Updated `TestRagAgentProperty` tests to remove `has_semantic_dependencies` mocks
- Fixed return value mock: `(mock_agent, [])` → `(mock_agent, [], None)`

---

### Issue 3: Cancellation State Corruption ✅ FIXED

**Location:** `codebase_rag/main.py`

**Problem:** After Ctrl+C cancellation, pydantic-ai error:
```
UserError: Cannot provide a new user prompt when the message history contains unprocessed tool calls.
```

**Root Cause:** When user cancels mid-execution, tool calls remain in message history without corresponding results.

**Fix Applied:**

1. Added helper function `_cleanup_unprocessed_tool_calls()`:
```python
def _cleanup_unprocessed_tool_calls(message_history: list) -> None:
    """Remove unprocessed tool call parts from message history."""
    from pydantic_ai.messages import ModelRequest, ToolCallPart

    cleaned = []
    for msg in message_history:
        if isinstance(msg, ModelRequest):
            parts = [p for p in msg.parts if not isinstance(p, ToolCallPart)]
            if parts:
                cleaned.append(ModelRequest(parts=parts))
        else:
            cleaned.append(msg)
    message_history[:] = cleaned
```

2. Integrated in cancellation handler:
```python
if isinstance(response, CancelledResult):
    # Clean up any unprocessed tool calls from message history
    # to allow the next prompt to work correctly
    _cleanup_unprocessed_tool_calls(message_history)
    log_session_event(config.cancelled_log)
    app_context.session.cancelled = True
    break
```

---

## Verification

```bash
uv run pytest codebase_rag/tests/test_mcp_update_and_search.py -v
# Result: 27 passed in 46.28s
```

---

## Files Changed

| File | Change |
|------|--------|
| `codebase_rag/mcp/tools.py` | Fixed enum: `MCPToolName` → `MCPParamName` |
| `codebase_rag/tests/test_mcp_update_and_search.py` | Rewrote tests for new architecture |
| `codebase_rag/main.py` | Added `_cleanup_unprocessed_tool_calls()` helper |

## Test Results

- **Before:** 28 tests (9 failed, 8 errors)
- **After:** 27 tests (all pass)
- **Reduction:** Merged 2 tests into 1 (semantic search always registered)
