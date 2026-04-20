# Data Modeling and Query Bugs: Summary and Implementation Priority

## Overview

This document summarizes all identified bugs in the Code-Graph-RAG data modeling and query subsystems, along with implementation priority and dependencies.

## Bug Summary

| ID | Bug | Severity | Impact | Related Spec |
|----|-----|----------|--------|--------------|
| V1 | Document vector search uses code indexes | Critical | Search returns no results | `document_vector_search_fix_spec.md` |
| V2 | Document backend initializes code indexes | Medium | Wasted resources | `vector_backend_optimization_spec.md` |
| Q1 | Code graph queried in DOCUMENT_ONLY mode | High | Incorrect results | `query_routing_fix_spec.md` |
| Q2 | Tool ignores query mode | High | Incorrect results | `tool_mode_awareness_spec.md` |
| Q3 | Workspace filter not handled | Medium | Wrong results | `document_vector_search_fix_spec.md` |

## Priority Order

### Phase 1: Critical Fixes (Immediate)

**V1: Document Vector Search Fix**

This is the root cause of "no results" for document queries.

1. Create `DocumentMemgraphBackend` class
2. Update `get_vector_backend()` factory
3. Implement proper `Chunk` node search with `doc_embeddings` index

**Estimated Impact**: Document semantic search will return results.

### Phase 2: High Priority

**Q1 + Q2: Query Routing and Tool Mode**

These bugs cause incorrect graph access in document-only mode.

1. Add mode filtering to tool registry
2. Update orchestrator to use filtered tools
3. Add mode guards to query router methods

**Estimated Impact**: Code graph will not be queried in DOCUMENT_ONLY mode.

### Phase 3: Medium Priority

**V2 + Q3: Optimization and Filter Support**

These improve efficiency and correctness.

1. Skip code index initialization for document backend
2. Add workspace filter support to document search
3. Move index creation to backend classes

**Estimated Impact**: Faster initialization, correct workspace filtering.

## Dependency Graph

```
V1 (Critical)
    │
    ├──→ Q3 (Filter support) [Medium]
    │
    └──→ V2 (Optimization) [Medium]

Q1 (Routing) ──→ Q2 (Tool mode) [High]
```

## Implementation Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | V1 | Document search returns results |
| 2 | Q1 + Q2 | Correct graph routing by mode |
| 3 | V2 + Q3 | Optimized initialization, workspace filters |
| 4 | Testing | Integration tests, documentation |

## Test Plan

### Unit Tests

1. `test_document_memgraph_backend.py` - New backend class
2. `test_vector_backend_factory.py` - Factory returns correct backend
3. `test_query_router_mode_guards.py` - Mode guards work
4. `test_tool_filtering.py` - Tools filtered by mode

### Integration Tests

1. Document-only repository → document search works
2. Code-only repository → code search works
3. Mixed repository → both_merged works
4. Mode switching → correct graph accessed

### End-to-End Tests

1. `cgr start --repo-path doc-repo` → semantic search returns chunks
2. `cgr start --repo-path code-repo` → function search works
3. Mode switch commands → correct behavior

## Related Files

### New Files (to create)

- `codebase_rag/vector_store_document.py` - DocumentMemgraphBackend
- `codebase_rag/tests/test_document_memgraph_backend.py`

### Modified Files

- `codebase_rag/vector_store_memgraph.py` - Add document mode
- `codebase_rag/vector_backend.py` - Update factory
- `codebase_rag/tools/document_query.py` - Mode awareness
- `codebase_rag/tools/__init__.py` - Tool filtering
- `codebase_rag/shared/query_router.py` - Mode guards
- `codebase_rag/main.py` - Use filtered tools
- `codebase_rag/document/tools/document_search.py` - Correct backend usage

## Verification Checklist

After implementation, verify:

- [x] Document semantic search returns results for document-only repos
- [x] Code graph is NOT queried in DOCUMENT_ONLY mode
- [x] Document backend initialization only checks `doc_embeddings` index
- [x] Workspace filter works correctly for multi-tenant setups
- [ ] Mode switching via `/mode` command works correctly
- [ ] All existing tests pass
- [ ] New tests cover all fixed bugs
