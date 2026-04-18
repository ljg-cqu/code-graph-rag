# Graph Traversal & Semantic Search - Consolidated Diagnosis

**Version:** 1.0 (Consolidated)
**Date:** 2026-04-18
**Status:** Accurate - Reflects Current Codebase State
**Scope:** `codebase_rag/` - Graph traversal, semantic search, vector retrieval

---

## Executive Summary

This document consolidates multiple diagnostic reports and has been corrected to reflect the actual codebase state. Some issues identified in earlier reports have already been implemented.

**System Health Score: 85/100** (estimated from code analysis - run `cgr doctor` for live metrics)

| System | Status | Notes |
|--------|--------|-------|
| Graph Traversal | ✅ Working | Correct Cypher generation with Memgraph syntax |
| Semantic Search | ⚠️ Degraded | Falls back to keyword search (0.5 scores) |
| Configuration | ✅ Fixed | `text_weight` removed, weights validated |
| Diagnostic Logging | ✅ Implemented | Full fallback chain logging in place |

---

## Issues Status

### Issue 1: Semantic Search Falls Back to Keyword 🔴 VALID

**Severity:** HIGH
**Location:** `codebase_rag/tools/semantic_search.py`, `codebase_rag/memgraph_advanced/hybrid_retrieval.py`

**Evidence:**
- All semantic search results show similarity score of 0.5 (keyword fallback score)
- The fallback chain is working correctly, but vector search returns empty results

**Root Cause:**
The `HybridRetriever._validate()` or vector backend may be failing due to:
1. Vector indexes not created in Memgraph
2. Embedding dimension mismatch (index dimension ≠ model output dimension)
3. No embeddings stored during indexing
4. Corrupted or incompatible embedding cache files

**Diagnostic Steps:**
```python
from codebase_rag.vector_backend import get_shared_backend
from codebase_rag.config import settings
from codebase_rag.embeddings import get_embedding_provider

# 1. Check vector backend health
backend = get_shared_backend()
print(f"Backend healthy: {backend.health_check()}")
print(f"Backend stats: {backend.get_stats()}")

# 2. Check embedding provider
config = settings.active_embedding_config
provider = get_embedding_provider(config.provider, config.model_id)
test_embed = provider.embed("test")
print(f"Embedding dimension: {len(test_embed)}")

# 3. Check if vector search works
results = backend.search(test_embed, top_k=5)
print(f"Vector search results: {len(results)}")
```

**Resolution:**
If vector search is broken, recreate indexes and re-index your data:
```bash
# Step 1: Recreate vector indexes with correct dimension
cgr vector recreate-indexes --code

# Step 2: Re-index your codebase to regenerate embeddings (REQUIRED)
cgr start --index-code
```

---

### Issue 2: HybridRetrievalConfig text_weight ✅ ALREADY FIXED

**Severity:** ~~MEDIUM~~ RESOLVED
**Location:** `codebase_rag/config.py:222-256`

**Original Problem:**
`text_weight: float = 0.2` was configured but `text_score=0.0` was always hardcoded.

**Current State:**
`text_weight` has been removed from `HybridRetrievalConfig`:
```python
@dataclass
class HybridRetrievalConfig:
    vector_weight: float = 0.7
    pagerank_weight: float = 0.2
    community_weight: float = 0.1
    top_k: int = 10
    max_context_depth: int = 2
    min_similarity_threshold: float = 0.1
```

---

### Issue 3: Diagnostic Logging ✅ ALREADY IMPLEMENTED

**Severity:** ~~MEDIUM~~ RESOLVED
**Location:** `codebase_rag/tools/semantic_search.py:233-267`

**Original Problem:**
No logging to identify which fallback level is active.

**Current State:**
Full diagnostic logging is implemented:
```python
# Level 0: Cache check
if cached is not None:
    logger.debug(f"Cache hit for query: {query[:50]}...")
    return cached

# Level 1: HybridRetriever
try:
    results = _search_with_hybrid_retriever(query, top_k)
    if results:
        logger.info(ls.SEMANTIC_FOUND.format(count=len(results), query=query))
        ...
except Exception as e:
    logger.warning(f"HybridRetriever failed: {e}")

# ... and so on for each fallback level
```

---

## Memgraph Cypher Syntax Reference

**CRITICAL:** The codebase uses **Memgraph**, not Neo4j. Syntax differs:

| Feature | Memgraph (CORRECT) | Neo4j (WRONG) |
|---------|-------------------|---------------|
| BFS range | `*BFS 1 TO 10` | `*BFS 1..10` |
| DFS range | `*DFS 1 TO 10` | `*DFS 1..10` |
| K-Shortest | `*KSHORTEST 5 1 TO 10` | N/A |

**Current prompt is correct** (`prompts.py:54-58`):
```python
- BFS: `*BFS 1 TO 3` (NOT `*BFS 1..3` — Neo4j syntax is wrong)
- DFS: `*DFS 1 TO 3`
- K-Shortest: `*KSHORTEST 5 1 TO 10`
```

---

## Actionable Items

### 1. Diagnose Vector Search (P0)

Run the diagnostic commands above to identify why vector search returns empty.

### 2. Recreate Vector Indexes (if needed)

```bash
cgr vector recreate-indexes --code
```

### 3. Verify Embedding Provider

Ensure `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` are correctly configured.

---

## Superseded Documents

The following documents have been consolidated into this one:
- `graph_traversal_semantic_search_diagnosis_v2.md`
- `graph_traversal_semantic_search_diagnosis_and_fix.md`
- `graph_traversal_semantic_search_comprehensive_diagnosis_v3.md`
- `graph_traversal_semantic_search_final_diagnosis.md`

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Semantic search using vector | 0% | >80% |
| Similarity score variance | All 0.5 | Varied |
| Graph query success | ~95% | >98% |
