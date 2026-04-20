# Query Mechanism Cypher Syntax Fix — Design Specification

## 1. Problem Statement

The hybrid retrieval query mechanism fails with a Cypher syntax error during vector search operations:

```
ERROR | Cypher syntax error, not retrying: Error on line 4 position 9.
The underlying parsing error is mismatched input 'WHERE' expecting {<EOF>, ';'}
```

This causes the semantic search to fall back to keyword-only search, significantly degrading query quality for code-related queries.

**Observed in logs:**
```
WARNING | codebase_rag.services.graph_service:_execute_query:421 - Cypher syntax error, not retrying
ERROR   | codebase_rag.services.graph_service:_execute_query:426 - Cypher Error
INFO    | codebase_rag.tools.semantic_search:semantic_code_search:262 - Keyword fallback found 10 results
```

---

## 2. Root Cause Analysis

### 2.1 `WHERE` After `YIELD` in `CALL vector_search.search` Is Invalid

**Location:** `codebase_rag/memgraph_advanced/hybrid_retrieval.py:187-233`

The `_search_atomic` method constructs a Cypher query with an invalid clause sequence:

```cypher
-- Current (BROKEN):
CALL vector_search.search($index_name, $overfetch, $embedding)
YIELD node AS seed, similarity AS vec_sim
WHERE vec_sim >= $min_similarity
```

**Problem:** In Memgraph 3.8.1, `WHERE` directly after `YIELD` in a `CALL vector_search.search(...)` procedure call is a syntax error. The parser expects `WITH` or `RETURN` after `YIELD`, not `WHERE`.

**Verified via direct Memgraph testing:**
- `CALL ... YIELD ... WHERE ...` → **FAIL** (`mismatched input 'WHERE' expecting {<EOF>, ';'}`)
- `CALL ... YIELD ... WITH ... WHERE ...` → **PASS**

**Correct syntax (from `document_search.py:135-143`):**
```cypher
CALL vector_search.search($index_name, $overfetch, $embedding)
YIELD node, distance, similarity
WITH node, distance, similarity
WHERE similarity >= $min_similarity
```

**Key insight:** Yielding only two values (`node` and `similarity`) is fine — Memgraph supports this variant. The issue is strictly the missing `WITH` clause between `YIELD` and `WHERE`.

### 2.2 BFS Bounds Syntax Error and Parameter Restriction

**Location:** `codebase_rag/memgraph_advanced/hybrid_retrieval.py:192`

```cypher
OPTIONAL MATCH path = (seed)-[:CALLS|:DEFINES|:IMPORTS *BFS 1 TO $max_depth]-(context_node)
```

**Problem:** Two separate issues were verified against Memgraph 3.8.1:

1. **`1 TO` syntax is invalid** — Memgraph 3.8.1 uses `1..$var` (range syntax), not `1 TO $var`.
2. **Parameters are not allowed inside BFS bounds** — `*BFS 1..$max_depth` produces `Property map matching not supported in MATCH/MERGE clause!`. The bound must be a variable, not a parameter.

**Correct syntax:**
```cypher
WITH $max_depth AS max_depth
...
OPTIONAL MATCH path = (seed)-[:CALLS|:DEFINES|:IMPORTS *BFS 1..max_depth]-(context_node)
```

The same bug exists in `vector_store_memgraph.py` (lines 428, 501) and `graph_algorithms.py` (line 210).

### 2.3 Index Name Mismatch

**Location:** `codebase_rag/memgraph_advanced/hybrid_retrieval.py:236`

The query uses `settings.MEMGRAPH_VECTOR_INDEX_NAME` which defaults to `"code_embeddings"`, but the actual indexes created are per-label:
- `function_embedding_index`
- `method_embedding_index`
- `class_embedding_index`
- etc.

**Evidence from `vector_store_memgraph.py:155`:**
```python
for label in self.LABELS_TO_INDEX:
    index_name = f"{label.lower()}_embedding_index"
```

This means the hybrid retrieval tries to query a **non-existent index**.

### 2.4 `LIMIT` with Variables Is Invalid

**Location:** `codebase_rag/vector_store_memgraph.py:437, 459`

```cypher
LIMIT top_k
```

**Problem:** Memgraph does not allow variables in `LIMIT`. Parameters (`$top_k`) or literals are required.

**Verified via direct Memgraph testing:**
- `LIMIT $top_k` → **PASS**
- `LIMIT top_k` (where `top_k` is a `WITH` alias) → **FAIL** (`Variables are not allowed in LIMIT`)

---

## 3. Comparison with Reference Implementations

### 3.1 `document_search.py` (Working)

```cypher
CALL vector_search.search($index_name, $overfetch, $embedding)
YIELD node, distance, similarity
WITH node, distance, similarity
WHERE node.workspace = $workspace AND similarity >= $min_similarity
```

**Key differences:**
1. Uses `WITH` clause after `YIELD` before `WHERE`
2. Uses correct index name (`settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME`)
3. Uses `$limit` (parameter) in `LIMIT`

### 3.2 `vector_store_memgraph.py` (Also Broken)

```cypher
WITH $embedding AS query_vec, $top_k AS top_k, $project_prefix AS project_prefix
CALL vector_search.search($index_name, top_k * 3, query_vec)
YIELD node AS n, similarity AS sim
WHERE ($project_prefix IS NULL OR n.qualified_name STARTS WITH $project_prefix)
```

**This query has the same `WHERE` after `YIELD` bug** as `hybrid_retrieval.py`. It was incorrectly assumed to be working. Direct testing confirmed it fails with the same syntax error. Both branches (`include_context=True` and `include_context=False`) need fixing.

---

## 4. Proposed Fix

### 4.1 Fix `_search_atomic` Cypher Query

**File:** `codebase_rag/memgraph_advanced/hybrid_retrieval.py`

Replace the `_search_atomic` method's Cypher query with a corrected version:

```cypher
CALL vector_search.search($index_name, $overfetch, $embedding)
YIELD node AS seed, similarity AS vec_sim
WITH seed, vec_sim
WHERE vec_sim >= $min_similarity

WITH seed, vec_sim, $max_depth AS max_depth
OPTIONAL MATCH path = (seed)-[:CALLS|:DEFINES|:IMPORTS *BFS 1..max_depth]-(context_node)
WHERE context_node:Function OR context_node:Class OR context_node:Method OR context_node:Module

WITH seed, vec_sim, context_node,
     CASE WHEN context_node IS NOT NULL THEN length(path) END AS path_depth

WITH seed, vec_sim,
     collect(DISTINCT CASE WHEN context_node IS NOT NULL THEN {
         qn: context_node.qualified_name,
         name: context_node.name,
         type: labels(context_node)[0],
         depth: path_depth
     } END) AS context,
     COALESCE(seed.pagerank_score, 0.1) AS pr_score,
     COALESCE(seed.community_importance, 0.0) AS ci_score,
     CASE WHEN ANY(kw IN $keywords WHERE
         toLower(seed.name) CONTAINS kw
         OR toLower(seed.qualified_name) CONTAINS kw
         OR toLower(COALESCE(seed.docstring, "")) CONTAINS kw
     ) THEN 1.0 ELSE 0.0 END AS text_match

WITH seed, vec_sim, context, pr_score, ci_score, text_match,
     (vec_sim * $vector_weight + pr_score * $pagerank_weight
      + ci_score * $community_weight + text_match * $text_weight) AS combined

ORDER BY combined DESC
LIMIT $top_k

RETURN id(seed) AS node_id,
       seed.name AS name,
       seed.qualified_name AS qualified_name,
       labels(seed)[0] AS node_type,
       seed.path AS file_path,
       seed.start_line AS start_line,
       seed.end_line AS end_line,
       vec_sim AS vector_score,
       text_match AS text_score,
       pr_score AS pagerank_score,
       ci_score AS community_score,
       combined AS combined_score,
       [c IN context WHERE c IS NOT NULL] AS context
```

**Key changes:**
- Added `WITH seed, vec_sim` after `YIELD` before `WHERE`
- Added `WITH seed, vec_sim, $max_depth AS max_depth` before `OPTIONAL MATCH`
- Changed `*BFS 1 TO $max_depth` to `*BFS 1..max_depth`
- Kept `LIMIT $top_k` (parameter, not variable)
- Relationship syntax `[:CALLS|:DEFINES|:IMPORTS` is valid — no change needed

### 4.2 Fix Index Name Resolution

**Option A: Use Per-Label Indexes (Recommended)**

Iterate over label-specific indexes similar to `vector_store_memgraph.py`:

```python
def _search_atomic(self, ...):
    """Atomic query with per-label index iteration."""
    all_results: list[HybridSearchResult] = []
    seen_node_ids: set[int] = set()
    cfg = self.config or HybridRetrievalConfig()

    for label in self.vector_backend.LABELS_TO_INDEX:
        index_name = f"{label.lower()}_embedding_index"
        params = {
            "index_name": index_name,
            "embedding": query_embedding,
            "overfetch": top_k * 3,
            "min_similarity": cfg.min_similarity_threshold,
            "max_depth": cfg.max_context_depth,
            "keywords": [kw.lower() for kw in query_keywords],
            "top_k": top_k,
            "vector_weight": cfg.vector_weight,
            "pagerank_weight": cfg.pagerank_weight,
            "community_weight": cfg.community_weight,
            "text_weight": cfg.text_weight,
        }
        try:
            records = self.graph_ingestor.fetch_all(atomic_cypher, params)
            for record in records:
                node_id = _coerce_int(record.get("node_id"), 0)
                if node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node_id)
                all_results.append(self._record_to_result(record, cfg))
        except Exception as e:
            logger.debug(f"Atomic hybrid query failed for label {label}: {e}")
            continue

    all_results.sort(key=lambda r: r.combined_score, reverse=True)
    return all_results[:top_k]
```

**Option B: Create a Unified Index**

Create a single `code_embeddings` index that covers all embeddable labels. This requires modifying `vector_store_memgraph.py` to create a unified index.

**Recommendation:** Option A is preferred as it maintains consistency with existing index creation logic and requires no migration.

### 4.3 Fix `vector_store_memgraph.py`

**File:** `codebase_rag/vector_store_memgraph.py`

Both `search()` branches need the same corrections:

1. Add `WITH` after `YIELD` before `WHERE`
2. Carry `max_depth` through the `WITH` projection so it remains in scope for BFS
3. Change `*BFS 1 TO max_depth` to `*BFS 1..max_depth`
4. Change `LIMIT top_k` to `LIMIT $top_k`

**Key insight:** The `search()` method uses an **importing CALL** (preceded by `WITH` variables). In Memgraph, imported variables remain in scope after `YIELD`, but they must be explicitly carried through any subsequent `WITH` projections. Dropping them (e.g., `WITH start_node, sim` without `max_depth`) makes them unbound. The fix keeps `max_depth` in the `WITH` projection alongside the yielded variables.

### 4.4 Fix `graph_algorithms.py`

**File:** `codebase_rag/graph_algorithms.py`

Change `*BFS 1 TO max_depth` to `*BFS 1..max_depth` and ensure `max_depth` is available as a variable:

```cypher
WITH $start_id AS start_id, $max_depth AS max_depth
MATCH (start) WHERE id(start) = start_id
MATCH path = (start)-[:CALLS|:DEFINES|:IMPORTS *BFS 1..max_depth]-(related)
```

---

## 5. Implementation Plan

### Phase 1: Critical Fixes (Immediate)

1. **Fix Cypher syntax errors in `hybrid_retrieval.py`**
   - Add `WITH` after `YIELD` before `WHERE`
   - Convert BFS bound parameter to variable (`$max_depth` → `max_depth`)
   - Change `1 TO` to `1..`

2. **Fix index name resolution**
   - Iterate over per-label indexes
   - Aggregate and deduplicate results
   - Apply global sort and `top_k`

3. **Fix `vector_store_memgraph.py`**
   - Apply same `WITH`/`YIELD`/`WHERE` fix
   - Fix BFS bounds syntax
   - Fix `LIMIT` to use parameter

4. **Fix `graph_algorithms.py`**
   - Fix BFS bounds syntax (`1 TO` → `1..`)

### Phase 2: Architecture Improvements (Follow-up)

1. **Centralize query generation**
   - Create a `HybridQueryBuilder` class in `memgraph_advanced/`
   - Share query logic between `hybrid_retrieval.py`, `document_search.py`, and `semantic_search.py`

2. **Add query validation**
   - Validate Cypher queries against Memgraph capabilities
   - Add pre-flight checks for index existence

---

## 6. Files to Modify

| File | Change |
|------|--------|
| `codebase_rag/memgraph_advanced/hybrid_retrieval.py` | Fix `_search_atomic` Cypher query; add per-label index iteration; add `_record_to_result` helper |
| `codebase_rag/vector_store_memgraph.py` | Fix `search()` Cypher queries (`WITH` after `YIELD`, BFS bounds, `LIMIT` parameter) |
| `codebase_rag/graph_algorithms.py` | Fix BFS bounds syntax (`1 TO` → `1..`) |
| `codebase_rag/tools/semantic_search.py` | Review for similar issues (if using atomic queries) |
| `codebase_rag/document/tools/document_search.py` | Document the correct pattern for reference |

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
def test_search_atomic_cypher_syntax():
    """Verify the generated Cypher query is syntactically correct."""
    retriever = HybridRetriever(...)
    captured_query = []
    retriever.graph_ingestor.fetch_all = lambda q, p: captured_query.append((q, p)) or []

    retriever.search("test query")

    query, params = captured_query[0]
    assert "YIELD node AS seed, similarity AS vec_sim" in query
    assert "WITH seed, vec_sim" in query  # Critical: WITH between YIELD and WHERE
    assert "*BFS 1.." in query  # Range syntax, not "1 TO"
    assert "LIMIT $top_k" in query  # Parameter in LIMIT
```

### 7.2 Integration Tests

```python
def test_hybrid_search_returns_results():
    """Verify hybrid search returns results without fallback."""
    with MemgraphIngestor(...) as ingestor:
        retriever = create_hybrid_retriever(ingestor)
        results = retriever.search("function definition")

        assert len(results) > 0
        assert all(r.vector_score >= 0 for r in results)
```

### 7.3 Direct Memgraph Syntax Verification

The following patterns were verified against Memgraph 3.8.1:

| Pattern | Result |
|---------|--------|
| `CALL ... YIELD ... WHERE` | ❌ Invalid |
| `CALL ... YIELD ... WITH ... WHERE` | ✅ Valid |
| `*BFS 1 TO 2` | ❌ Invalid |
| `*BFS 1..2` | ✅ Valid |
| `*BFS 1..$param` | ❌ Invalid |
| `WITH $p AS p THEN *BFS 1..p` | ✅ Valid |
| `LIMIT variable` | ❌ Invalid |
| `LIMIT $param` | ✅ Valid |
| `[:TYPE1|:TYPE2]` | ✅ Valid |
| `[:TYPE1|TYPE2]` | ✅ Valid |

### 7.4 Manual Verification

1. Start Memgraph with vector indexes
2. Index a codebase: `cgr index ./my-repo`
3. Run semantic search: `cgr query "error handling functions"`
4. Verify logs show `Atomic hybrid search returned N results` (not `Keyword fallback`)

---

## 8. Related Issues

| Issue | Relationship |
|-------|--------------|
| `hybrid_retrieval_settings_fix.md` | Pre-requisite: settings import must work (already applied) |
| OpenAI embedding fallback | Causes delays but is separate (fallback working correctly) |

---

## 9. Backward Compatibility

| Component | Impact |
|-----------|--------|
| Query API | No change — same input/output |
| Index structure | No change — uses existing per-label indexes |
| Configuration | No change — same settings used |
| Result quality | **Improved** — vector search now works instead of falling back to keyword search |

---

*Specification version: 2.0*
*Updated: 2026-04-20*
*Target: codebase_rag v0.x*
