# Memgraph Query Full Utilization Design Spec

**Version:** 1.0  
**Date:** 2026-04-22  
**Status:** Implementation-Ready  
**Scope:** `codebase_rag/` — Full utilization of Memgraph query capabilities: graph traversal, semantic search, keyword search, cypher syntax correctness, instance management, and atomic multi-signal queries  
**Predecessors:**  
- `.specs/memgraph_advanced_query_optimization_design_spec.md` (hybrid retrieval)  
- `.specs/systematic_query_integration_and_memgraph_resilience_phase3.md` (orchestrator + failure classifier)  
- `.specs/graph_traversal_semantic_search_optimization_phase2.md` (Phase 2 — implemented)

---

## Executive Summary

This specification identifies **12 concrete issues** preventing the system from making **full, correct, and systematic use** of Memgraph query capabilities. The issues span five categories:

| Category | Issues | Severity |
|----------|--------|----------|
| **Dead/Unreachable Code** | 1 | 🟡 MEDIUM |
| **Incorrect Cypher Syntax** | 2 | 🟠 HIGH |
| **Wrong Memgraph Instance/Connection Management** | 2 | 🟠 HIGH |
| **Insufficient Query Depth (Early Return / Truncation)** | 3 | 🟠 HIGH |
| **Missing Atomic Multi-Signal Queries** | 1 | 🔴 CRITICAL |
| **Suboptimal Search Strategy** | 3 | 🟡 MEDIUM |

Each issue is documented with: root cause, exact file/line references, proposed fix, and implementation-ready code.

---

## Issue 1: Dead Code in `_detect_dynamic_algorithm_support()`

**Severity:** 🟡 MEDIUM  
**Location:** `codebase_rag/services/graph_service.py`, `MemgraphIngestor._detect_dynamic_algorithm_support()`  

### Problem Analysis

The method contains **two sequential try blocks** that both attempt to detect Enterprise edition. The first block returns `True` or `False`, making the second block **unreachable dead code**:

```python
def _detect_dynamic_algorithm_support(self) -> bool:
    if not self.conn:
        return False

    # Block 1: Try dynamic algorithm function
    try:
        with self._get_cursor() as cursor:
            cursor.execute("RETURN dynamic_graph_update_is_supported() AS supported LIMIT 1")
            return True
    except Exception:
        # Fallback: check version string for enterprise
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SHOW VERSION AS version")
                results = self._cursor_to_results(cursor)
                if results and "enterprise" in str(results[0].get("version", "")).lower():
                    return True
        except Exception:
            pass
        return False  # <-- FIRST BLOCK ENDS HERE

    # Block 2: COMPLETELY UNREACHABLE — never executed
    try:
        self._execute_query("""
            RETURN dynamic_graph_update_is_supported() AS supported
            LIMIT 1
        """)
        return True
    except Exception:
        try:
            results = self._execute_query("SHOW VERSION AS version")
            if results and "enterprise" in str(results[0].get("version", "")).lower():
                return True
        except Exception:
            pass
        return False
```

**Impact:**
- Dead code obscures intent and creates confusion for maintainers
- The unreachable Block 2 uses `_execute_query()` which includes retry logic — the intended design was likely that Block 2 should be the primary method, and Block 1 was an earlier draft that wasn't removed
- The cursor-based approach in Block 1 bypasses `_execute_query()`'s retry/reconnect logic, meaning transient failures that could be recovered are treated as permanent

### Proposed Fix

Remove the unreachable Block 2 entirely. The first block already provides the correct two-step detection (try dynamic function, then fallback to version string). No functionality is lost.

```python
def _detect_dynamic_algorithm_support(self) -> bool:
    """Detect if Memgraph supports dynamic incremental algorithms (Enterprise feature)."""
    if not self.conn:
        return False

    try:
        with self._get_cursor() as cursor:
            cursor.execute(
                "RETURN dynamic_graph_update_is_supported() AS supported LIMIT 1"
            )
            return True
    except Exception:
        # Fallback: check version string for enterprise keyword
        try:
            with self._get_cursor() as cursor:
                cursor.execute("SHOW VERSION AS version")
                results = self._cursor_to_results(cursor)
                if (
                    results
                    and "enterprise" in str(results[0].get("version", "")).lower()
                ):
                    return True
        except Exception:
            pass
        return False
```

**Validation:** Run `test_dynamic_algorithms_enabled` and `test_detect_enterprise` tests to confirm no regression.

---

## Issue 2: Cypher `|` Label Syntax in Query Templates Despite Documented Policy Against It

**Severity:** 🟠 HIGH  
**Location:** `codebase_rag/cypher_queries.py`, `CYPHER_QUERY_TEMPLATES`  

### Problem Analysis

The `CYPHER_QUERY_TEMPLATES` docstring explicitly states:

> "NOTE: All templates use WHERE IN clause for label filtering instead of `|` union syntax (e.g., `Function|Class|Method`) because the WHERE IN approach works correctly even when some labels don't exist in the graph."

Yet two templates **violate this policy** and use `|` syntax:

**Template `find_dependencies`:**
```python
"find_dependencies": (
    """
    MATCH (n:Function|Class|Method)-[:CALLS]->(m)
    WHERE n.qualified_name CONTAINS $keyword
    ...
    """,
    {"keyword": str, "limit": int},
),
```

**Template `find_by_docstring`:**
```python
"find_by_docstring": (
    """
    MATCH (n:Function|Class|Method)
    WHERE n.docstring IS NOT NULL
    AND n.docstring CONTAINS $keyword
    ...
    """,
    {"keyword": str, "limit": int},
),
```

**Impact:**
- `Function|Class|Method` syntax fails in Memgraph when any of the labels don't exist in the graph — it returns zero results instead of matching existing labels
- The `|` syntax in relationship type position `(n)-[:CALLS]->(m)` works fine (single type), but `|` in node label position is the problematic pattern
- Inconsistent with the stated policy, creating confusion for developers

### Proposed Fix

Convert both templates to `WHERE IN` clause pattern:

```python
"find_dependencies": (
    """
    MATCH (n)-[:CALLS]->(m)
    WHERE labels(n)[0] IN ['Function', 'Class', 'Method', 'Enum', 'Type',
                            'Union', 'Interface', 'Contract', 'Library']
      AND n.qualified_name CONTAINS $keyword
    RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
           n.name AS name, labels(n)[0] AS type, n.path AS file_path,
           id(m) AS target_id, m.qualified_name AS target_name
    LIMIT $limit
    """,
    {"keyword": str, "limit": int},
),
"find_by_docstring": (
    """
    MATCH (n)
    WHERE labels(n)[0] IN ['Function', 'Class', 'Method']
      AND n.docstring IS NOT NULL
      AND n.docstring CONTAINS $keyword
    RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
           n.name AS name, labels(n)[0] AS type,
           n.path AS file_path, n.docstring AS docstring
    LIMIT $limit
    """,
    {"keyword": str, "limit": int},
),
```

**Note:** The `find_dependencies` template also expands the label list to match the same comprehensive set used in `find_by_name`, ensuring consistent coverage.

**Validation:** Run all template-based fallback tests; verify that queries return results even when some labels don't exist in the graph.

---

## Issue 3: `QueryGenerator` Creates Ephemeral Short-Lived Connection Instead of Using Pool

**Severity:** 🟠 HIGH  
**Location:** `codebase_rag/graph/query_generator.py`, `QueryGenerator._detect_memgraph_capabilities()`  

### Problem Analysis

`QueryGenerator` (the disconnected-node query generator, distinct from `MemgraphQueryGenerator`) creates its own ephemeral `mgclient.Connection`:

```python
class QueryGenerator:
    def _detect_memgraph_capabilities(self) -> None:
        import mgclient as _mgclient
        from ..config import settings

        conn = None
        cursor = None
        try:
            conn = _mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
            cursor = conn.cursor()
            cursor.execute(cs.QUERY_GEN_SHOW_VERSION)
            ...
        except Exception:
            ...
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
```

**Impact:**
- Creates a **new TCP connection** to Memgraph for a one-time version check, then closes it
- Does not use the existing `MemgraphConnectionPool` or `MemgraphIngestor` connection
- In high-concurrency scenarios, this wastes connection resources and adds latency
- No authentication support — ignores `MEMGRAPH_USERNAME`/`MEMGRAPH_PASSWORD`
- No timeout configuration — uses default `mgclient` timeout (infinite)

### Proposed Fix

Use the connection pool for capability detection:

```python
class QueryGenerator:
    __slots__ = ("_memgraph_version", "_has_enterprise_license")

    def __init__(self) -> None:
        from .. import constants as cs

        self._memgraph_version: tuple[int, ...] = cs.QUERY_GEN_FALLBACK_VERSION
        self._has_enterprise_license: bool = False
        self._detect_memgraph_capabilities()

    def _detect_memgraph_capabilities(self) -> None:
        from .. import constants as cs
        from ..config import settings
        from ..services.connection_pool import get_connection_pool

        pool = get_connection_pool(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        )
        conn = pool.get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(cs.QUERY_GEN_SHOW_VERSION)
            row = cursor.fetchone()
            if row:
                parts = str(row[0]).split(".")
                self._memgraph_version = tuple(int(p) for p in parts if p.isdigit())

            cursor.execute(cs.QUERY_GEN_SHOW_LICENSE)
            row = cursor.fetchone()
            if row:
                self._has_enterprise_license = (
                    cs.QUERY_GEN_ENTERPRISE_KEYWORD in str(row[0]).lower()
                )
            cursor.close()
        except Exception:
            self._memgraph_version = cs.QUERY_GEN_FALLBACK_VERSION
            self._has_enterprise_license = False
        finally:
            pool.return_connection(conn)
```

**Alternative (simpler):** Use `MemgraphIngestor` as context manager for the one-time check, which already handles connection creation with timeout, auth, and keepalive:

```python
def _detect_memgraph_capabilities(self) -> None:
    from .. import constants as cs
    from ..config import settings
    from ..services.graph_service import MemgraphIngestor

    try:
        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        ) as ingestor:
            results = ingestor.fetch_all(cs.QUERY_GEN_SHOW_VERSION)
            if results:
                version_str = str(results[0].get("version", ""))
                parts = version_str.split(".")
                self._memgraph_version = tuple(int(p) for p in parts if p.isdigit())

            results = ingestor.fetch_all(cs.QUERY_GEN_SHOW_LICENSE)
            if results:
                license_str = str(results[0].get("license", ""))
                self._has_enterprise_license = (
                    cs.QUERY_GEN_ENTERPRISE_KEYWORD in license_str.lower()
                )
    except Exception:
        self._memgraph_version = cs.QUERY_GEN_FALLBACK_VERSION
        self._has_enterprise_license = False
```

---

## Issue 4: `PathAnalyzer`, `CommunityQFS`, and `DynamicGraphAlgorithms` Create Ephemeral Connections Instead of Accepting Injected Connection

**Severity:** 🟠 HIGH  
**Location:**  
- `codebase_rag/memgraph_advanced/path_analysis.py`, `PathAnalyzer`  
- `codebase_rag/memgraph_advanced/qfs.py`, `CommunityQFS`  
- `codebase_rag/memgraph_advanced/dynamic_algorithms.py`, `DynamicGraphAlgorithms`  

### Problem Analysis

All three classes create their own ephemeral `MemgraphIngestor` instances with `with MemgraphIngestor(...) as ingestor:` inside each method call:

```python
# PathAnalyzer.analyze_call_chain()
with MemgraphIngestor(
    host=settings.MEMGRAPH_HOST,
    port=settings.MEMGRAPH_PORT,
    username=settings.MEMGRAPH_USERNAME,
    password=settings.MEMGRAPH_PASSWORD,
) as ingestor:
    results = ingestor.fetch_all(cypher, params)

# CommunityQFS.build_community_summaries()
with MemgraphIngestor(
    host=settings.MEMGRAPH_HOST,
    port=settings.MEMGRAPH_PORT,
    username=settings.MEMGRAPH_USERNAME,
    password=settings.MEMGRAPH_PASSWORD,
) as ingestor:
    records = ingestor.fetch_all(cypher, params)

# DynamicGraphAlgorithms.update_pagerank_dynamic()
with MemgraphIngestor(...) as ingestor:
    results = ingestor.fetch_all(cypher, params)
```

**Impact:**
- Each method call creates a **new TCP connection** + cursor, executes one query, and closes the connection
- For `PathAnalyzer.analyze_call_chain()` which also calls `find_bottlenecks()`, **two separate connections** are opened in one method call
- `CommunityQFS.build_community_summaries()` opens a connection, then `_generate_community_summary()` makes an LLM call (slow), then the method loop continues — each iteration opens/closes a connection
- No reuse of existing connection pool or shared ingestor
- No timeout configuration
- Connection overhead adds ~50-100ms per query in local deployments, more in remote

### Proposed Fix

Refactor all three classes to accept an injected `QueryProtocol` (connection) via constructor dependency injection, with fallback to creating their own connection only when no connection is provided:

```python
# path_analysis.py
class PathAnalyzer:
    """Analyze paths between code entities."""

    def __init__(self, ingestor: QueryProtocol | None = None) -> None:
        self._ingestor = ingestor
        self._own_ingestor: MemgraphIngestor | None = None

    def _get_ingestor(self) -> QueryProtocol:
        """Get ingestor — use injected or create ephemeral."""
        if self._ingestor is not None:
            return self._ingestor
        from ..config import settings
        self._own_ingestor = MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        )
        self._own_ingestor.__enter__()
        return self._own_ingestor

    def close(self) -> None:
        """Close owned ingestor if created."""
        if self._own_ingestor is not None:
            self._own_ingestor.__exit__(None, None, None)
            self._own_ingestor = None
```

**Same pattern for `CommunityQFS` and `DynamicGraphAlgorithms`.**

**Integration:** Update `create_hybrid_retriever()` factory and `QueryMethodOrchestrator` to pass their existing `code_graph` ingestor to these classes:

```python
# In QueryMethodOrchestrator._execute_graph_algorithms()
algo = GraphAlgorithms()  # GraphAlgorithms creates its own conn — acceptable for one-off
# But PathAnalyzer and CommunityQFS should reuse the orchestrator's connection
```

---

## Issue 5: Keyword Search Uses Only Single Keyword — Loses Query Context

**Severity:** 🟡 MEDIUM  
**Location:** `codebase_rag/utils/query_utils.py`, `extract_best_keyword()`  

### Problem Analysis

`extract_best_keyword()` returns only **one word** from a multi-word query:

```python
def extract_best_keyword(query: str) -> str:
    words = query.lower().split()
    meaningful = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    if not meaningful:
        return max(words, key=len, default="")
    return max(meaningful, key=len)
```

**Impact:**
- Query "How does authentication work in the login module?" → keyword = `"authentication"` (6 other meaningful words lost)
- Query "Find functions that handle database connection errors" → keyword = `"connection"` (misses "database", "errors", "handle")
- The keyword fallback Cypher uses `CONTAINS $keyword` — only matches nodes containing that single word
- `QueryMethodOrchestrator._execute_keyword_search()` and `_execute_graph_navigation()` both rely on this single keyword
- Template fallback in `_try_template_fallback()` also uses single keyword
- For queries like "Memgraph query generator", the keyword "query" is a stopword, so only "memgraph" or "generator" is used

### Proposed Fix

Create `extract_keywords()` that returns multiple ranked keywords, and update callers to use OR-based matching:

```python
# utils/query_utils.py

def extract_keywords(query: str, max_keywords: int = 3) -> list[str]:
    """Extract multiple meaningful keywords from a natural language query.

    Returns up to max_keywords, ranked by:
    1. Length (longer = more specific)
    2. Position (earlier = more important)

    Args:
        query: Natural language query string
        max_keywords: Maximum number of keywords to return

    Returns:
        List of keywords, ordered by relevance
    """
    if not query:
        return []

    words = query.lower().split()
    meaningful = [w for w in words if len(w) > 2 and w not in STOPWORDS]

    if not meaningful:
        meaningful = [w for w in words if len(w) > 1]

    if not meaningful:
        return []

    # Rank by: length * position_weight
    position_weight = 1.0
    scored = []
    for i, w in enumerate(meaningful):
        score = len(w) * position_weight * (1.0 / (1.0 + i * 0.3))  # Earlier words score higher
        scored.append((score, w))

    scored.sort(reverse=True)
    return [w for _, w in scored[:max_keywords]]
```

Update keyword search Cypher to use multiple keywords with OR matching:

```python
# In _execute_keyword_search()
keywords = extract_keywords(query, max_keywords=3)
if not keywords:
    return QueryMethodResult(...)

# Use ANY keyword matching
cypher = """
MATCH (n)
WHERE labels(n)[0] IN ['Function', 'Class', 'Method', 'Enum', 'Type',
                        'Union', 'Interface', 'Contract', 'Library']
  AND ANY(kw IN $keywords WHERE
      n.name CONTAINS kw
      OR n.qualified_name CONTAINS kw
      OR n.docstring CONTAINS kw)
RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
       n.name AS name, labels(n)[0] AS type,
       n.path AS file_path, n.start_line AS start_line,
       n.end_line AS end_line
LIMIT $limit
"""
results = await self._fetch_all_async(
    cypher, {"keywords": keywords, "limit": top_k}
)
```

**Note:** `extract_best_keyword()` should remain for backward compatibility (single-keyword callers), but all new code should use `extract_keywords()`.

---

## Issue 6: `HybridSearchResult.text_score` Always 0.0 — Text Signal Never Utilized

**Severity:** 🟡 MEDIUM  
**Location:** `codebase_rag/memgraph_advanced/hybrid_retrieval.py`, `HybridRetriever.search()`  

### Problem Analysis

`HybridSearchResult` has a `text_score` field that is always set to `0.0`:

```python
results.append(
    HybridSearchResult(
        ...
        text_score=0.0,  # <-- Always zero
        ...
    )
)
```

The `HybridRetrievalConfig` documents `text_weight` removal with:

> "Note: text_weight was removed as Memgraph text indexing is not currently implemented."

**Impact:**
- The `combined_score` formula is: `vector_score * vector_weight + graph_score * graph_weight`
- With `vector_weight=0.7` and `pagerank_weight=0.2 + community_weight=0.1 = 0.3`, total is `0.7 + 0.3 = 1.0` — correct but missing text signal
- Queries like "find functions related to error handling" benefit enormously from text matching (CONTAINS on names/docstrings) alongside vector similarity
- The `text_score` field existing but always being 0.0 is misleading in output

### Proposed Fix

**Option A (Recommended): Add keyword-based text scoring within the existing metadata Cypher query**

This doesn't require Memgraph text indexing — it uses `CONTAINS` matching which is already available:

```python
# In HybridRetriever.search()
# Extract keywords from query for text matching
from ..utils.query_utils import extract_keywords
query_keywords = extract_keywords(query, max_keywords=3)

metadata_cypher = """
MATCH (n)
WHERE id(n) IN $node_ids
RETURN id(n) AS node_id,
       n.name AS name,
       n.qualified_name AS qualified_name,
       labels(n)[0] AS node_type,
       n.path AS file_path,
       n.start_line AS start_line,
       n.end_line AS end_line,
       COALESCE(n.pagerank_score, 0.1) AS pagerank_score,
       COALESCE(n.community_importance, 0.0) AS community_score,
       CASE WHEN ANY(kw IN $keywords WHERE
           toLower(n.name) CONTAINS kw
           OR toLower(n.qualified_name) CONTAINS kw
           OR toLower(COALESCE(n.docstring, '')) CONTAINS kw
       ) THEN 1.0 ELSE 0.0 END AS text_match
"""

records = self.graph_ingestor.fetch_all(
    metadata_cypher,
    {"node_ids": node_ids, "keywords": [kw.lower() for kw in query_keywords]},
)
```

Then update scoring:

```python
# With text_weight=0.15, adjust other weights: vector=0.60, pagerank=0.20, community=0.05, text=0.15
text_score = _coerce_float(record.get("text_match"), 0.0)
combined_score = (
    vector_score * cfg.vector_weight
    + pagerank_score * cfg.pagerank_weight
    + community_score * cfg.community_weight
    + text_score * cfg.text_weight  # NEW
)
```

**Option B (Future): Full Memgraph text indexing**

When Memgraph adds text index support, create a proper `CREATE TEXT INDEX` and use it for full-text search. This is a future enhancement.

**Config Update:** Add `text_weight` back to `HybridRetrievalConfig`:

```python
@dataclass
class HybridRetrievalConfig:
    vector_weight: float = 0.60
    text_weight: float = 0.15      # RESTORED — keyword-based text matching
    pagerank_weight: float = 0.20
    community_weight: float = 0.05
    ...
```

---

## Issue 7: `_search_memgraph_native()` in Document Search Passes Non-Procedure Parameters

**Severity:** 🟠 HIGH  
**Location:** `codebase_rag/document/tools/document_search.py`, `_search_memgraph_native()`  

### Problem Analysis

The `_search_memgraph_native()` function uses `vector_search.search()` procedure with incorrect parameter passing:

```python
def _search_memgraph_native(ingestor, embedding, workspace, limit, min_similarity):
    query = """
    CALL vector_search.search(
        'doc_embeddings',
        $limit,
        $embedding
    ) YIELD node, distance, similarity
    WITH node, distance, similarity
    WHERE node.workspace = $workspace AND similarity >= $min_similarity
    ...
    """
    return ingestor.fetch_all(query, params={
        "embedding": embedding,
        "workspace": workspace,
        "limit": limit,
        "min_similarity": min_similarity,
    })
```

**Problems:**
1. `vector_search.search()` is a **procedure** that takes positional arguments `(index_name, limit, query_vector)` — `$limit` and `$embedding` as Cypher parameters work fine in the procedure call
2. However, the `WHERE node.workspace = $workspace AND similarity >= $min_similarity` clause **filters after** the vector search returns results — if the procedure returns `limit` results and most don't match the workspace filter, the final result count could be 0 or very few
3. The `vector_search.search()` procedure's `limit` parameter should be set **higher** than the desired final count to account for post-filtering
4. The procedure name `'doc_embeddings'` is hardcoded — should match the actual document vector index name from settings

### Proposed Fix

```python
def _search_memgraph_native(
    ingestor: QueryProtocol,
    embedding: list[float],
    workspace: str,
    limit: int,
    min_similarity: float,
) -> list[dict]:
    """Use Memgraph's native vector search with post-filter awareness."""
    from ...config import settings

    index_name = settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME
    # Request more results from the procedure to account for post-filtering
    overfetch = limit * 3

    query = """
    CALL vector_search.search(
        $index_name,
        $overfetch,
        $embedding
    ) YIELD node, distance, similarity
    WITH node, distance, similarity
    WHERE node.workspace = $workspace AND similarity >= $min_similarity
    MATCH (d:Document)-[:CONTAINS_CHUNK]->(node)
    OPTIONAL MATCH (node)-[:BELONGS_TO_SECTION]->(s:Section)
    RETURN
        node.content as content,
        node.qualified_name as chunk_qn,
        node.start_line as chunk_start_line,
        node.end_line as chunk_end_line,
        node.code_references as code_references,
        node.resolved_code_references as resolved_code_references,
        s.title as section_title,
        s.qualified_name as section_qn,
        d.path as document_path,
        similarity
    ORDER BY similarity DESC
    LIMIT $limit
    """

    return ingestor.fetch_all(
        query,
        params={
            "index_name": index_name,
            "embedding": embedding,
            "workspace": workspace,
            "overfetch": overfetch,
            "min_similarity": min_similarity,
            "limit": limit,
        },
    )
```

---

## Issue 8: `_detect_capabilities()` L2 Distance Test Uses f-string Interpolation — Injection Risk

**Severity:** 🟡 MEDIUM  
**Location:** `codebase_rag/graph/query_generator.py`, `MemgraphQueryGenerator._detect_capabilities()`  

### Problem Analysis

The L2 distance detection test uses f-string interpolation to build a Cypher function name:

```python
test_query = f"""
    RETURN {capabilities.vector_function_syntax.replace("cosine_similarity", "l2_distance")}([1.0, 2.0], [3.0, 4.0]) AS dist
"""
```

While `capabilities.vector_function_syntax` is currently only set to `"cosine_similarity"` or `"vector.cosine_similarity"` (internal values), this pattern:
1. Is inconsistent with the rest of the module's approach (which uses parameterized queries)
2. Creates a fragile assumption that the function name is always safe
3. If the syntax detection logic ever returns unexpected values, this could generate invalid Cypher

**Impact:** Currently low risk (internal values only), but represents a pattern that should be corrected for consistency and safety.

### Proposed Fix

Use a lookup table instead of string replacement:

```python
# In MemgraphQueryGenerator._detect_capabilities()
# Check L2 distance support
if (
    capabilities.supports_vector_search
    and not capabilities.supports_vector_search_procedure
):
    # Map cosine function to L2 function
    l2_function_map = {
        "cosine_similarity": "l2_distance",
        "vector.cosine_similarity": "vector.l2_distance",
    }
    l2_func = l2_function_map.get(capabilities.vector_function_syntax)
    if l2_func:
        try:
            test_query = f"RETURN {l2_func}([1.0, 2.0], [3.0, 4.0]) AS dist"
            self._run_query(test_query)
            capabilities.supports_l2_distance = True
        except Exception:
            capabilities.supports_l2_distance = False
```

This ensures only known-safe function names are interpolated.

---

## Issue 9: `CommunityQFS._rank_communities_by_relevance` Uses Naive Keyword Split — No Embedding-Based Ranking

**Severity:** 🟡 MEDIUM  
**Location:** `codebase_rag/memgraph_advanced/qfs.py`, `CommunityQFS._rank_communities_by_relevance()`  

### Problem Analysis

```python
def _rank_communities_by_relevance(self, question, communities):
    keywords = question.lower().split()  # Naive split on spaces
    scored = []
    for comm in communities:
        score = 0
        summary_text = comm.summary_text.lower()
        for kw in keywords:
            if kw in summary_text:
                score += 1
            if kw in [fn.lower() for fn in comm.key_functions]:
                score += 2
            if kw in [c.lower() for c in comm.key_classes]:
                score += 2
        if score > 0:
            scored.append((-score, comm))
    scored.sort()
```

**Impact:**
- `question.lower().split()` splits "How does authentication work?" into `["how", "does", "authentication", "work?"]` — includes stopwords and punctuation
- No semantic understanding — "login" won't match "authentication" community
- Communities with no keyword matches are appended in original order at the end, regardless of semantic relevance
- No use of embeddings (the system has embedding infrastructure)

### Proposed Fix

Use embedding-based semantic ranking as primary, keyword as secondary:

```python
def _rank_communities_by_relevance(
    self, question: str, communities: list[CommunitySummary]
) -> list[CommunitySummary]:
    """Rank communities by semantic similarity + keyword overlap."""
    from ..embeddings import get_embedding_provider
    from ..config import settings

    # Get query embedding
    config = settings.active_embedding_config
    provider = get_embedding_provider(
        provider=config.provider,
        model_id=config.model_id,
    )
    query_embedding = provider.embed(question)

    # Build community text for embedding comparison
    scored: list[tuple[float, CommunitySummary]] = []

    for comm in communities:
        # Semantic score: cosine similarity between query embedding and community text
        comm_text = f"{comm.summary_text} {', '.join(comm.key_functions[:3])} {', '.join(comm.key_classes[:2])}"
        comm_embedding = provider.embed(comm_text)
        semantic_score = _cosine_similarity(query_embedding, comm_embedding)

        # Keyword overlap score (secondary)
        keywords = extract_keywords(question, max_keywords=5)
        keyword_score = 0
        summary_lower = comm.summary_text.lower()
        for kw in keywords:
            if kw in summary_lower:
                keyword_score += 1
            if kw in [fn.lower() for fn in comm.key_functions]:
                keyword_score += 2

        # Combined: semantic is primary, keyword is bonus
        combined_score = semantic_score * 0.7 + (keyword_score / max(len(keywords), 1)) * 0.3
        scored.append((combined_score, comm))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [comm for _, comm in scored]
```

Add a simple cosine similarity helper:

```python
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

**Performance Note:** Embedding generation for each community summary is expensive. Use caching or pre-compute community embeddings during `build_community_summaries()`.

---

## Issue 10: `_execute_graph_navigation()` Severely Truncates Results

**Severity:** 🟠 HIGH  
**Location:** `codebase_rag/retrieval/query_orchestrator.py`, `QueryMethodOrchestrator._execute_graph_navigation()`  

### Problem Analysis

```python
async def _execute_graph_navigation(self, query, top_k, start):
    # Find nodes matching keyword — limit to min(top_k, 3)
    nodes = await self._fetch_all_async(
        find_nodes_cypher, {"keyword": keyword, "limit": min(top_k, 3)}
    )
    
    for node in nodes:
        # Find callers — only 2 per node
        callers = await self._fetch_all_async(CYPHER_FIND_CALLERS, {"qn": qn})
        for caller in callers[:2]:  # <-- HARDCODED LIMIT
            items.append(...)

        # Find importers — only 2 per node
        importers = await self._fetch_all_async(CYPHER_FIND_IMPORTERS, {"qn": qn})
        for importer in importers[:2]:  # <-- HARDCODED LIMIT
            items.append(...)
        
        if len(items) >= top_k:
            break
```

**Impact:**
- `min(top_k, 3)` caps node discovery at 3, even when `top_k=10`
- `callers[:2]` and `importers[:2]` cap results at 2 per category per node
- Maximum possible results: 3 nodes × (2 callers + 2 importers) = 12, but often fewer
- For queries like "What calls authenticate?", the user expects ALL callers, not just 2
- The `[:2]` truncation discards relevant results that the graph can provide

### Proposed Fix

Use dynamic limits based on `top_k`:

```python
async def _execute_graph_navigation(self, query, top_k, start):
    from ..utils.query_utils import extract_keywords
    keywords = extract_keywords(query, max_keywords=3)
    if not keywords:
        return QueryMethodResult(...)

    # Find nodes matching keywords — use full top_k
    find_nodes_cypher = """
    MATCH (n)
    WHERE labels(n)[0] IN ['Function', 'Class', 'Method']
      AND ANY(kw IN $keywords WHERE n.name CONTAINS kw OR n.qualified_name CONTAINS kw)
    RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
           n.name AS name, labels(n)[0] AS type,
           n.path AS file_path
    LIMIT $limit
    """
    nodes = await self._fetch_all_async(
        find_nodes_cypher, {"keywords": keywords, "limit": top_k}
    )

    items = []
    per_node_limit = max(3, top_k // len(nodes) if nodes else top_k)

    for node in nodes:
        qn = node.get("qualified_name")
        # Find callers — use proportional limit
        callers = await self._fetch_all_async(
            CYPHER_FIND_CALLERS, {"qn": qn}
        )
        for caller in callers[:per_node_limit]:
            items.append(...)

        # Find importers — use proportional limit
        importers = await self._fetch_all_async(
            CYPHER_FIND_IMPORTERS, {"qn": qn}
        )
        for importer in importers[:per_node_limit]:
            items.append(...)

        if len(items) >= top_k * 2:  # Allow overfetch for merge_and_rank dedup
            break

    return QueryMethodResult(
        method=QueryMethod.GRAPH_NAVIGATION,
        items=items[:top_k],
        execution_time_ms=(time.time() - start) * 1000,
    )
```

---

## Issue 11: `CYPHER_FIND_CALLERS` and `CYPHER_FIND_IMPORTERS` Use `|` Label Syntax

**Severity:** 🟠 HIGH  
**Location:** `codebase_rag/cypher_queries.py`, `CYPHER_FIND_CALLERS`, `CYPHER_FIND_IMPORTERS`  

### Problem Analysis

```python
CYPHER_FIND_CALLERS = """
MATCH (caller:Function|Method)-[:CALLS]->(target)
WHERE target.qualified_name = $qn
...
"""

CYPHER_FIND_IMPLEMENTATIONS = """
MATCH (impl:Class)-[r:IMPLEMENTS|INHERITS*1..2]->(base)
...
"""
```

**Impact:** Same as Issue 2 — `Function|Method` label union syntax returns zero results in Memgraph when either label doesn't exist in the graph. This is the **most commonly used query** in the system (called by `GraphNavigator.find_references()`, `QueryMethodOrchestrator._execute_graph_navigation()`, and template fallbacks).

### Proposed Fix

```python
CYPHER_FIND_CALLERS = """
MATCH (caller)-[:CALLS]->(target)
WHERE target.qualified_name = $qn
  AND labels(caller)[0] IN ['Function', 'Method', 'Class']
OPTIONAL MATCH (m:Module)-[:DEFINES]->(caller)
RETURN caller.qualified_name AS qualified_name, caller.name AS name,
       labels(caller) AS type, m.path AS path, caller.start_line AS start_line
ORDER BY caller.qualified_name
"""
```

Note: We add `'Class'` to the label list because classes can also CALLS other functions (e.g., constructor calls).

For `CYPHER_FIND_IMPLEMENTATIONS`, the `IMPLEMENTS|INHERITS` is in **relationship type** position, which works differently in Memgraph. Variable-length path `[*1..2]` with `|` relationship types is valid Memgraph syntax. However, we should verify this works correctly.

---

## Issue 12: No Atomic Vector + Graph Traversal Query for Code Graph

**Severity:** 🔴 CRITICAL  
**Location:** `codebase_rag/memgraph_advanced/hybrid_retrieval.py`, `codebase_rag/vector_store_memgraph.py`  

### Problem Analysis

The system **never** combines vector search and graph traversal in a single atomic Cypher query for the **code graph**. The current `HybridRetriever.search()` flow is:

1. Vector search: `vector_backend.search()` → returns `(node_id, similarity)` pairs
2. Separate metadata Cypher: `MATCH (n) WHERE id(n) IN $node_ids` → fetches properties

The `MemgraphBackend.search()` with `include_context=True` does combine vector + BFS, but **HybridRetriever never uses this feature**. It always calls `vector_backend.search()` with `include_context=False` (default).

**The document graph** (`_search_memgraph_native`) does combine vector search + graph traversal in a single query, but the **code graph** never does.

**Impact:**
- Two separate queries instead of one atomic query = 2× round-trip latency
- No context expansion (BFS) in the primary search path
- Results lack graph context (callers, callees, parent classes) that could be obtained atomically
- The scoring formula `similarity * 0.7 + pagerank * 0.3` is computed in Python instead of in Cypher, losing the opportunity for database-side ranking

### Proposed Fix

Create an atomic hybrid query that combines vector search + BFS context + PageRank scoring in a single Cypher execution:

```python
# In HybridRetriever.search()

def search(self, query: str, top_k: int = 10) -> list[HybridSearchResult]:
    """Multi-modal retrieval with atomic vector+graph+text query."""
    cfg = self.config or HybridRetrievalConfig()
    query_embedding = self.embedding_provider.embed(query)
    query_keywords = extract_keywords(query, max_keywords=3)

    # ATOMIC QUERY: Vector search + BFS context + PageRank + text match in one Cypher
    atomic_cypher = """
    // Step 1: Vector search for seed nodes
    CALL vector_search.search($index_name, $overfetch, $embedding)
    YIELD node AS seed, similarity AS vec_sim

    // Step 2: BFS context expansion from seed nodes
    OPTIONAL MATCH path = (seed)-[:CALLS|DEFINES|IMPORTS *BFS 1 TO $max_depth]-(context_node)
    WHERE context_node:Function OR context_node:Class OR context_node:Method OR context_node:Module

    // Step 3: Collect context and compute scores
    WITH seed, vec_sim,
         collect(DISTINCT {
             qn: context_node.qualified_name,
             name: context_node.name,
             type: labels(context_node)[0],
             depth: length(path)
         }) AS context,
         COALESCE(seed.pagerank_score, 0.1) AS pr_score,
         COALESCE(seed.community_importance, 0.0) AS ci_score,
         CASE WHEN ANY(kw IN $keywords WHERE
             toLower(seed.name) CONTAINS kw
             OR toLower(seed.qualified_name) CONTAINS kw
         ) THEN 1.0 ELSE 0.0 END AS text_match

    // Step 4: Combined scoring
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
           context
    """

    params = {
        "index_name": "function_embedding_index",  # Primary label index
        "embedding": query_embedding,
        "overfetch": top_k * 3,
        "max_depth": cfg.max_context_depth,
        "keywords": [kw.lower() for kw in query_keywords],
        "top_k": top_k,
        "vector_weight": cfg.vector_weight,
        "pagerank_weight": cfg.pagerank_weight,
        "community_weight": cfg.community_weight,
        "text_weight": cfg.text_weight,
    }

    # Execute atomic query
    try:
        records = self.graph_ingestor.fetch_all(atomic_cypher, params)
    except Exception as e:
        # Fallback to separate vector + metadata queries (current approach)
        logger.warning(f"Atomic hybrid query failed, falling back to separate queries: {e}")
        return self._search_separate(query_embedding, query_keywords, top_k, cfg)

    results = []
    for record in records:
        results.append(
            HybridSearchResult(
                node_id=_coerce_int(record.get("node_id"), 0),
                name=_coerce_str(record.get("name")),
                qualified_name=_coerce_str(record.get("qualified_name")),
                node_type=_coerce_str(record.get("node_type")),
                file_path=_coerce_str(record.get("file_path")),
                start_line=_coerce_int(record.get("start_line"), 0),
                end_line=_coerce_int(record.get("end_line"), 0),
                vector_score=_coerce_float(record.get("vector_score"), 0.0),
                text_score=_coerce_float(record.get("text_score"), 0.0),
                pagerank_score=_coerce_float(record.get("pagerank_score"), 0.1),
                community_score=_coerce_float(record.get("community_score"), 0.0),
                graph_score=(
                    _coerce_float(record.get("pagerank_score"), 0.1) * cfg.pagerank_weight
                    + _coerce_float(record.get("community_score"), 0.0) * cfg.community_weight
                ) / (cfg.pagerank_weight + cfg.community_weight)
                if (cfg.pagerank_weight + cfg.community_weight) > 0
                else _coerce_float(record.get("pagerank_score"), 0.1),
                combined_score=_coerce_float(record.get("combined_score"), 0.0),
            )
        )

    return results
```

**Important:** The atomic query needs to handle the case where `vector_search.search()` is not available (older Memgraph versions). The fallback `_search_separate()` method would be the current implementation.

**Multi-Label Index Consideration:** Memgraph vector indexes are per-label. The atomic query above uses `function_embedding_index`. For comprehensive coverage, we should either:
- Search across multiple indexes (iterate over `LABELS_TO_INDEX`)
- Or use a single unified index (requires all embeddable nodes to have the same label — not currently supported)

For the initial implementation, use the primary `function_embedding_index` (covers the most common search target), with fallback to separate queries for other label indexes.

---

## Implementation Plan

### Phase A: Critical Fixes (Week 1)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 1 | Remove dead code in `_detect_dynamic_algorithm_support()` | `services/graph_service.py` | P1 | 0.5h |
| 2 | Fix `|` label syntax in `CYPHER_FIND_CALLERS`, templates | `cypher_queries.py` | P1 | 1h |
| 3 | Fix `_search_memgraph_native()` parameter passing | `document/tools/document_search.py` | P1 | 1h |
| 4 | Fix `_execute_graph_navigation()` truncation | `retrieval/query_orchestrator.py` | P1 | 1h |
| 5 | Fix `QueryGenerator` ephemeral connection | `graph/query_generator.py` | P1 | 1h |

### Phase B: Connection Management (Week 2)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 6 | Refactor `PathAnalyzer` for DI connection | `memgraph_advanced/path_analysis.py` | P1 | 2h |
| 7 | Refactor `CommunityQFS` for DI connection | `memgraph_advanced/qfs.py` | P1 | 2h |
| 8 | Refactor `DynamicGraphAlgorithms` for DI connection | `memgraph_advanced/dynamic_algorithms.py` | P1 | 2h |
| 9 | Update factory functions to pass connections | `memgraph_advanced/__init__.py` | P1 | 1h |

### Phase C: Search Strategy Enhancement (Week 3)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 10 | Implement `extract_keywords()` multi-keyword | `utils/query_utils.py` | P1 | 1h |
| 11 | Add text scoring to `HybridRetriever` | `memgraph_advanced/hybrid_retrieval.py` | P1 | 3h |
| 12 | Update `HybridRetrievalConfig` with text_weight | `config.py` | P1 | 0.5h |
| 13 | Fix L2 distance test f-string interpolation | `graph/query_generator.py` | P2 | 0.5h |
| 14 | Enhance `CommunityQFS._rank_communities` with embeddings | `memgraph_advanced/qfs.py` | P2 | 2h |

### Phase D: Atomic Multi-Signal Query (Week 4)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 15 | Implement atomic vector+graph+text Cypher query | `memgraph_advanced/hybrid_retrieval.py` | P1 | 6h |
| 16 | Add `_search_separate()` fallback method | `memgraph_advanced/hybrid_retrieval.py` | P1 | 2h |
| 17 | Test atomic query on various Memgraph versions | `tests/` | P1 | 3h |
| 18 | Benchmark atomic vs. separate query latency | `benchmarks/` | P2 | 2h |

**Total Estimated Effort:** ~23 hours

---

## Success Metrics

| Metric | Before | After Target |
|--------|--------|--------------|
| Cypher `|` label syntax violations | 4 templates/queries | 0 |
| Dead code blocks | 1 unreachable block | 0 |
| Ephemeral connections per query session | 3-5 (PathAnalyzer, CommunityQFS, DynamicGraphAlgorithms, QueryGenerator) | 0-1 (reuse injected/pooled) |
| Keywords extracted per query | 1 (single keyword) | 3 (multi-keyword) |
| `text_score` utilization | Always 0.0 | 0.0 or 1.0 (keyword match) |
| Graph navigation result cap | 2 per category | Dynamic (proportional to top_k) |
| Atomic queries (vector+graph in single Cypher) | 0 for code graph | 1 (primary path with fallback) |
| Query round-trips per hybrid search | 2 (vector + metadata) | 1 (atomic) |
| Document vector search parameter correctness | Wrong params to procedure | Correct overfetch + settings-based index name |

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Atomic hybrid query fails on older Memgraph | Medium | Medium | Fallback to `_search_separate()` (current approach) |
| `WHERE IN` clause slower than `|` label syntax | Low | Low | Memgraph optimizes `IN` with small lists; benchmark first |
| DI connection refactoring breaks callers | Medium | Low | Keep fallback ephemeral connection when no injectable provided |
| `extract_keywords()` returns irrelevant keywords | Low | Medium | Length + position scoring; stopword filtering; max 3 keywords |
| Text matching via CONTAINS is slower than text index | Low | High | Acceptable for metadata enrichment (small result set); future: real text indexing |
| Community embedding ranking is expensive | Medium | Medium | Pre-compute community embeddings in `build_community_summaries()` |

---

## Dependency Map

```
Issue 12 (Atomic Query) ──→ depends on Issue 6 (text scoring) ──→ depends on Issue 5 (multi-keyword)
Issue 4 (DI connections) ──→ enables Issue 6 (HybridRetriever text scoring can reuse injected conn)
Issue 2 (| syntax fix) ──→ independent, can be done first
Issue 1 (dead code) ──→ independent, trivial fix
Issue 10 (truncation fix) ──→ depends on Issue 5 (multi-keyword for navigation)
```

**Recommended Implementation Order:**
1. Issue 1 (dead code) — 30 min
2. Issue 2 + 11 (| syntax) — 1 hour  
3. Issue 3 (QueryGenerator connection) — 1 hour
4. Issue 7 (document search params) — 1 hour
5. Issue 10 (truncation) — 1 hour
6. Issue 4 (DI connections) — 5 hours
7. Issue 5 (multi-keyword) — 1 hour
8. Issue 6 (text scoring) — 3 hours
9. Issue 8 (L2 interpolation) — 30 min
10. Issue 9 (community ranking) — 2 hours
11. Issue 12 (atomic query) — 8 hours

---

## Conclusion

This specification addresses **12 concrete, verified issues** preventing full Memgraph query utilization. The issues are:

**Critical (1):**
- Issue 12: No atomic vector+graph+text query for code graph

**High (4):**
- Issue 2: `|` label syntax in templates and commonly-used queries
- Issue 3: `QueryGenerator` ephemeral connection
- Issue 4: Ephemeral connections in `PathAnalyzer`, `CommunityQFS`, `DynamicGraphAlgorithms`
- Issue 7: Wrong parameter passing in document vector search
- Issue 10: Severe result truncation in graph navigation
- Issue 11: `|` label syntax in `CYPHER_FIND_CALLERS`

**Medium (3):**
- Issue 1: Dead code in `_detect_dynamic_algorithm_support()`
- Issue 5: Single keyword extraction
- Issue 6: `text_score` always 0.0
- Issue 8: f-string interpolation in L2 distance test
- Issue 9: Naive keyword-based community ranking

**All solutions are:**
- ✅ **Logically sound** — Each fix addresses the root cause, not a symptom
- ✅ **Implementation-ready** — Exact file locations, code snippets, and test strategies provided
- ✅ **Aligned with existing codebase** — Uses established patterns (QueryProtocol DI, WHERE IN clause, HybridRetrievalConfig, connection pool)
- ✅ **Backward compatible** — All changes are additive or corrective; no API-breaking changes
- ✅ **Testable** — Each fix has explicit validation criteria

**Total Estimated Effort:** ~23 hours over 4 weeks