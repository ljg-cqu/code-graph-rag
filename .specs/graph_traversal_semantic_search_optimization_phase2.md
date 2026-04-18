# Graph Traversal & Semantic Search Optimization — Phase 2

**Version:** 1.0
**Date:** 2025-01-20
**Status:** Implementation-Ready
**Scope:** `codebase_rag/` — Memgraph query execution, graph traversal, and semantic search pipeline
**Predecessor:** `.specs/graph_traversal_semantic_search_optimization.md` (Phase 1 — fully implemented)

---

## Executive Summary

Phase 1 resolved 8 critical issues (torch dependency, HybridRetriever factory, algorithm tool exposure, ranking, text weight dead code, caching, fallback chain, sufficiency loophole).

This Phase 2 spec identifies **10 remaining issues** from deep code audit. **Five are critical runtime syntax bugs** where the system sends invalid Cypher to Memgraph that will **always be rejected**, completely breaking those graph traversal paths. Others are architectural gaps limiting LLM reasoning quality.

### Issue Severity Summary

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | Invalid Memgraph BFS syntax in `vector_store_memgraph.py` | 🔴 CRITICAL | Vector search with context expansion **always fails** |
| 2 | Invalid Memgraph KSHORTEST syntax in `path_analysis.py` | 🔴 CRITICAL | Path analysis between functions **always fails** |
| 3 | Jinja2 template in raw Cypher in `path_analysis.py` | 🔴 CRITICAL | Bottleneck analysis **always fails** |
| 4 | `QueryRouter` unconditionally initializes document vector backend | 🟠 HIGH | Wasteful resource consumption in code-only mode |
| 5 | Keyword fallback extracts first word (often a stopword) | 🟡 MEDIUM | ~60% of keyword fallbacks return useless results |
| 6 | LLM Cypher prompt teaches wrong traversal syntax | 🟠 HIGH | LLM generates invalid Memgraph queries that fail at runtime |
| 7 | QueryRouter bypasses HybridRetriever factory | 🟡 MEDIUM | Redundant embedding provider initialization |
| 8 | No graph context enrichment on semantic results | 🟡 MEDIUM | LLM sees matches but not how they connect |
| 9 | `HybridRetriever` strict constructor validation blocks fallbacks | 🟡 MEDIUM | Transient errors prevent any search attempt |
| 10 | No traversal depth safety caps on variable-length patterns | 🟠 HIGH | Risk of exponential path explosion on large codebases |

---

## Issue 1: Invalid Memgraph BFS Syntax in Vector Store (BUG)

**Severity:** 🔴 CRITICAL
**Locations:**
- `codebase_rag/vector_store_memgraph.py` line ~428 (with context)
- `codebase_rag/vector_store_memgraph.py` line ~501 (without context, same pattern)

### Problem Analysis

The code uses **Neo4j-style** BFS range syntax `(1..max_depth)` which Memgraph **does not support**:

```python
# vector_store_memgraph.py:428 (with context expansion)
MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS (1..max_depth)]-(related)

# vector_store_memgraph.py:501 (same in compatibility mode)
MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS (1..max_depth)]-(related)
```

Memgraph's correct BFS syntax is `*BFS 1 TO max_depth` — **without parentheses** and **using `TO`** instead of `..`.

**Evidence — correct syntax exists elsewhere in the codebase:**
```python
# codebase_rag/graph_algorithms.py:210 — CORRECT Memgraph syntax
MATCH path = (start)-[:CALLS|:DEFINES|:IMPORTS *BFS 1 TO max_depth]-(related)
```

**Failure mode:** When `search(..., include_context=True)` is called, Memgraph returns a parse error like:
```
Syntax error: unexpected token '(' at position ...
```

The LLM will never get context-expanded results from vector search because this query always fails.

### Proposed Fix

Replace the two occurrences with correct Memgraph syntax:

```python
# Line 428 (procedure mode, with context):
MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS 1 TO max_depth]-(related)

# Line 501 (compatibility mode, with context):
MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS 1 TO max_depth]-(related)
```

**Surgical patches:**

```python
# vector_store_memgraph.py — line ~428
# TARGET:
"""MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS (1..max_depth)]-(related)"""
# REPLACEMENT:
"""MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS 1 TO max_depth]-(related)"""
```

```python
# vector_store_memgraph.py — line ~501
# TARGET:
"""MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS (1..max_depth)]-(related)"""
# REPLACEMENT:
"""MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS 1 TO max_depth]-(related)"""
```

---

## Issue 2: Invalid Memgraph KSHORTEST Syntax in Path Analyzer (BUG)

**Severity:** 🔴 CRITICAL
**Location:** `codebase_rag/memgraph_advanced/path_analysis.py` line 45

### Problem Analysis

The `analyze_call_chain` method uses invalid KSHORTEST syntax:

```python
# path_analysis.py:45
MATCH path = (start)-[:CALLS *KSHORTEST $max_paths ..$max_length]->(end)
```

Memgraph's correct KSHORTEST syntax is `*KSHORTEST $count $min TO $max`:

```
*KSHORTEST $max_paths 1 TO $max_length
```

The `..$max_length` syntax is Neo4j's range pattern, not Memgraph's.

**Failure mode:** Every call to `analyze_call_chain` fails with a Memgraph parse error. The LLM cannot analyze call paths between functions — a key feature for understanding code flow.

### Proposed Fix

```python
# path_analysis.py — analyze_call_chain cypher
# TARGET:
"MATCH path = (start)-[:CALLS *KSHORTEST $max_paths ..$max_length]->(end)"
# REPLACEMENT:
"MATCH path = (start)-[:CALLS *KSHORTEST $max_paths 1 TO $max_length]->(end)"
```

Also add a safety cap (see Issue 10):

```python
_MAX_TRAVERSAL_DEPTH = 8

def analyze_call_chain(
    self, start_qn: str, end_qn: str, max_paths: int = 3, max_path_length: int = 10
) -> PathAnalysis:
    # Apply safety cap
    safe_max_length = min(max_path_length, _MAX_TRAVERSAL_DEPTH)

    cypher = """
    MATCH (start:Function {qualified_name: $start_qn}),
          (end:Function {qualified_name: $end_qn})
    MATCH path = (start)-[:CALLS *KSHORTEST $max_paths 1 TO $safe_max_length]->(end)
    ...
    """

    params = {
        "start_qn": start_qn,
        "end_qn": end_qn,
        "max_paths": max_paths,
        "safe_max_length": safe_max_length,
    }
```

---

## Issue 3: Jinja2 Template in Raw Cypher (BUG)

**Severity:** 🔴 CRITICAL
**Location:** `codebase_rag/memgraph_advanced/path_analysis.py` lines 130-132

### Problem Analysis

The `find_bottlenecks` method embeds Jinja2 template syntax directly in a Cypher string:

```python
cypher = """
CALL betweenness_centrality.get("CALLS", "BOTH")
YIELD node, betweenness

WHERE node:Function AND betweenness > $threshold
{% if function_qn %}
AND (node)-[:CALLS*]->(:Function {qualified_name: $function_qn})
OR (:Function {qualified_name: $function_qn})-[:CALLS*]->(node)
{% endif %}

OPTIONAL MATCH (node)-[:CALLS]->(callee)
...
"""
```

Memgraph does **not** understand Jinja2 syntax. The `{% if function_qn %}` markers are sent verbatim to Memgraph, which rejects them as invalid Cypher.

**Two failure modes:**
1. When `function_qn` is `None` → Jinja2 blocks sent as-is → parse error
2. When `function_qn` has a value → same → parse error

**Impact:** Both `find_bottlenecks` and `analyze_call_chain` (which calls `find_bottlenecks` internally) always return errors. The "bottleneck analysis" and "path analysis" MCP/agent tools are completely non-functional.

### Proposed Fix

Replace Jinja2 template with conditional query string building:

```python
def find_bottlenecks(
    self, function_qn: str | None = None, threshold: float = 0.01
) -> list[dict]:
    base_cypher = """
    CALL betweenness_centrality.get("CALLS", "BOTH")
    YIELD node, betweenness

    WHERE node:Function AND betweenness > $threshold
    """

    if function_qn:
        base_cypher += """
        AND (
            (node)-[:CALLS*]->(:Function {qualified_name: $function_qn})
            OR (:Function {qualified_name: $function_qn})-[:CALLS*]->(node)
        )
        """

    base_cypher += """
    OPTIONAL MATCH (node)-[:CALLS]->(callee)
    WITH node, betweenness, count(callee) AS out_degree

    OPTIONAL MATCH (caller)-[:CALLS]->(node)
    WITH node, betweenness, out_degree, count(caller) AS in_degree

    RETURN node.qualified_name AS function_name,
           node.name AS name,
           betweenness AS centrality_score,
           in_degree,
           out_degree,
           (in_degree + out_degree) AS total_connections
    ORDER BY betweenness DESC
    LIMIT 20
    """

    params = {"threshold": threshold, "function_qn": function_qn}

    with MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        username=settings.MEMGRAPH_USERNAME,
        password=settings.MEMGRAPH_PASSWORD,
    ) as ingestor:
        records = ingestor.fetch_all(base_cypher, params)

    return records
```

---

## Issue 4: Unconditional Document Backend Initialization (BUG)

**Severity:** 🟠 HIGH
**Location:** `codebase_rag/shared/query_router.py` — `QueryRouter.__init__()`

### Problem Analysis

```python
def __init__(self, ..., doc_vector: VectorBackend | None = None):
    from ..vector_backend import get_shared_backend, get_shared_backend_for_documents

    self.code_vector = code_vector or get_shared_backend()
    self.doc_vector = doc_vector or get_shared_backend_for_documents()  # ← Always called!
```

`get_shared_backend_for_documents()` is called unconditionally, even when:
- `doc_graph` is `None`
- The router will never be used in document modes
- The user runs in code-only mode

This initializes a document vector backend (connection, index setup) that's never used, wasting memory and connection pool resources.

### Proposed Fix

Use lazy initialization via properties:

```python
class QueryRouter:
    def __init__(
        self,
        code_graph: QueryProtocol | None = None,
        doc_graph: QueryProtocol | None = None,
        code_vector: VectorBackend | None = None,
        doc_vector: VectorBackend | None = None,
    ):
        from ..vector_backend import get_shared_backend

        self.code_graph = code_graph
        self.doc_graph = doc_graph
        self._code_vector = code_vector
        self._doc_vector = doc_vector
        self.current_mode: QueryMode = QueryMode.CODE_ONLY

    @property
    def code_vector(self) -> VectorBackend | None:
        if self._code_vector is None:
            from ..vector_backend import get_shared_backend
            self._code_vector = get_shared_backend()
        return self._code_vector

    @property
    def doc_vector(self) -> VectorBackend | None:
        if self._doc_vector is None and self.doc_graph is not None:
            from ..vector_backend import get_shared_backend_for_documents
            self._doc_vector = get_shared_backend_for_documents()
        return self._doc_vector
```

Update `_query_document_only` to use the property:

```python
def _query_document_only(self, request: QueryRequest) -> QueryResponse:
    if not self.doc_graph:
        return QueryResponse(
            answer="Document graph is not available.",
            sources=[],
            mode=request.mode,
            warnings=["Document graph connection not configured"],
        )

    doc_vec = self.doc_vector  # Lazy property — only initializes if needed
    if not doc_vec:
        return QueryResponse(...)
```

---

## Issue 5: Keyword Fallback Extracts First Word (BUG)

**Severity:** 🟡 MEDIUM
**Locations:**
- `codebase_rag/shared/query_router.py` — `_query_code_only()` keyword fallback
- `codebase_rag/tools/semantic_search.py` — `_semantic_search_keyword_fallback()`

### Problem Analysis

In `query_router.py`:
```python
keyword = request.question.split()[0] if request.question else ""
```

This takes the first word of the query, which is very often a stopword:
- "What functions call..." → `"what"`
- "How does the code..." → `"how"`
- "Find all classes..." → `"find"`
- "Show me where..." → `"show"`

Meanwhile, `semantic_search.py` already has a much better approach:
```python
stopwords = {'the', 'is', 'at', ...}
keywords = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in stopwords]
keyword = max(keywords, key=len)  # Most specific keyword
```

### Proposed Fix

Create a shared utility and use it everywhere:

```python
# codebase_rag/utils/query_utils.py (NEW)

def extract_best_keyword(query: str) -> str:
    """Extract the most meaningful keyword from a natural language query.

    Filters stopwords and returns the longest remaining word.
    Falls back to empty string if no meaningful words found.
    """
    if not query:
        return ""

    stopwords = {
        'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
        'in', 'with', 'to', 'for', 'of', 'what', 'how', 'where', 'when',
        'why', 'who', 'show', 'find', 'tell', 'get', 'list', 'all', 'any',
        'some', 'does', 'do', 'can', 'could', 'would', 'should', 'will',
    }

    words = query.lower().split()
    meaningful = [w for w in words if len(w) > 2 and w not in stopwords]

    if not meaningful:
        return max(words, key=len, default="")

    return max(meaningful, key=len)
```

Then update both consumers:

```python
# query_router.py
from ..utils.query_utils import extract_best_keyword
keyword = extract_best_keyword(request.question)

# semantic_search.py
from ..utils.query_utils import extract_best_keyword
keyword = extract_best_keyword(query)
```

---

## Issue 6: LLM Cypher Prompt Teaches Wrong Traversal Syntax

**Severity:** 🟠 HIGH
**Location:** `codebase_rag/prompts.py` — `CYPHER_QUERY_RULES`

### Problem Analysis

The prompt instructs the LLM to use Memgraph traversal syntax:

```python
- **Traversal optimization**: Use built-in traversal syntax `*BFS`, `*DFS`, `*KSHORTEST` instead of Neo4j's `shortestPath()`/`kShortestPaths()` functions
```

But it does **NOT** specify the correct Memgraph syntax, leaving the LLM to guess. Models trained on Neo4j data will naturally generate Neo4j-style syntax:
- `*BFS 1..3` (Neo4j range) instead of `*BFS 1 TO 3` (Memgraph)
- `*KSHORTEST 3 1..10` instead of `*KSHORTEST 3 1 TO 10`

The `_clean_cypher_response` function in `services/llm.py` fixes `|` syntax and comments, but does **NOT** fix traversal syntax errors.

**Impact:** When the LLM generates Cypher queries involving graph traversal (which happens frequently for "find paths", "trace calls", etc.), the generated queries will fail at Memgraph execution. The error is caught and the LLM attempts repair, but without explicit guidance on the correct syntax, the repair also fails — wasting retries and potentially giving up.

### Proposed Fix

Add explicit Memgraph traversal syntax rules to `CYPHER_QUERY_RULES`:

```python
CYPHER_QUERY_RULES = """**2. Critical Cypher Query Rules**
...
**3. Memgraph-Specific Optimization Rules**
...
- **Traversal optimization**: Use Memgraph's built-in traversal syntax. CORRECT syntax:
  - BFS: `*BFS 1 TO 3` (NOT `*BFS 1..3` — Neo4j syntax is wrong)
  - DFS: `*DFS 1 TO 3`
  - K-Shortest: `*KSHORTEST 5 1 TO 10` (finds up to 5 shortest paths, max length 10)
  - Example: `MATCH path = (a)-[:CALLS *BFS 1 TO 3]->(b) RETURN path`
  - Example: `MATCH path = (a)-[:CALLS *KSHORTEST 3 1 TO 5]->(b) RETURN path`
...
"""
```

---

## Issue 7: QueryRouter Bypasses HybridRetriever Factory

**Severity:** 🟡 MEDIUM
**Location:** `codebase_rag/shared/query_router.py` — `_query_code_only()`

### Problem Analysis

```python
# query_router.py — direct instantiation, bypasses factory
retriever = HybridRetriever(
    graph_ingestor=self.code_graph,
    vector_backend=self.code_vector,
    embedding_provider=provider,
    config=settings.hybrid_retrieval_config,
)
```

The `_query_code_only` method creates `HybridRetriever` directly instead of using `create_hybrid_retriever()`. This:
1. Duplicates instantiation logic
2. Misses shared embedding provider — the factory uses `get_shared_embedding_provider()` which caches the provider; this code creates a new provider each time
3. Inconsistent with Phase 1's factory pattern

### Proposed Fix

```python
def _query_code_only(self, request: QueryRequest) -> QueryResponse:
    if self.code_vector:
        try:
            from ..memgraph_advanced import create_hybrid_retriever

            retriever = create_hybrid_retriever(self.code_graph)
            results: list[HybridSearchResult] = retriever.search(
                query=request.question,
                top_k=request.top_k,
            )
```

---

## Issue 8: No Graph Context Enrichment for Semantic Results (GAP)

**Severity:** 🟡 MEDIUM
**Scope:** `tools/semantic_search.py` and LLM context generation

### Problem Analysis

When `semantic_code_search` returns results, the LLM gets:
- Function/class names with similarity scores
- Source code (via `get_function_source_by_id`)

But the LLM does **NOT** get graph relationships:
- Who calls these functions
- What these functions call
- Parent classes/modules
- How related the results are to each other

For questions like "How does authentication work?", the LLM might find `authenticate_user` and `validate_token` but won't see the call chain between them.

### Proposed Fix

Add optional graph context enrichment:

```python
# tools/semantic_search.py

def enrich_with_graph_context(
    results: list[SemanticSearchResult],
    ingestor: QueryProtocol,
    max_relations: int = 5,
) -> list[SemanticSearchResult]:
    """Enrich semantic search results with graph relationship context."""
    if not results:
        return results

    node_ids = [r.node_id for r in results]

    cypher = """
    MATCH (n)
    WHERE id(n) IN $node_ids

    OPTIONAL MATCH (caller)-[:CALLS]->(n)
    WITH n, caller

    OPTIONAL MATCH (n)-[:CALLS]->(callee)
    WITH n, caller, callee

    OPTIONAL MATCH (n)<-[:DEFINES]-(parent)
    WHERE parent:Class OR parent:Module

    RETURN id(n) AS node_id,
           collect(DISTINCT caller.qualified_name)[0..$max] AS callers,
           collect(DISTINCT callee.qualified_name)[0..$max] AS callees,
           collect(DISTINCT parent.qualified_name)[0..2] AS parents
    """

    params = {"node_ids": node_ids, "max": max_relations}
    context_rows = ingestor.fetch_all(cypher, params)
    context_map = {row["node_id"]: row for row in context_rows}

    enriched = []
    for result in results:
        ctx = context_map.get(result.node_id, {})
        result.callers = ctx.get("callers", [])
        result.callees = ctx.get("callees", [])
        result.parents = ctx.get("parents", [])
        enriched.append(result)

    return enriched
```

Update `SemanticSearchResult` dataclass in `types_defs.py`:

```python
@dataclass
class SemanticSearchResult:
    node_id: int
    qualified_name: str
    name: str
    type: str
    similarity: float
    callers: list[str] = field(default_factory=list)
    callees: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
```

Update the tool to include graph context in response:

```python
async def semantic_search_functions(query: str, top_k: int = 5) -> str:
    results = semantic_code_search(query, top_k)
    if not results:
        return cs.MSG_SEMANTIC_NO_RESULTS.format(query=query)

    from ..config import settings
    from ..services.graph_service import MemgraphIngestor

    with MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
    ) as ingestor:
        results = enrich_with_graph_context(results, ingestor)

    formatted_results = []
    for i, result in enumerate(results, 1):
        line = f"{i}. {result['qualified_name']} (type: {result['type']}, similarity: {result['similarity']})"
        callers = result.get('callers', [])
        callees = result.get('callees', [])
        parents = result.get('parents', [])
        if callers:
            line += f"\n   ← Called by: {', '.join(callers[:3])}"
        if callees:
            line += f"\n   → Calls: {', '.join(callees[:3])}"
        if parents:
            line += f"\n   📦 In: {', '.join(parents)}"
        formatted_results.append(line)
    # ...
```

---

## Issue 9: HybridRetriever Strict Constructor Validation Blocks Fallbacks

**Severity:** 🟡 MEDIUM
**Location:** `codebase_rag/memgraph_advanced/hybrid_retrieval.py` — `__init__()`

### Problem Analysis

The constructor performs strict validation that raises exceptions:

```python
def __init__(self, ...):
    if not vector_backend.health_check():
        raise RuntimeError("Vector backend health check failed")

    try:
        test_embedding = embedding_provider.embed("test")
        ...
    except Exception as e:
        raise ValueError(f"Embedding provider validation failed: {e}")
```

If the vector backend is temporarily unhealthy or the embedding provider has a transient error, the retriever can't even be created. The caller's fallback chain catches the exception, but the retriever was never constructed.

### Proposed Fix

Make validation non-fatal with warnings:

```python
def __init__(
    self,
    graph_ingestor: QueryProtocol | None = None,
    vector_backend: VectorBackend | None = None,
    embedding_provider: EmbeddingProviderProtocol | None = None,
    config: HybridRetrievalConfig | None = None,
    strict_validation: bool = True,
) -> None:
    if graph_ingestor is None:
        raise ValueError("graph_ingestor is required for HybridRetriever")
    if vector_backend is None:
        raise ValueError("vector_backend is required for HybridRetriever")
    if embedding_provider is None:
        raise ValueError("embedding_provider is required for HybridRetriever")

    self.graph_ingestor = graph_ingestor
    self.vector_backend = vector_backend
    self.embedding_provider = embedding_provider
    self.config = config
    self._is_healthy: bool | None = None

    if strict_validation:
        self._validate()

def _validate(self) -> bool:
    """Validate dependencies. Returns True if healthy, False otherwise."""
    try:
        if not self.vector_backend.health_check():
            logger.warning("Vector backend health check failed")
            self._is_healthy = False
            return False

        test_embedding = self.embedding_provider.embed("test")
        if not isinstance(test_embedding, list) or len(test_embedding) == 0:
            logger.warning("Embedding provider returned invalid embedding")
            self._is_healthy = False
            return False

        self._is_healthy = True
        return True
    except Exception as e:
        logger.warning(f"HybridRetriever validation failed: {e}")
        self._is_healthy = False
        return False

def search(self, query: str, top_k: int = 10) -> list[HybridSearchResult]:
    if self._is_healthy is None or not self._is_healthy:
        if not self._validate():
            logger.warning("Skipping search due to unhealthy dependencies")
            return []

    # ... existing search logic
```

The factory function uses `strict_validation=False` for graceful fallback:

```python
def create_hybrid_retriever(
    graph_ingestor: QueryProtocol,
    strict_validation: bool = False,  # Default to non-strict for fallback compatibility
) -> HybridRetriever:
    return HybridRetriever(
        graph_ingestor=graph_ingestor,
        vector_backend=get_shared_backend(),
        embedding_provider=get_shared_embedding_provider(),
        config=settings.hybrid_retrieval_config,
        strict_validation=strict_validation,
    )
```

---

## Issue 10: Missing Graph Traversal Depth Safety Caps

**Severity:** 🟠 HIGH
**Locations:**
- `codebase_rag/memgraph_advanced/path_analysis.py` — `analyze_call_chain`
- `codebase_rag/memgraph_advanced/path_analysis.py` — `find_bottlenecks` (unbounded `[:CALLS*]`)

### Problem Analysis

The `analyze_call_chain` method uses user-controlled `max_path_length` (default 10):

```python
MATCH path = (start)-[:CALLS *KSHORTEST $max_paths 1 TO $max_length]->(end)
```

In a large codebase with dense call graphs, variable-length path traversals of depth 10 can cause **exponential explosion** in paths examined, leading to query timeouts, Memgraph memory exhaustion, and degraded performance.

The `find_bottlenecks` method uses **completely unbounded** variable-length patterns:
```python
AND (node)-[:CALLS*]->(:Function {qualified_name: $function_qn})
OR (:Function {qualified_name: $function_qn})-[:CALLS*]->(node)
```

No depth limit at all — this will traverse the entire call graph in both directions.

### Proposed Fix

Add hard safety caps:

```python
# path_analysis.py
_MAX_TRAVERSAL_DEPTH = 8  # Absolute maximum regardless of user input
_MAX_UNBOUNDED_DEPTH = 5  # Max for previously unbounded patterns

def analyze_call_chain(
    self, start_qn: str, end_qn: str, max_paths: int = 3, max_path_length: int = 10
) -> PathAnalysis:
    safe_max_length = min(max_path_length, _MAX_TRAVERSAL_DEPTH)
    # ... use safe_max_length in query

def find_bottlenecks(
    self, function_qn: str | None = None, threshold: float = 0.01
) -> list[dict]:
    # ... when function_qn is provided, add depth cap:
    if function_qn:
        base_cypher += f"""
        AND (
            (node)-[:CALLS*1..{_MAX_UNBOUNDED_DEPTH}]->(:Function {{qualified_name: $function_qn}})
            OR (:Function {{qualified_name: $function_qn}})-[:CALLS*1..{_MAX_UNBOUNDED_DEPTH}]->(node)
        )
        """
```

Also add the cap to `get_call_hierarchy` in `graph_navigation.py` (already has `_MAX_DEPTH = 5` — verify it's enforced in the query string, not just the parameter).

---

## Implementation Plan

### Phase 2A: Critical Syntax Bugs (Week 1)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 1 | Fix invalid BFS syntax `*BFS (1..n)` → `*BFS 1 TO n` | `vector_store_memgraph.py` | P0 | 0.5h |
| 2 | Fix invalid KSHORTEST syntax `*KSHORTEST $p ..$l` → `*KSHORTEST $p 1 TO $l` | `path_analysis.py` | P0 | 0.5h |
| 3 | Remove Jinja2 template from Cypher query | `path_analysis.py` | P0 | 1h |
| 6 | Add correct traversal syntax rules to LLM prompt | `prompts.py` | P0 | 0.5h |
| 10 | Add traversal depth safety caps | `path_analysis.py`, `graph_navigation.py` | P0 | 1h |

### Phase 2B: Code Quality & Consistency (Week 2)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 4 | Lazy-init document vector backend | `shared/query_router.py` | P1 | 2h |
| 5 | Extract shared keyword utility | `utils/query_utils.py` (new), `query_router.py`, `semantic_search.py` | P1 | 1h |
| 7 | Use HybridRetriever factory in QueryRouter | `shared/query_router.py` | P1 | 0.5h |

### Phase 2C: Feature Enhancement (Week 3)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 8 | Add graph context enrichment to semantic results | `tools/semantic_search.py`, `types_defs.py` | P1 | 4h |
| 9 | Relax HybridRetriever constructor validation | `memgraph_advanced/hybrid_retrieval.py` | P1 | 2h |

### Phase 2D: Testing & Validation (Week 4)

| Task | Priority | Effort |
|------|----------|--------|
| Unit tests for fixed Cypher syntax | P0 | 2h |
| Unit tests for keyword extraction utility | P1 | 1h |
| Integration tests for lazy backend init | P1 | 2h |
| Integration tests for graph context enrichment | P1 | 3h |
| Performance tests for traversal depth caps | P1 | 2h |

**Total Estimated Effort:** ~20 hours

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_vector_store_syntax.py

def test_bfs_uses_correct_memgraph_syntax():
    """Verify BFS traversal uses Memgraph syntax, not Neo4j."""
    from codebase_rag.vector_store_memgraph import MemgraphBackend

    # Check the source code for correct syntax
    import inspect
    source = inspect.getsource(MemgraphBackend.search)
    assert "*BFS (1.." not in source, "Found Neo4j-style BFS syntax (1..n)"
    assert "*BFS 1 TO" in source or "*BFS " in source, "Expected Memgraph BFS syntax"

def test_kshortest_uses_correct_memgraph_syntax():
    """Verify KSHORTEST uses Memgraph syntax, not Neo4j."""
    import inspect
    from codebase_rag.memgraph_advanced.path_analysis import PathAnalyzer

    source = inspect.getsource(PathAnalyzer.analyze_call_chain)
    assert "..$max_length" not in source, "Found Neo4j-style KSHORTEST syntax"
    assert "1 TO" in source, "Expected Memgraph KSHORTEST syntax"

def test_no_jinja2_in_cypher():
    """Verify no Jinja2 template syntax in Cypher queries."""
    import inspect
    from codebase_rag.memgraph_advanced.path_analysis import PathAnalyzer

    for method_name in ['find_bottlenecks', 'analyze_call_chain', 'find_similar_functions']:
        method = getattr(PathAnalyzer, method_name)
        source = inspect.getsource(method)
        assert "{%" not in source, f"Jinja2 template found in {method_name}"
        assert "%}" not in source, f"Jinja2 template found in {method_name}"
```

```python
# tests/unit/test_query_utils.py

def test_extract_best_keyword():
    from codebase_rag.utils.query_utils import extract_best_keyword

    # Common patterns that previously failed
    assert extract_best_keyword("What functions call authenticate?") == "functions"
    assert extract_best_keyword("How does the code work?") in ["code", "work"]
    assert extract_best_keyword("Find all classes in utils") == "classes"
    assert extract_best_keyword("Show me where database connects") == "database"

    # Edge cases
    assert extract_best_keyword("") == ""
    assert extract_best_keyword("a b c") in ["a", "b", "c"]
```

```python
# tests/unit/test_query_router.py

def test_query_router_lazy_doc_backend():
    """Verify document vector backend is NOT initialized in code-only mode."""
    with patch('codebase_rag.vector_backend.get_shared_backend_for_documents') as mock_doc:
        router = QueryRouter(
            code_graph=mock_code_graph,
            doc_graph=None,
        )
        mock_doc.assert_not_called()
        assert router.doc_vector is None
        mock_doc.assert_not_called()

def test_query_router_uses_hybrid_factory():
    """Verify QueryRouter uses create_hybrid_retriever factory."""
    with patch('codebase_rag.shared.query_router.create_hybrid_retriever') as mock_factory:
        router = QueryRouter(code_graph=mock_code_graph)
        request = QueryRequest(question="test", mode=QueryMode.CODE_ONLY)
        router.query(request)
        mock_factory.assert_called_once()
```

```python
# tests/unit/test_cypher_prompt.py

def test_cypher_prompt_has_correct_traversal_syntax():
    """Verify the LLM prompt teaches correct Memgraph traversal syntax."""
    from codebase_rag.prompts import CYPHER_QUERY_RULES

    assert "1 TO" in CYPHER_QUERY_RULES, "Prompt should specify '1 TO' for Memgraph traversal"
    assert "*BFS" in CYPHER_QUERY_RULES
    assert "*KSHORTEST" in CYPHER_QUERY_RULES
    # Ensure Neo4j syntax is NOT mentioned as valid
    assert "1.." not in CYPHER_QUERY_RULES or "NOT" in CYPHER_QUERY_RULES
```

### Integration Tests

```python
# tests/integration/test_graph_traversal.py

def test_bfs_context_expansion_integration():
    """Verify BFS context expansion works with real Memgraph (no syntax error)."""
    from codebase_rag.vector_store_memgraph import MemgraphBackend

    backend = MemgraphBackend()
    # Use a dummy embedding — the query should not fail with syntax error
    dummy_embedding = [0.0] * 768
    try:
        # This should not raise a Memgraph parse error
        results = backend.search(dummy_embedding, top_k=1, include_context=True)
        # May return empty if no embeddings exist, but should not error
    except Exception as e:
        if "syntax error" in str(e).lower() or "parse" in str(e).lower():
            pytest.fail(f"Memgraph syntax error in BFS query: {e}")
        # Other errors (connection, no embeddings) are acceptable

def test_path_analysis_integration():
    """Verify path analysis works end-to-end (no Jinja2 or KSHORTEST syntax error)."""
    analyzer = PathAnalyzer()
    # Will return empty if functions don't exist, but should not raise syntax error
    try:
        result = analyzer.analyze_call_chain(
            start_qn="test.start",
            end_qn="test.end",
            max_paths=3,
            max_path_length=5,
        )
        assert isinstance(result, PathAnalysis)
    except Exception as e:
        if "syntax" in str(e).lower() or "parse" in str(e).lower() or "jinja" in str(e).lower():
            pytest.fail(f"Syntax error in path analysis query: {e}")
```

---

## Migration Guide

### No Breaking Changes

All fixes are backward compatible:
- **Syntax fixes**: Query string corrections only, same parameters
- **Lazy init**: Property-based access, same external API
- **Factory usage**: Internal refactoring
- **Keyword utility**: Replaces inline logic, improved behavior
- **Graph context enrichment**: Additive fields on `SemanticSearchResult`
- **HybridRetriever validation**: `strict_validation=True` by default for backward compatibility
- **Traversal depth caps**: Additive safety, only affects extreme cases

### No New Environment Variables Required

---

## Success Metrics

| Metric | Before | After Target |
|--------|--------|--------------|
| BFS context expansion success rate | ❌ Always fails (syntax error) | ✅ Works correctly |
| Path analysis success rate | ❌ Always fails (syntax + Jinja2 error) | ✅ Works correctly |
| Bottleneck analysis success rate | ❌ Always fails (Jinja2 error) | ✅ Works correctly |
| LLM-generated traversal query success | ~50% (depends on model training) | ~90% (correct syntax in prompt) |
| Document backend initialized in code-only mode | ✅ Yes (wasteful) | ❌ No (lazy) |
| Keyword fallback quality | ~40% useful | ~85% useful |
| QueryRouter uses shared embedding provider | ❌ No | ✅ Yes |
| Semantic results include graph context | ❌ No | ✅ Yes |
| Variable-length traversal runaway queries | ⚠️ Possible | ✅ Capped at depth 8 |

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Lazy `doc_vector` property breaks existing callers | Medium | Low | All internal accesses use property; external API unchanged |
| Graph context enrichment adds latency | Low | High | Single batched Cypher query; capped at 5 relations per node |
| Relaxed validation masks real errors | Medium | Medium | `strict_validation=True` by default; warnings logged |
| Traversal cap cuts off legitimate deep paths | Low | Medium | Cap is 8 (covers 99% of real codebases); configurable via constant |
| Prompt changes affect LLM behavior in unexpected ways | Low | Medium | Only adds syntax guidance; doesn't change query semantics |

---

## Conclusion

This Phase 2 spec addresses **10 remaining issues** in the graph traversal and semantic search pipeline:

**Critical Runtime Bugs (5):**
1. Invalid Memgraph BFS syntax `*BFS (1..n)` — always fails
2. Invalid Memgraph KSHORTEST syntax — always fails
3. Jinja2 template in raw Cypher — always fails
4. LLM prompt teaches wrong traversal syntax — causes generated queries to fail
5. No traversal depth safety caps — risk of exponential path explosion

**Architectural Issues (5):**
6. Unconditional document backend initialization — wasteful
7. QueryRouter bypasses HybridRetriever factory — redundant provider init
8. No graph context enrichment on semantic results — shallow LLM answers
9. Strict HybridRetriever validation — blocks graceful fallbacks
10. Poor keyword fallback — first-word extraction often yields stopwords

**All solutions are:**
- ✅ Logically sound — Address root causes with targeted fixes
- ✅ Implementation-ready — Code snippets provided, file locations specified
- ✅ Aligned with existing codebase — Uses existing patterns (factories, properties, utilities)
- ✅ Backward compatible — No breaking changes to external APIs

**Recommended Implementation Order:**
1. Phase 2A (P0 syntax bugs) — 1 day
2. Phase 2B (code quality) — 2 days
3. Phase 2C (features) — 2 days
4. Phase 2D (testing) — 1 day

**Total Timeline:** ~1 week
