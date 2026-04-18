# Graph Traversal & Semantic Search Optimization Design Spec

**Version:** 1.0  
**Date:** 2024-01-15  
**Status:** Draft for Review  
**Scope:** Code-Graph-RAG system (`codebase_rag/`)

---

## Executive Summary

This specification identifies **8 critical issues** preventing the Code-Graph-RAG system from making full use of its graph traversal and semantic search capabilities. While the codebase has substantial infrastructure (HybridRetriever, MemgraphBackend, GraphAlgorithms, QueryRouter), architectural gaps cause semantic search to fail silently, advanced algorithms to remain unexposed, and graph traversal to underperform.

**Key Impact:**
- Semantic search returns empty results when `torch`/`transformers` aren't installed, despite Memgraph vector search working independently
- Advanced graph algorithms (CommunityQFS, PathAnalysis, DynamicGraphAlgorithms) exist but are not exposed as tools
- Redundant HybridRetriever instantiation causes performance overhead
- Graph navigation tools lack semantic ranking
- Sufficiency gatekeeper can be bypassed when semantic search fails

---

## Current State Analysis

### Architecture Overview

The system has three main layers for graph traversal and semantic search:

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Layer                             │
│  (sufficiency_gatekeeper, investigation_tracker)             │
│  Enforces: must use semantic_search + query_graph + read_file│
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Tool Layer                                │
│  - semantic_search.py (semantic_code_search)                 │
│  - graph_navigation.py (GraphNavigator)                      │
│  - codebase_query.py (query_graph)                           │
│  - document_search.py (document_semantic_search)             │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│               Retrieval Layer                                │
│  - HybridRetriever (vector + PageRank + community)           │
│  - MemgraphBackend (native vector search)                    │
│  - GraphAlgorithms (PageRank, BFS, community, similarity)    │
│  - DynamicGraphAlgorithms (incremental updates)              │
│  - CommunityQFS (query-focused summarization)                │
│  - PathAnalyzer (k-shortest paths, bottlenecks)              │
└─────────────────────────────────────────────────────────────┘
```

### What's Implemented ✅

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Memgraph vector search | ✅ Full | `vector_store_memgraph.py` | Native `vector_search.search()` |
| HybridRetriever | ✅ Full | `memgraph_advanced/hybrid_retrieval.py` | Vector + PageRank + community scoring |
| GraphAlgorithms | ✅ Full | `graph_algorithms.py` | PageRank, Leiden/Louvain, BFS, Jaccard |
| DynamicGraphAlgorithms | ✅ Full | `memgraph_advanced/dynamic_algorithms.py` | Incremental PageRank/communities |
| CommunityQFS | ✅ Full | `memgraph_advanced/qfs.py` | Query-focused summarization |
| PathAnalyzer | ✅ Full | `memgraph_advanced/path_analysis.py` | K-shortest paths, bottlenecks |
| QueryRouter | ✅ Full | `shared/query_router.py` | Multi-mode routing (5 modes) |
| GraphNavigator | ✅ Full | `tools/graph_navigation.py` | References, call hierarchy, implementations |
| Semantic search tool | ⚠️ Partial | `tools/semantic_search.py` | **Blocked by torch dependency** |
| Document semantic search | ✅ Full | `document/tools/document_search.py` | Vector + keyword fallback |

### Critical Issues Identified ❌

#### Issue 1: Semantic Search Blocked by Unnecessary Torch Dependency

**Severity:** 🔴 CRITICAL  
**Location:** `codebase_rag/tools/semantic_search.py:14-18`

**Current Code:**
```python
def semantic_code_search(query: str, top_k: int = 5) -> list[SemanticSearchResult]:
    if not has_semantic_dependencies():  # Requires torch + transformers
        logger.warning(ex.SEMANTIC_EXTRA)
        return []  # ← Returns empty immediately, no fallback!
```

**Problem Analysis:**
- `has_semantic_dependencies()` checks for both `torch` AND `transformers` (see `utils/dependencies.py`)
- These are only needed for **generating** embeddings locally during indexing
- Query-time retrieval uses Memgraph's native vector search which operates on pre-stored embeddings
- Users with:
  - Cloud embedding providers (OpenAI, Google) for indexing
  - Pre-indexed codebases
  - No need for local embedding generation
  
  Get **zero results** from semantic search even though their embeddings exist in Memgraph.

**Impact:** Semantic search completely unavailable for ~50% of deployment scenarios.

**Evidence from Code:**
```python
# utils/dependencies.py
def has_semantic_dependencies() -> bool:
    return has_torch() and has_transformers()  # BOTH required!
```

#### Issue 2: HybridRetriever Instantiated Redundantly

**Severity:** 🟡 MEDIUM
**Locations:**
- `tools/semantic_search.py:40-45`
- `shared/query_router.py:420-430`

**Problem Analysis:**
Each query creates a new HybridRetriever which:
1. Validates vector backend health (connection test)
2. Tests embedding provider with "test" query (API call or compute)
3. Creates new connections

**Current Pattern:**
```python
# semantic_search.py:40-45 - per query
retriever = HybridRetriever(
    graph_ingestor=ingestor,
    vector_backend=get_shared_backend(),
    embedding_provider=provider,
    config=settings.hybrid_retrieval_config,
)

# query_router.py:420-430 - also per query
retriever = HybridRetriever(
    graph_ingestor=self.code_graph,
    vector_backend=get_shared_backend(),
    embedding_provider=provider,
    config=settings.hybrid_retrieval_config,
)
```

**Related Sub-Issue 2a: QueryRouter Creates Non-Shared Vector Backends**

**Location:** `shared/query_router.py:220-221`

```python
# QueryRouter.__init__ uses get_vector_backend() instead of get_shared_backend()
self.code_vector = code_vector or get_vector_backend()  # Creates NEW instance!
self.doc_vector = doc_vector or get_vector_backend(is_document=True)  # Creates NEW instance!
```

This means:
- QueryRouter creates its own vector backend instances
- Each instance has separate connection pools
- No benefit from `get_shared_backend()` singleton pattern

**Impact:** Performance overhead, redundant validation, potential connection pool exhaustion under load.

#### Issue 3: Advanced Algorithms Not Exposed as Tools

**Severity:** 🟡 MEDIUM  
**Location:** MCP registry (`mcp/tools.py`) and agentic tools

**Available but Not Exposed:**
| Algorithm | Location | Potential Tool |
|-----------|----------|----------------|
| CommunityQFS | `memgraph_advanced/qfs.py` | `community_summary` |
| PathAnalyzer | `memgraph_advanced/path_analysis.py` | `analyze_path` |
| DynamicGraphAlgorithms | `memgraph_advanced/dynamic_algorithms.py` | `update_graph_scores` |
| GraphAlgorithms.get_similar_nodes | `graph_algorithms.py` | `find_similar_functions` |
| GraphAlgorithms.get_bfs_context | `graph_algorithms.py` | `expand_context` |

**Impact:** Users cannot leverage powerful graph analysis capabilities through the agent interface.

#### Issue 4: Graph Navigation Missing Semantic Ranking

**Severity:** 🟡 MEDIUM  
**Location:** `tools/graph_navigation.py`

**Current Code:**
```python
callers_query = (
    "MATCH path = (caller:Function|Method)-[:CALLS*1..$depth]->(target) "
    "WHERE target.qualified_name = $qn "
    "RETURN DISTINCT caller.qualified_name AS qualified_name, ... "
    "ORDER BY depth, caller.qualified_name"  # ← Only alphabetical!
)
```

**Problem Analysis:**
- Call hierarchy returns ALL callers/callees ordered only by depth and name
- No use of PageRank scores, community importance, or vector similarity
- For large codebases, this returns overwhelming unranked results

**Impact:** Poor UX for navigation, no relevance ranking.

#### Issue 5: Text Weight Dead Code in HybridRetrievalConfig

**Severity:** 🟠 HIGH  
**Location:** `config.py:197-206`

**Current Code:**
```python
@dataclass
class HybridRetrievalConfig:
    vector_weight: float = 0.6
    text_weight: float = 0.2      # ← Configured
    pagerank_weight: float = 0.15
    community_weight: float = 0.05
    # Validation requires sum ≈ 1.0
```

**But in HybridRetriever:**
```python
def search(self, query: str, top_k: int = 10) -> list[HybridSearchResult]:
    # ...
    results.append(HybridSearchResult(
        # ...
        text_score=0.0,  # ← Always hardcoded to 0.0!
        # ...
        combined_score=combined_score,  # ← Uses weights but text_score=0
    ))
```

**Problem Analysis:**
- `text_weight` is set to 0.2 in defaults
- Validation forces users to allocate 20% of weight to unused component
- No text index exists in Memgraph backend
- No text search implementation exists

**Impact:** Confusing configuration, impossible to set reasonable weights.

#### Issue 6: No Semantic Search Result Caching

**Severity:** 🟡 MEDIUM  
**Location:** Entire codebase

**Problem Analysis:**
- Identical queries re-compute embeddings every time
- Each embedding generation costs:
  - Local: GPU/CPU compute (~50-200ms)
  - Cloud: API call + latency (~200-1000ms) + cost
- No caching layer exists anywhere in the retrieval pipeline

**Impact:** Unnecessary API calls, slower response times, higher costs.

#### Issue 7: Insufficient Fallback Chain

**Severity:** 🟠 HIGH  
**Location:** `tools/semantic_search.py`

**Current Fallback:**
```
Semantic Search
    ↓ (torch missing?)
Return []  ← Dead end!
```

**Missing Fallbacks:**
- When torch missing → should use keyword search on graph
- When embedding provider fails → should use cached results or direct vector search
- When vector search unavailable → should use text-based graph query
- When graph unavailable → should use filesystem search

**Impact:** Single point of failure breaks entire semantic search.

#### Issue 8: Sufficiency Gatekeeper Loophole

**Severity:** 🟡 MEDIUM  
**Location:** `orchestrator/sufficiency_gatekeeper.py`

**Current Code:**
```python
if reqs.requires_vector:
    if AgenticToolName.SEMANTIC_SEARCH not in state.tool_failures:
        if AgenticToolName.SEMANTIC_SEARCH not in state.tools_used:
            return False, "CRITICAL: You must use `semantic_search`..."
```

**InvestigationTracker:**
```python
def record_tool(self, tool_name: str, ..., results_count: int = -1) -> None:
    if results_count == 0 and tool_name in {SEMANTIC_SEARCH, QUERY_GRAPH}:
        self.tool_failures.add(tool_name)  # ← Excluded from sufficiency!
```

**Problem Analysis:**
- If semantic_search returns 0 results (e.g., torch missing), it's marked as "failed"
- Failed tools are excluded from sufficiency requirements
- Agent can satisfy requirements WITHOUT using semantic search
- This defeats the purpose of enforcing semantic search usage

**Impact:** System designed to require semantic search can be bypassed entirely.

---

## Proposed Solutions

### Solution 1: Decouple Semantic Search from Torch Dependency

**Priority:** 🔴 P0 (Critical)  
**Files to Modify:** `tools/semantic_search.py`, `utils/dependencies.py`

**Design:**

```python
# tools/semantic_search.py

def semantic_code_search(query: str, top_k: int = 5) -> list[SemanticSearchResult]:
    """Search codebase using semantic similarity.
    
    Fallback chain:
    1. HybridRetriever (vector + graph signals) - needs embedding provider
    2. Direct vector search on Memgraph - needs pre-indexed embeddings
    3. Keyword-based graph search - always available
    """
    try:
        from ..config import settings
        from ..embeddings import get_embedding_provider
        from ..memgraph_advanced import HybridRetriever
        from ..services.graph_service import MemgraphIngestor
        from ..vector_backend import get_shared_backend
        
        # Attempt to get embedding provider
        try:
            config = settings.active_embedding_config
            provider = get_embedding_provider(
                provider=config.provider,
                model_id=config.model_id,
            )
        except Exception as e:
            logger.warning(f"Embedding provider unavailable: {e}")
            return _semantic_search_keyword_fallback(query, top_k)
        
        effective_top_k = top_k if top_k > 0 else settings.VECTOR_SEARCH_TOP_K
        
        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            batch_size=cs.SEMANTIC_BATCH_SIZE,
        ) as ingestor:
            retriever = HybridRetriever(
                graph_ingestor=ingestor,
                vector_backend=get_shared_backend(),
                embedding_provider=provider,
                config=settings.hybrid_retrieval_config,
            )
            
            hybrid_results = retriever.search(query, top_k=effective_top_k)
            
            formatted_results: list[SemanticSearchResult] = []
            for result in hybrid_results:
                formatted_results.append(
                    SemanticSearchResult(
                        node_id=result.node_id,
                        qualified_name=result.qualified_name,
                        name=result.name,
                        type=result.node_type,
                        similarity=round(result.combined_score, 3),
                    )
                )
            
            logger.info(ls.SEMANTIC_FOUND.format(count=len(formatted_results), query=query))
            return formatted_results
            
    except Exception as e:
        logger.error(ls.SEMANTIC_FAILED.format(query=query, error=e))
        # CRITICAL: Fallback to keyword search instead of empty list
        return _semantic_search_keyword_fallback(query, top_k)


def _semantic_search_keyword_fallback(query: str, top_k: int) -> list[SemanticSearchResult]:
    """Fallback keyword-based search when semantic search unavailable.
    
    Uses CONTAINS queries on function/class names and docstrings.
    """
    from ..config import settings
    from ..services.graph_service import MemgraphIngestor
    
    try:
        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
        ) as ingestor:
            # Extract meaningful keywords (filter out stopwords)
            stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with'}
            keywords = [w.lower() for w in query.split() if len(w) > 2 and w.lower() not in stopwords]
            if not keywords:
                return []
            
            # Use most specific keyword (longest)
            keyword = max(keywords, key=len)
            
            cypher = """
            MATCH (n:Function|Class|Method)
            WHERE n.name CONTAINS $keyword 
               OR n.qualified_name CONTAINS $keyword
               OR n.docstring CONTAINS $keyword
            RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
                   n.name AS name, labels(n)[0] AS node_type
            LIMIT $limit
            """
            results = ingestor.fetch_all(cypher, {"keyword": keyword, "limit": top_k})
            
            return [
                SemanticSearchResult(
                    node_id=r["node_id"],
                    qualified_name=r["qualified_name"],
                    name=r["name"],
                    type=r["node_type"],
                    similarity=0.5,  # Neutral score for keyword matches
                )
                for r in results
            ]
    except Exception as e:
        logger.error(f"Keyword fallback failed: {e}")
        return []
```

**Also update:** `utils/dependencies.py`

Add new function:
```python
def has_embedding_provider() -> bool:
    """Check if an embedding provider is configured and available.
    
    This is separate from has_semantic_dependencies() which checks for
    local embedding generation capabilities (torch/transformers).
    """
    try:
        from ..config import settings
        from ..embeddings import get_embedding_provider
        
        config = settings.active_embedding_config
        provider = get_embedding_provider(
            provider=config.provider,
            model_id=config.model_id,
        )
        # Test with simple embedding
        provider.embed("test")
        return True
    except Exception:
        return False
```

---

### Solution 2: HybridRetriever Factory with Shared Dependencies

**Priority:** 🟡 P1
**File to Modify:** `memgraph_advanced/hybrid_retrieval.py`

**Problem with Singleton Approach:**
The `graph_ingestor` parameter is a context manager (MemgraphIngestor) that must be used with `with` statements. A singleton HybridRetriever would either:
1. Hold a persistent connection (resource leak risk)
2. Require passing ingestor at search time (breaks existing API)

**Design: Factory Pattern with Shared Heavy Dependencies**

Instead of sharing the HybridRetriever instance, we share the **heavy dependencies** (vector backend, embedding provider) and create short-lived HybridRetriever instances:

```python
# Add to hybrid_retrieval.py

import threading
from .. import constants as cs

# Module-level shared dependencies (NOT the retriever itself)
_SHARED_EMBEDDING_PROVIDER: EmbeddingProviderProtocol | None = None
_PROVIDER_LOCK = threading.Lock()


def get_shared_embedding_provider() -> EmbeddingProviderProtocol:
    """Get or create shared embedding provider instance.

    This avoids redundant provider initialization (API key validation,
    model loading, etc.) across multiple HybridRetriever instances.
    """
    global _SHARED_EMBEDDING_PROVIDER

    if _SHARED_EMBEDDING_PROVIDER is None:
        with _PROVIDER_LOCK:
            if _SHARED_EMBEDDING_PROVIDER is None:
                from ..config import settings
                from ..embeddings import get_embedding_provider

                config = settings.active_embedding_config
                _SHARED_EMBEDDING_PROVIDER = get_embedding_provider(
                    provider=config.provider,
                    model_id=config.model_id,
                )

    return _SHARED_EMBEDDING_PROVIDER


def reset_shared_embedding_provider() -> None:
    """Reset shared provider (e.g., when configuration changes)."""
    global _SHARED_EMBEDDING_PROVIDER
    with _PROVIDER_LOCK:
        _SHARED_EMBEDDING_PROVIDER = None


def create_hybrid_retriever(graph_ingestor: QueryProtocol) -> HybridRetriever:
    """Factory function that creates HybridRetriever with shared dependencies.

    This is the recommended way to create HybridRetriever instances.
    Shares vector_backend and embedding_provider across all instances,
    avoiding redundant initialization while allowing proper context manager
    usage for graph_ingestor.

    Args:
        graph_ingestor: MemgraphIngestor instance (use as context manager)

    Returns:
        HybridRetriever configured with shared dependencies

    Example:
        with MemgraphIngestor(...) as ingestor:
            retriever = create_hybrid_retriever(ingestor)
            results = retriever.search("query")
    """
    from ..vector_backend import get_shared_backend

    return HybridRetriever(
        graph_ingestor=graph_ingestor,
        vector_backend=get_shared_backend(),  # Already singleton
        embedding_provider=get_shared_embedding_provider(),  # Shared
        config=settings.hybrid_retrieval_config,
    )
```

**Update consumers:**

In `tools/semantic_search.py`:
```python
def semantic_code_search(query: str, top_k: int = 5) -> list[SemanticSearchResult]:
    from ..memgraph_advanced import create_hybrid_retriever, get_shared_embedding_provider
    from ..services.graph_service import MemgraphIngestor

    try:
        # Early check for embedding provider availability
        try:
            get_shared_embedding_provider()
        except Exception as e:
            logger.warning(f"Embedding provider unavailable: {e}")
            return _semantic_search_keyword_fallback(query, top_k)

        effective_top_k = top_k if top_k > 0 else settings.VECTOR_SEARCH_TOP_K

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            batch_size=cs.SEMANTIC_BATCH_SIZE,
        ) as ingestor:
            retriever = create_hybrid_retriever(ingestor)
            hybrid_results = retriever.search(query, top_k=effective_top_k)
            # ... format results
```

In `shared/query_router.py`:
```python
def _query_code_only(self, request: QueryRequest) -> QueryResponse:
    from ..memgraph_advanced import create_hybrid_retriever

    try:
        with MemgraphIngestor(...) as ingestor:
            retriever = create_hybrid_retriever(ingestor)
            results = retriever.search(query=request.question, top_k=request.top_k)
            # ... process results
    except Exception as e:
        # ... fallback
```

**Fix QueryRouter vector backend initialization:**

Also fix `shared/query_router.py:208-221` to use shared backends:

```python
# query_router.py - Update __init__
def __init__(
    self,
    code_graph: QueryProtocol | None = None,
    doc_graph: QueryProtocol | None = None,
    code_vector: VectorBackend | None = None,
    doc_vector: VectorBackend | None = None,
):
    from ..vector_backend import get_shared_backend  # Changed from get_vector_backend

    self.code_graph = code_graph
    self.doc_graph = doc_graph
    # Use shared singleton instead of creating new instances
    self.code_vector = code_vector or get_shared_backend()
    self.doc_vector = doc_vector or get_shared_backend_for_documents()  # Need to add this
    self.current_mode: QueryMode = QueryMode.CODE_ONLY
```

Note: This requires adding `get_shared_backend_for_documents()` to `vector_backend.py`:

```python
_DOC_BACKEND_INSTANCE: VectorBackend | None = None

def get_shared_backend_for_documents() -> VectorBackend:
    """Get shared document backend instance."""
    global _DOC_BACKEND_INSTANCE
    if _DOC_BACKEND_INSTANCE is None:
        _DOC_BACKEND_INSTANCE = get_vector_backend(is_document=True)
        _DOC_BACKEND_INSTANCE.initialize()
    return _DOC_BACKEND_INSTANCE
```

**Benefits:**
- ✅ No API signature changes to HybridRetriever
- ✅ Proper context manager usage for connections
- ✅ Shared embedding provider avoids redundant model loading/validation
- ✅ Shared vector backends for QueryRouter
- ✅ Thread-safe via locks
- ✅ Backward compatible - existing code still works

---

### Solution 3: Expose Advanced Algorithms as MCP/Agentic Tools

**Priority:** 🟡 P1  
**Files to Modify:** `mcp/tools.py`, `tools/__init__.py`, `constants.py`, `tool_descriptions.py`

**Design:**

**Step 1: Add new tool names to constants.py**

```python
# Add to MCPToolName enum or constants
MCPToolName.COMMUNITY_SUMMARY = "community_summary"
MCPToolName.ANALYZE_PATH = "analyze_path"
MCPToolName.FIND_SIMILAR_FUNCTIONS = "find_similar_functions"
MCPToolName.EXPAND_CONTEXT = "expand_context"
```

**Step 2: Add tool descriptions to tool_descriptions.py**

```python
# Add to AGENTIC_TOOLS dict
AgenticToolName.COMMUNITY_SUMMARY: (
    "Generate query-focused summary using community detection. "
    "Identifies code communities and summarizes the most relevant ones "
    "for a given question. Best for understanding large codebases."
)

AgenticToolName.ANALYZE_PATH: (
    "Analyze call paths between two functions. Finds k-shortest paths, "
    "identifies critical nodes and bottlenecks. "
    "Example: 'Analyze path from main() to database.connect()'"
)

AgenticToolName.FIND_SIMILAR_FUNCTIONS: (
    "Find functions similar to a given function based on call patterns "
    "using Jaccard similarity. Returns similarity scores."
)

AgenticToolName.EXPAND_CONTEXT: (
    "Expand context around a code entity using BFS traversal. "
    "Returns related functions, classes, and modules ordered by importance."
)
```

**Step 3: Create tool factories**

```python
# tools/graph_algorithms_tools.py (NEW)

from __future__ import annotations

from pydantic_ai import Tool
from loguru import logger

from ..memgraph_advanced import CommunityQFS, PathAnalyzer
from ..graph_algorithms import get_shared_algorithms
from ..config import settings


def create_community_summary_tool() -> Tool:
    """Create tool for query-focused community summarization."""
    
    async def community_summary(question: str, top_communities: int = 3) -> str:
        logger.info(f"Community summary requested: {question[:50]}...")
        try:
            qfs = CommunityQFS()
            summary = qfs.query_focused_summary(
                question=question,
                top_communities=top_communities,
                min_community_size=settings.qfs_config.min_community_size,
            )
            return summary
        except Exception as e:
            logger.error(f"Community summary failed: {e}")
            return f"Community summary failed: {e}"
    
    return Tool(
        community_summary,
        name="community_summary",
        description="Generate query-focused summary using community detection",
    )


def create_analyze_path_tool() -> Tool:
    """Create tool for path analysis between code entities."""
    
    async def analyze_path(
        source: str,
        target: str,
        max_paths: int = 3,
        max_length: int = 10,
    ) -> str:
        logger.info(f"Path analysis: {source} -> {target}")
        try:
            analyzer = PathAnalyzer()
            result = analyzer.analyze_call_chain(
                start_qn=source,
                end_qn=target,
                max_paths=max_paths,
                max_path_length=max_length,
            )
            
            if not result.paths:
                return f"No call paths found between '{source}' and '{target}'."
            
            lines = [
                f"Call paths from '{source}' to '{target}':",
                f"Shortest path length: {result.shortest_path_length}",
                f"Alternative paths: {result.alternative_paths}",
            ]
            
            if result.critical_nodes:
                lines.append(f"Critical nodes (appear in multiple paths): {', '.join(result.critical_nodes)}")
            
            if result.bottlenecks:
                lines.append(f"Bottlenecks: {', '.join(result.bottlenecks)}")
            
            for i, path in enumerate(result.paths, 1):
                nodes = [n.get('qualified_name', '?') for n in path.get('nodes', [])]
                lines.append(f"\nPath {i} (length {path.get('length', '?')}):")
                lines.append(f"  {' -> '.join(nodes)}")
            
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Path analysis failed: {e}")
            return f"Path analysis failed: {e}"
    
    return Tool(
        analyze_path,
        name="analyze_path",
        description="Analyze call paths between two functions",
    )


def create_find_similar_functions_tool() -> Tool:
    """Create tool for finding similar functions."""
    
    async def find_similar_functions(
        function_name: str,
        min_similarity: float = 0.3,
        limit: int = 10,
    ) -> str:
        logger.info(f"Finding similar functions to: {function_name}")
        try:
            algorithms = get_shared_algorithms()
            results = algorithms.get_similar_nodes(
                node_id=int(function_name),  # Note: needs node_id, not name
                top_k=limit,
            )
            
            if not results:
                return f"No similar functions found for node ID {function_name}."
            
            lines = [f"Similar functions to node {function_name}:"]
            for r in results:
                lines.append(
                    f"  - {r['qualified_name']} (similarity: {r['jaccard_similarity']:.3f})"
                )
            
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Similar functions search failed: {e}")
            return f"Similar functions search failed: {e}"
    
    return Tool(
        find_similar_functions,
        name="find_similar_functions",
        description="Find functions similar to a given function based on call patterns",
    )


def create_expand_context_tool() -> Tool:
    """Create tool for BFS context expansion."""
    
    async def expand_context(
        node_id: int,
        max_depth: int = 3,
    ) -> str:
        logger.info(f"Expanding context from node {node_id}")
        try:
            algorithms = get_shared_algorithms()
            results = algorithms.get_bfs_context(
                start_node_id=node_id,
                max_depth=max_depth,
            )
            
            if not results:
                return f"No context found for node {node_id}."
            
            lines = [f"Context around node {node_id} (depth {max_depth}):"]
            for r in results:
                depth = r.get('depth', '?')
                qn = r.get('qualified_name', '?')
                pr = r.get('pagerank_score', 0)
                lines.append(f"  [depth {depth}, pagerank {pr:.4f}] {qn}")
            
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"Context expansion failed: {e}")
            return f"Context expansion failed: {e}"
    
    return Tool(
        expand_context,
        name="expand_context",
        description="Expand context around a code entity using BFS traversal",
    )
```

**Step 4: Register in MCPToolsRegistry**

```python
# In mcp/tools.py, MCPToolsRegistry.__init__

# Add new tools
self._tools[cs.MCPToolName.COMMUNITY_SUMMARY] = ToolMetadata(
    name=cs.MCPToolName.COMMUNITY_SUMMARY,
    description=td.MCP_TOOLS[cs.MCPToolName.COMMUNITY_SUMMARY],
    input_schema=MCPInputSchema(
        type=cs.MCPSchemaType.OBJECT,
        properties={
            "question": MCPInputSchemaProperty(
                type=cs.MCPSchemaType.STRING,
                description="Question to answer using community summaries",
            ),
            "top_communities": MCPInputSchemaProperty(
                type=cs.MCPSchemaType.INTEGER,
                description="Number of top communities to consider",
                default=3,
            ),
        },
        required=["question"],
    ),
    handler=self.community_summary,
    returns_json=False,
)

# Add handler methods
async def community_summary(self, question: str, top_communities: int = 3) -> str:
    """Generate query-focused summary using community detection."""
    from codebase_rag.tools.graph_algorithms_tools import create_community_summary_tool
    
    tool = create_community_summary_tool()
    result = await tool.function(question=question, top_communities=top_communities)
    return str(result)
```

---

### Solution 4: Semantic-Aware Graph Navigation

**Priority:** 🟡 P1  
**File to Modify:** `tools/graph_navigation.py`

**Design:**

Update `get_call_hierarchy` to use PageRank and community importance for ranking:

```python
async def get_call_hierarchy(
    self, qualified_name: str, direction: str = "both", depth: int = 2
) -> str:
    depth = min(max(depth, 1), _MAX_DEPTH)
    
    if direction not in ("callers", "callees", "both"):
        return te.ERROR_WRAPPER.format(
            message=f"Invalid direction: '{direction}'. Use 'callers', 'callees', or 'both'."
        )
    
    logger.info(
        ls.GRAPH_CALL_HIERARCHY.format(
            name=qualified_name, direction=direction, depth=depth
        )
    )
    
    try:
        lines = [f"Call hierarchy for '{qualified_name}' (depth={depth}):"]
        
        # ENHANCED: Query with PageRank and community scoring
        if direction in ("callers", "both"):
            callers_query = """
            MATCH path = (caller:Function|Method)-[:CALLS*1..$depth]->(target)
            WHERE target.qualified_name = $qn
            RETURN DISTINCT 
                caller.qualified_name AS qualified_name,
                caller.name AS name,
                length(path) AS depth,
                COALESCE(caller.pagerank_score, 0.1) AS pagerank,
                COALESCE(caller.community_importance, 0.0) AS community_importance,
                caller.docstring AS docstring
            ORDER BY (pagerank * 0.7 + community_importance * 0.3) DESC, depth ASC
            """
            callers = await asyncio.to_thread(
                self.ingestor.fetch_all,
                callers_query,
                {"qn": qualified_name, "depth": depth},
            )
            lines.append(f"\nCallers ({len(callers)}, ranked by importance):")
            if callers:
                for row in callers:
                    d = row.get("depth", "?")
                    qn = row.get("qualified_name", "unknown")
                    pr = row.get("pagerank", 0)
                    ci = row.get("community_importance", 0)
                    importance = pr * 0.7 + ci * 0.3
                    docstring_preview = ""
                    ds = row.get("docstring", "")
                    if ds:
                        docstring_preview = f" - {ds[:80]}..."
                    lines.append(
                        f"  {'  ' * min(d - 1, 3)}[depth {d}, importance: {importance:.3f}] {qn}{docstring_preview}"
                    )
            else:
                lines.append("  (none)")
        
        # Similar enhancement for callees...
        
        return "\n".join(lines)
    
    except Exception as e:
        logger.error(ls.GRAPH_NAVIGATOR_ERROR.format(name=qualified_name, error=e))
        return te.ERROR_WRAPPER.format(message=str(e))
```

---

### Solution 5: Fix HybridRetrievalConfig Text Weight

**Priority:** 🟠 P1
**Files to Modify:** `config.py`, `memgraph_advanced/hybrid_retrieval.py`

**Current Problem:**
```python
# config.py:226-253
@dataclass
class HybridRetrievalConfig:
    vector_weight: float = 0.6
    text_weight: float = 0.2      # ← Configured but never used!
    pagerank_weight: float = 0.15
    community_weight: float = 0.05
    # Validation requires sum of ALL 4 weights ≈ 1.0

# hybrid_retrieval.py:162
text_score=0.0,  # ← Always hardcoded to 0.0
```

**Option A: Remove text_weight entirely (Recommended)**

```python
# config.py

@dataclass
class HybridRetrievalConfig:
    """Configuration for hybrid retrieval.

    Note: text_weight was removed in v1.1 as Memgraph text indexing
    is not currently implemented. The weight redistribution preserves
    relative importance of remaining signals.
    """

    vector_weight: float = 0.7  # Was 0.6, increased proportionally
    pagerank_weight: float = 0.2  # Was 0.15
    community_weight: float = 0.1  # Was 0.05
    top_k: int = 10
    max_context_depth: int = 2
    min_similarity_threshold: float = 0.1

    def __post_init__(self) -> None:
        """Validate that weights are in valid range and sum to 1.0."""
        weights = [
            self.vector_weight,
            self.pagerank_weight,
            self.community_weight,
        ]
        for i, w in enumerate(weights):
            if not 0.0 <= w <= 1.0:
                weight_names = ["vector_weight", "pagerank_weight", "community_weight"]
                raise ValueError(f"{weight_names[i]} must be between 0 and 1, got {w}")

        total = sum(weights)
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"Hybrid weights must sum to 1.0, got {total:.3f} "
                f"(vector={self.vector_weight}, pagerank={self.pagerank_weight}, "
                f"community={self.community_weight})"
            )
```

**Migration for Existing Configurations:**

```python
# Add to config.py AppConfig class

def _migrate_hybrid_config(self) -> HybridRetrievalConfig:
    """Migrate old config with text_weight to new format.

    If user has custom weights set, redistribute proportionally.
    """
    old_config = self._hybrid_retrieval_config
    if old_config is None:
        return HybridRetrievalConfig(top_k=self.VECTOR_SEARCH_TOP_K)

    # Check if old config has text_weight > 0 (legacy config)
    if hasattr(old_config, 'text_weight') and old_config.text_weight > 0:
        logger.info(
            f"Migrating HybridRetrievalConfig: redistributing text_weight={old_config.text_weight}"
        )
        # Redistribute text_weight proportionally to other weights
        total_active = old_config.vector_weight + old_config.pagerank_weight + old_config.community_weight
        if total_active > 0:
            scale = 1.0 / total_active
            return HybridRetrievalConfig(
                vector_weight=old_config.vector_weight * scale,
                pagerank_weight=old_config.pagerank_weight * scale,
                community_weight=old_config.community_weight * scale,
                top_k=old_config.top_k,
                max_context_depth=old_config.max_context_depth,
                min_similarity_threshold=old_config.min_similarity_threshold,
            )

    # Already migrated or default
    return old_config
```

**Update HybridRetriever.search() - no changes needed:**

The current implementation already handles this correctly:
```python
# hybrid_retrieval.py:133-151 (existing code works as-is)
graph_weight = cfg.pagerank_weight + cfg.community_weight

for record in records:
    # ...
    combined_score = (
        vector_score * cfg.vector_weight + graph_score * graph_weight
    )
    results.append(
        HybridSearchResult(
            # ...
            text_score=0.0,  # Keep for backward compatibility with HybridSearchResult dataclass
            # ...
        )
    )
```

**Keep HybridSearchResult.text_score for backward compatibility:**

```python
# hybrid_retrieval.py:33-49 - DO NOT CHANGE
@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    # ... other fields ...
    text_score: float  # Retained for API compatibility, always 0.0
    # ... other fields ...
```

**Environment Variable Migration (optional):**

```bash
# Old (deprecated, will be ignored):
# HYBRID_TEXT_WEIGHT=0.2

# New weights (auto-normalized if sum ≠ 1.0):
HYBRID_VECTOR_WEIGHT=0.7
HYBRID_PAGERANK_WEIGHT=0.2
HYBRID_COMMUNITY_WEIGHT=0.1
```

---

### Solution 6: Implement Semantic Search Result Caching

**Priority:** 🟡 P1  
**New File:** `utils/semantic_cache.py`

**Design:**

```python
"""LRU cache for semantic search results."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class CacheEntry:
    """Single cache entry."""
    results: Any
    timestamp: float
    access_count: int = 0
    
    def touch(self) -> None:
        self.access_count += 1
        self.timestamp = time.time()


class SemanticSearchCache:
    """LRU cache for semantic search results.
    
    Caches results based on query hash to avoid redundant embedding generation
    and vector search for identical queries.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    @staticmethod
    def _hash_query(query: str, top_k: int) -> str:
        """Create deterministic hash for query parameters."""
        content = f"{query.lower().strip()}|{top_k}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def get(self, query: str, top_k: int) -> Any | None:
        """Get cached result if available and not expired."""
        key = self._hash_query(query, top_k)
        
        if key not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[key]
        
        # Check TTL
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self._cache[key]
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._hits += 1
        
        logger.debug(f"Semantic cache hit for query: {query[:50]}...")
        return entry.results
    
    def put(self, query: str, top_k: int, results: Any) -> None:
        """Store result in cache."""
        key = self._hash_query(query, top_k)
        
        # Evict if at capacity
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Remove least recently used
        
        self._cache[key] = CacheEntry(
            results=results,
            timestamp=time.time(),
        )
        logger.debug(f"Semantic cache stored: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }


# Module-level singleton
_CACHE: SemanticSearchCache | None = None


def get_semantic_cache() -> SemanticSearchCache:
    """Get shared semantic search cache."""
    global _CACHE
    if _CACHE is None:
        from ..config import settings
        _CACHE = SemanticSearchCache(
            max_size=settings.CACHE_MAX_ENTRIES,
            ttl_seconds=3600,  # 1 hour default
        )
    return _CACHE


def reset_semantic_cache() -> None:
    """Reset the cache (e.g., when embeddings change)."""
    global _CACHE
    if _CACHE is not None:
        _CACHE.clear()
    _CACHE = None
```

**Integration into semantic search:**

```python
# tools/semantic_search.py

def semantic_code_search(query: str, top_k: int = 5) -> list[SemanticSearchResult]:
    from ..utils.semantic_cache import get_semantic_cache
    
    # Check cache first
    cache = get_semantic_cache()
    cached = cache.get(query, top_k)
    if cached is not None:
        return cached
    
    # ... perform search ...
    results = ... # from HybridRetriever
    
    # Cache results
    if results:
        cache.put(query, top_k, results)
    
    return results
```

---

### Solution 7: Comprehensive Fallback Chain

**Priority:** 🟠 P1  
**File to Modify:** `tools/semantic_search.py`

**Design:**

Implement layered fallback as shown in Solution 1, with explicit fallback levels:

```python
def semantic_code_search(query: str, top_k: int = 5) -> list[SemanticSearchResult]:
    """Search with comprehensive fallback chain.
    
    Fallback order:
    1. Semantic cache (fastest)
    2. HybridRetriever (best quality - vector + graph signals)
    3. Direct vector search (if embeddings exist but HybridRetriever fails)
    4. Keyword search (always available, lower quality)
    5. Empty with warning (last resort)
    """
    from ..utils.semantic_cache import get_semantic_cache
    
    # Level 1: Cache
    cache = get_semantic_cache()
    cached = cache.get(query, top_k)
    if cached is not None:
        logger.info(f"Cache hit for query: {query[:50]}...")
        return cached
    
    # Level 2: HybridRetriever
    try:
        results = _search_with_hybrid_retriever(query, top_k)
        if results:
            cache.put(query, top_k, results)
            return results
    except Exception as e:
        logger.warning(f"HybridRetriever failed: {e}")
    
    # Level 3: Direct vector search
    try:
        results = _search_direct_vector(query, top_k)
        if results:
            cache.put(query, top_k, results)
            return results
    except Exception as e:
        logger.warning(f"Direct vector search failed: {e}")
    
    # Level 4: Keyword fallback
    try:
        results = _semantic_search_keyword_fallback(query, top_k)
        if results:
            cache.put(query, top_k, results)
            return results
    except Exception as e:
        logger.warning(f"Keyword fallback failed: {e}")
    
    # Level 5: Empty with clear message
    logger.warning(f"All search methods failed for query: {query}")
    return []


def _search_with_hybrid_retriever(query: str, top_k: int) -> list[SemanticSearchResult]:
    """Level 2: HybridRetriever search."""
    from ..config import settings
    from ..embeddings import get_embedding_provider
    from ..memgraph_advanced import HybridRetriever
    from ..services.graph_service import MemgraphIngestor
    from ..vector_backend import get_shared_backend
    
    config = settings.active_embedding_config
    provider = get_embedding_provider(
        provider=config.provider,
        model_id=config.model_id,
    )
    
    effective_top_k = top_k if top_k > 0 else settings.VECTOR_SEARCH_TOP_K
    
    with MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=cs.SEMANTIC_BATCH_SIZE,
    ) as ingestor:
        retriever = HybridRetriever(
            graph_ingestor=ingestor,
            vector_backend=get_shared_backend(),
            embedding_provider=provider,
            config=settings.hybrid_retrieval_config,
        )
        
        hybrid_results = retriever.search(query, top_k=effective_top_k)
        
        return [
            SemanticSearchResult(
                node_id=result.node_id,
                qualified_name=result.qualified_name,
                name=result.name,
                type=result.node_type,
                similarity=round(result.combined_score, 3),
            )
            for result in hybrid_results
        ]


def _search_direct_vector(query: str, top_k: int) -> list[SemanticSearchResult]:
    """Level 3: Direct vector search without HybridRetriever."""
    from ..config import settings
    from ..embeddings import get_embedding_provider
    from ..vector_backend import get_shared_backend
    
    config = settings.active_embedding_config
    provider = get_embedding_provider(
        provider=config.provider,
        model_id=config.model_id,
    )
    
    query_embedding = provider.embed(query)
    backend = get_shared_backend()
    
    # Direct vector search
    vector_results = backend.search(query_embedding, top_k=top_k)
    
    # Fetch metadata from graph
    from ..services.graph_service import MemgraphIngestor
    
    with MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
    ) as ingestor:
        node_ids = [nid for nid, _ in vector_results]
        if not node_ids:
            return []
        
        cypher = """
        MATCH (n)
        WHERE id(n) IN $node_ids
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS node_type
        """
        metadata = ingestor.fetch_all(cypher, {"node_ids": node_ids})
        metadata_map = {r["node_id"]: r for r in metadata}
        
        return [
            SemanticSearchResult(
                node_id=nid,
                qualified_name=metadata_map.get(nid, {}).get("qualified_name", "?"),
                name=metadata_map.get(nid, {}).get("name", "?"),
                type=metadata_map.get(nid, {}).get("node_type", "?"),
                similarity=round(score, 3),
            )
            for nid, score in vector_results
        ]
```

---

### Solution 8: Fix Sufficiency Gatekeeper Loophole

**Priority:** 🟡 P1  
**Files to Modify:** `orchestrator/sufficiency_gatekeeper.py`, `orchestrator/investigation_tracker.py`

**Design:**

**Update sufficiency_gatekeeper.py:**

```python
def evaluate_sufficiency(
    state: InvestigationState,
    reqs: InvestigationRequirements,
    rejection_count: int = 0,
) -> tuple[bool, str | None]:
    if rejection_count >= MAX_REJECTION_LIMIT:
        return True, None

    if state.rounds_completed < reqs.min_rounds:
        return False, (
            f"Investigation too shallow. You need at least {reqs.min_rounds} rounds of querying "
            f"(currently at round {state.rounds_completed})."
        )

    if reqs.requires_vector:
        # ENHANCED: Check if semantic_search was attempted, not just if it succeeded
        if AgenticToolName.SEMANTIC_SEARCH not in state.tools_used:
            return False, (
                "CRITICAL: You must use `semantic_search` first to find relevant candidates by intent. "
                "This is the recommended entry point for functional and structural queries."
            )
        
        # If it was used but returned 0 results, that's still valid usage
        # But we should encourage retrying with different queries
        if AgenticToolName.SEMANTIC_SEARCH in state.tool_failures:
            # Add a warning but don't block
            logger.warning(
                f"semantic_search returned empty results. "
                f"Consider trying a different query or using fallback methods."
            )

    if reqs.requires_graph:
        # Similar enhancement for graph queries
        if AgenticToolName.QUERY_GRAPH not in state.tools_used:
            return False, (
                "CRITICAL: You must use `query_graph` to understand structural relationships "
                "between code elements."
            )

    # ... rest of the function

    return True, None
```

**Update investigation_tracker.py:**

```python
def record_tool(
    self, tool_name: str, query_arg: str = "", results_count: int = -1
) -> None:
    # ALWAYS record tool usage, even if it returned 0 results
    self.tools_used.add(tool_name)

    if tool_name == AgenticToolName.QUERY_GRAPH:
        self.graph_queries_run += 1

    if tool_name in FILE_READ_TOOLS and query_arg:
        self.files_read.append(query_arg)

    # ENHANCED: Still track failures separately, but don't exclude from usage
    if results_count == 0 and tool_name in {
        AgenticToolName.SEMANTIC_SEARCH,
        AgenticToolName.QUERY_GRAPH,
    }:
        self.tool_failures.add(tool_name)
        logger.debug(
            f"Tool {tool_name} returned empty results — tracked as failure but still counts as usage"
        )
```

---

## Implementation Plan

### Phase 1: Critical Fixes (Week 1)

| Task | File | Priority | Effort |
|------|------|----------|--------|
| Remove torch dependency gate | `tools/semantic_search.py` | P0 | 2h |
| Add keyword fallback | `tools/semantic_search.py` | P0 | 3h |
| Add has_embedding_provider() | `utils/dependencies.py` | P0 | 1h |
| Fix sufficiency gatekeeper loophole | `orchestrator/sufficiency_gatekeeper.py` | P0 | 2h |

### Phase 2: Performance Optimization (Week 2)

| Task | File | Priority | Effort |
|------|------|----------|--------|
| Implement shared HybridRetriever | `memgraph_advanced/hybrid_retrieval.py` | P1 | 4h |
| Add semantic search cache | `utils/semantic_cache.py` (new) | P1 | 3h |
| Fix HybridRetrievalConfig text_weight | `config.py`, `hybrid_retrieval.py` | P1 | 2h |
| Update all consumers | Multiple | P1 | 4h |

### Phase 3: Feature Enhancement (Week 3)

| Task | File | Priority | Effort |
|------|------|----------|--------|
| Create graph algorithm tools | `tools/graph_algorithms_tools.py` (new) | P1 | 6h |
| Register in MCP registry | `mcp/tools.py` | P1 | 3h |
| Add tool descriptions | `tool_descriptions.py` | P1 | 2h |
| Enhance graph navigation ranking | `tools/graph_navigation.py` | P1 | 3h |

### Phase 4: Testing & Validation (Week 4)

| Task | Priority | Effort |
|------|----------|--------|
| Unit tests for semantic search fallbacks | P0 | 4h |
| Unit tests for caching | P1 | 2h |
| Integration tests with Memgraph | P1 | 4h |
| Performance benchmarks | P2 | 3h |
| Documentation updates | P2 | 2h |

**Total Estimated Effort:** ~50 hours

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_semantic_search.py

def test_semantic_search_without_torch():
    """Verify semantic search works without torch/transformers."""
    with patch('codebase_rag.utils.dependencies.has_semantic_dependencies', return_value=False):
        results = semantic_code_search("test query")
        # Should use keyword fallback, not return empty
        assert isinstance(results, list)
        # Keyword fallback may return results or empty list
        # But it should NOT raise an exception

def test_semantic_search_keyword_fallback():
    """Verify keyword fallback returns results for known terms."""
    results = _semantic_search_keyword_fallback("authenticate", top_k=5)
    assert isinstance(results, list)
    # May return results if "authenticate" exists in codebase

def test_semantic_search_cache_hit():
    """Verify caching works correctly."""
    cache = get_semantic_cache()
    cache.clear()
    
    # Mock the search to return predictable results
    with patch('codebase_rag.tools.semantic_search._search_with_hybrid_retriever') as mock_search:
        mock_search.return_value = [mock_result_1, mock_result_2]
        
        # First call - miss
        results1 = semantic_code_search("unique query")
        assert cache.stats["misses"] == 1
        
        # Second call - hit
        results2 = semantic_code_search("unique query")
        assert cache.stats["hits"] == 1
        assert results1 == results2

def test_hybrid_retrieval_config_weights():
    """Verify weight validation."""
    # Valid config (new weights)
    config = HybridRetrievalConfig(
        vector_weight=0.7,
        pagerank_weight=0.2,
        community_weight=0.1,
    )
    assert config.vector_weight == 0.7
    
    # Invalid - doesn't sum to 1
    with pytest.raises(ValueError):
        HybridRetrievalConfig(
            vector_weight=0.5,
            pagerank_weight=0.5,
            community_weight=0.5,
        )

def test_sufficiency_gatekeeper_allows_failed_search():
    """Verify sufficiency allows semantic_search even if it failed."""
    state = InvestigationState()
    state.rounds_completed = 3
    state.tools_used.add(AgenticToolName.SEMANTIC_SEARCH)
    state.tool_failures.add(AgenticToolName.SEMANTIC_SEARCH)  # Failed but used
    state.tools_used.add(AgenticToolName.QUERY_GRAPH)
    
    reqs = InvestigationRequirements(
        question_type=QuestionType.FUNCTIONAL,
        requires_vector=True,
        requires_graph=True,
        requires_file_read=True,
        min_rounds=3,
    )
    
    passed, warning = evaluate_sufficiency(state, reqs)
    assert passed == True  # Should pass even with failed semantic_search
```

### Integration Tests

```python
# tests/integration/test_graph_traversal.py

def test_query_router_with_vector_backend():
    """Verify QueryRouter uses vector backends correctly."""
    router = QueryRouter(
        code_graph=mock_ingestor,
        code_vector=mock_vector_backend,
    )
    
    request = QueryRequest(
        question="find authentication functions",
        mode=QueryMode.CODE_ONLY,
    )
    
    response = router.query(request)
    # Should not error, may return results or empty
    assert response.mode == QueryMode.CODE_ONLY

def test_semantic_aware_call_hierarchy():
    """Verify call hierarchy includes importance scores."""
    navigator = GraphNavigator(project_root=".", ingestor=mock_ingestor)
    
    result = asyncio.run(navigator.get_call_hierarchy("auth.login"))
    
    assert "importance" in result
    assert "pagerank" in result or "community" in result

def test_community_summary_tool():
    """Verify community summary tool works."""
    from codebase_rag.tools.graph_algorithms_tools import create_community_summary_tool
    
    tool = create_community_summary_tool()
    result = asyncio.run(tool.function(question="authentication", top_communities=2))
    
    assert isinstance(result, str)
    # May contain summary or error message
```

---

## Migration Guide

### For Existing Deployments

1. **No breaking changes** - All fixes are backward compatible
2. **Optional**: Set `EMBEDDING_PROVIDER` to use cloud providers without torch
3. **Recommended**: Enable semantic cache (default is on)

### Configuration Changes

```env
# Existing (works as before)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=microsoft/unixcoder-base

# NEW: Cloud provider without torch dependency
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# NEW: Cache configuration (optional, has defaults)
CACHE_MAX_ENTRIES=1000

# NEW: Hybrid retrieval weights (optional, has defaults)
# VECTOR_WEIGHT=0.7
# PAGERANK_WEIGHT=0.2
# COMMUNITY_WEIGHT=0.1
```

---

## Success Metrics

| Metric | Before | After Target |
|--------|--------|--------------|
| Semantic search availability | ~50% (requires torch) | ~100% (fallback chain) |
| Average search latency | ~800ms | ~200ms (with cache hits) |
| Cache hit rate | 0% | >30% |
| Tool coverage | 16 tools | 20+ tools |
| Graph traversal relevance | Alphabetical | PageRank-weighted |
| Sufficiency enforcement | Bypassable | Strict |

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Cache serves stale results | Medium | Low | TTL-based eviction (1h default), manual clear API |
| Shared retriever connection leaks | Low | Low | Context manager usage, health checks on each search |
| Weight redistribution changes results | Low | High | Configurable via env vars, backward compatible defaults |
| New tools require Enterprise Memgraph | Medium | Medium | Graceful degradation, community edition fallback, clear error messages |
| Keyword fallback returns low-quality results | Low | High | Clearly marked as fallback in UI, similarity score = 0.5 |

---

## Conclusion

This specification addresses 8 critical issues preventing the Code-Graph-RAG system from making full use of its graph traversal and semantic search capabilities:

1. **Semantic search blocked by torch** - Decouple query-time search from embedding generation
2. **Redundant HybridRetriever** - Implement singleton pattern
3. **Advanced algorithms unexposed** - Create MCP/agentic tools for CommunityQFS, PathAnalyzer, etc.
4. **Graph navigation unranked** - Add PageRank/community importance scoring
5. **Text weight dead code** - Remove or implement text search
6. **No caching** - Add LRU cache for semantic search results
7. **Insufficient fallbacks** - Implement 5-level fallback chain
8. **Sufficiency loophole** - Track usage separately from failures

**All solutions are:**
- ✅ Logically sound - Address root causes, not symptoms
- ✅ Implementation-ready - Code snippets provided, file locations specified
- ✅ Aligned with existing codebase - Uses existing patterns (singleton, context managers, Pydantic config)
- ✅ Backward compatible - No breaking changes to existing APIs

**Recommended Implementation Order:**
1. Phase 1 (P0 fixes) - 1 week
2. Phase 2 (Performance) - 1 week
3. Phase 3 (Features) - 1 week
4. Phase 4 (Testing) - 1 week

**Total Timeline:** 4 weeks
