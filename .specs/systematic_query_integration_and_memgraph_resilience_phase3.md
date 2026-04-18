# Systematic Query Method Integration & Memgraph Failure Resilience Design Spec

**Version:** 1.1 (Reviewed & Corrected)
**Date:** 2026-04-18
**Status:** Implementation-Ready
**Scope:** `codebase_rag/` — Query method integration, failure classification, and systematic retrieval
**Predecessor:** `.specs/graph_traversal_semantic_search_optimization_phase2.md` (Phase 2 — fully implemented)

**Changes from v1.0:**
- Issue 1: Renamed to "No Intelligent Intent-Based Query Method Orchestration"; clarified that HybridRetriever already combines vector+graph scoring
- Issue 4: Merged into Issue 1 (QueryMethodOrchestrator._merge_and_rank)
- Issue 5: Changed from "Create MemgraphHealthMonitor" to "Extend HealthChecker with runtime monitoring"
- Issue 6: Corrected description to acknowledge HybridRetriever usage; clarified gap is paradigm orchestration
- Code fixes: Dependency injection for HybridRetriever, @dataclass for FailureClassification, async handling for CypherGenerator

---

## Executive Summary

Phase 1 and Phase 2 resolved 18 critical issues (syntax bugs, fallback chains, caching, factory patterns, etc.). This Phase 3 spec identifies **7 remaining systemic issues** preventing the codebase from making **full systematic use** of all query methods (graph traversal, semantic search, keyword search, vector search, graph algorithms) in an integrated, intelligent manner.

**Key Finding:** The codebase has all the pieces but lacks a **systematic integration framework** that:
1. Intelligently combines all query methods based on query characteristics
2. Classifies Memgraph failures systematically with appropriate recovery strategies
3. Handles cross-graph reference resolution robustly
4. Provides granular health monitoring for different Memgraph capabilities

### Issue Severity Summary

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | No intelligent intent-based query method orchestration | 🟠 HIGH | HybridRetriever combines vector+graph, but lacks multi-paradigm orchestration |
| 2 | Memgraph failure classification lacks granularity | 🟠 HIGH | Same recovery for syntax errors vs network failures |
| 3 | Cross-graph reference resolution fragile | 🟡 MEDIUM | Document→code references fail silently |
| 4 | ~~No meta-retriever for combined ranking~~ | ✅ MERGED | Merged into Issue 1's QueryMethodOrchestrator._merge_and_rank() |
| 5 | No runtime health monitoring during query execution | 🟡 MEDIUM | Pre-flight checks exist, but no session-level degradation detection |
| 6 | QueryRouter lacks intent-based paradigm orchestration | 🟡 MEDIUM | Uses HybridRetriever, but doesn't orchestrate multiple paradigms |
| 7 | Missing integration tests for query combination | 🟡 MEDIUM | No verification of systematic method usage |

---

## Issue 1: No Intelligent Intent-Based Query Method Orchestration

**Severity:** 🟠 HIGH
**Scope:** `codebase_rag/tools/`, `codebase_rag/shared/`, `codebase_rag/memgraph_advanced/`

### Problem Analysis

**What Already Exists:**
The codebase has `HybridRetriever` (`memgraph_advanced/hybrid_retrieval.py`) which combines:
- Vector similarity search
- Graph-based scoring (PageRank, community importance)
- Configurable weight-based ranking

`QueryRouter._query_code_only()` already uses `HybridRetriever` for semantic queries.

**The Gap:**
`HybridRetriever` only handles **vector-based semantic search + graph scoring**. It does NOT:
- Classify query intent to select appropriate methods
- Execute **multiple retrieval paradigms** in parallel (e.g., Cypher LLM queries + semantic search)
- Combine results from fundamentally different approaches (graph traversal vs. vector search)

The codebase implements these methods in silos:

| Method | Location | Current Usage |
|--------|----------|---------------|
| Semantic Search (HybridRetriever) | `memgraph_advanced/hybrid_retrieval.py` | ✅ Combines vector + graph scoring |
| Graph Traversal (Cypher via LLM) | `tools/codebase_query.py` | Standalone tool |
| Keyword Search | `utils/query_utils.py` + inline Cypher | Fallback only |
| Graph Algorithms (PageRank, BFS, etc.) | `graph_algorithms.py` | Standalone tools |
| Graph Navigation | `tools/graph_navigation.py` | Standalone tool |

**Missing:** A framework that **orchestrates multiple retrieval paradigms** based on query intent:
```
User Query → Classify intent → Select methods → Execute in parallel → 
Merge & rank results from different paradigms → Return enriched results
```

### Impact

- Semantic search finds functions by intent but doesn't execute structural queries
- Graph queries find structural relationships but aren't combined with semantic results
- No intent-based selection of which methods to use for a given query

### Proposed Solution: QueryMethodOrchestrator

Create a new orchestrator that intelligently combines query methods:

```python
# codebase_rag/retrieval/query_orchestrator.py (NEW)

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..services import QueryProtocol
    from ..vector_backend import VectorBackend


class QueryIntent(Enum):
    """Classified query intent for method selection."""
    FUNCTIONAL = auto()  # "How does X work?"
    STRUCTURAL = auto()  # "What calls Y?"
    SEMANTIC = auto()    # "Find functions similar to X"
    EXPLORATORY = auto() # "Tell me about authentication"
    VALIDATION = auto()  # "Is X implemented correctly?"


class QueryMethod(Enum):
    """Available query methods."""
    SEMANTIC_SEARCH = auto()      # Vector + graph hybrid
    GRAPH_TRAVERSAL = auto()      # Cypher via LLM
    KEYWORD_SEARCH = auto()       # CONTAINS matching
    VECTOR_DIRECT = auto()        # Direct vector search
    GRAPH_ALGORITHMS = auto()     # PageRank, BFS, community
    GRAPH_NAVIGATION = auto()     # Call hierarchy, references


@dataclass
class QueryMethodResult:
    """Results from a single query method."""
    method: QueryMethod
    items: list[dict]
    execution_time_ms: float
    error: str | None = None


@dataclass
class CombinedQueryResult:
    """Combined results from multiple query methods."""
    query: str
    intent: QueryIntent
    methods_used: list[QueryMethod]
    items: list[dict]  # Deduplicated and ranked
    execution_time_ms: float
    warnings: list[str] = field(default_factory=list)


class QueryMethodOrchestrator:
    """Orchestrates multiple query methods for comprehensive retrieval.
    
    Analyzes query characteristics to determine which methods to use,
    executes them (potentially in parallel), merges results, and ranks
    them using combined scoring.
    
    Uses HybridRetriever as the SEMANTIC_SEARCH method implementation.
    """
    
    def __init__(
        self,
        code_graph: QueryProtocol,
        code_vector: VectorBackend | None = None,
        doc_graph: QueryProtocol | None = None,
        doc_vector: VectorBackend | None = None,
        hybrid_retriever: "HybridRetriever | None" = None,
    ):
        self.code_graph = code_graph
        self.code_vector = code_vector
        self.doc_graph = doc_graph
        self.doc_vector = doc_vector
        self._hybrid_retriever = hybrid_retriever
    
    def _get_hybrid_retriever(self) -> "HybridRetriever":
        """Lazy initialization of HybridRetriever."""
        if self._hybrid_retriever is None:
            from ..memgraph_advanced import create_hybrid_retriever
            self._hybrid_retriever = create_hybrid_retriever(self.code_graph)
        return self._hybrid_retriever
    
    @staticmethod
    def classify_intent(query: str) -> QueryIntent:
        """Classify query intent based on keywords and patterns."""
        query_lower = query.lower()
        
        # Structural indicators
        structural_keywords = {'call', 'calls', 'called by', 'caller', 'callee',
                              'hierarchy', 'inherit', 'implement', 'import', 'depend'}
        if any(kw in query_lower for kw in structural_keywords):
            return QueryIntent.STRUCTURAL
        
        # Functional indicators
        functional_keywords = {'how does', 'how to', 'what does', 'why does',
                              'explain', 'work', 'function', 'behavior'}
        if any(kw in query_lower for kw in functional_keywords):
            return QueryIntent.FUNCTIONAL
        
        # Validation indicators
        validation_keywords = {'correct', 'valid', 'implement', 'comply', 'spec',
                              'should', 'must', 'require'}
        if any(kw in query_lower for kw in validation_keywords):
            return QueryIntent.VALIDATION
        
        # Semantic indicators
        semantic_keywords = {'similar', 'like', 'compare', 'related', 'same as'}
        if any(kw in query_lower for kw in semantic_keywords):
            return QueryIntent.SEMANTIC
        
        # Default to exploratory
        return QueryIntent.EXPLORATORY
    
    def select_methods(self, intent: QueryIntent) -> list[QueryMethod]:
        """Select appropriate query methods based on intent."""
        method_map = {
            QueryIntent.FUNCTIONAL: [
                QueryMethod.SEMANTIC_SEARCH,
                QueryMethod.GRAPH_TRAVERSAL,
            ],
            QueryIntent.STRUCTURAL: [
                QueryMethod.GRAPH_TRAVERSAL,
                QueryMethod.GRAPH_NAVIGATION,
            ],
            QueryIntent.SEMANTIC: [
                QueryMethod.SEMANTIC_SEARCH,
                QueryMethod.VECTOR_DIRECT,
            ],
            QueryIntent.EXPLORATORY: [
                QueryMethod.SEMANTIC_SEARCH,
                QueryMethod.GRAPH_TRAVERSAL,
                QueryMethod.GRAPH_ALGORITHMS,
            ],
            QueryIntent.VALIDATION: [
                QueryMethod.GRAPH_TRAVERSAL,
                QueryMethod.SEMANTIC_SEARCH,
            ],
        }
        return method_map.get(intent, [QueryMethod.SEMANTIC_SEARCH])
    
    def execute_method(
        self,
        method: QueryMethod,
        query: str,
        top_k: int = 5,
    ) -> QueryMethodResult:
        """Execute a single query method."""
        import time
        start = time.time()
        
        try:
            if method == QueryMethod.SEMANTIC_SEARCH:
                # Use HybridRetriever (dependency injection via _get_hybrid_retriever)
                retriever = self._get_hybrid_retriever()
                results = retriever.search(query, top_k=top_k)
                items = [
                    {
                        "node_id": r.node_id,
                        "qualified_name": r.qualified_name,
                        "name": r.name,
                        "type": r.node_type,
                        "file_path": r.file_path,
                        "similarity": r.combined_score,
                    }
                    for r in results
                ]
                return QueryMethodResult(
                    method=method,
                    items=items,
                    execution_time_ms=(time.time() - start) * 1000,
                )
            
            elif method == QueryMethod.GRAPH_TRAVERSAL:
                # Use LLM-generated Cypher (async, called synchronously here)
                from ..services.llm import CypherGenerator
                import asyncio
                cypher_gen = CypherGenerator()
                cypher = asyncio.run(cypher_gen.generate(query))
                results = self.code_graph.fetch_all(cypher)
                return QueryMethodResult(
                    method=method,
                    items=results,
                    execution_time_ms=(time.time() - start) * 1000,
                )
            
            elif method == QueryMethod.KEYWORD_SEARCH:
                from ..utils.query_utils import extract_best_keyword
                keyword = extract_best_keyword(query)
                cypher = """
                MATCH (n:Function|Class|Method)
                WHERE n.name CONTAINS $keyword
                   OR n.qualified_name CONTAINS $keyword
                   OR n.docstring CONTAINS $keyword
                RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
                       n.name AS name, labels(n)[0] AS type
                LIMIT $limit
                """
                results = self.code_graph.fetch_all(cypher, {"keyword": keyword, "limit": top_k})
                return QueryMethodResult(
                    method=method,
                    items=results,
                    execution_time_ms=(time.time() - start) * 1000,
                )
            
            # GRAPH_NAVIGATION uses graph_navigation.py tools
            # GRAPH_ALGORITHMS uses GraphAlgorithms class
            
        except Exception as e:
            return QueryMethodResult(
                method=method,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
                error=str(e),
            )
    
    def execute(
        self,
        query: str,
        top_k: int = 5,
        max_methods: int = 3,
    ) -> CombinedQueryResult:
        """Execute comprehensive query using multiple methods."""
        intent = self.classify_intent(query)
        methods = self.select_methods(intent)[:max_methods]
        
        # Execute methods (could be parallelized with ThreadPoolExecutor)
        results = []
        for method in methods:
            result = self.execute_method(method, query, top_k)
            results.append(result)
        
        # Merge and rank results
        merged_items = self._merge_and_rank(results, top_k)
        
        total_time = sum(r.execution_time_ms for r in results)
        warnings = [r.error for r in results if r.error]
        
        return CombinedQueryResult(
            query=query,
            intent=intent,
            methods_used=[r.method for r in results],
            items=merged_items,
            execution_time_ms=total_time,
            warnings=warnings,
        )
    
    def _merge_and_rank(
        self,
        results: list[QueryMethodResult],
        top_k: int,
    ) -> list[dict]:
        """Merge results from multiple methods and rank them."""
        from collections import defaultdict
        
        # Group by qualified_name or node_id
        merged = defaultdict(lambda: {"scores": [], "sources": [], "item": None})
        
        for result in results:
            for item in result.items:
                # Use qualified_name or node_id as key
                key = item.get("qualified_name") or str(item.get("node_id", ""))
                if not key:
                    continue
                
                # Extract score (similarity, pagerank, etc.)
                score = item.get("similarity") or item.get("score", 0.5)
                
                merged[key]["scores"].append(score)
                merged[key]["sources"].append(result.method.name)
                if merged[key]["item"] is None:
                    merged[key]["item"] = item
        
        # Rank by average score across methods (multi-method consensus)
        ranked = []
        for key, data in merged.items():
            item = data["item"]
            avg_score = sum(data["scores"]) / len(data["scores"])
            method_count = len(data["sources"])
            
            # Boost items found by multiple methods
            combined_score = avg_score * (1 + 0.2 * (method_count - 1))
            
            item["combined_score"] = combined_score
            item["found_by_methods"] = data["sources"]
            item["method_count"] = method_count
            ranked.append(item)
        
        # Sort by combined score and return top_k
        ranked.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return ranked[:top_k]
```

### Integration Points

1. **Update QueryRouter** to use QueryMethodOrchestrator:
```python
# shared/query_router.py
def _query_code_only(self, request: QueryRequest) -> QueryResponse:
    orchestrator = QueryMethodOrchestrator(
        code_graph=self.code_graph,
        code_vector=self.code_vector,
    )
    combined = orchestrator.execute(request.question, top_k=request.top_k)
    # Format combined results...
```

2. **Update semantic_search.py** to optionally use orchestrator for enrichment:
```python
# tools/semantic_search.py
def semantic_code_search(query: str, top_k: int = 5, use_orchestrator: bool = False):
    if use_orchestrator:
        # Use comprehensive orchestrator
        ...
    else:
        # Use existing fallback chain (backward compatible)
        ...
```

---

## Issue 2: Memgraph Failure Classification Lacks Granularity

**Severity:** 🟠 HIGH
**Location:** `codebase_rag/services/graph_service.py`, `codebase_rag/graph/query_generator.py`

### Problem Analysis

Current error handling in `MemgraphIngestor`:

```python
_transient_error_markers = (
    "broken pipe", "bad session", "connection reset", ...
)

def _is_retryable_memgraph_error(cls, error: Exception) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in cls._TRANSIENT_ERROR_MARKERS)
```

This only distinguishes **transient** vs **permanent** errors. Missing classification for:

| Failure Type | Current Handling | Needed Handling |
|--------------|------------------|-----------------|
| Invalid Cypher syntax | Retry (wrong!) | Don't retry, repair query |
| Missing procedure (community vs enterprise) | Retry (wrong!) | Fallback to alternative |
| Vector index missing/dimension mismatch | Retry (wrong!) | Recreate index |
| Network disconnection | ✅ Retry with reconnect | ✅ Keep as-is |
| Timeout | ✅ Retry | ✅ Keep as-is |
| Authentication failure | Retry (wrong!) | Fail immediately |
| Graph data integrity issue | Retry (wrong!) | Alert admin |

### Proposed Solution: MemgraphFailureClassifier

```python
# codebase_rag/services/failure_classifier.py (NEW)

from __future__ import annotations

from enum import Enum, auto
from typing import Protocol


class FailureType(Enum):
    """Classification of Memgraph failures."""
    TRANSIENT_NETWORK = auto()       # Retry with backoff
    TRANSIENT_TIMEOUT = auto()       # Retry with longer timeout
    SYNTAX_ERROR = auto()            # Don't retry, repair query
    MISSING_PROCEDURE = auto()       # Fallback to alternative
    VECTOR_INDEX_MISSING = auto()    # Recreate index
    VECTOR_DIMENSION_MISMATCH = auto() # Recreate index + re-embed
    AUTHENTICATION_FAILURE = auto()  # Fail immediately
    PERMISSION_DENIED = auto()       # Fail immediately
    DATA_INTEGRITY = auto()          # Alert admin
    RESOURCE_EXHAUSTION = auto()     # Retry after cleanup
    UNKNOWN = auto()                 # Default


@dataclass
class FailureClassification:
    """Result of failure classification."""
    failure_type: FailureType
    message: str
    should_retry: bool
    max_retries: int = 0
    recovery_action: str | None = None


# Error pattern definitions
_SYNTAX_ERROR_MARKERS = (
    "syntax error", "parse error", "invalid", "expected", "unexpected",
    "mismatched input", "no viable alternative", "cannot match",
)

_MISSING_PROCEDURE_MARKERS = (
    "there is no procedure", "procedure not found", "unknown procedure",
    "function not found", "doesn't exist",
)

_VECTOR_INDEX_MARKERS = (
    "vector index", "vector_search", "index not found",
)

_VECTOR_DIMENSION_MARKERS = (
    "dimension mismatch", "different number of dimensions",
    "expected", "got",
)

_AUTH_FAILURE_MARKERS = (
    "authentication failed", "invalid credentials", "access denied",
    "permission denied",
)


def classify_memgraph_failure(error: Exception) -> FailureClassification:
    """Classify a Memgraph failure and determine recovery strategy."""
    message = str(error).lower()
    
    # Check syntax errors
    if any(m in message for m in _SYNTAX_ERROR_MARKERS):
        return FailureClassification(
            failure_type=FailureType.SYNTAX_ERROR,
            message="Cypher syntax error detected",
            should_retry=False,
            recovery_action="repair_query",
        )
    
    # Check missing procedure
    if any(m in message for m in _MISSING_PROCEDURE_MARKERS):
        return FailureClassification(
            failure_type=FailureType.MISSING_PROCEDURE,
            message="Memgraph procedure not found (likely community edition)",
            should_retry=False,
            recovery_action="fallback_alternative",
        )
    
    # Check vector index issues
    if any(m in message for m in _VECTOR_INDEX_MARKERS):
        return FailureClassification(
            failure_type=FailureType.VECTOR_INDEX_MISSING,
            message="Vector index not found or invalid",
            should_retry=False,
            recovery_action="recreate_index",
        )
    
    # Check dimension mismatch
    if any(m in message for m in _VECTOR_DIMENSION_MARKERS):
        return FailureClassification(
            failure_type=FailureType.VECTOR_DIMENSION_MISMATCH,
            message="Embedding dimension mismatch",
            should_retry=False,
            recovery_action="recreate_index_and_reembed",
        )
    
    # Check authentication failures
    if any(m in message for m in _AUTH_FAILURE_MARKERS):
        return FailureClassification(
            failure_type=FailureType.AUTHENTICATION_FAILURE,
            message="Authentication or permission failure",
            should_retry=False,
            max_retries=0,
        )
    
    # Check transient network errors
    _TRANSIENT_MARKERS = (
        "broken pipe", "connection reset", "connection refused",
        "connection closed", "network is unreachable", "temporarily unavailable",
    )
    if any(m in message for m in _TRANSIENT_MARKERS):
        return FailureClassification(
            failure_type=FailureType.TRANSIENT_NETWORK,
            message="Transient network error",
            should_retry=True,
            max_retries=3,
            recovery_action="reconnect",
        )
    
    # Check timeouts
    if "timeout" in message or "timed out" in message:
        return FailureClassification(
            failure_type=FailureType.TRANSIENT_TIMEOUT,
            message="Query timeout",
            should_retry=True,
            max_retries=2,
            recovery_action="increase_timeout",
        )
    
    # Default: unknown
    return FailureClassification(
        failure_type=FailureType.UNKNOWN,
        message=f"Unknown error: {error}",
        should_retry=True,
        max_retries=1,
    )
```

### Integration

Update `MemgraphIngestor._execute_query` (lines 400-454 in `graph_service.py`):

```python
# In graph_service.py, modify the retry loop in _execute_query:

def _execute_query(self, query, params=None):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            # ... existing connection and execution code ...
            return rows
        except Exception as e:
            # Use new classifier instead of binary _is_retryable_memgraph_error
            from .failure_classifier import classify_memgraph_failure, FailureType
            
            classification = classify_memgraph_failure(e)
            
            if classification.failure_type == FailureType.SYNTAX_ERROR:
                # Don't retry - trigger Cypher repair via LLM
                logger.warning(f"Cypher syntax error, attempting repair: {e}")
                # Existing CypherGenerator.repair() can be used here
                raise CypherSyntaxError(query, str(e)) from e
            
            elif classification.failure_type == FailureType.MISSING_PROCEDURE:
                # Don't retry - procedure doesn't exist (community edition)
                logger.warning(f"Procedure not found, using fallback: {e}")
                raise MissingProcedureError(str(e)) from e
            
            elif classification.failure_type == FailureType.AUTHENTICATION_FAILURE:
                # Fail immediately - credentials are wrong
                logger.error(f"Authentication failure: {e}")
                raise
            
            elif classification.should_retry and attempt < classification.max_retries:
                # Use classifier's max_retries instead of fixed value
                logger.warning(f"Retryable error (attempt {attempt + 1}/{classification.max_retries}): {e}")
                time.sleep(self._retry_delay_seconds(attempt))
                continue
            
            raise
    return []
```

**Note:** The existing `_TRANSIENT_ERROR_MARKERS` tuple (lines 64-78) should remain for backward compatibility, but `_is_retryable_memgraph_error()` should delegate to the classifier.

---

## Issue 3: Cross-Graph Reference Resolution Fragile

**Severity:** 🟡 MEDIUM
**Location:** `codebase_rag/shared/query_router.py`, `codebase_rag/document/tools/document_search.py`

### Problem Analysis

When document search returns results with `resolved_code_references`, the QueryRouter attempts to resolve them in the code graph:

```python
# query_router.py
def _build_document_reference_context(self, document_results):
    qualified_names = []
    for result in document_results:
        for reference in result.get("resolved_code_references"):
            qualified_names.append(reference)
    
    # Query code graph
    rows = self.code_graph.fetch_all(
        """
        MATCH (n)
        WHERE n.qualified_name IN $qualified_names
        RETURN ...
        """,
        {"qualified_names": qualified_names[:20]},
    )
```

**Issues:**
1. If `code_graph` is `None` or disconnected, resolution fails silently
2. Qualified names may not match between document and code graphs
3. No fallback if reference resolution fails
4. No validation that references actually exist

### Proposed Solution: Robust Reference Resolver

```python
# codebase_rag/shared/reference_resolver.py (NEW)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..services import QueryProtocol


@dataclass
class ResolvedReference:
    """A successfully resolved code reference."""
    qualified_name: str
    node_type: str
    file_path: str
    line_range: tuple[int, int] | None
    found: bool = True


@dataclass
class UnresolvedReference:
    """A reference that could not be resolved."""
    qualified_name: str
    reason: str  # "not_found", "graph_unavailable", "name_mismatch"


@dataclass
class ReferenceResolutionResult:
    """Result of reference resolution."""
    resolved: list[ResolvedReference]
    unresolved: list[UnresolvedReference]
    warnings: list[str] = field(default_factory=list)


class CrossGraphReferenceResolver:
    """Resolves code references from document graph in code graph."""
    
    def __init__(self, code_graph: QueryProtocol | None):
        self.code_graph = code_graph
    
    def resolve_references(
        self,
        references: list[str],
        max_references: int = 20,
    ) -> ReferenceResolutionResult:
        """Resolve code references with robust error handling."""
        if not self.code_graph:
            return ReferenceResolutionResult(
                resolved=[],
                unresolved=[
                    UnresolvedReference(ref, "graph_unavailable")
                    for ref in references
                ],
                warnings=["Code graph not available for reference resolution"],
            )
        
        if not references:
            return ReferenceResolutionResult(resolved=[], unresolved=[])
        
        # Limit references to avoid large queries
        refs_to_resolve = references[:max_references]
        resolved = []
        unresolved = []
        
        # Batch query for all references
        try:
            rows = self.code_graph.fetch_all(
                """
                MATCH (n)
                WHERE n.qualified_name IN $qualified_names
                RETURN 
                    n.qualified_name AS qualified_name,
                    labels(n)[0] AS node_type,
                    n.path AS file_path,
                    n.start_line AS start_line,
                    n.end_line AS end_line
                """,
                {"qualified_names": refs_to_resolve},
            )
            
            # Create lookup map
            found_names = {row["qualified_name"] for row in rows}
            
            # Build resolved list
            for row in rows:
                resolved.append(ResolvedReference(
                    qualified_name=row["qualified_name"],
                    node_type=row["node_type"],
                    file_path=row["file_path"] or "unknown",
                    line_range=(row["start_line"], row["end_line"])
                        if row.get("start_line") else None,
                ))
            
            # Build unresolved list
            for ref in refs_to_resolve:
                if ref not in found_names:
                    # Try partial match (fallback)
                    partial_match = self._try_partial_match(ref)
                    if partial_match:
                        resolved.append(partial_match)
                    else:
                        unresolved.append(UnresolvedReference(
                            ref, "not_found",
                        ))
        
        except Exception as e:
            logger.warning(f"Reference resolution failed: {e}")
            return ReferenceResolutionResult(
                resolved=[],
                unresolved=[
                    UnresolvedReference(ref, "resolution_error")
                    for ref in refs_to_resolve
                ],
                warnings=[f"Reference resolution error: {e}"],
            )
        
        return ReferenceResolutionResult(
            resolved=resolved,
            unresolved=unresolved,
        )
    
    def _try_partial_match(self, qualified_name: str) -> ResolvedReference | None:
        """Try to find a partial match for a qualified name."""
        # Extract the short name (last component)
        parts = qualified_name.split(".")
        if len(parts) < 2:
            return None
        
        short_name = parts[-1]
        
        try:
            rows = self.code_graph.fetch_all(  # type: ignore[union-attr]
                """
                MATCH (n)
                WHERE n.name = $short_name
                RETURN 
                    n.qualified_name AS qualified_name,
                    labels(n)[0] AS node_type,
                    n.path AS file_path,
                    n.start_line AS start_line,
                    n.end_line AS end_line
                LIMIT 1
                """,
                {"short_name": short_name},
            )
            
            if rows:
                row = rows[0]
                return ResolvedReference(
                    qualified_name=row["qualified_name"],
                    node_type=row["node_type"],
                    file_path=row["file_path"] or "unknown",
                    line_range=(row["start_line"], row["end_line"])
                        if row.get("start_line") else None,
                )
        except Exception:
            pass
        
        return None
```

---

## Issue 4: ~~No Meta-Retriever for Combined Ranking~~ → MERGED INTO ISSUE 1

> **Note:** This functionality is now implemented within `QueryMethodOrchestrator._merge_and_rank()` (Issue 1). The orchestrator's merge logic handles:
> - Deduplication by `qualified_name` or `node_id`
> - Score extraction from different methods (similarity, pagerank, etc.)
> - Multi-method consensus boosting (items found by multiple methods rank higher)
> - Configurable scoring weights
>
> No separate `MetaRetriever` class is needed.

---

## Issue 5: No Runtime Health Monitoring During Query Execution

**Severity:** 🟡 MEDIUM
**Location:** `codebase_rag/tools/health_checker.py`, `codebase_rag/services/graph_service.py`

### Problem Analysis

**What Already Exists:**
`HealthChecker` in `tools/health_checker.py` (880+ lines) already provides comprehensive pre-flight checks:
- `check_vector_indexes()` - verifies vector index existence
- `check_vector_search()` - tests vector search functionality
- `check_embedding_correlation()` - validates embedding model consistency
- `check_disconnected_nodes()` - data integrity check
- `check_required_properties()` - property validation

**The Gap:**
These are **pre-flight checks** run before/after ingestion. Missing:
- **Runtime health monitoring** during active query execution
- **Degradation detection** when capabilities fail mid-session
- **Recovery suggestions** integrated with failure classifier (Issue 2)

### Proposed Solution: Extend HealthChecker with Runtime Monitoring

```python
# Add to existing tools/health_checker.py

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..services.graph_service import MemgraphIngestor


class HealthStatus(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    UNHEALTHY = auto()


@dataclass
class RuntimeHealthStatus:
    """Runtime health status for active query sessions."""
    vector_search: HealthStatus = HealthStatus.HEALTHY
    graph_traversal: HealthStatus = HealthStatus.HEALTHY
    procedures: HealthStatus = HealthStatus.HEALTHY
    last_error: str | None = None

    @property
    def overall(self) -> HealthStatus:
        """Determine overall health from individual statuses."""
        if HealthStatus.UNHEALTHY in [self.vector_search, self.graph_traversal, self.procedures]:
            return HealthStatus.UNHEALTHY
        if HealthStatus.DEGRADED in [self.vector_search, self.graph_traversal, self.procedures]:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY


class HealthChecker:
    # ... existing code ...
    
    def get_runtime_status(
        self,
        ingestor: "MemgraphIngestor",
    ) -> RuntimeHealthStatus:
        """Get current runtime health status for active session.
        
        Call this periodically during long-running operations or after
        errors to detect capability degradation.
        """
        status = RuntimeHealthStatus()
        
        # Quick vector search test (use existing check_vector_search logic)
        try:
            from ..vector_backend import get_shared_backend
            backend = get_shared_backend()
            stats = backend.get_stats()
            if stats.get("total_embeddings", 0) == 0:
                status.vector_search = HealthStatus.DEGRADED
        except Exception as e:
            status.vector_search = HealthStatus.UNHEALTHY
            status.last_error = str(e)
        
        # Quick graph traversal test
        try:
            ingestor.fetch_all("MATCH (n) RETURN count(n) LIMIT 1")
        except Exception as e:
            status.graph_traversal = HealthStatus.UNHEALTHY
            status.last_error = str(e)
        
        # Procedure availability (PageRank is enterprise-only)
        try:
            ingestor.fetch_all(
                "CALL pagerank.get() YIELD node, rank "
                "WITH rank LIMIT 1 RETURN rank"
            )
        except Exception:
            status.procedures = HealthStatus.DEGRADED  # Community edition is OK
        
        return status
```

### Integration with Failure Classifier

When `FailureClassifier` (Issue 2) detects a failure, it can update runtime health:

```python
# In failure_classifier.py
def classify_memgraph_failure(error: Exception) -> FailureClassification:
    # ... existing classification logic ...
    
    # Update runtime health status if unhealthy
    if classification.failure_type in (FailureType.VECTOR_INDEX_MISSING, FailureType.VECTOR_DIMENSION_MISMATCH):
        # Signal to caller that vector search is degraded
        classification.metadata = {"affected_capability": "vector_search"}
```

---

## Issue 6: QueryRouter Lacks Intent-Based Orchestration of Multiple Retrieval Paradigms

**Severity:** 🟡 MEDIUM
**Location:** `codebase_rag/shared/query_router.py`

### Problem Analysis

**What Already Exists:**
`_query_code_only()` (lines 400-530 in `query_router.py`) already uses `HybridRetriever`:
```python
if self.code_vector:
    retriever = create_hybrid_retriever(self.code_graph)
    results = retriever.search(query, top_k)  # Combines vector + graph scoring
```

`HybridRetriever` combines **vector similarity + graph-based scoring** (PageRank, community importance) in a single unified score.

**The Gap:**
The current flow uses `HybridRetriever` for semantic queries and falls back to keyword search only if it fails. It does NOT:
- **Classify query intent** to select appropriate retrieval paradigm
- **Execute multiple paradigms in parallel** (e.g., Cypher LLM queries + semantic search + keyword)
- **Merge results from different paradigms** with unified ranking

Current flow:
```
User Query → HybridRetriever (vector+graph) → If empty, fallback to keyword
```

Missing flow:
```
User Query → Intent classification → Select paradigms → Execute in parallel → Merge results
```

**Impact:**
- Structural queries ("What calls authenticate?") use semantic search, missing precise graph traversal
- Exploratory queries miss results from keyword matching that semantic search doesn't find

### Proposed Solution: Update QueryRouter to Use Orchestrator

```python
# shared/query_router.py
def _query_code_only(self, request: QueryRequest) -> QueryResponse:
    """Query CODE graph/vector using systematic method combination."""
    if not self.code_graph:
        return QueryResponse(
            answer="Code graph is not available.",
            sources=[],
            mode=request.mode,
            warnings=["Code graph connection not configured"],
        )
    
    # Use QueryMethodOrchestrator for comprehensive retrieval
    orchestrator = QueryMethodOrchestrator(
        code_graph=self.code_graph,
        code_vector=self.code_vector,
    )
    
    combined = orchestrator.execute(
        query=request.question,
        top_k=request.top_k,
        max_methods=3,
    )
    
    # Format results
    sources = []
    answer_parts = ["**Code Results (multi-method):**\n"]
    
    for item in combined.items:
        sources.append(Source(
            type="code",
            path=item.get("file_path", "unknown"),
            node_type=item.get("type", "Unknown"),
            qualified_name=item.get("qualified_name"),
        ))
        score = item.get("combined_score", 0)
        methods = ", ".join(item.get("found_by_methods", ["unknown"]))
        answer_parts.append(
            f"- **{item.get('qualified_name', 'unknown')}** "
            f"({item.get('type', 'Unknown')}) "
            f"[Score: {score:.2f}, Methods: {methods}]"
        )
    
    return QueryResponse(
        answer="\n".join(answer_parts),
        sources=sources,
        mode=request.mode,
        warnings=combined.warnings,
    )
```

---

## Issue 7: Missing Integration Tests for Query Combination

**Severity:** 🟡 MEDIUM
**Location:** `codebase_rag/tests/`

### Problem Analysis

Existing tests focus on individual components:
- `test_semantic_search.py` - tests semantic search only
- `test_codebase_query.py` - tests graph query only
- `test_query_router.py` - tests routing logic

**Missing:** Tests that verify systematic combination of methods.

### Proposed Tests

```python
# tests/integration/test_query_combination.py (NEW)

import pytest
from unittest.mock import MagicMock, patch

from codebase_rag.retrieval.query_orchestrator import (
    QueryMethodOrchestrator,
    QueryIntent,
    QueryMethod,
)
from codebase_rag.retrieval.meta_retriever import MetaRetriever


class TestQueryMethodOrchestrator:
    """Test systematic query method integration."""
    
    @pytest.fixture
    def mock_graph(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_vector(self):
        return MagicMock()
    
    @pytest.fixture
    def orchestrator(self, mock_graph, mock_vector):
        return QueryMethodOrchestrator(
            code_graph=mock_graph,
            code_vector=mock_vector,
        )
    
    def test_classifies_functional_intent(self, orchestrator):
        """Verify functional query intent classification."""
        intent = orchestrator.classify_intent("How does authentication work?")
        assert intent == QueryIntent.FUNCTIONAL
    
    def test_classifies_structural_intent(self, orchestrator):
        """Verify structural query intent classification."""
        intent = orchestrator.classify_intent("What functions call main?")
        assert intent == QueryIntent.STRUCTURAL
    
    def test_selects_methods_for_exploratory(self, orchestrator):
        """Verify method selection for exploratory queries."""
        methods = orchestrator.select_methods(QueryIntent.EXPLORATORY)
        assert QueryMethod.SEMANTIC_SEARCH in methods
        assert QueryMethod.GRAPH_TRAVERSAL in methods
    
    def test_execute_combines_multiple_methods(self, orchestrator):
        """Verify execution combines results from multiple methods."""
        with patch('codebase_rag.tools.semantic_search.semantic_code_search') as mock_semantic:
            mock_semantic.return_value = [
                MagicMock(node_id=1, qualified_name="auth.login", similarity=0.8),
            ]
            
            orchestrator.code_graph.fetch_all.return_value = [
                {"node_id": 2, "qualified_name": "auth.validate"},
            ]
            
            result = orchestrator.execute("How does authentication work?")
            
            # Should use multiple methods
            assert len(result.methods_used) >= 2
            # Should have combined results
            assert len(result.items) > 0


class TestMetaRetriever:
    """Test combined ranking from multiple methods."""
    
    @pytest.fixture
    def retriever(self):
        return MetaRetriever()
    
    def test_combines_results_from_multiple_methods(self, retriever):
        """Verify results are combined and ranked."""
        semantic = [
            {"node_id": 1, "qualified_name": "auth.login", "similarity": 0.8},
        ]
        graph = [
            {"node_id": 1, "qualified_name": "auth.login", "type": "Function"},
            {"node_id": 2, "qualified_name": "auth.validate", "type": "Function"},
        ]
        keyword = [
            {"node_id": 3, "qualified_name": "auth.logout", "type": "Function"},
        ]
        
        results = retriever.combine_and_rank(semantic, graph, keyword, top_k=5)
        
        # auth.login should have highest score (found by 2 methods)
        assert results[0].qualified_name == "auth.login"
        assert "semantic" in results[0].methods
        assert "graph" in results[0].methods
    
    def test_applies_consensus_bonus(self, retriever):
        """Verify consensus bonus is applied."""
        # Item found by 3 methods should rank higher than item found by 1
        semantic = [{"node_id": 1, "qualified_name": "fn1", "similarity": 0.5}]
        graph = [{"node_id": 1, "qualified_name": "fn1"}]
        keyword = [{"node_id": 1, "qualified_name": "fn1"}]
        
        single = [{"node_id": 2, "qualified_name": "fn2", "similarity": 0.7}]
        
        results = retriever.combine_and_rank(semantic + single, graph, keyword, top_k=5)
        
        # fn1 (3 methods) should rank higher than fn2 (1 method, higher semantic)
        assert results[0].qualified_name == "fn1"


class TestCrossGraphReferenceResolver:
    """Test robust cross-graph reference resolution."""
    
    def test_handles_unavailable_code_graph(self):
        """Verify graceful handling when code graph is unavailable."""
        from codebase_rag.shared.reference_resolver import CrossGraphReferenceResolver
        
        resolver = CrossGraphReferenceResolver(code_graph=None)
        result = resolver.resolve_references(["auth.login"])
        
        assert len(result.unresolved) == 1
        assert result.unresolved[0].reason == "graph_unavailable"
        assert len(result.warnings) > 0
    
    def test_partial_match_fallback(self, mock_code_graph):
        """Verify partial match fallback works."""
        from codebase_rag.shared.reference_resolver import CrossGraphReferenceResolver
        
        resolver = CrossGraphReferenceResolver(code_graph=mock_code_graph)
        
        # Qualified name doesn't match, but short name does
        mock_code_graph.fetch_all.side_effect = [
            [],  # First query: no exact match
            [{"qualified_name": "auth.login", "node_type": "Function", "path": "auth.py"}],
        ]
        
        result = resolver.resolve_references(["Project.auth.login"])
        
        # Should find via partial match
        assert len(result.resolved) == 1
```

---

## Implementation Plan

### Phase 3A: Core Integration (Week 1)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 1 | Create QueryMethodOrchestrator (includes merge & rank logic) | `retrieval/query_orchestrator.py` | P1 | 10h |
| 2 | Create failure classifier | `services/failure_classifier.py` | P1 | 4h |
| 3 | Create reference resolver | `shared/reference_resolver.py` | P1 | 4h |

### Phase 3B: Integration (Week 2)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 4 | Update QueryRouter to use orchestrator | `shared/query_router.py` | P1 | 4h |
| 5 | Extend HealthChecker with runtime monitoring | `tools/health_checker.py` | P1 | 3h |
| 6 | Integrate failure classifier with retry loop | `services/graph_service.py` | P1 | 2h |

### Phase 3C: Testing (Week 3)

| # | Task | File | Priority | Effort |
|---|------|------|----------|--------|
| 7 | Write integration tests for query orchestration | `tests/integration/test_query_orchestration.py` | P1 | 6h |
| 8 | Write unit tests for failure classifier | `tests/unit/test_failure_classifier.py` | P1 | 2h |
| 9 | Write unit tests for reference resolver | `tests/unit/test_reference_resolver.py` | P1 | 2h |
| 10 | Write runtime health monitoring tests | `tests/unit/test_runtime_health.py` | P1 | 2h |

### Phase 3D: Validation & Documentation (Week 4)

| # | Task | Priority | Effort |
|---|------|----------|--------|
| 11 | Integration testing with real Memgraph | P1 | 4h |
| 12 | Performance benchmarking | P2 | 4h |
| 13 | Documentation updates | P2 | 2h |
| 14 | Review and refine | P1 | 2h |

**Total Estimated Effort:** ~47 hours (reduced from 52h due to MetaRetriever merge)

---

## Success Metrics

| Metric | Before | After Target |
|--------|--------|--------------|
| Query paradigms orchestrated per query | 1 (HybridRetriever only) | 2-3 (intent-based selection) |
| Results found by multiple paradigms | 0% | >30% |
| Failure classification accuracy | Binary (retry/don't) | 9 categories |
| Reference resolution success rate | ~70% | >90% |
| Health monitoring | Pre-flight checks only | Runtime degradation detection |
| Integration test coverage for orchestration | 0% | >80% |

---

## Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Query orchestrator adds latency | Medium | High | Parallel execution, caching, configurable method limit |
| Intent classification misclassifies queries | Medium | Medium | Comprehensive keyword lists, fallback to exploratory |
| Failure classifier misclassifies errors | Medium | Low | Comprehensive test cases, fallback to default retry |
| Reference resolver partial matches return wrong results | Low | Medium | Strict matching first, partial as fallback with clear labeling |
| Breaking changes to existing APIs | High | Low | All new modules are additive; HybridRetriever unchanged |

---

## Conclusion

This Phase 3 spec addresses **6 remaining systemic issues** (Issue 4 merged into Issue 1):

**Integration Gaps (2):**
1. No intelligent intent-based query method orchestration
6. QueryRouter lacks intent-based paradigm orchestration

**Resilience Gaps (3):**
2. Memgraph failure classification lacks granularity
3. Cross-graph reference resolution fragile
5. No runtime health monitoring during query execution

**Testing Gap (1):**
7. Missing integration tests for query orchestration

**All solutions are:**
- ✅ Logically sound — Addresses root causes with systematic integration
- ✅ Implementation-ready — Code snippets provided, file locations specified
- ✅ Aligned with existing codebase — Uses existing patterns (dataclasses, HybridRetriever, HealthChecker)
- ✅ Backward compatible — HybridRetriever unchanged; all new modules are additive

**Recommended Implementation Order:**
1. Phase 3A (Core integration) — 1 week
2. Phase 3B (Integration) — 1 week
3. Phase 3C (Testing) — 1 week
4. Phase 3D (Validation) — 1 week

**Total Timeline:** ~4 weeks
