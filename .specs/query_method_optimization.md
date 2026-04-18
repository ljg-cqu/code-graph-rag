# Query Method Optimization Design Specifications
## Version 1.1
## Status: Implementation-Ready

### Executive Summary
This specification addresses critical gaps in the current multi-paradigm query system to ensure optimal LLM generation reliability. The existing system provides solid foundations — hybrid retrieval (`HybridRetriever`), Cypher safety validation (`_validate_cypher_read_only`), failure classification (`failure_classifier.py`), connection health checks (`_check_connection_health`), sufficiency gatekeeping (`sufficiency_gatekeeper.py`), and bidirectional code↔document validation (`CodeVsDocValidator`, `DocVsCodeValidator`). However, these mechanisms operate independently without coordination, leaving systemic gaps: uncoordinated method execution, a sync/async mismatch in graph traversal, no template-based Cypher fallback, no graph-to-disk integrity verification, and insufficient integration between the orchestrator and the existing failure/recovery infrastructure.

This spec proposes **incremental enhancements to existing modules** rather than new standalone components, ensuring every change builds on proven code.

### Problem Statement
Current LLM-based query systems can fail silently or return incomplete results due to:
1. **Cypher Generation Failures**: LLM-generated Cypher may have syntax errors or unsupported features; the existing `CypherGenerator.repair()` provides recovery but the orchestrator doesn't invoke it on execution failures. Observed: LLM sometimes generates concatenated multi-query output where a broken second fragment leaks past `_clean_cypher_response()` markdown extraction
2. **Hardcoded Cypher Parameterization Bugs**: `get_call_hierarchy()` in `graph_navigation.py` uses `[:CALLS*1..$depth]` — Memgraph does NOT support parameterized bounds in variable-length path expressions (`$depth` is misclassified as "Property map matching" and rejected). This causes **100% failure** of call hierarchy queries
3. **Network/Backend Issues**: `MemgraphIngestor` has per-query retry via `failure_classifier.py`, but the orchestrator's `_execute_graph_traversal()` bypasses this by using `asyncio.run()` — a sync/async mismatch that crashes inside async contexts (MCP server, pydantic-ai agent loop)
4. **Embedding Cold-Start Cascade**: When OpenAI auth fails (401), the local embedding fallback loads `BAAI/bge-large-en-v1.5` from scratch each time (observed: 3+ redundant loads in one session). No caching or warm-up mechanism, causing multi-second latency spikes on every vector search attempt
5. **Query Method Imbalance**: The orchestrator executes methods sequentially with fixed ordering and no early-termination intelligence; results from different methods aren't cross-validated
6. **Graph Data Integrity Gap**: No systematic verification that graph-stored metadata (file paths, line numbers, qualified names) matches actual source files on disk — distinct from the existing code↔document semantic validation
7. **Premature Termination**: The `sufficiency_gatekeeper` in `main.py` enforces minimum rounds, but the orchestrator's `execute()` has no analogous mechanism — it returns after one pass regardless of result quality
8. **Uncoordinated Health Checks**: `MemgraphIngestor._check_connection_health()`, `HybridRetriever._validate()`, and `GraphAlgorithms.health_check()` all exist but are not consulted before the orchestrator selects methods

### Core Design Principles
1. **Enhance, Not Replace**: Build on existing `QueryMethodOrchestrator`, `CypherGenerator`, `failure_classifier.py`, and `sufficiency_gatekeeper.py` — no standalone replacement modules
2. **Defense in Depth**: Multiple validation layers at different system levels, coordinated rather than siloed
3. **Graceful Degradation**: System should continue functioning with reduced capabilities rather than failing completely
4. **Async-Native Execution**: All orchestrator methods must be async-compatible to work inside the pydantic-ai agent loop and MCP server
5. **Transparent Failure Reporting**: Clear error messages that guide recovery actions, leveraging existing `FailureClassification` infrastructure

### Detailed Specifications

## 1. Enhanced Query Method Orchestration

### 1.1 Async-Native Execution Pipeline
The current `QueryMethodOrchestrator.execute()` is synchronous. `_execute_graph_traversal()` calls `asyncio.run(cypher_gen.generate(query))`, which **crashes inside existing async contexts** (the pydantic-ai agent loop in `main.py`, MCP server). All orchestrator methods must be converted to async.

**Changes to `codebase_rag/retrieval/query_orchestrator.py`:**

```
STAGE 1: Intent Classification & Method Selection
├── classify_intent() — unchanged (keyword matching)
├── select_methods() — enhanced with primary/secondary designation
└── Health pre-check: consult existing health mechanisms before execution

STAGE 2: Adaptive Sequential Execution (NOT parallel)
├── Execute primary methods first with early-termination check
├── On failure: retry via existing failure_classifier.py recovery actions
├── On primary success: optionally execute secondary methods for validation
└── If primary fails after retries: fall through to secondary methods

STAGE 3: Result Quality Assessment
├── Minimum evidence threshold: require results from ≥2 methods
├── Cross-method consistency check: flag contradictions between methods
├── Confidence scoring: weighted average of method scores × method diversity
└── Sufficient-results gate: skip remaining methods if threshold met

STAGE 4: Graph Data Integrity Spot-Check (lightweight)
├── For top results: verify file exists on disk
├── For top results: verify start_line/end_line within file bounds
├── Log discrepancies as warnings; don't block responses
└── Full integrity verification available on demand (§4)
```

**Why adaptive sequential, not parallel:**
- `_execute_graph_traversal()` calls an LLM — expensive, error-prone, should only run when simpler methods don't suffice
- `_execute_graph_algorithms()` runs MAGE procedures — slow, may conflict
- Cheap methods (keyword_search, semantic_search) can run first to narrow scope
- Parallel execution of all methods wastes LLM budget when early methods succeed
- Selective cheap parallelism (semantic_search + keyword_search simultaneously) is a future optimization, not a baseline requirement

### 1.2 Intent Classification — Heuristic Enhancement
The existing `classify_intent()` uses keyword matching via `_INTENT_KEYWORD_MAP`. This is fast, deterministic, and doesn't require LLM calls. The enhancement adds a **confidence score derived from keyword match quality**, not from an LLM classifier (which would add latency and its own failure modes).

**Confidence computation:**
```python
@staticmethod
def classify_intent_with_confidence(query: str) -> tuple[QueryIntent, float]:
    """Classify intent with heuristic confidence score.
    
    Confidence is based on:
    - Number of matching keywords (more matches → higher confidence)
    - Keyword specificity (longer/more-specific phrases → higher confidence)
    - Whether only one intent matched (single intent → higher confidence)
    
    Returns (intent, confidence) where confidence is 0.0–1.0.
    """
    query_lower = query.lower()
    scores: dict[QueryIntent, float] = {}
    
    for intent, keywords in _INTENT_KEYWORD_MAP.items():
        match_count = sum(1 for kw in keywords if kw in query_lower)
        if match_count > 0:
            # Score: match ratio + bonus for specificity
            avg_keyword_len = sum(len(kw) for kw in keywords if kw in query_lower) / match_count
            specificity_bonus = min(avg_keyword_len / 20.0, 0.3)  # Longer keywords are more specific
            scores[intent] = min(match_count / 3.0 + specificity_bonus, 1.0)
    
    if not scores:
        return QueryIntent.EXPLORATORY, 0.3  # Low confidence for default
    
    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]
    
    # If multiple intents scored similarly, reduce confidence
    if len(scores) > 1:
        second_best = max(v for k, v in scores.items() if k != best_intent)
        if second_best > best_score * 0.5:
            best_score *= 0.7  # Penalize ambiguous classification
    
    return best_intent, best_score
```

- **Fallback behavior**: When confidence < 0.5, designate all methods as primary (exploratory-style full coverage)
- **Compound intents**: Not implemented in v1.1 — a compound intent like "functional + structural" is handled by the exploratory fallback, which executes all methods. A dedicated compound-intent mechanism can be added in a future iteration when confidence scoring is validated empirically.

### 1.3 Method Selection Strategy — Primary/Secondary Designation
The existing `_INTENT_METHOD_MAP` provides a flat list of methods per intent. The enhancement adds **primary/secondary designation** so the adaptive sequencer knows which methods to execute first and which to use for validation/fallback.

| Intent Type | Primary Methods | Secondary Methods | Health Pre-check (via BackendHealthCoordinator) |
|-------------|----------------|-------------------|------------------|
| FUNCTIONAL | Semantic Search, Graph Traversal | Keyword Search, Vector Direct | `check_vector_health()` + `check_graph_health()` |
| STRUCTURAL | Graph Traversal, Graph Navigation | Keyword Search, Semantic Search | `check_graph_health()` |
| SEMANTIC | Semantic Search, Vector Direct | Graph Traversal, Keyword Search | `check_vector_health()` + `check_graph_health()` |
| EXPLORATORY | All available methods (adaptive order) | None | All available checks |
| VALIDATION | Graph Traversal, Keyword Search | Semantic Search, Graph Algorithms | `check_graph_health()` + `check_algorithm_health()` |

> **Note on private-API references**: The original spec referenced `_check_connection_health()` and `HybridRetriever._validate()` as health pre-checks. These are private methods that are not part of the `QueryProtocol` interface (see §3.1 design notes for details). The `BackendHealthCoordinator` resolves this by using `QueryProtocol.fetch_all("RETURN 1")` for graph health checks, which works for all `QueryProtocol` implementations including `PooledMemgraphProxy`.

**Implementation**: Extend `_INTENT_METHOD_MAP` to return `(primary, secondary)` tuples:
```python
_INTENT_METHOD_MAP: dict[QueryIntent, tuple[list[QueryMethod], list[QueryMethod]]] = {
    QueryIntent.FUNCTIONAL: (
        [QueryMethod.SEMANTIC_SEARCH, QueryMethod.GRAPH_TRAVERSAL],
        [QueryMethod.KEYWORD_SEARCH, QueryMethod.VECTOR_DIRECT],
    ),
    QueryIntent.STRUCTURAL: (
        [QueryMethod.GRAPH_TRAVERSAL, QueryMethod.GRAPH_NAVIGATION],
        [QueryMethod.KEYWORD_SEARCH, QueryMethod.SEMANTIC_SEARCH],
    ),
    # ... etc.
}
```

## 2. Enhanced Cypher Generation with Template Fallback

### 2.1 Enhancement Strategy — Add Fallback Layer, Not Replacement
The existing `CypherGenerator` in `codebase_rag/services/llm.py` already provides:
- `_clean_cypher_response()`: strips markdown, comments, unsupported syntax
- `_validate_cypher_read_only()`: blocks all write operations via `CYPHER_DANGEROUS_KEYWORDS`
- `repair()`: LLM-based Cypher repair with specialized prompt

Several orchestrator methods already use **parameterized Cypher templates** directly:
- `_execute_keyword_search()`: hardcoded parameterized Cypher
- `_execute_graph_navigation()`: uses `CYPHER_FIND_CALLERS`, `CYPHER_FIND_IMPORTERS` constants
- `_execute_graph_algorithms()`: hardcoded BFS and similarity queries

**The enhancement adds a template catalog as a fallback layer**, not a replacement:

```
Cypher Generation Flow (enhanced):

1. Try existing CypherGenerator.generate() (LLM-based)
   ├── On success: validate via existing _validate_cypher_read_only()
   ├── On failure: classify error via existing failure_classifier.py
   │   ├── Syntax error → Step 2 (template fallback)
   │   ├── Transient network → retry with existing CypherGenerator.repair()
   │   └── Other → Step 2 (template fallback)
   └── On timeout → Step 2 (template fallback)

2. Template Fallback: match query to pre-built parameterized template
   ├── Template catalog: common patterns (find by name, find callers,
   │   find importers, find similar, find by type, find dependencies)
   ├── Parameter extraction: use extract_best_keyword() + intent classification
   └── All templates are read-only and parameterized — no LLM involvement

3. Keyword Fallback: existing _execute_keyword_search() logic
   ├── Simple CONTAINS-based Cypher with extracted keyword
   └── Always succeeds (worst case: empty results, not an error)
```

### 2.2 Template Catalog
Add a `CYPHER_QUERY_TEMPLATES` dictionary in `codebase_rag/cypher_queries.py` (existing module) mapping common query patterns to parameterized Cypher:

```python
# NOTE: All templates use Memgraph-native `|` label syntax (e.g., `Function|Class|Method`)
# because these are HARDCODED queries, not LLM-generated. The `_clean_cypher_response()`
# function in services/llm.py converts `|` to WHERE IN clauses only for LLM output,
# to protect against malformed syntax. Hardcoded templates bypass `_clean_cypher_response()`
# and go directly to `fetch_all()`, so they must use valid native Memgraph syntax.
#
# NOTE: Labels like `Enum`, `Type`, `Union`, `Interface`, `Contract`, `Library` are defined
# in NodeLabel constants but may not appear in all graph instances (they depend on the
# language being parsed — e.g., `Interface` appears for Java, `Contract` for Solidity).
# For broad compatibility, templates use WHERE-based filtering that works regardless of
# which labels exist in a particular project's graph, rather than `|` label unions
# that assume all labels are present.

CYPHER_QUERY_TEMPLATES: dict[str, tuple[str, dict[str, type]]] = {
    "find_by_name": (
        """
        MATCH (n)
        WHERE labels(n)[0] IN ['Function', 'Class', 'Method', 'Enum', 'Type',
                                'Union', 'Interface', 'Contract', 'Library']
          AND (n.name CONTAINS $keyword OR n.qualified_name CONTAINS $keyword)
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type,
               n.path AS file_path, n.start_line AS start_line,
               n.end_line AS end_line
        LIMIT $limit
        """,
        {"keyword": str, "limit": int},
    ),
    "find_callers_of": (
        CYPHER_FIND_CALLERS,  # Already exists as constant in cypher_queries.py
        {"qn": str},
    ),
    "find_importers_of": (
        CYPHER_FIND_IMPORTERS,  # Already exists as constant in cypher_queries.py
        {"qn": str},
    ),
    "find_by_type": (
        # NOTE: Cypher does not support parameterized node labels (MATCH (n:$label) is invalid).
        # Instead, we use string interpolation for the label with strict validation,
        # consistent with how build_merge_node_query() works in cypher_queries.py.
        """
        MATCH (n:{label})
        WHERE n.project_name = $project
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type, n.path AS file_path
        LIMIT $limit
        """,
        # The {label} placeholder is interpolated at runtime, NOT via Cypher parameters.
        # Validated against NodeLabel enum before interpolation to prevent injection.
        {"label": str, "project": str, "limit": int},
    ),
    "find_dependencies": (
        """
        MATCH (n:Function|Class|Method)-[:CALLS]->(m)
        WHERE n.qualified_name CONTAINS $keyword
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type, n.path AS file_path,
               id(m) AS target_id, m.qualified_name AS target_name
        LIMIT $limit
        """,
        {"keyword": str, "limit": int},
    ),
}
```

Template matching is done via intent classification + keyword extraction — no LLM needed. If no template matches, fall through to keyword search.

**Template label handling**: For `find_by_type`, the `{label}` placeholder is interpolated into the query string at runtime (since Cypher doesn't support `$label` as a parameter for node labels). The label value must be validated against the `NodeLabel` enum in `constants.py` before interpolation, preventing Cypher injection. This is the same pattern used by `build_merge_node_query()` and `build_create_node_query()` in `cypher_queries.py`.

**Template label compatibility**: For `find_by_name`, the WHERE IN clause `labels(n)[0] IN [...]` works correctly even when some labels don't exist in the graph — Memgraph simply returns no nodes for non-existent labels, unlike `|` union syntax which may cause issues if a label doesn't exist in some Memgraph versions. The three labels that always exist (`Function`, `Class`, `Method`) are listed first for optimal query planning.

### 2.3 Cypher Safety — Existing Mechanisms Preserved + Multi-Query Output Enhancement
The existing safety infrastructure is mostly preserved, with one targeted enhancement:

- **`_validate_cypher_read_only()`**: blocks CREATE, DELETE, SET, REMOVE, MERGE, DROP, LOAD CSV, FOREACH, CALL, CREATE INDEX/CONSTRAINT, OVER() window functions, USING PARALLEL EXECUTION, and multiple semicolon-separated queries — **preserved as-is**
- **`_clean_cypher_response()`**: strips markdown, comments, `|` label syntax (converts to WHERE IN clause), unsupported clauses — **preserved with one enhancement** (see below)
- **`CYPHER_DANGEROUS_KEYWORDS`**: comprehensive list with word-boundary regex patterns — **preserved as-is**

**Multi-Query LLM Output Enhancement**: Observed in production logs, the LLM sometimes generates two concatenated Cypher queries where the second fragment is broken syntax (e.g., `.qualified_name) CONTAINS 'constant'...`). `_clean_cypher_response()` splits on `;` and keeps the first query, which handles the logged case. However, the markdown extraction logic (`parts = query.split("```")`) can fail when the LLM generates output where both queries are inside a single markdown block without `;` separation. The enhancement adds a **trailing-fragment detection** step:

```python
# Add after existing _clean_cypher_response() logic, before final return:

# Detect and discard trailing broken query fragments that leak past
# markdown extraction. LLMs sometimes concatenate two queries where
# the second starts with a fragment (no MATCH keyword at beginning).
# This happens when the LLM generates output like:
#   ```cypher MATCH (c:Class)...RETURN...; .qualified_name)...LIMIT 50;```
# After markdown extraction and semicolon splitting, the first query
# is valid but a trailing fragment without MATCH may remain.
if query and not query.lstrip().upper().startswith("MATCH"):
    # Discard trailing fragment — valid Cypher queries always start
    # with MATCH (our use case is strictly read-only queries)
    logger.warning(f"Discarding trailing Cypher fragment without MATCH: {query[:80]}")
    return ""  # Will be caught by _validate_cypher_read_only() or downstream checks
```

This is a **minimal enhancement** — one conditional check after the existing processing pipeline. It doesn't change any existing behavior for valid single-query output; it only adds protection against the observed multi-query concatenation failure mode.

The template catalog inherently satisfies all safety constraints because templates are pre-validated, read-only, and parameterized.

### 2.4 Retry Integration with Existing Infrastructure
The orchestrator's `_execute_graph_traversal()` currently has **no retry logic** — if Cypher generation fails, the method returns an error result. The enhancement integrates with existing retry mechanisms:

```python
async def _execute_graph_traversal(
    self, query: str, top_k: int, start: float
) -> QueryMethodResult:
    """Execute graph traversal with retry and template fallback."""
    from ..exceptions import LLMGenerationError  # In codebase_rag/exceptions.py
    from ..services.llm import CypherGenerator
    from ..services.failure_classifier import classify_memgraph_failure
    
    cypher_gen = CypherGenerator()
    
    # Step 1: Try LLM-based generation (async-native, no asyncio.run)
    cypher = None
    try:
        cypher = await cypher_gen.generate(query)
        results = await self._fetch_all_async(cypher)
        return QueryMethodResult(
            method=QueryMethod.GRAPH_TRAVERSAL,
            items=results[:top_k],
            execution_time_ms=(time.time() - start) * 1000,
        )
    except LLMGenerationError as e:
        logger.warning(f"LLM Cypher generation failed: {e}")
    except Exception as e:
        # Classify the execution failure
        classification = classify_memgraph_failure(e)
        if classification.should_retry and classification.recovery_action == "repair_query":
            # Step 1b: Try CypherGenerator.repair() for syntax errors
            try:
                cypher = await cypher_gen.repair(query, cypher or "", str(e))
                results = await self._fetch_all_async(cypher)
                return QueryMethodResult(
                    method=QueryMethod.GRAPH_TRAVERSAL,
                    items=results[:top_k],
                    execution_time_ms=(time.time() - start) * 1000,
                )
            except Exception:
                pass  # Fall through to template fallback
    
    # Step 2: Template fallback (async — uses _fetch_all_async internally)
    template_result = await self._try_template_fallback(query, top_k, start)
    if template_result is not None:
        return template_result
    
    # Step 3: Keyword fallback (async — all execute_* methods are now async)
    return await self._execute_keyword_search(query, top_k, start)
```

**Async consistency**: All `_execute_*()` methods must be async since they're called from `execute_method_async()` which runs inside the pydantic-ai agent loop. This includes `_execute_keyword_search()`, `_execute_semantic_search()`, `_execute_vector_direct()`, `_execute_graph_navigation()`, and `_execute_graph_algorithms()`. Each method must wrap its `self.code_graph.fetch_all()` calls with `await self._fetch_all_async()`, which is a helper that delegates to `asyncio.to_thread(self.code_graph.fetch_all, ...)` for the sync mgclient I/O.

The `_fetch_all_async()` helper needs to be added to `QueryMethodOrchestrator` as a convenience method, and a proper `fetch_all_async()` method needs to be added to `MemgraphIngestor` and `PooledMemgraphProxy` (using `asyncio.to_thread()` for the sync mgclient call).

## 3. Coordinated Backend Health Monitoring

### 3.1 Integration with Existing Health Mechanisms
The system already has three independent health-check mechanisms:
- **`MemgraphIngestor._check_connection_health()`**: periodic (30s interval) TCP connectivity + `RETURN 1` query
- **`HybridRetriever._validate()`**: vector backend health + embedding provider validation + vector search test
- **`GraphAlgorithms.health_check()`**: MAGE procedure availability + basic connection

The enhancement adds a **`BackendHealthCoordinator`** (lightweight utility, NOT a new standalone module) in `codebase_rag/retrieval/query_orchestrator.py` that consults these existing checks before method execution:

```python
class BackendHealthCoordinator:
    """Consults existing health mechanisms before method execution.
    
    Uses the QueryProtocol interface's fetch_all() for graph health checks
    (works for both MemgraphIngestor and PooledMemgraphProxy), and delegates
    to concrete class health methods only when the concrete type is known.
    
    Design note: QueryProtocol only defines fetch_all() and execute_write().
    Private methods like _check_connection_health() (on MemgraphIngestor) and
    _validate() (on HybridRetriever) are NOT part of the protocol interface.
    PooledMemgraphProxy implements QueryProtocol but has no _check_connection_health().
    Therefore, health checks must use the public interface (RETURN 1 probe) for
    QueryProtocol objects, and only call private health methods on objects that
    are guaranteed to be the concrete type (not proxies).
    
    For HybridRetriever._validate() and GraphAlgorithms.health_check(), these
    are called on objects the coordinator owns directly (not via QueryProtocol),
    so the concrete type is always known. However, _validate() is a private
    method — a future refactor should expose it as a public health_check() method.
    For now, this is an intentional private-API dependency documented here.
    """
    
    def __init__(
        self,
        graph_ingestor: QueryProtocol,
        hybrid_retriever: HybridRetriever | None = None,
        graph_algorithms: GraphAlgorithms | None = None,
    ) -> None:
        self._graph = graph_ingestor
        self._hybrid = hybrid_retriever
        self._algorithms = graph_algorithms
        self._last_check_time: float = 0.0
        self._check_interval: float = 30.0  # Same as MemgraphIngestor
        self._cached_status: dict[str, bool] = {}
    
    def check_graph_health(self) -> bool:
        """Check graph backend health via the QueryProtocol interface.
        
        Uses fetch_all("RETURN 1") which works for ALL QueryProtocol
        implementations (MemgraphIngestor, PooledMemgraphProxy, etc.).
        Does NOT call _check_connection_health() because that is a private
        method on MemgraphIngestor that PooledMemgraphProxy doesn't have,
        and accessing private methods via hasattr() on a Protocol interface
        violates the interface contract.
        """
        try:
            result = self._graph.fetch_all("RETURN 1 AS health")
            return len(result) > 0
        except Exception:
            return False
    
    def check_vector_health(self) -> bool:
        """Check vector backend health via HybridRetriever._validate().
        
        Intentional private-API dependency: HybridRetriever._validate() is
        private but provides the only comprehensive health check for the
        vector+embedding subsystem. A future refactor should add a public
        health_check() method to HybridRetriever. Until then, this is the
        only viable approach since HybridRetriever is owned by the coordinator
        (not received via a Protocol interface).
        """
        if self._hybrid is not None:
            return self._hybrid._validate()
        return False  # No vector backend available
    
    def check_algorithm_health(self) -> bool:
        """Check MAGE algorithm availability via GraphAlgorithms.health_check().
        
        GraphAlgorithms.health_check() is a public method, so no private-API
        concern here. The coordinator owns the GraphAlgorithms instance directly.
        """
        if self._algorithms is not None:
            return self._algorithms.health_check()
        return False  # Algorithms not available
    
    def get_method_availability(self) -> dict[QueryMethod, bool]:
        """Map each query method to its backend availability."""
        graph_ok = self.check_graph_health()
        vector_ok = self.check_vector_health()
        algo_ok = self.check_algorithm_health()
        
        return {
            QueryMethod.SEMANTIC_SEARCH: vector_ok,
            QueryMethod.GRAPH_TRAVERSAL: graph_ok,
            QueryMethod.KEYWORD_SEARCH: graph_ok,  # Only needs basic graph
            QueryMethod.VECTOR_DIRECT: vector_ok,
            QueryMethod.GRAPH_NAVIGATION: graph_ok,
            QueryMethod.GRAPH_ALGORITHMS: algo_ok,
        }
```

### 3.2 Health-Based Method Selection
Before executing methods, the orchestrator consults `BackendHealthCoordinator.get_method_availability()` to:
1. **Skip unavailable methods**: Don't attempt `SEMANTIC_SEARCH` if vector backend is down
2. **Adjust method order**: Promote available secondary methods when primary methods are unhealthy
3. **Set expectations**: Mark results from degraded backends with lower confidence weights

### 3.3 Circuit Breaker — Lightweight Implementation
When a method consistently fails, temporarily disable it for the current query session:

```python
# In QueryMethodOrchestrator:
self._circuit_breaker: dict[QueryMethod, int] = {}  # consecutive_failures count

def _is_method_available(self, method: QueryMethod) -> bool:
    """Check circuit breaker and health coordinator."""
    failures = self._circuit_breaker.get(method, 0)
    if failures >= 3:  # Same threshold as existing _should_retry_shared_connection_error
        return False
    return self._health_coordinator.get_method_availability().get(method, False)
```

This is **not** a new module — it's a few fields added to the existing `QueryMethodOrchestrator`.

### 3.5 Local Embedding Provider Singleton — Prevent Redundant Model Loads
Problem Statement #4 identifies an **embedding cold-start cascade**: when OpenAI auth fails (401), the local fallback (`BAAI/bge-large-en-v1.5`) is loaded from scratch each time. Production logs show **3+ redundant model loads in a single session** — each load takes ~2 seconds and 391 weight tensors, totaling ~6+ seconds of unnecessary latency per query session.

The root cause is that `LocalEmbeddingProvider._ensure_model_loaded()` in `codebase_rag/embeddings/local.py` creates a new model instance on each call when the OpenAI provider fails. Multiple components (semantic search, vector direct, HybridRetriever validation) all trigger independent fallbacks, each causing a separate model load.

**Fix: Make `LocalEmbeddingProvider` a process-level singleton with lazy initialization.**

```python
# In codebase_rag/embeddings/local.py:

_module_model_lock = threading.Lock()
_module_model_instance: LocalEmbeddingProvider | None = None

def get_local_embedding_provider(
    model_id: str = "BAAI/bge-large-en-v1.5",
    device: str = "cpu",
) -> LocalEmbeddingProvider:
    """Get or create the process-level singleton LocalEmbeddingProvider.
    
    Thread-safe: uses module-level lock to prevent concurrent model loads.
    The model is loaded once and reused across all fallback invocations,
    eliminating the 3+ redundant loads observed in production.
    
    This is NOT a new module — it's a module-level singleton pattern
    added to the existing local.py, consistent with how get_embedding_provider()
    in embeddings/__init__.py already creates providers.
    """
    global _module_model_instance
    if _module_model_instance is not None:
        return _module_model_instance
    with _module_model_lock:
        if _module_model_instance is None:
            _module_model_instance = LocalEmbeddingProvider(
                model_id=model_id, device=device
            )
            _module_model_instance._ensure_model_loaded()
        return _module_model_instance
```

**Integration with existing fallback chain**: `OpenAIEmbeddingProvider._make_request()` currently calls `local = LocalEmbeddingProvider(...)` on each 401 failure. Change this to `local = get_local_embedding_provider()` so the model is loaded once and reused.

**Integration with `BackendHealthCoordinator`**: The `check_vector_health()` method calls `HybridRetriever._validate()` which triggers embedding provider validation. With the singleton, the first validation call loads the model (~2s), and subsequent calls reuse it (~0ms). The health coordinator should log the cold-start latency on first check:

```python
def check_vector_health(self) -> bool:
    if self._hybrid is not None:
        start = time.time()
        result = self._hybrid._validate()
        latency = time.time() - start
        if latency > 1.0 and not self._vector_was_checked:
            logger.info(f"Vector health check cold-start latency: {latency:.1f}s (model loaded)")
            self._vector_was_checked = True
        return result
    return False
```

This doesn't change any existing behavior — it only makes the local model persist across invocations instead of being recreated each time.

### 3.4 Data Consistency Check — Defined Concretely
The `_test_data_consistency()` concept from the original spec was undefined. Here is the concrete implementation:

**`_check_graph_data_consistency()`** (utility method in orchestrator):
```python
def _check_graph_data_consistency(self) -> ConsistencyStatus:
    """Lightweight graph-vs-disk consistency check.
    
    Compares graph node counts with disk file counts.
    NOT a full integrity verification (that's §4).
    """
    # Count indexed files in graph
    graph_files = self.code_graph.fetch_all(
        "MATCH (n:File) RETURN count(n) AS file_count"
    )
    graph_count = graph_files[0].get("file_count", 0) if graph_files else 0
    
    # Count actual source files on disk (fast walk, no reading)
    from ..config import settings
    repo_path = Path(settings.TARGET_REPO_PATH)
    source_extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".rs", ".go", ".lua", ".sol"}
    disk_count = sum(1 for _ in repo_path.rglob("*") if _.suffix in source_extensions)
    
    # Allow 10% tolerance (graph may exclude files via .cgrignore)
    ratio = graph_count / max(disk_count, 1)
    if ratio < 0.5:
        return ConsistencyStatus.STALE  # Graph may need re-indexing
    if ratio < 0.9:
        return ConsistencyStatus.PARTIAL  # Some files not indexed
    return ConsistencyStatus.CONSISTENT
```

This is a **quick heuristic** for the health coordinator, not a full integrity scan. Full integrity verification is in §4.

## 4. Graph Data Integrity Verification

> **Important distinction**: This section addresses **graph-to-disk metadata integrity** — verifying that node properties stored in Memgraph (file_path, start_line, end_line, qualified_name) match actual source files on disk. This is DIFFERENT from the existing **code↔document semantic validation** in `codebase_rag/shared/validation/` (`CodeVsDocValidator`, `DocVsCodeValidator`), which validates whether code implementations comply with document specifications. Both are valuable but serve different purposes.

### 4.1 Graph-to-Disk Integrity Check Protocol
For graph query results that reference source code, verify metadata accuracy:

1. **File Existence Verification**: `Path(item["file_path"]).exists()` — confirm the file referenced by the graph node actually exists on disk
2. **Line Number Validation**: `start_line <= total_lines and end_line <= total_lines` — verify line ranges are within current file bounds (files may have changed since indexing)
3. **Qualified Name Resolution**: Search the file for the named entity — confirm the function/class still exists at the reported location
4. **Staleness Detection**: Compare file modification time with graph indexing timestamp — flag files that were modified after indexing

### 4.2 Implementation — Utility in Existing Module
Add `verify_graph_result_integrity()` as a utility function in `codebase_rag/retrieval/query_orchestrator.py` (NOT a new module):

```python
def verify_graph_result_integrity(
    items: list[dict[str, Any]],
    repo_path: Path,
    max_checks: int = 5,  # Only check top-N results (performance)
) -> list[IntegrityWarning]:
    """Spot-check graph results against actual source files.
    
    Lightweight verification for top results only.
    Full verification available via separate integrity-audit command.
    """
    warnings: list[IntegrityWarning] = []
    
    for item in items[:max_checks]:
        file_path = item.get("file_path")
        if not file_path:
            continue
        
        full_path = repo_path / file_path
        start_line = item.get("start_line", 0)
        end_line = item.get("end_line", 0)
        qualified_name = item.get("qualified_name", "")
        
        # Check 1: File existence
        if not full_path.exists():
            warnings.append(IntegrityWarning(
                severity="hard",
                item=qualified_name,
                issue=f"File {file_path} does not exist on disk",
                action="Graph may be stale — consider re-indexing",
            ))
            continue
        
        # Check 2: Line number bounds
        total_lines = sum(1 for _ in full_path.open(encoding="utf-8", errors="replace"))
        if start_line > total_lines or end_line > total_lines:
            warnings.append(IntegrityWarning(
                severity="soft",
                item=qualified_name,
                issue=f"Line range ({start_line}-{end_line}) exceeds file length ({total_lines})",
                action="File may have been modified since indexing",
            ))
        
        # Check 3: Staleness (modification time)
        # (Detailed implementation uses graph indexing timestamp comparison)
    
    return warnings
```

### 4.3 Integrity Failure Handling
- **Soft Failures** (line range slightly off, file modified since indexing): Log warning, include `IntegrityWarning` in `CombinedQueryResult.warnings`, continue processing
- **Hard Failures** (file doesn't exist, qualified name completely wrong): Log error, reduce confidence score of affected results, suggest re-indexing in user-facing output
- **Recovery**: No automatic re-indexing triggered (that's an administrative action). The warning message includes the re-indexing command: `cgr index --code`

### 4.4 Integrity Metrics Collection
Add metrics to `CombinedQueryResult` for monitoring over time:
```python
@dataclass
class CombinedQueryResult:
    # ... existing fields ...
    integrity_warnings: list[IntegrityWarning] = field(default_factory=list)
    integrity_check_count: int = 0  # How many items were checked
    integrity_pass_count: int = 0   # How many passed all checks
```

Track aggregate metrics via logging (not a new metrics service):
- Log integrity pass rate per query at INFO level
- Existing monitoring infrastructure can aggregate from logs

## 5. Adaptive Query Sequencing and Completion

### 5.1 Adaptive Sequencing — Replaces Fixed Order
The existing `execute()` method iterates through `_INTENT_METHOD_MAP[intent]` in fixed order, limited to `max_methods`. The enhancement replaces this with adaptive sequencing that considers:
- Method availability (from `BackendHealthCoordinator`)
- Intent confidence (from `classify_intent_with_confidence()`)
- Early termination when sufficient results found

```python
async def execute(
    self,
    query: str,
    top_k: int = 5,
    min_methods: int = 2,  # NEW: minimum methods required
) -> CombinedQueryResult:
    """Execute comprehensive query with adaptive sequencing."""
    intent, confidence = self.classify_intent_with_confidence(query)
    primary_methods, secondary_methods = self.select_methods(intent)
    
    # Health pre-check: filter out unavailable methods
    availability = self._health_coordinator.get_method_availability()
    primary_methods = [m for m in primary_methods if availability.get(m, False)]
    secondary_methods = [m for m in secondary_methods if availability.get(m, False)]
    
    # If confidence is low, treat all methods as primary (exploratory)
    if confidence < 0.5:
        primary_methods = [m for m in QueryMethod if availability.get(m, False)]
        secondary_methods = []
    
    results: list[QueryMethodResult] = []
    sufficient = False
    
    # Stage 1: Execute primary methods
    for method in primary_methods:
        result = await self.execute_method_async(method, query, top_k)
        results.append(result)
        self._update_circuit_breaker(method, result)
        
        # Early termination check: sufficient results from multiple methods?
        successful_results = [r for r in results if r.error is None and len(r.items) > 0]
        if len(successful_results) >= min_methods and self._results_cover_query(query, successful_results):
            sufficient = True
            break
    
    # Stage 2: If not sufficient, execute secondary methods
    if not sufficient:
        for method in secondary_methods:
            if self._is_method_available(method):  # Circuit breaker check
                result = await self.execute_method_async(method, query, top_k)
                results.append(result)
                self._update_circuit_breaker(method, result)
                
                successful_results = [r for r in results if r.error is None and len(r.items) > 0]
                if len(successful_results) >= min_methods:
                    break
    
    # Stage 3: Merge, rank, and integrity spot-check
    merged_items = self._merge_and_rank(results, top_k)
    integrity_warnings = verify_graph_result_integrity(
        merged_items, Path(settings.TARGET_REPO_PATH), max_checks=5
    )
    
    # Build final result
    total_time = sum(r.execution_time_ms for r in results)
    errors = [r.error for r in results if r.error]
    
    return CombinedQueryResult(
        query=query,
        intent=intent,
        methods_used=[r.method for r in results],
        items=merged_items,
        execution_time_ms=total_time,
        warnings=errors,
        integrity_warnings=integrity_warnings,
        integrity_check_count=min(5, len(merged_items)),
        integrity_pass_count=min(5, len(merged_items)) - len([w for w in integrity_warnings if w.severity == "hard"]),
    )
```

### 5.2 Sufficient Information Detection
**`_results_cover_query()`** heuristic (lightweight, no LLM):
```python
def _results_cover_query(
    self, query: str, results: list[QueryMethodResult]
) -> bool:
    """Heuristic check: do results cover the query's likely scope?
    
    Criteria:
    1. At least min_methods different methods returned non-empty results
    2. Combined item count >= 3 (enough for context)
    3. Items come from diverse file paths (not all from one file)
    """
    total_items = sum(len(r.items) for r in results)
    if total_items < 3:
        return False
    
    unique_files = len(set(
        item.get("file_path", "") 
        for r in results 
        for item in r.items
        if item.get("file_path")
    ))
    if unique_files < 2 and total_items < 5:
        return False  # All from one file — likely incomplete
    
    return True
```

### 5.3 Integration with Existing Sufficiency Gatekeeper
The orchestrator-level sufficiency (§5.2) is **complementary** to the existing agent-level sufficiency gatekeeper in `main.py`. They operate at different levels:

| Level | Component | Scope | Enforcement |
|-------|-----------|-------|-------------|
| **Orchestrator** | `_results_cover_query()` | Single `execute()` call | Early termination / method continuation |
| **Agent loop** | `evaluate_sufficiency()` | Full investigation session | Reject/force continuation of LLM response |

The orchestrator ensures its own output is comprehensive. The agent-level gatekeeper ensures the LLM uses the orchestrator's output correctly (reads source files, cross-validates, etc.). No duplication — they're complementary.

### 5.4 Early Termination Prevention
- **Minimum methods**: `min_methods=2` parameter prevents returning after just one method succeeds
- **Circuit breaker bypass**: When all primary methods are circuit-broken, force at least one secondary method
- **No empty-result returns**: If all methods fail, return a `CombinedQueryResult` with clear warnings listing what failed and why, rather than an empty result silently

## 6. Enhanced Error Handling and Recovery

### 6.1 Error Classification — Built on Existing `failure_classifier.py`
The existing `FailureType` enum in `codebase_rag/services/failure_classifier.py` already covers all necessary categories:

| Existing `FailureType` | Spec Error Category | Existing Recovery Action |
|------------------------|--------------------|-----------------------|
| `SYNTAX_ERROR` | Syntax Errors | `repair_query` |
| `TRANSIENT_NETWORK` | Connectivity Issues | `reconnect` |
| `TRANSIENT_TIMEOUT` | Connectivity Issues | `increase_timeout` |
| `MISSING_PROCEDURE` | Backend Misuse | `fallback_alternative` |
| `VECTOR_INDEX_MISSING` | Backend Misuse | `recreate_index` |
| `VECTOR_DIMENSION_MISMATCH` | Backend Misuse | `recreate_index_and_reembed` |
| `AUTHENTICATION_FAILURE` | Auth Issues | (none — raise) |
| `PERMISSION_DENIED` | Auth Issues | (none — raise) |
| `DATA_INTEGRITY` | Data Inconsistencies | `alert_admin` |
| `RESOURCE_EXHAUSTION` | Resource Constraints | `cleanup` |
| `UNKNOWN` | Unknown | retry once |

**No new error classification is needed.** The enhancement ensures the orchestrator **uses** the existing classifier:

```python
# In execute_method_async():
try:
    result = await self._execute_X(query, top_k, start)
    self._circuit_breaker[method] = 0  # Reset on success
    return result
except Exception as e:
    classification = classify_memgraph_failure(e)
    
    # Use existing recovery actions
    if classification.recovery_action == "repair_query" and method == QueryMethod.GRAPH_TRAVERSAL:
        # Try CypherGenerator.repair() — §2.4
        repaired = await self._retry_with_repair(query, str(e))
        if repaired:
            return repaired
    
    if classification.should_retry:
        # Retry with existing backoff logic
        for attempt in range(classification.max_retries):
            await asyncio.sleep(settings.MEMGRAPH_RETRY_BASE_DELAY * (attempt + 1))
            try:
                result = await self._execute_X(query, top_k, start)
                self._circuit_breaker[method] = 0
                return result
            except Exception:
                continue
    
    # All retries exhausted — record in circuit breaker
    self._circuit_breaker[method] = self._circuit_breaker.get(method, 0) + 1
    
    return QueryMethodResult(
        method=method,
        items=[],
        execution_time_ms=(time.time() - start) * 1000,
        error=f"{classification.failure_type.name}: {classification.message}",
    )
```

### 6.2 Graceful Degradation Hierarchy
This already exists implicitly in the system. The enhancement makes it **explicit** in the orchestrator:

1. **Full Capability**: All methods available and healthy → use primary + secondary methods
2. **Reduced Capability**: Some methods circuit-broken → promote available secondary methods to primary
3. **Basic Capability**: Only `KEYWORD_SEARCH` and `GRAPH_NAVIGATION` available → use with lower confidence weights in `_merge_and_rank()`
4. **Emergency Mode**: No graph methods available → return `CombinedQueryResult` with clear error message listing what's down and suggested recovery steps (e.g., "Memgraph unreachable — check if service is running: `docker ps`")

### 6.3 User-Facing Error Communication
Enhanced `CombinedQueryResult.warnings` format:
```python
# Existing: warnings = [r.error for r in results if r.error]
# Enhanced: structured warnings with recovery context
warnings: list[str] = []
for r in results:
    if r.error:
        classification = classify_memgraph_failure(Exception(r.error))
        recovery = classification.recovery_action or "no_recovery"
        warnings.append(
            f"[{r.method.name}] {r.error} (recovery: {recovery})"
        )
```

The `QueryRouter` in `codebase_rag/shared/query_router.py` already surfaces warnings in `QueryResponse.warnings` — no new user-facing format needed.

## 7. Implementation Requirements

### 7.1 Required Code Changes — Targeted Enhancements to Existing Files

| # | Change | Target File | Type |
|---|--------|------------|------|
| 1 | Convert orchestrator to async-native (`execute_method_async`, all `_execute_*()` methods) | `codebase_rag/retrieval/query_orchestrator.py` | Refactor |
| 2 | Add `_fetch_all_async()` convenience helper on orchestrator | `codebase_rag/retrieval/query_orchestrator.py` | Add method |
| 3 | Add `classify_intent_with_confidence()` | `codebase_rag/retrieval/query_orchestrator.py` | Add method |
| 4 | Add primary/secondary method designation | `codebase_rag/retrieval/query_orchestrator.py` | Modify `_INTENT_METHOD_MAP` |
| 5 | Add adaptive `execute()` with early termination | `codebase_rag/retrieval/query_orchestrator.py` | Refactor |
| 6 | Add `BackendHealthCoordinator` (inner class) | `codebase_rag/retrieval/query_orchestrator.py` | Add class |
| 7 | Add circuit breaker fields + `_is_method_available()` + `_update_circuit_breaker()` | `codebase_rag/retrieval/query_orchestrator.py` | Add fields |
| 8 | Add template fallback in `_execute_graph_traversal()` | `codebase_rag/retrieval/query_orchestrator.py` | Modify method |
| 9 | Add `CYPHER_QUERY_TEMPLATES` catalog | `codebase_rag/cypher_queries.py` | Add constant |
| 10 | Add `fetch_all_async()` method (async wrapper via `asyncio.to_thread`) | `codebase_rag/services/graph_service.py` | Add method |
| 11 | Add `fetch_all_async()` method (async wrapper via `asyncio.to_thread`) | `codebase_rag/services/connection_pool.py` | Add method |
| 12 | Add `verify_graph_result_integrity()` utility + `IntegrityWarning` dataclass | `codebase_rag/retrieval/query_orchestrator.py` | Add function + dataclass |
| 13 | Add `ConsistencyStatus` enum + `_check_graph_data_consistency()` | `codebase_rag/retrieval/query_orchestrator.py` | Add enum + method |
| 14 | Update `CombinedQueryResult` with integrity fields | `codebase_rag/retrieval/query_orchestrator.py` | Modify dataclass |
| 15 | Update `QueryRouter._query_code_with_orchestrator()` for async + integrity warnings | `codebase_rag/shared/query_router.py` | Modify method |
| 16 | Add 6 new `AppConfig` fields for query optimization parameters (§7.2) | `codebase_rag/config.py` | Add fields |
| 17 | Fix `$depth` parameterization bug in `get_call_hierarchy()` — replace `[:CALLS*1..$depth]` with string-interpolated literal (Memgraph doesn't support parameterized variable-length path bounds) | `codebase_rag/tools/graph_navigation.py` | Bug fix |
| 18 | Add trailing-fragment detection in `_clean_cypher_response()` — discard Cypher output that doesn't start with MATCH keyword (§2.3) | `codebase_rag/services/llm.py` | Enhancement |
| 19 | Make local embedding provider (`LocalEmbeddingProvider`) a process-level singleton with lazy initialization to prevent redundant model loads (§3.5) | `codebase_rag/embeddings/local.py` | Refactor |

**No new Python modules are created.** All changes target existing files.

### 7.1.1 Prerequisite Bug Fixes (Must Complete Before Phase 1)
Two existing bugs in the codebase must be fixed before the enhancement phases begin. These are independent of the orchestrator enhancement but directly cause the failure modes observed in production logs:

**Bug Fix #17: `$depth` Parameterization in `get_call_hierarchy()`**

`get_call_hierarchy()` in `graph_navigation.py` uses `[:CALLS*1..$depth]` where `$depth` is passed as a Cypher parameter. Memgraph does NOT support parameterized bounds in variable-length path expressions — the `$depth` parameter is misclassified as "Property map matching" and the query fails with `Property map matching not supported in MATCH/MERGE clause!`. This causes **100% failure of all call hierarchy queries**.

The fix is to use **string interpolation** for the depth value (with integer validation to prevent injection), consistent with how `build_merge_node_query()` handles `{label}` interpolation in `cypher_queries.py`:

```python
# BEFORE (broken — $depth parameterized in variable-length path):
callers_query = """
MATCH path = (caller:Function|Method)-[:CALLS*1..$depth]->(target)
WHERE target.qualified_name = $qn
..."""
callers = await asyncio.to_thread(
    self.ingestor.fetch_all, callers_query,
    {"qn": qualified_name, "depth": depth},  # $depth causes "Property map matching" error
)

# AFTER (fixed — depth interpolated as literal integer with validation):
depth = min(max(depth, 1), _MAX_DEPTH)  # Already validated as int
callers_query = f"""
MATCH path = (caller:Function|Method)-[:CALLS*1..{depth}]->(target)
WHERE target.qualified_name = $qn
RETURN DISTINCT
    caller.qualified_name AS qualified_name,
    caller.name AS name,
    length(path) AS depth,
    COALESCE(caller.pagerank_score, 0.1) AS pagerank,
    COALESCE(caller.community_importance, 0.0) AS community_importance,
    caller.docstring AS docstring
ORDER BY (pagerank * 0.7 + community_importance * 0.3) DESC, depth ASC
LIMIT 50
"""
callers = await asyncio.to_thread(
    self.ingestor.fetch_all, callers_query,
    {"qn": qualified_name},  # Only $qn remains as parameter (it works fine)
)
```

The same fix applies to the callees query. The `_MAX_DEPTH` cap (already defined as `5`) prevents abuse. `depth` is validated as a positive integer before interpolation, making Cypher injection impossible (only digits can be interpolated).

**Bug Fix #18: `_clean_cypher_response()` Trailing-Fragment Detection**

See §2.3 for the detailed enhancement. This adds a single conditional check after the existing `_clean_cypher_response()` pipeline to detect and discard trailing broken query fragments that lack a MATCH keyword — a failure mode observed in production logs where the LLM concatenates two queries and the second fragment leaks past markdown extraction.

### 7.2 Configuration Parameters — Mapped to `AppConfig`
All new parameters are added to the existing `AppConfig` class in `codebase_rag/config.py` as environment-variable-backed fields (consistent with existing pattern):

| Parameter | AppConfig Field | Env Var | Default | Type |
|-----------|----------------|---------|---------|------|
| Intent confidence threshold | `QUERY_INTENT_CONFIDENCE_THRESHOLD` | `CGR_QUERY_INTENT_CONFIDENCE_THRESHOLD` | `0.5` | `float` |
| Minimum methods required | `QUERY_MIN_METHODS` | `CGR_QUERY_MIN_METHODS` | `2` | `int` |
| Circuit breaker threshold | `QUERY_CIRCUIT_BREAKER_THRESHOLD` | `CGR_QUERY_CIRCUIT_BREAKER_THRESHOLD` | `3` | `int` |
| Integrity spot-check count | `QUERY_INTEGRITY_CHECK_COUNT` | `CGR_QUERY_INTEGRITY_CHECK_COUNT` | `5` | `int` |
| Enable integrity verification | `QUERY_ENABLE_INTEGRITY_CHECK` | `CGR_QUERY_ENABLE_INTEGRITY_CHECK` | `True` | `bool` |
| Template fallback enabled | `QUERY_TEMPLATE_FALLBACK_ENABLED` | `CGR_QUERY_TEMPLATE_FALLBACK_ENABLED` | `True` | `bool` |

No YAML configuration file is introduced. All settings follow the existing `pydantic-settings` + `.env` pattern.

### 7.3 Testing Requirements — Specific Acceptance Criteria

1. **Async Execution Tests** (`test_query_orchestrator.py`):
   - `_execute_graph_traversal_async()` works inside an async context (no `asyncio.run()` crash)
   - `execute()` works when called from `asyncio.run()` and from within an existing event loop
   - `fetch_all_async()` returns same results as `fetch_all()`

2. **Template Fallback Tests** (`test_query_orchestrator.py`):
   - When `CypherGenerator.generate()` raises `LLMGenerationError`, `_execute_graph_traversal()` falls back to template matching
   - When no template matches, falls back to keyword search
   - Template safety by construction: verify all templates in `CYPHER_QUERY_TEMPLATES` contain only read operations (MATCH, RETURN, WHERE, LIMIT — no CREATE/DELETE/SET/MERGE/DROP), use parameterized values ($keyword, $limit, $qn), and that `_try_template_fallback()` validates `{label}` interpolation against the `NodeLabel` enum before executing

3. **Adaptive Sequencing Tests** (`test_query_orchestrator.py`):
   - With high intent confidence, only primary methods execute
   - With low intent confidence (< 0.5), all available methods execute
   - Early termination: stops after 2 successful methods if results cover query
   - Circuit breaker: skips method after 3 consecutive failures

4. **Health Coordination Tests** (`test_query_orchestrator.py`):
   - `BackendHealthCoordinator` correctly delegates to existing health checks
   - Unavailable methods are skipped (not attempted)
   - When primary methods unavailable, secondary methods promoted

5. **Integrity Verification Tests** (`test_query_orchestrator.py`):
   - `verify_graph_result_integrity()` detects missing files
   - `verify_graph_result_integrity()` detects out-of-bounds line numbers
   - Integrity warnings included in `CombinedQueryResult`
   - Hard failures reduce confidence; soft failures logged only

6. **Failure Injection Tests** (`test_query_orchestrator.py`):
   - Simulate Memgraph connection failure → orchestrator degrades gracefully
   - Simulate Cypher syntax error → template fallback triggered
   - Simulate vector backend down → semantic_search skipped, keyword promoted
   - Simulate all methods failing → emergency mode with clear error message

7. **Backward Compatibility Tests** (`test_query_router.py`):
   - `QueryRouter.query()` with `use_orchestrator=False` works unchanged
   - `QueryRouter.query()` with `use_orchestrator=True` returns same `QueryResponse` shape
   - `CombinedQueryResult` new fields don't break existing consumers

## 8. Backward Compatibility

- **Existing APIs**: `QueryMethodOrchestrator.execute()` signature preserved (query, top_k, max_methods → query, top_k, min_methods). `max_methods` parameter retained as alias for backward compat
- **`QueryRouter`**: `_query_code_with_orchestrator()` updated for async; `_query_code_legacy()` untouched
- **Existing tool interfaces**: All MCP tools, CLI commands, and direct API calls continue working. The orchestrator is only invoked when `use_orchestrator=True` is set in `QueryRequest`
- **Configuration**: New `AppConfig` fields have defaults matching current behavior (e.g., `QUERY_MIN_METHODS=2`, `QUERY_ENABLE_INTEGRITY_CHECK=True`). No `.env` changes required for existing deployments
- **Performance**: No regression for simple queries — health checks are cached (30s interval), integrity spot-checks limited to 5 items, template fallback only invoked on LLM failure

## 9. Success Metrics — Measurable Engineering Targets

| Metric | Current Baseline (estimated) | Target | Measurement Method |
|--------|------------------------------|--------|--------------------|
| Cypher generation recovery rate | ~0% (no retry/fallback on failure) | >80% of failures recovered via template fallback or repair | Count: successful retries / total Cypher failures |
| Orchestrator method diversity | 1–2 methods per query (fixed order, max_methods=3) | ≥2 methods per query for 90% of non-trivial queries | Count: distinct methods with non-empty results / total queries |
| Empty-result query rate | Unknown | Reduce by 30% via adaptive sequencing + fallbacks | Count: queries returning 0 items / total queries |
| Async context crash rate | `asyncio.run()` crashes in MCP/agent contexts | 0% (async-native execution) | Count: asyncio crashes / total async-context calls |
| Graph data accuracy | Unknown | <5% hard integrity failures on top-5 results | Count: hard integrity warnings / total integrity checks |
| Transient failure recovery | Existing `_execute_query()` retries per-query | Orchestrator-level recovery for all methods | Count: recovered failures / total transient failures |

**Note**: Baselines marked "Unknown" should be measured for 2 weeks before implementation begins to establish actual starting values. Targets are relative improvements once baselines are established.

## 10. Implementation Timeline

- **Phase 0 (Week 0 — Prerequisite Bug Fixes)**: Must complete before Phase 1
  - Fix `$depth` parameterization bug in `get_call_hierarchy()` (change #17)
  - Add trailing-fragment detection in `_clean_cypher_response()` (change #18)
  - Make local embedding provider a process-level singleton (change #19)
  - Tests: call hierarchy works end-to-end, `_clean_cypher_response()` discards non-MATCH fragments, embedding model loaded only once per session

- **Phase 1 (Week 1–2)**: Async conversion + health coordination
  - Convert `QueryMethodOrchestrator` methods to async
  - Add `fetch_all_async()` to `MemgraphIngestor` and `PooledMemgraphProxy`
  - Add `BackendHealthCoordinator` inner class
  - Add circuit breaker logic
  - Tests: async execution, health coordination, circuit breaker

- **Phase 2 (Week 3–4)**: Template fallback + adaptive sequencing
  - Add `CYPHER_QUERY_TEMPLATES` catalog to `cypher_queries.py`
  - Enhance `_execute_graph_traversal()` with retry → template → keyword fallback chain
  - Add `classify_intent_with_confidence()` and primary/secondary method designation
  - Rewrite `execute()` with adaptive sequencing and early termination
  - Tests: template fallback, adaptive sequencing, intent confidence

- **Phase 3 (Week 5–6)**: Integrity verification + integration
  - Add `verify_graph_result_integrity()` and `IntegrityWarning` dataclass
  - Add `ConsistencyStatus` and `_check_graph_data_consistency()`
  - Update `CombinedQueryResult` with integrity fields
  - Update `QueryRouter._query_code_with_orchestrator()` for async + integrity
  - Add `AppConfig` fields for new parameters
  - Tests: integrity verification, integration, backward compatibility

- **Phase 4 (Week 7–8)**: Failure injection + baseline measurement + documentation
  - Failure injection tests for all degradation scenarios
  - Measure baselines for success metrics (2-week data collection)
  - Update user documentation and architecture docs
  - Final review and deployment

### Conclusion
This specification provides targeted, incremental enhancements to the existing multi-paradigm query system. By converting to async-native execution, adding template-based Cypher fallback, coordinating existing health checks, introducing adaptive method sequencing, and adding lightweight graph data integrity verification, the system addresses its core failure modes without introducing new standalone modules or duplicating existing infrastructure. Every change builds on proven code in `query_orchestrator.py`, `services/llm.py`, `failure_classifier.py`, `sufficiency_gatekeeper.py`, and `query_router.py`.