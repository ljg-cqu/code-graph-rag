"""Query method orchestration for multi-paradigm retrieval.

Orchestrates multiple query methods (semantic search, graph traversal,
keyword search, etc.) based on query intent and combines results with
unified ranking.

Enhanced with:
- Async-native execution for MCP server / pydantic-ai agent loop compatibility
- Backend health coordination before method selection
- Circuit breaker for failing methods
- Template-based Cypher fallback
- Adaptive sequencing with early termination
- Graph data integrity verification
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from ..config import settings
from ..shared.utils.file_classifier import is_code_file

if TYPE_CHECKING:
    from ..embeddings.base import EmbeddingProviderProtocol
    from ..graph_algorithms import GraphAlgorithms
    from ..memgraph_advanced.hybrid_retrieval import HybridRetriever
    from ..services import QueryProtocol
    from ..vector_backend import VectorBackend


class QueryIntent(Enum):
    """Classified query intent for method selection."""

    FUNCTIONAL = auto()
    STRUCTURAL = auto()
    SEMANTIC = auto()
    EXPLORATORY = auto()
    VALIDATION = auto()


class QueryMethod(Enum):
    """Available query methods."""

    SEMANTIC_SEARCH = auto()
    GRAPH_TRAVERSAL = auto()
    KEYWORD_SEARCH = auto()
    VECTOR_DIRECT = auto()
    GRAPH_ALGORITHMS = auto()
    GRAPH_NAVIGATION = auto()


class ConsistencyStatus(Enum):
    """Graph-vs-disk consistency status."""

    CONSISTENT = "consistent"
    PARTIAL = "partial"
    STALE = "stale"


@dataclass
class QueryMethodResult:
    """Results from a single query method."""

    method: QueryMethod
    items: list[dict[str, Any]]
    execution_time_ms: float
    error: str | None = None


@dataclass
class IntegrityWarning:
    """Warning about graph data integrity issues."""

    severity: Literal["hard", "soft"]
    item: str
    issue: str
    action: str


@dataclass
class CombinedQueryResult:
    """Combined results from multiple query methods."""

    query: str
    intent: QueryIntent
    intent_confidence: float = 0.0
    methods_used: list[QueryMethod] = field(default_factory=list)
    items: list[dict[str, Any]] = field(default_factory=list)
    execution_time_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    integrity_warnings: list[IntegrityWarning] = field(default_factory=list)
    integrity_check_count: int = 0
    integrity_pass_count: int = 0


# Intent keyword mapping for classification
_INTENT_KEYWORD_MAP: dict[QueryIntent, frozenset[str]] = {
    QueryIntent.STRUCTURAL: frozenset(
        {
            "call",
            "calls",
            "called by",
            "caller",
            "callee",
            "hierarchy",
            "inherit",
            "import",
            "depend",
        }
    ),
    QueryIntent.FUNCTIONAL: frozenset(
        {
            "how does",
            "how to",
            "what does",
            "why does",
            "explain",
            "work",
            "behavior",
        }
    ),
    QueryIntent.VALIDATION: frozenset(
        {
            "correct",
            "valid",
            "implement",
            "comply",
            "spec",
            "should",
            "must",
            "require",
        }
    ),
    QueryIntent.SEMANTIC: frozenset(
        {"similar", "like", "compare", "related", "same as", "function"}
    ),
}

# Primary/secondary method designation per intent
_INTENT_METHOD_MAP: dict[QueryIntent, tuple[list[QueryMethod], list[QueryMethod]]] = {
    QueryIntent.FUNCTIONAL: (
        [QueryMethod.SEMANTIC_SEARCH, QueryMethod.GRAPH_TRAVERSAL],
        [QueryMethod.KEYWORD_SEARCH, QueryMethod.VECTOR_DIRECT],
    ),
    QueryIntent.STRUCTURAL: (
        [QueryMethod.GRAPH_TRAVERSAL, QueryMethod.GRAPH_NAVIGATION],
        [QueryMethod.KEYWORD_SEARCH, QueryMethod.SEMANTIC_SEARCH],
    ),
    QueryIntent.SEMANTIC: (
        [QueryMethod.SEMANTIC_SEARCH, QueryMethod.VECTOR_DIRECT],
        [QueryMethod.GRAPH_TRAVERSAL, QueryMethod.KEYWORD_SEARCH],
    ),
    QueryIntent.EXPLORATORY: (
        [QueryMethod.SEMANTIC_SEARCH, QueryMethod.GRAPH_TRAVERSAL, QueryMethod.GRAPH_ALGORITHMS],
        [],
    ),
    QueryIntent.VALIDATION: (
        [QueryMethod.GRAPH_TRAVERSAL, QueryMethod.KEYWORD_SEARCH],
        [QueryMethod.SEMANTIC_SEARCH, QueryMethod.GRAPH_ALGORITHMS],
    ),
}


class BackendHealthCoordinator:
    """Consults existing health mechanisms before method execution.

    Uses the QueryProtocol interface's fetch_all() for graph health checks
    (works for both MemgraphIngestor and PooledMemgraphProxy), and delegates
    to concrete class health methods only when the concrete type is known.
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
        self._check_interval: float = 30.0
        self._cached_status: dict[str, bool] = {}
        self._vector_was_checked: bool = False

    def check_graph_health(self) -> bool:
        """Check graph backend health via the QueryProtocol interface."""
        try:
            result = self._graph.fetch_all("RETURN 1 AS health")
            return len(result) > 0
        except Exception:
            return False

    def check_vector_health(self) -> bool:
        """Check vector backend health via HybridRetriever._validate().

        Note: This is an intentional private-API dependency. HybridRetriever._validate()
        is private but provides the only comprehensive health check for the
        vector+embedding subsystem.
        """
        if self._hybrid is not None:
            start = time.time()
            result = self._hybrid._validate()
            latency = time.time() - start
            if latency > 1.0 and not self._vector_was_checked:
                logger.info(f"Vector health check cold-start latency: {latency:.1f}s (model loaded)")
                self._vector_was_checked = True
            return result
        return False

    def check_algorithm_health(self) -> bool:
        """Check MAGE algorithm availability via GraphAlgorithms.health_check()."""
        if self._algorithms is not None:
            return self._algorithms.health_check()
        return False

    def get_method_availability(self) -> dict[QueryMethod, bool]:
        """Map each query method to its backend availability."""
        graph_ok = self.check_graph_health()
        vector_ok = self.check_vector_health()
        algo_ok = self.check_algorithm_health()

        return {
            QueryMethod.SEMANTIC_SEARCH: vector_ok,
            QueryMethod.GRAPH_TRAVERSAL: graph_ok,
            QueryMethod.KEYWORD_SEARCH: graph_ok,
            QueryMethod.VECTOR_DIRECT: vector_ok,
            QueryMethod.GRAPH_NAVIGATION: graph_ok,
            QueryMethod.GRAPH_ALGORITHMS: algo_ok,
        }


def verify_graph_result_integrity(
    items: list[dict[str, Any]],
    repo_path: Path,
    max_checks: int = 5,
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
            warnings.append(
                IntegrityWarning(
                    severity="hard",
                    item=qualified_name,
                    issue=f"File {file_path} does not exist on disk",
                    action="Graph may be stale — consider re-indexing",
                )
            )
            continue

        # Check 2: Line number bounds
        try:
            with full_path.open(encoding="utf-8", errors="replace") as f:
                total_lines = sum(1 for _ in f)
            if start_line > total_lines or end_line > total_lines:
                warnings.append(
                    IntegrityWarning(
                        severity="soft",
                        item=qualified_name,
                        issue=f"Line range ({start_line}-{end_line}) exceeds file length ({total_lines})",
                        action="File may have been modified since indexing",
                    )
                )
        except Exception as e:
            warnings.append(
                IntegrityWarning(
                    severity="soft",
                    item=qualified_name,
                    issue=f"Could not verify file: {e}",
                    action="File may have been modified since indexing",
                )
            )

    return warnings


class QueryMethodOrchestrator:
    """Orchestrates multiple query methods for comprehensive retrieval.

    Analyzes query characteristics to determine which methods to use,
    executes them adaptively, merges results, and ranks them using
    combined scoring.

    Enhanced with:
    - Async-native execution for compatibility with async contexts
    - Backend health coordination
    - Circuit breaker for failing methods
    - Template-based Cypher fallback
    - Adaptive sequencing with early termination
    - Graph data integrity verification
    """

    def __init__(
        self,
        code_graph: QueryProtocol,
        code_vector: VectorBackend | None = None,
        doc_graph: QueryProtocol | None = None,
        doc_vector: VectorBackend | None = None,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        self.code_graph = code_graph
        self.code_vector = code_vector
        self.doc_graph = doc_graph
        self.doc_vector = doc_vector
        self._hybrid_retriever = hybrid_retriever
        self._circuit_breaker: dict[QueryMethod, int] = {}
        self._health_coordinator: BackendHealthCoordinator | None = None

    def _get_hybrid_retriever(self) -> HybridRetriever:
        """Lazy initialization of HybridRetriever."""
        if self._hybrid_retriever is None:
            from ..memgraph_advanced import create_hybrid_retriever

            self._hybrid_retriever = create_hybrid_retriever(self.code_graph)
        return self._hybrid_retriever

    def _get_health_coordinator(self) -> BackendHealthCoordinator:
        """Lazy initialization of BackendHealthCoordinator."""
        if self._health_coordinator is None:
            try:
                from ..graph_algorithms import GraphAlgorithms

                algo = GraphAlgorithms()
            except Exception:
                algo = None

            self._health_coordinator = BackendHealthCoordinator(
                graph_ingestor=self.code_graph,
                hybrid_retriever=self._hybrid_retriever,
                graph_algorithms=algo,
            )
        return self._health_coordinator

    def _check_graph_data_consistency(self) -> ConsistencyStatus:
        """Lightweight graph-vs-disk consistency check.

        Compares graph node counts with disk file counts.
        This is a quick heuristic for the health coordinator,
        not a full integrity scan.
        """
        try:
            # Count indexed files in graph
            graph_files = self.code_graph.fetch_all(
                "MATCH (n:File) RETURN count(n) AS file_count"
            )
            graph_count = graph_files[0].get("file_count", 0) if graph_files else 0

            # Count actual source files on disk (fast walk, no reading)
            repo_path = Path(settings.TARGET_REPO_PATH)
            disk_count = sum(
                1 for _ in repo_path.rglob("*") if _.is_file() and is_code_file(_)
            )

            # Allow 10% tolerance (graph may exclude files via .cgrignore)
            ratio = graph_count / max(disk_count, 1)
            if ratio < 0.5:
                return ConsistencyStatus.STALE  # Graph may need re-indexing
            if ratio < 0.9:
                return ConsistencyStatus.PARTIAL  # Some files not indexed
            return ConsistencyStatus.CONSISTENT
        except Exception as e:
            logger.debug(f"Graph data consistency check failed: {e}")
            return ConsistencyStatus.CONSISTENT  # Assume OK on error

    @staticmethod
    def classify_intent(query: str) -> QueryIntent:
        """Classify query intent based on keywords and patterns."""
        query_lower = query.lower()

        for intent, keywords in _INTENT_KEYWORD_MAP.items():
            if any(kw in query_lower for kw in keywords):
                return intent

        return QueryIntent.EXPLORATORY

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
                avg_keyword_len = (
                    sum(len(kw) for kw in keywords if kw in query_lower) / match_count
                )
                specificity_bonus = min(avg_keyword_len / 20.0, 0.3)
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

    def select_methods(
        self, intent: QueryIntent
    ) -> tuple[list[QueryMethod], list[QueryMethod]]:
        """Select appropriate query methods based on intent.

        Returns (primary_methods, secondary_methods).
        """
        return _INTENT_METHOD_MAP.get(
            intent, ([QueryMethod.SEMANTIC_SEARCH], [QueryMethod.KEYWORD_SEARCH])
        )

    def _is_method_available(self, method: QueryMethod) -> bool:
        """Check circuit breaker and health coordinator."""
        failures = self._circuit_breaker.get(method, 0)
        threshold = getattr(settings, "QUERY_CIRCUIT_BREAKER_THRESHOLD", 3)
        if failures >= threshold:
            return False
        if self._health_coordinator is None:
            return True
        return self._health_coordinator.get_method_availability().get(method, False)

    def _update_circuit_breaker(
        self, method: QueryMethod, result: QueryMethodResult
    ) -> None:
        """Update circuit breaker state based on result."""
        if result.error is None:
            self._circuit_breaker[method] = 0
        else:
            self._circuit_breaker[method] = self._circuit_breaker.get(method, 0) + 1

    async def _fetch_all_async(
        self, query: str, params: dict | None = None
    ) -> list[dict[str, Any]]:
        """Async helper for fetch_all that works with QueryProtocol."""
        # Use fetch_all_async if available (async-native)
        if hasattr(self.code_graph, "fetch_all_async"):
            return await self.code_graph.fetch_all_async(query, params)
        # Fallback to asyncio.to_thread for sync fetch_all
        return await asyncio.to_thread(self.code_graph.fetch_all, query, params)

    async def execute_method_async(
        self,
        method: QueryMethod,
        query: str,
        top_k: int = 5,
    ) -> QueryMethodResult:
        """Execute a single query method (async)."""
        start = time.time()

        try:
            match method:
                case QueryMethod.SEMANTIC_SEARCH:
                    return await self._execute_semantic_search(query, top_k, start)
                case QueryMethod.GRAPH_TRAVERSAL:
                    return await self._execute_graph_traversal(query, top_k, start)
                case QueryMethod.KEYWORD_SEARCH:
                    return await self._execute_keyword_search(query, top_k, start)
                case QueryMethod.VECTOR_DIRECT:
                    return await self._execute_vector_direct(query, top_k, start)
                case QueryMethod.GRAPH_NAVIGATION:
                    return await self._execute_graph_navigation(query, top_k, start)
                case QueryMethod.GRAPH_ALGORITHMS:
                    return await self._execute_graph_algorithms(query, top_k, start)
                case _:
                    return QueryMethodResult(
                        method=method,
                        items=[],
                        execution_time_ms=(time.time() - start) * 1000,
                        error=f"Method {method.name} not implemented",
                    )
        except Exception as e:
            return QueryMethodResult(
                method=method,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    async def _execute_semantic_search(
        self,
        query: str,
        top_k: int,
        start: float,
    ) -> QueryMethodResult:
        """Execute semantic search using HybridRetriever.

        Uses asyncio.to_thread() to wrap synchronous search() call
        for async-native execution in MCP server / pydantic-ai contexts.
        """
        retriever = self._get_hybrid_retriever()
        # Wrap synchronous call in asyncio.to_thread for async-native execution
        results = await asyncio.to_thread(retriever.search, query, top_k)
        items = [
            {
                "node_id": r.node_id,
                "qualified_name": r.qualified_name,
                "name": r.name,
                "type": r.node_type,
                "file_path": r.file_path,
                "start_line": getattr(r, "start_line", None),
                "end_line": getattr(r, "end_line", None),
                "similarity": r.combined_score,
            }
            for r in results
        ]
        return QueryMethodResult(
            method=QueryMethod.SEMANTIC_SEARCH,
            items=items,
            execution_time_ms=(time.time() - start) * 1000,
        )

    async def _execute_graph_traversal(
        self,
        query: str,
        top_k: int,
        start: float,
    ) -> QueryMethodResult:
        """Execute graph traversal with retry and template fallback.

        Uses async CypherGenerator.generate()/repair() and _fetch_all_async()
        for async-native execution in MCP server / pydantic-ai contexts.
        """
        from ..exceptions import LLMGenerationError
        from ..services.failure_classifier import classify_memgraph_failure
        from ..services.llm import CypherGenerator

        cypher_gen = CypherGenerator()
        cypher: str | None = None

        # Step 1: Try LLM-based generation (async-native)
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
                    if cypher:
                        cypher = await cypher_gen.repair(query, cypher, str(e))
                        results = await self._fetch_all_async(cypher)
                        return QueryMethodResult(
                            method=QueryMethod.GRAPH_TRAVERSAL,
                            items=results[:top_k],
                            execution_time_ms=(time.time() - start) * 1000,
                        )
                except Exception:
                    pass  # Fall through to template fallback

        # Step 2: Template fallback
        template_result = await self._try_template_fallback(query, top_k, start)
        if template_result is not None:
            return template_result

        # Step 3: Keyword fallback
        return await self._execute_keyword_search(query, top_k, start)

    async def _try_template_fallback(
        self, query: str, top_k: int, start: float
    ) -> QueryMethodResult | None:
        """Try to match query to a pre-built Cypher template.

        Template matching is based on intent classification + keyword extraction.
        Tries multiple templates in order based on query intent.
        Uses _fetch_all_async() for async-native database access.
        """
        if not getattr(settings, "QUERY_TEMPLATE_FALLBACK_ENABLED", True):
            return None

        from ..cypher_queries import CYPHER_QUERY_TEMPLATES
        from ..utils.query_utils import extract_best_keyword

        keyword = extract_best_keyword(query)
        if not keyword:
            return None

        # Determine which templates to try based on query content
        query_lower = query.lower()

        # Template selection based on query patterns
        templates_to_try: list[tuple[str, dict[str, Any]]] = []

        # Structural queries: callers/importers
        if any(kw in query_lower for kw in ["call", "caller", "calls", "called by"]):
            templates_to_try.append(("find_callers_of", {"qn": keyword}))

        if any(kw in query_lower for kw in ["import", "importer", "imports"]):
            templates_to_try.append(("find_importers_of", {"qn": keyword}))

        # Dependency queries
        if any(kw in query_lower for kw in ["depend", "uses", "reference"]):
            templates_to_try.append(("find_dependencies", {"keyword": keyword, "limit": top_k}))

        # Default: try find_by_name for any query
        templates_to_try.append(("find_by_name", {"keyword": keyword, "limit": top_k}))

        # Try each template in order
        for template_name, params in templates_to_try:
            template_entry = CYPHER_QUERY_TEMPLATES.get(template_name)
            if not template_entry:
                continue

            template_cypher, param_types = template_entry

            # For find_by_type, validate and interpolate the label placeholder
            if template_name == "find_by_type" and "label" in params:
                from ..constants import NodeLabel
                label = params.get("label", "")
                if label not in [e.value for e in NodeLabel]:
                    continue  # Skip invalid label
                template_cypher = template_cypher.replace("{label}", label)

            try:
                results = await self._fetch_all_async(template_cypher, params)
                if results:
                    return QueryMethodResult(
                        method=QueryMethod.GRAPH_TRAVERSAL,
                        items=results,
                        execution_time_ms=(time.time() - start) * 1000,
                    )
            except Exception as e:
                logger.debug(f"Template '{template_name}' fallback failed: {e}")

        return None

    async def _execute_keyword_search(
        self,
        query: str,
        top_k: int,
        start: float,
    ) -> QueryMethodResult:
        """Execute keyword-based search.

        Uses _fetch_all_async() for async-native database access.
        """
        from ..utils.query_utils import extract_keywords

        keywords = extract_keywords(query, max_keywords=3)
        if not keywords:
            return QueryMethodResult(
                method=QueryMethod.KEYWORD_SEARCH,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
            )

        # Use WHERE IN clause instead of | union syntax for label filtering.
        # This works correctly even when some labels don't exist in the graph.
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
        return QueryMethodResult(
            method=QueryMethod.KEYWORD_SEARCH,
            items=results,
            execution_time_ms=(time.time() - start) * 1000,
        )

    async def _execute_vector_direct(
        self,
        query: str,
        top_k: int,
        start: float,
    ) -> QueryMethodResult:
        """Execute direct vector search.

        Uses asyncio.to_thread() to wrap synchronous embedding generation
        and vector search for async-native execution in MCP server / pydantic-ai contexts.
        """
        if self.code_vector is None:
            return QueryMethodResult(
                method=QueryMethod.VECTOR_DIRECT,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
                error="Vector backend not available",
            )

        from ..embeddings import get_embedding_provider

        config = settings.active_embedding_config
        provider = get_embedding_provider(
            provider=config.provider,
            model_id=config.model_id,
        )
        # Wrap synchronous calls in asyncio.to_thread for async-native execution
        embedding = await asyncio.to_thread(provider.embed, query)
        pairs = await asyncio.to_thread(self.code_vector.search, embedding, top_k)

        if not pairs:
            return QueryMethodResult(
                method=QueryMethod.VECTOR_DIRECT,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
            )

        node_ids = [pair[0] for pair in pairs]
        similarity_map = {pair[0]: pair[1] for pair in pairs}

        cypher = """
        MATCH (n)
        WHERE id(n) IN $node_ids
        RETURN id(n) AS node_id,
               n.qualified_name AS qualified_name,
               n.name AS name,
               labels(n)[0] AS type,
               n.path AS file_path
        """
        records = await self._fetch_all_async(cypher, {"node_ids": node_ids})
        items = []
        for record in records:
            node_id = record.get("node_id")
            item = {
                "node_id": node_id,
                "qualified_name": record.get("qualified_name"),
                "name": record.get("name"),
                "type": record.get("type"),
                "file_path": record.get("file_path"),
                "similarity": similarity_map.get(node_id, 0.0),
            }
            items.append(item)

        return QueryMethodResult(
            method=QueryMethod.VECTOR_DIRECT,
            items=items,
            execution_time_ms=(time.time() - start) * 1000,
        )

    async def _execute_graph_navigation(
        self, query: str, top_k: int, start: float
    ) -> QueryMethodResult:
        """Execute graph navigation using pre-built Cypher queries.

        Uses _fetch_all_async() for async-native database access.
        """
        from ..cypher_queries import CYPHER_FIND_CALLERS, CYPHER_FIND_IMPORTERS
        from ..utils.query_utils import extract_keywords

        keywords = extract_keywords(query, max_keywords=3)
        if not keywords:
            return QueryMethodResult(
                method=QueryMethod.GRAPH_NAVIGATION,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
                error="No keyword extracted from query",
            )

        # First, find nodes matching any keyword
        # Use WHERE IN clause instead of | union syntax for label filtering
        find_nodes_cypher = """
        MATCH (n)
        WHERE labels(n)[0] IN ['Function', 'Class', 'Method']
          AND ANY(kw IN $keywords WHERE n.name CONTAINS kw OR n.qualified_name CONTAINS kw)
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type,
               n.path AS file_path
        LIMIT $limit
        """
        try:
            nodes = await self._fetch_all_async(
                find_nodes_cypher, {"keywords": keywords, "limit": top_k}
            )
            if not nodes:
                return QueryMethodResult(
                    method=QueryMethod.GRAPH_NAVIGATION,
                    items=[],
                    execution_time_ms=(time.time() - start) * 1000,
                )

            items = []
            per_node_limit = max(3, top_k // len(nodes))

            for node in nodes:
                qn = node.get("qualified_name")
                # Find callers
                callers = await self._fetch_all_async(
                    CYPHER_FIND_CALLERS, {"qn": qn}
                )
                for caller in callers[:per_node_limit]:
                    items.append(
                        {
                            "node_id": caller.get("node_id", node.get("node_id")),
                            "qualified_name": caller.get("qualified_name", qn),
                            "name": caller.get("name", node.get("name")),
                            "type": caller.get("type", node.get("type")),
                            "file_path": caller.get("path", node.get("file_path")),
                            "score": 0.7,
                        }
                    )

                # Find importers
                importers = await self._fetch_all_async(
                    CYPHER_FIND_IMPORTERS, {"qn": qn}
                )
                for importer in importers[:per_node_limit]:
                    items.append(
                        {
                            "node_id": importer.get("node_id", node.get("node_id")),
                            "qualified_name": importer.get("qualified_name", qn),
                            "name": importer.get("name", node.get("name")),
                            "type": "Module",
                            "file_path": importer.get("path", node.get("file_path")),
                            "score": 0.6,
                        }
                    )

                if len(items) >= top_k * 2:
                    break

            return QueryMethodResult(
                method=QueryMethod.GRAPH_NAVIGATION,
                items=items[:top_k],
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return QueryMethodResult(
                method=QueryMethod.GRAPH_NAVIGATION,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
                error=str(e),
            )

    async def _execute_graph_algorithms(
        self, query: str, top_k: int, start: float
    ) -> QueryMethodResult:
        """Execute graph algorithms using GraphAlgorithms.

        Uses asyncio.to_thread() to wrap synchronous MAGE algorithm calls
        for async-native execution in MCP server / pydantic-ai contexts.
        """
        from ..graph_algorithms import GraphAlgorithms
        from ..utils.query_utils import extract_keywords

        try:
            keywords = extract_keywords(query, max_keywords=3)
            if not keywords:
                return QueryMethodResult(
                    method=QueryMethod.GRAPH_ALGORITHMS,
                    items=[],
                    execution_time_ms=(time.time() - start) * 1000,
                )

            # Find node(s) matching any keyword
            # Use WHERE IN clause instead of | union syntax for label filtering
            find_cypher = """
            MATCH (n)
            WHERE labels(n)[0] IN ['Function', 'Class', 'Method']
              AND ANY(kw IN $keywords WHERE n.name CONTAINS kw OR n.qualified_name CONTAINS kw)
            RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
                   n.name AS name, labels(n)[0] AS type,
                   n.path AS file_path
            LIMIT 1
            """
            nodes = await self._fetch_all_async(find_cypher, {"keywords": keywords})
            if not nodes:
                return QueryMethodResult(
                    method=QueryMethod.GRAPH_ALGORITHMS,
                    items=[],
                    execution_time_ms=(time.time() - start) * 1000,
                )

            node = nodes[0]
            node_id = node.get("node_id")

            # Determine algorithm based on query content
            query_lower = query.lower()
            algo = GraphAlgorithms()

            if any(
                word in query_lower
                for word in ["similar", "like", "same as", "analogous"]
            ):
                # Structural similarity
                similar_nodes = await asyncio.to_thread(
                    algo.get_similar_nodes, node_id, top_k
                )
                items = [
                    {
                        "node_id": sn.get("node_id"),
                        "qualified_name": sn.get("qualified_name"),
                        "name": sn.get("name"),
                        "type": "Function",
                        "file_path": sn.get("file_path", ""),
                        "similarity": sn.get("jaccard_similarity", 0.0),
                        "score": sn.get("jaccard_similarity", 0.0),
                    }
                    for sn in similar_nodes
                ]
            elif any(
                word in query_lower
                for word in ["context", "neighbors", "around", "surrounding"]
            ):
                # BFS context
                bfs_nodes = await asyncio.to_thread(
                    algo.get_bfs_context, node_id, 3
                )
                items = [
                    {
                        "node_id": bn.get("node_id"),
                        "qualified_name": bn.get("qualified_name"),
                        "name": bn.get("name"),
                        "type": bn.get("type", "Function"),
                        "file_path": bn.get("file_path", ""),
                        "depth": bn.get("depth", 0),
                        "score": 0.9 - (bn.get("depth", 0) * 0.1),
                    }
                    for bn in bfs_nodes[:top_k]
                ]
            else:
                # Default: use similar nodes
                similar_nodes = await asyncio.to_thread(
                    algo.get_similar_nodes, node_id, top_k
                )
                items = [
                    {
                        "node_id": sn.get("node_id"),
                        "qualified_name": sn.get("qualified_name"),
                        "name": sn.get("name"),
                        "type": "Function",
                        "file_path": sn.get("file_path", ""),
                        "similarity": sn.get("jaccard_similarity", 0.0),
                        "score": sn.get("jaccard_similarity", 0.0),
                    }
                    for sn in similar_nodes
                ]

            return QueryMethodResult(
                method=QueryMethod.GRAPH_ALGORITHMS,
                items=items,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return QueryMethodResult(
                method=QueryMethod.GRAPH_ALGORITHMS,
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
        """Execute comprehensive query using multiple methods.

        Synchronous wrapper for backward compatibility.
        For async contexts, use execute_async() instead.
        """
        return asyncio.run(self.execute_async(query, top_k, max_methods))

    async def execute_async(
        self,
        query: str,
        top_k: int = 5,
        max_methods: int = 3,
    ) -> CombinedQueryResult:
        """Execute comprehensive query with adaptive sequencing.

        Args:
            query: Natural language query
            top_k: Maximum number of results to return
            max_methods: Maximum number of methods to try (for backward compat)

        Returns:
            CombinedQueryResult with merged items and metadata
        """
        min_methods = getattr(settings, "QUERY_MIN_METHODS", 2)
        confidence_threshold = getattr(settings, "QUERY_INTENT_CONFIDENCE_THRESHOLD", 0.5)
        enable_integrity = getattr(settings, "QUERY_ENABLE_INTEGRITY_CHECK", True)
        integrity_check_count = getattr(settings, "QUERY_INTEGRITY_CHECK_COUNT", 5)

        # Classify intent with confidence
        intent, confidence = self.classify_intent_with_confidence(query)
        primary_methods, secondary_methods = self.select_methods(intent)

        # Initialize health coordinator
        health = self._get_health_coordinator()

        # Health pre-check: filter out unavailable methods
        availability = health.get_method_availability()
        primary_methods = [m for m in primary_methods if availability.get(m, False)]
        secondary_methods = [m for m in secondary_methods if availability.get(m, False)]

        # If confidence is low, treat all methods as primary (exploratory)
        if confidence < confidence_threshold:
            primary_methods = [
                m for m in QueryMethod if availability.get(m, False)
            ]
            secondary_methods = []

        # Limit methods for backward compatibility
        if len(primary_methods) > max_methods:
            primary_methods = primary_methods[:max_methods]

        results: list[QueryMethodResult] = []
        sufficient = False

        # Stage 1: Execute primary methods
        for method in primary_methods:
            if not self._is_method_available(method):
                continue
            result = await self.execute_method_async(method, query, top_k)
            results.append(result)
            self._update_circuit_breaker(method, result)

            # Early termination check
            successful_results = [
                r for r in results if r.error is None and len(r.items) > 0
            ]
            if (
                len(successful_results) >= min_methods
                and self._results_cover_query(query, successful_results)
            ):
                sufficient = True
                break

        # Stage 2: If not sufficient, execute secondary methods
        if not sufficient:
            for method in secondary_methods:
                if not self._is_method_available(method):
                    continue
                result = await self.execute_method_async(method, query, top_k)
                results.append(result)
                self._update_circuit_breaker(method, result)

                successful_results = [
                    r for r in results if r.error is None and len(r.items) > 0
                ]
                if len(successful_results) >= min_methods:
                    break

        # Stage 3: Merge, rank, and integrity spot-check
        merged_items = self._merge_and_rank(results, top_k)

        integrity_warnings: list[IntegrityWarning] = []
        if enable_integrity and merged_items:
            repo_path = Path(settings.TARGET_REPO_PATH)
            integrity_warnings = verify_graph_result_integrity(
                merged_items, repo_path, max_checks=integrity_check_count
            )

        # Build final result
        total_time = sum(r.execution_time_ms for r in results)
        errors = [r.error for r in results if r.error]

        hard_warnings = [w for w in integrity_warnings if w.severity == "hard"]

        return CombinedQueryResult(
            query=query,
            intent=intent,
            intent_confidence=confidence,
            methods_used=[r.method for r in results],
            items=merged_items,
            execution_time_ms=total_time,
            warnings=errors,
            integrity_warnings=integrity_warnings,
            integrity_check_count=min(integrity_check_count, len(merged_items)),
            integrity_pass_count=min(integrity_check_count, len(merged_items))
            - len(hard_warnings),
        )

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

        unique_files = len(
            set(
                item.get("file_path", "")
                for r in results
                for item in r.items
                if item.get("file_path")
            )
        )
        if unique_files < 2 and total_items < 5:
            return False  # All from one file — likely incomplete

        return True

    def _merge_and_rank(
        self,
        results: list[QueryMethodResult],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Merge results from multiple methods and rank them."""
        merged: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"scores": [], "sources": [], "item": None}
        )

        for result in results:
            for item in result.items:
                key = item.get("qualified_name") or str(item.get("node_id", ""))
                if not key:
                    continue

                score = item.get("similarity") or item.get("score", 0.5)

                merged[key]["scores"].append(score)
                merged[key]["sources"].append(result.method.name)
                if merged[key]["item"] is None:
                    merged[key]["item"] = item

        ranked: list[dict[str, Any]] = []
        for key, data in merged.items():
            item = data["item"]
            avg_score = sum(data["scores"]) / len(data["scores"])
            method_count = len(data["sources"])

            combined_score = avg_score * (1 + 0.2 * (method_count - 1))

            item["combined_score"] = combined_score
            item["found_by_methods"] = data["sources"]
            item["method_count"] = method_count
            ranked.append(item)

        ranked.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        return ranked[:top_k]
