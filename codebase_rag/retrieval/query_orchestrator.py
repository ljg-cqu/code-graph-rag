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
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger

from ..config import settings
from ..orchestrator.llm_query_planner import (
    GraphAlgorithm,
    LLMQueryPlanner,
    QueryIntent,
    QueryMethod,
    QueryPlan,
)
from ..shared.utils.file_classifier import is_code_file

if TYPE_CHECKING:
    from ..graph_algorithms import GraphAlgorithms
    from ..memgraph_advanced.hybrid_retrieval import HybridRetriever
    from ..services import QueryProtocol
    from ..vector_backend import VectorBackend


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
        plan: QueryPlan | None = None,
    ) -> QueryMethodResult:
        """Execute a single query method (async)."""
        start = time.time()

        try:
            match method:
                case QueryMethod.SEMANTIC_SEARCH:
                    return await self._execute_semantic_search(query, top_k, start)
                case QueryMethod.GRAPH_TRAVERSAL:
                    return await self._execute_graph_traversal(query, top_k, start, plan)
                case QueryMethod.KEYWORD_SEARCH:
                    return await self._execute_keyword_search(query, top_k, start, plan)
                case QueryMethod.VECTOR_DIRECT:
                    return await self._execute_vector_direct(query, top_k, start)
                case QueryMethod.GRAPH_NAVIGATION:
                    return await self._execute_graph_navigation(query, top_k, start, plan)
                case QueryMethod.GRAPH_ALGORITHMS:
                    return await self._execute_graph_algorithms(query, top_k, start, plan)
                case _:
                    return QueryMethodResult(
                        method=method,
                        items=[],
                        execution_time_ms=(time.time() - start) * 1000,
                        error=f"Method {method.value} not implemented",
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
        plan: QueryPlan | None,
    ) -> QueryMethodResult:
        """Execute graph traversal with retry and LLM fallback chain.

        Fallback chain:
        1. LLM-based Cypher generation
        1b. Cypher repair on syntax errors
        2. Conservative Cypher fallback (generate_fallback)
        3. Keyword search using plan.expected_entities
        4. Semantic fallback
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
                    pass  # Fall through to fallback chain

        # Step 2: Conservative Cypher fallback
        try:
            cypher = await cypher_gen.generate_fallback(query)
            results = await self._fetch_all_async(cypher)
            return QueryMethodResult(
                method=QueryMethod.GRAPH_TRAVERSAL,
                items=results[:top_k],
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.warning(f"Conservative Cypher fallback failed: {e}")

        # Step 3: Keyword search using LLM-extracted entities
        if plan and plan.expected_entities:
            return await self._execute_keyword_search(query, top_k, start, plan)

        # Step 4: Semantic fallback
        return await self._execute_semantic_search(query, top_k, start)

    async def _execute_keyword_search(
        self,
        query: str,
        top_k: int,
        start: float,
        plan: QueryPlan | None = None,
    ) -> QueryMethodResult:
        """Execute keyword-based search using LLM-extracted entities.

        Uses _fetch_all_async() for async-native database access.
        """
        keywords = [kw.lower() for kw in plan.expected_entities[:3]] if plan and plan.expected_entities else []
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
              toLower(n.name) CONTAINS kw
              OR toLower(n.qualified_name) CONTAINS kw
              OR toLower(COALESCE(n.docstring, '')) CONTAINS kw)
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
        provider = get_embedding_provider(config=config)
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
        self,
        query: str,
        top_k: int,
        start: float,
        plan: QueryPlan | None = None,
    ) -> QueryMethodResult:
        """Execute graph navigation using pre-built Cypher queries.

        Uses _fetch_all_async() for async-native database access.
        """
        from ..cypher_queries import CYPHER_FIND_CALLERS, CYPHER_FIND_IMPORTERS

        keywords = [kw.lower() for kw in plan.expected_entities[:3]] if plan and plan.expected_entities else []
        if not keywords:
            return QueryMethodResult(
                method=QueryMethod.GRAPH_NAVIGATION,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
                error="No entities extracted from query",
            )

        # First, find nodes matching any keyword
        # Use WHERE IN clause instead of | union syntax for label filtering
        find_nodes_cypher = """
        MATCH (n)
        WHERE labels(n)[0] IN ['Function', 'Class', 'Method']
          AND ANY(kw IN $keywords WHERE toLower(n.name) CONTAINS kw OR toLower(n.qualified_name) CONTAINS kw)
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
        self,
        query: str,
        top_k: int,
        start: float,
        plan: QueryPlan | None = None,
    ) -> QueryMethodResult:
        """Execute graph algorithms based on LLM plan.

        Algorithm selection is mechanical: the LLM outputs plan.algorithm
        as a structured enum, and Python executes the corresponding method.
        No keyword matching on reasoning text.
        """
        from ..graph_algorithms import GraphAlgorithms

        if not self._health_coordinator:
            return QueryMethodResult(
                method=QueryMethod.GRAPH_ALGORITHMS,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
                error="Graph algorithms unavailable",
            )

        entities = plan.expected_entities if plan else []
        algorithm = plan.algorithm if plan else GraphAlgorithm.NONE

        try:
            algo = GraphAlgorithms()
            items: list[dict[str, Any]] = []

            match algorithm:
                case GraphAlgorithm.SHORTEST_PATH:
                    if len(entities) >= 2:
                        result = await asyncio.to_thread(
                            algo.find_shortest_path,
                            entities[0],
                            entities[1],
                        )
                        if result:
                            items = [result]

                case GraphAlgorithm.SIMILARITY:
                    if entities:
                        # Find node by name first
                        find_cypher = """
                        MATCH (n)
                        WHERE labels(n)[0] IN ['Function', 'Class', 'Method']
                          AND (n.name = $name OR n.qualified_name CONTAINS $name)
                        RETURN id(n) AS node_id
                        LIMIT 1
                        """
                        nodes = await self._fetch_all_async(
                            find_cypher, {"name": entities[0]}
                        )
                        if nodes:
                            similar = await asyncio.to_thread(
                                algo.get_similar_nodes,
                                nodes[0].get("node_id"),
                                top_k,
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
                                for sn in similar
                            ]

                case GraphAlgorithm.COMMUNITY:
                    communities = await asyncio.to_thread(
                        algo.detect_communities,
                    )
                    items = communities

                case GraphAlgorithm.BFS:
                    if entities:
                        find_cypher = """
                        MATCH (n)
                        WHERE labels(n)[0] IN ['Function', 'Class', 'Method']
                          AND (n.name = $name OR n.qualified_name CONTAINS $name)
                        RETURN id(n) AS node_id
                        LIMIT 1
                        """
                        nodes = await self._fetch_all_async(
                            find_cypher, {"name": entities[0]}
                        )
                        if nodes:
                            bfs = await asyncio.to_thread(
                                algo.get_bfs_context,
                                nodes[0].get("node_id"),
                                3,
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
                                for bn in bfs[:top_k]
                            ]

                case GraphAlgorithm.NONE | _:
                    # Default: similarity search using first entity
                    if entities:
                        find_cypher = """
                        MATCH (n)
                        WHERE labels(n)[0] IN ['Function', 'Class', 'Method']
                          AND (n.name = $name OR n.qualified_name CONTAINS $name)
                        RETURN id(n) AS node_id
                        LIMIT 1
                        """
                        nodes = await self._fetch_all_async(
                            find_cypher, {"name": entities[0]}
                        )
                        if nodes:
                            similar = await asyncio.to_thread(
                                algo.get_similar_nodes,
                                nodes[0].get("node_id"),
                                top_k,
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
                                for sn in similar
                            ]

            return QueryMethodResult(
                method=QueryMethod.GRAPH_ALGORITHMS,
                items=items,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"Graph algorithm execution failed: {e}")
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

        Uses LLM-driven planning instead of keyword-based intent classification.

        Args:
            query: Natural language query
            top_k: Maximum number of results to return
            max_methods: Maximum number of methods to try (for backward compat)

        Returns:
            CombinedQueryResult with merged items and metadata
        """
        start = time.time()
        min_methods = getattr(settings, "QUERY_MIN_METHODS", 2)
        enable_integrity = getattr(settings, "QUERY_ENABLE_INTEGRITY_CHECK", True)
        integrity_check_count = getattr(settings, "QUERY_INTEGRITY_CHECK_COUNT", 5)

        # LLM-driven planning replaces keyword-based intent classification
        planner = LLMQueryPlanner()
        plan = await planner.plan(query)

        methods_to_run = plan.methods[:max_methods]
        if not methods_to_run:
            methods_to_run = [QueryMethod.SEMANTIC_SEARCH]

        # Initialize health coordinator
        health = self._get_health_coordinator()
        availability = health.get_method_availability()
        methods_to_run = [m for m in methods_to_run if availability.get(m, False)]

        results: list[QueryMethodResult] = []

        # Execute primary methods
        for method in methods_to_run:
            if not self._is_method_available(method):
                continue
            result = await self.execute_method_async(method, query, top_k, plan)
            results.append(result)
            self._update_circuit_breaker(method, result)

            successful_results = [
                r for r in results if r.error is None and len(r.items) > 0
            ]
            if (
                len(successful_results) >= min_methods
                and self._results_cover_query(query, successful_results)
            ):
                break

        # If primary methods returned empty, try fallbacks
        if not any(r.items for r in results) and plan.fallback_methods:
            logger.info(f"Primary methods returned empty, trying fallbacks: {plan.fallback_methods[:2]}")
            for method in plan.fallback_methods[:2]:
                if not availability.get(method, False):
                    continue
                if not self._is_method_available(method):
                    continue
                result = await self.execute_method_async(method, query, top_k, plan)
                results.append(result)
                self._update_circuit_breaker(method, result)

        # Final fallback: semantic search if everything else failed
        if not any(r.items for r in results):
            logger.info("All methods returned empty, falling back to semantic search")
            result = await self._execute_semantic_search(query, top_k, start)
            results.append(result)

        # Merge, rank, and integrity spot-check
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
            intent=plan.intent or QueryIntent.EXPLORATORY,
            intent_confidence=1.0 if plan.intent else 0.0,
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
                merged[key]["sources"].append(result.method.value)
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
