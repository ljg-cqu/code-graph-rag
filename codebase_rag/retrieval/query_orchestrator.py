"""Query method orchestration for multi-paradigm retrieval.

Orchestrates multiple query methods (semantic search, graph traversal,
keyword search, etc.) based on query intent and combines results with
unified ranking.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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


@dataclass
class QueryMethodResult:
    """Results from a single query method."""

    method: QueryMethod
    items: list[dict[str, Any]]
    execution_time_ms: float
    error: str | None = None


@dataclass
class CombinedQueryResult:
    """Combined results from multiple query methods."""

    query: str
    intent: QueryIntent
    methods_used: list[QueryMethod]
    items: list[dict[str, Any]]
    execution_time_ms: float
    warnings: list[str] = field(default_factory=list)


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

_INTENT_METHOD_MAP: dict[QueryIntent, list[QueryMethod]] = {
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
        hybrid_retriever: Any | None = None,
    ) -> None:
        self.code_graph = code_graph
        self.code_vector = code_vector
        self.doc_graph = doc_graph
        self.doc_vector = doc_vector
        self._hybrid_retriever = hybrid_retriever

    def _get_hybrid_retriever(self) -> Any:
        """Lazy initialization of HybridRetriever."""
        if self._hybrid_retriever is None:
            from ..memgraph_advanced import create_hybrid_retriever

            self._hybrid_retriever = create_hybrid_retriever(self.code_graph)
        return self._hybrid_retriever

    @staticmethod
    def classify_intent(query: str) -> QueryIntent:
        """Classify query intent based on keywords and patterns."""
        query_lower = query.lower()

        for intent, keywords in _INTENT_KEYWORD_MAP.items():
            if any(kw in query_lower for kw in keywords):
                return intent

        return QueryIntent.EXPLORATORY

    def select_methods(self, intent: QueryIntent) -> list[QueryMethod]:
        """Select appropriate query methods based on intent."""
        return _INTENT_METHOD_MAP.get(intent, [QueryMethod.SEMANTIC_SEARCH])

    def execute_method(
        self,
        method: QueryMethod,
        query: str,
        top_k: int = 5,
    ) -> QueryMethodResult:
        """Execute a single query method."""
        start = time.time()

        try:
            match method:
                case QueryMethod.SEMANTIC_SEARCH:
                    return self._execute_semantic_search(query, top_k, start)
                case QueryMethod.GRAPH_TRAVERSAL:
                    return self._execute_graph_traversal(query, top_k, start)
                case QueryMethod.KEYWORD_SEARCH:
                    return self._execute_keyword_search(query, top_k, start)
                case QueryMethod.VECTOR_DIRECT:
                    return self._execute_vector_direct(query, top_k, start)
                case QueryMethod.GRAPH_NAVIGATION:
                    return self._execute_graph_navigation(query, top_k, start)
                case QueryMethod.GRAPH_ALGORITHMS:
                    return self._execute_graph_algorithms(query, top_k, start)
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

    def _execute_semantic_search(
        self,
        query: str,
        top_k: int,
        start: float,
    ) -> QueryMethodResult:
        """Execute semantic search using HybridRetriever."""
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
            method=QueryMethod.SEMANTIC_SEARCH,
            items=items,
            execution_time_ms=(time.time() - start) * 1000,
        )

    def _execute_graph_traversal(
        self,
        query: str,
        top_k: int,
        start: float,
    ) -> QueryMethodResult:
        """Execute graph traversal using LLM-generated Cypher."""
        from ..services.llm import CypherGenerator

        cypher_gen = CypherGenerator()
        import asyncio

        cypher = asyncio.run(cypher_gen.generate(query))
        results = self.code_graph.fetch_all(cypher)
        return QueryMethodResult(
            method=QueryMethod.GRAPH_TRAVERSAL,
            items=results[:top_k],
            execution_time_ms=(time.time() - start) * 1000,
        )

    def _execute_keyword_search(
        self,
        query: str,
        top_k: int,
        start: float,
    ) -> QueryMethodResult:
        """Execute keyword-based search."""
        from ..utils.query_utils import extract_best_keyword

        keyword = extract_best_keyword(query)
        cypher = """
        MATCH (n:Function|Class|Method)
        WHERE n.name CONTAINS $keyword
           OR n.qualified_name CONTAINS $keyword
           OR n.docstring CONTAINS $keyword
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type,
               n.path AS file_path, n.start_line AS start_line,
               n.end_line AS end_line
        LIMIT $limit
        """
        results = self.code_graph.fetch_all(
            cypher, {"keyword": keyword, "limit": top_k}
        )
        return QueryMethodResult(
            method=QueryMethod.KEYWORD_SEARCH,
            items=results,
            execution_time_ms=(time.time() - start) * 1000,
        )

    def _execute_vector_direct(
        self,
        query: str,
        top_k: int,
        start: float,
    ) -> QueryMethodResult:
        """Execute direct vector search."""
        if self.code_vector is None:
            return QueryMethodResult(
                method=QueryMethod.VECTOR_DIRECT,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
                error="Vector backend not available",
            )

        from ..config import settings
        from ..embeddings import get_embedding_provider

        config = settings.active_embedding_config
        provider = get_embedding_provider(
            provider=config.provider,
            model_id=config.model_id,
        )
        embedding = provider.embed(query)
        pairs = self.code_vector.search(embedding, top_k=top_k)

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
        records = self.code_graph.fetch_all(cypher, {"node_ids": node_ids})
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

    def _execute_graph_navigation(self, query: str, top_k: int, start: float) -> QueryMethodResult:
        """Execute graph navigation using GraphNavigator tools."""
        from ..utils.query_utils import extract_best_keyword
        from ..cypher_queries import CYPHER_FIND_CALLERS, CYPHER_FIND_IMPORTERS

        keyword = extract_best_keyword(query)
        if not keyword:
            return QueryMethodResult(
                method=QueryMethod.GRAPH_NAVIGATION,
                items=[],
                execution_time_ms=(time.time() - start) * 1000,
                error="No keyword extracted from query",
            )

        # First, find nodes matching the keyword
        find_nodes_cypher = """
        MATCH (n:Function|Class|Method)
        WHERE n.name CONTAINS $keyword OR n.qualified_name CONTAINS $keyword
        RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
               n.name AS name, labels(n)[0] AS type,
               n.path AS file_path
        LIMIT $limit
        """
        try:
            nodes = self.code_graph.fetch_all(
                find_nodes_cypher, {"keyword": keyword, "limit": min(top_k, 3)}
            )
            if not nodes:
                return QueryMethodResult(
                    method=QueryMethod.GRAPH_NAVIGATION,
                    items=[],
                    execution_time_ms=(time.time() - start) * 1000,
                )

            items = []
            for node in nodes:
                qn = node.get("qualified_name")
                # Find callers
                callers = self.code_graph.fetch_all(CYPHER_FIND_CALLERS, {"qn": qn})
                for caller in callers[:2]:  # Limit per node
                    items.append({
                        "node_id": caller.get("node_id", node.get("node_id")),
                        "qualified_name": caller.get("qualified_name", qn),
                        "name": caller.get("name", node.get("name")),
                        "type": caller.get("type", node.get("type")),
                        "file_path": caller.get("path", node.get("file_path")),
                        "score": 0.7,  # Navigation results get medium score
                    })

                # Find importers
                importers = self.code_graph.fetch_all(CYPHER_FIND_IMPORTERS, {"qn": qn})
                for importer in importers[:2]:
                    items.append({
                        "node_id": importer.get("node_id", node.get("node_id")),
                        "qualified_name": importer.get("qualified_name", qn),
                        "name": importer.get("name", node.get("name")),
                        "type": "Module",
                        "file_path": importer.get("path", node.get("file_path")),
                        "score": 0.6,
                    })

                if len(items) >= top_k:
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

    def _execute_graph_algorithms(self, query: str, top_k: int, start: float) -> QueryMethodResult:
        """Execute graph algorithms using GraphAlgorithms."""
        from ..graph_algorithms import GraphAlgorithms
        from ..utils.query_utils import extract_best_keyword

        try:
            keyword = extract_best_keyword(query)
            if not keyword:
                return QueryMethodResult(
                    method=QueryMethod.GRAPH_ALGORITHMS,
                    items=[],
                    execution_time_ms=(time.time() - start) * 1000,
                )

            # Find node(s) matching the keyword
            find_cypher = """
            MATCH (n:Function|Class|Method)
            WHERE n.name CONTAINS $keyword OR n.qualified_name CONTAINS $keyword
            RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
                   n.name AS name, labels(n)[0] AS type,
                   n.path AS file_path
            LIMIT 1
            """
            nodes = self.code_graph.fetch_all(find_cypher, {"keyword": keyword})
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

            if any(word in query_lower for word in ["similar", "like", "same as", "analogous"]):
                # Structural similarity
                similar_nodes = algo.get_similar_nodes(node_id, top_k=top_k)
                items = [
                    {
                        "node_id": sn.get("node_id"),
                        "qualified_name": sn.get("qualified_name"),
                        "name": sn.get("name"),
                        "type": "Function",  # Default
                        "file_path": sn.get("file_path", ""),
                        "similarity": sn.get("jaccard_similarity", 0.0),
                        "score": sn.get("jaccard_similarity", 0.0),
                    }
                    for sn in similar_nodes
                ]
            elif any(word in query_lower for word in ["context", "neighbors", "around", "surrounding"]):
                # BFS context
                bfs_nodes = algo.get_bfs_context(node_id, max_depth=3)
                items = [
                    {
                        "node_id": bn.get("node_id"),
                        "qualified_name": bn.get("qualified_name"),
                        "name": bn.get("name"),
                        "type": bn.get("type", "Function"),
                        "file_path": bn.get("file_path", ""),
                        "depth": bn.get("depth", 0),
                        "score": 0.9 - (bn.get("depth", 0) * 0.1),  # Higher score for closer nodes
                    }
                    for bn in bfs_nodes[:top_k]
                ]
            else:
                # Default: use similar nodes
                similar_nodes = algo.get_similar_nodes(node_id, top_k=top_k)
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
        """Execute comprehensive query using multiple methods."""
        intent = self.classify_intent(query)
        methods = self.select_methods(intent)[:max_methods]

        results: list[QueryMethodResult] = []
        for method in methods:
            result = self.execute_method(method, query, top_k)
            results.append(result)

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
