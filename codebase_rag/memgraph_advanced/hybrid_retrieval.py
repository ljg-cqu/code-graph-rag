"""Hybrid retrieval combining vector, text, and graph signals."""

from dataclasses import dataclass
from typing import Any, Optional

from ..graph_algorithms import GraphAlgorithms
from ..services.graph_service import MemgraphIngestor
from ..config import settings


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""

    node_id: int
    name: str
    qualified_name: str
    node_type: str
    vector_score: float
    text_score: float
    pagerank_score: float
    community_score: float
    final_score: float
    metadata: dict[str, Any]


class HybridRetriever:
    """Multi-modal retrieval combining vector, text, and graph signals."""

    def __init__(self, algorithms: Optional[GraphAlgorithms] = None):
        self.algo = algorithms or GraphAlgorithms()
        self.default_weights = {
            "vector": 0.6,
            "text": 0.2,
            "pagerank": 0.15,
            "community": 0.05,
        }

    def search(
        self,
        query_embedding: list[float],
        keywords: list[str],
        top_k: int = 10,
        weights: Optional[dict[str, float]] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[HybridSearchResult]:
        """
        Execute hybrid search combining multiple signals.

        Args:
            query_embedding: Vector representation of query
            keywords: Text keywords for text search
            top_k: Number of results to return
            weights: Optional custom weights for scoring components
            filters: Optional filters (e.g., project_prefix, node_types)

        Returns:
            List of hybrid search results ranked by combined score
        """
        final_weights = {**self.default_weights, **(weights or {})}

        cypher = """
        WITH $embedding AS query_vec, $keywords AS keywords, $top_k AS top_k, $weights AS weights

        // Step 1: Parallel vector searches across multiple indexes
        CALL vector_search.search("function_embedding_index", top_k * 2, query_vec)
        YIELD node AS vf, similarity AS sim_f

        UNION ALL

        CALL vector_search.search("class_embedding_index", top_k * 2, query_vec)
        YIELD node AS vc, similarity AS sim_c

        WITH collect({node: coalesce(vf, vc), sim: coalesce(sim_f, sim_c)}) AS vector_results,
             weights, keywords, top_k

        // Step 2: Text search for keyword matches
        OPTIONAL UNWIND keywords AS keyword
        OPTIONAL MATCH (t:Function|Class|Method)
        WHERE t.name CONTAINS keyword OR t.docstring CONTAINS keyword
        WITH vector_results, collect(DISTINCT t) AS text_results, weights, top_k

        // Step 3: Merge and rank with graph centrality
        UNWIND vector_results AS vr
        WITH vr.node AS n, vr.sim AS vector_score, text_results, weights, top_k
        WHERE n:Function OR n:Class OR n:Method

        // Apply filters if provided
        {% if filters and filters.get('project_prefix') %}
        WHERE n.qualified_name STARTS WITH $filters.project_prefix
        {% endif %}
        {% if filters and filters.get('node_types') %}
        WHERE any(label IN labels(n) WHERE label IN $filters.node_types)
        {% endif %}

        // Check if also in text results (boost score)
        WITH n, vector_score,
             CASE WHEN n IN text_results THEN 1.0 ELSE 0.0 END AS text_match,
             COALESCE(n.pagerank_score, 0.1) AS pagerank,
             COALESCE(n.community_importance, 0.0) AS community_score,
             weights, top_k

        // Final ranking formula
        WITH n,
             vector_score,
             text_match,
             pagerank,
             community_score,
             (vector_score * weights.vector) + 
             (text_match * weights.text) + 
             (pagerank * weights.pagerank) + 
             (community_score * weights.community) AS final_score
        ORDER BY final_score DESC
        LIMIT top_k

        RETURN id(n) AS node_id, 
               n.name AS name, 
               n.qualified_name AS qualified_name,
               labels(n)[0] AS node_type,
               vector_score,
               text_match AS text_score,
               pagerank AS pagerank_score,
               community_score AS community_score,
               final_score
        """

        params = {
            "embedding": query_embedding,
            "keywords": keywords,
            "top_k": top_k,
            "weights": final_weights,
            "filters": filters or {},
        }

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        ) as ingestor:
            records = ingestor.fetch_all(cypher, params)

        return [
            HybridSearchResult(
                node_id=record["node_id"],
                name=record["name"],
                qualified_name=record["qualified_name"],
                node_type=record["node_type"],
                vector_score=record["vector_score"],
                text_score=record["text_score"],
                pagerank_score=record["pagerank_score"],
                community_score=record["community_score"],
                final_score=record["final_score"],
                metadata={},
            )
            for record in records
        ]
