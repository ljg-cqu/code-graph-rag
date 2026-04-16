"""Advanced path analysis for code relationships."""

from dataclasses import dataclass
from enum import Enum

from ..config import settings
from ..services.graph_service import MemgraphIngestor


class PathType(Enum):
    CALL_CHAIN = "call_chain"
    DEPENDENCY = "dependency"
    INHERITANCE = "inheritance"
    IMPORT = "import"


@dataclass
class PathAnalysis:
    """Analysis of paths between code entities."""

    path_type: PathType
    paths: list[dict]
    shortest_path_length: int
    alternative_paths: int
    critical_nodes: list[str]
    bottlenecks: list[str]


class PathAnalyzer:
    """Analyze paths between code entities."""

    def analyze_call_chain(
        self, start_qn: str, end_qn: str, max_paths: int = 3, max_path_length: int = 10
    ) -> PathAnalysis:
        """
        Analyze call chains between two functions.

        Uses k-shortest paths to find multiple call chains.
        """
        cypher = """
        MATCH (start:Function {qualified_name: $start_qn}),
              (end:Function {qualified_name: $end_qn})

        // Find k shortest paths using CALLS relationship
        MATCH path = (start)-[:CALLS *KSHORTEST $max_paths ..$max_length]->(end)

        WITH path,
             length(path) AS path_length,
             [n IN nodes(path) | {
                 qualified_name: n.qualified_name,
                 name: n.name,
                 type: labels(n)[0],
                 pagerank: COALESCE(n.pagerank_score, 0)
             }] AS node_details,
             [r IN relationships(path) | type(r)] AS relationship_types

        // Find critical nodes (high betweenness - appear in multiple paths)
        WITH collect({
            path: path,
            length: path_length,
            nodes: node_details,
            rels: relationship_types
        }) AS all_paths

        UNWIND all_paths AS p
        UNWIND p.nodes AS node
        WITH all_paths, node, count(*) AS node_frequency
        WHERE node_frequency > 1

        RETURN all_paths,
               collect(DISTINCT node.qualified_name) AS critical_nodes,
               size(all_paths) AS total_paths,
               reduce(min_len = 1000, p IN all_paths | CASE WHEN p.length < min_len THEN p.length ELSE min_len END) AS shortest_length
        """

        params = {
            "start_qn": start_qn,
            "end_qn": end_qn,
            "max_paths": max_paths,
            "max_length": max_path_length,
        }

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        ) as ingestor:
            results = ingestor.fetch_all(cypher, params)
            record = results[0] if results else None

        if not record:
            return PathAnalysis(
                path_type=PathType.CALL_CHAIN,
                paths=[],
                shortest_path_length=0,
                alternative_paths=0,
                critical_nodes=[],
                bottlenecks=[],
            )

        # Find bottlenecks separately
        bottlenecks = self.find_bottlenecks(start_qn)

        return PathAnalysis(
            path_type=PathType.CALL_CHAIN,
            paths=record["all_paths"],
            shortest_path_length=record["shortest_length"],
            alternative_paths=record["total_paths"],
            critical_nodes=record["critical_nodes"],
            bottlenecks=[b["function_name"] for b in bottlenecks],
        )

    def find_bottlenecks(
        self, function_qn: str | None = None, threshold: float = 0.01
    ) -> list[dict]:
        """
        Find functions that are bottlenecks (high betweenness centrality).

        Uses betweenness centrality to identify critical functions that
        control flow between different parts of the codebase.
        """
        cypher = """
        // Use betweenness centrality to find bottlenecks
        CALL betweenness_centrality.get("CALLS", "BOTH")
        YIELD node, betweenness

        WHERE node:Function AND betweenness > $threshold
        {% if function_qn %}
        AND (node)-[:CALLS*]->(:Function {qualified_name: $function_qn})
        OR (:Function {qualified_name: $function_qn})-[:CALLS*]->(node)
        {% endif %}

        // Get additional context
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
            records = ingestor.fetch_all(cypher, params)

        return records

    def find_similar_functions(
        self, target_qn: str, min_similarity: float = 0.3, limit: int = 10
    ) -> list[dict]:
        """Find structurally similar functions based on call patterns using Jaccard similarity."""
        cypher = """
        // Find functions similar to a given function based on call patterns
        MATCH (f:Function {qualified_name: $target_qn})-[:CALLS]->(target_calls)
        WITH f, collect(DISTINCT id(target_calls)) AS f_calls

        MATCH (candidate:Function)-[:CALLS]->(callee)
        WHERE candidate <> f
        WITH f, f_calls, candidate, collect(DISTINCT id(callee)) AS c_calls

        // Calculate Jaccard similarity
        WITH f, candidate,
             size([x IN f_calls WHERE x IN c_calls]) AS intersection,
             size(f_calls) + size(c_calls) - size([x IN f_calls WHERE x IN c_calls]) AS union

        WITH f, candidate, toFloat(intersection) / union AS jaccard_score
        WHERE jaccard_score > $min_similarity

        RETURN candidate.qualified_name AS similar_function,
               candidate.name AS name,
               jaccard_score
        ORDER BY jaccard_score DESC
        LIMIT $limit;
        """

        params = {
            "target_qn": target_qn,
            "min_similarity": min_similarity,
            "limit": limit,
        }

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        ) as ingestor:
            return ingestor.fetch_all(cypher, params)
