"""Memgraph graph algorithm integration using MAGE library.

Implements community detection, centrality measures, and optimized traversal algorithms
to improve retrieval relevance and performance.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import mgclient
from loguru import logger

from . import logs as ls
from .config import settings


class GraphAlgorithms:
    """Wrapper for Memgraph MAGE graph algorithms."""

    def __init__(self) -> None:
        self._conn: mgclient.Connection | None = None

    def _create_connection(self) -> mgclient.Connection:
        """Create a new Memgraph connection."""
        if settings.MEMGRAPH_USERNAME:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                username=settings.MEMGRAPH_USERNAME,
                password=settings.MEMGRAPH_PASSWORD,
            )
        else:
            conn = mgclient.connect(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            )
        conn.autocommit = True
        return conn

    @contextmanager
    def _get_connection(self) -> Generator[mgclient.Connection, None, None]:
        """Get connection (use existing or create new)."""
        if self._conn is not None:
            yield self._conn
        else:
            conn = self._create_connection()
            try:
                yield conn
            finally:
                conn.close()

    def _execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute Cypher query and return results."""
        params = params or {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                if not cursor.description:
                    return []
                columns = [desc.name for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            finally:
                cursor.close()

    @staticmethod
    def _is_missing_procedure_error(error: Exception) -> bool:
        message = str(error).lower()
        return "there is no procedure named" in message or (
            "procedure" in message and "not found" in message
        )

    def run_pagerank(self) -> int:
        """Run PageRank algorithm and store scores as node properties.

        Uses Memgraph MAGE pagerank.get() procedure (Enterprise-only feature).
        Falls back gracefully for Community edition users.
        Scores are stored in `pagerank_score` property on all nodes.

        Returns:
            Number of nodes updated with PageRank score
        """
        logger.info("Running PageRank algorithm...")

        try:
            cypher = """
            CALL pagerank.get() YIELD node, rank
            SET node.pagerank_score = rank
            RETURN count(node) AS updated_count;
            """
            results = self._execute_query(cypher)
            updated = results[0].get("updated_count", 0) if results else 0
            logger.info(f"PageRank algorithm completed, updated {updated} nodes")
            return updated
        except Exception as e:
            logger.debug(f"PageRank algorithm failed: {e}")
            logger.info(
                "Skipping PageRank optimization (not supported in your Memgraph edition)"
            )
            return 0

    def run_community_detection(self, use_leiden: bool = True) -> int:
        """Run community detection algorithm and store community IDs on nodes.

        Args:
            use_leiden: Use Leiden algorithm (higher quality) if True,
                       fallback to Louvain if False or Leiden fails.

        Returns:
            Number of nodes updated with community ID
        """
        algo_name = "Leiden" if use_leiden else "Louvain"
        logger.info(f"Running {algo_name} community detection algorithm...")

        attempts: list[tuple[str, str]] = []
        if use_leiden:
            attempts.append(
                (
                    "Leiden",
                    """
                    CALL graph_algorithms.leiden(
                        "CALLS",
                        "OUTGOING",
                        { community_property: "community_id", weight_property: "weight" }
                    ) YIELD node, community_id
                    RETURN count(node) AS updated_count;
                    """,
                )
            )
        attempts.append(
            (
                "Louvain",
                """
                CALL graph_algorithms.louvain(
                    "CALLS",
                    "OUTGOING",
                    { community_property: "community_id", weight_property: "weight" }
                ) YIELD node, community_id
                RETURN count(node) AS updated_count;
                """,
            )
        )

        unsupported_attempts = 0

        for index, (name, cypher) in enumerate(attempts):
            try:
                results = self._execute_query(cypher)
                updated = results[0].get("updated_count", 0) if results else 0
                logger.info(
                    f"{name} community detection completed, updated {updated} nodes"
                )
                return updated
            except Exception as e:
                if self._is_missing_procedure_error(e):
                    unsupported_attempts += 1
                    logger.info(
                        f"Skipping {name} community detection: procedure not available"
                    )
                    continue

                if index < len(attempts) - 1:
                    logger.warning(
                        f"{name} algorithm failed, falling back to {attempts[index + 1][0]}: {e}"
                    )
                    logger.info(
                        f"Falling back to {attempts[index + 1][0]} algorithm..."
                    )
                    continue

                logger.error(f"Community detection algorithm failed: {e}")
                return 0

        if unsupported_attempts == len(attempts):
            logger.info(
                "Skipping community detection optimization (not supported in your Memgraph edition)"
            )
        return 0

    def get_bfs_context(
        self, start_node_id: int, max_depth: int = 3
    ) -> list[dict[str, Any]]:
        """Get BFS traversal context starting from a node.

        Args:
            start_node_id: Internal Memgraph ID of the start node
            max_depth: Maximum traversal depth

        Returns:
            List of nodes in the BFS traversal path
        """
        cypher = """
        WITH $start_id AS start_id, $max_depth AS max_depth
        MATCH (start) WHERE id(start) = start_id
        // Fixed Memgraph BFS syntax
        MATCH path = (start)-[:CALLS|:DEFINES|:IMPORTS *BFS 1 TO max_depth]-(related)
        WHERE related:Function OR related:Class OR related:Module
        WITH DISTINCT related, length(path) AS depth
        RETURN
            id(related) AS node_id,
            related.name AS name,
            related.qualified_name AS qualified_name,
            related.path AS file_path,
            related.pagerank_score AS pagerank_score,
            depth
        ORDER BY depth ASC, pagerank_score DESC;
        """

        params = {"start_id": start_node_id, "max_depth": max_depth}

        try:
            return self._execute_query(cypher, params)
        except Exception as e:
            logger.warning(f"BFS context expansion failed: {e}")
            return []

    def get_similar_nodes(self, node_id: int, top_k: int = 5) -> list[dict[str, Any]]:
        """Get structurally similar nodes using node similarity algorithm.

        Args:
            node_id: Node ID to find similar nodes for
            top_k: Number of similar nodes to return

        Returns:
            List of similar nodes with similarity scores
        """
        cypher = """
        MATCH (n) WHERE id(n) = $node_id
        MATCH (n)-[:CALLS]->(neighbor)
        WITH collect(DISTINCT id(neighbor)) AS n_neighbors
        MATCH (m)
        WHERE m:Function OR m:Class
        AND id(m) <> $node_id
        MATCH (m)-[:CALLS]->(m_neighbor)
        WITH
            m,
            n_neighbors,
            collect(DISTINCT id(m_neighbor)) AS m_neighbors,
            [x IN n_neighbors WHERE x IN m_neighbors | x] AS intersection
        WITH
            m,
            toFloat(size(intersection)) / (size(n_neighbors) + size(m_neighbors) - size(intersection)) AS jaccard_similarity
        WHERE jaccard_similarity > 0.1
        RETURN
            id(m) AS node_id,
            m.name AS name,
            m.qualified_name AS qualified_name,
            jaccard_similarity
        ORDER BY jaccard_similarity DESC
        LIMIT $top_k;
        """

        params = {"node_id": node_id, "top_k": top_k}

        try:
            return self._execute_query(cypher, params)
        except Exception as e:
            logger.warning(f"Structural similarity search failed: {e}")
            return []

    def analyze_graph(self) -> None:
        """Run ANALYZE GRAPH to update query planner statistics.

        Should be run after every ingestion/update operation to improve
        query performance by providing up-to-date graph statistics to the planner.
        """
        logger.debug("Running ANALYZE GRAPH to update query planner statistics")
        try:
            self._execute_query("ANALYZE GRAPH;")
        except Exception as e:
            logger.warning(f"ANALYZE GRAPH failed: {e}")

    def health_check(self) -> bool:
        """Check if MAGE algorithms are available."""
        try:
            # Test if pagerank procedure exists
            self._execute_query("CALL pagerank.get() YIELD node, rank LIMIT 1;")
            return True
        except Exception:
            try:
                # Fallback test for basic connection
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("RETURN 1 AS health;")
                    cursor.close()
                    return True
            except Exception:
                return False

    def close(self) -> None:
        """Close Memgraph connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None


# Global shared instance
_ALGO_INSTANCE: GraphAlgorithms | None = None


def get_shared_algorithms() -> GraphAlgorithms:
    """Get shared graph algorithms instance."""
    global _ALGO_INSTANCE
    if _ALGO_INSTANCE is None:
        _ALGO_INSTANCE = GraphAlgorithms()
    return _ALGO_INSTANCE


def close_shared_algorithms() -> None:
    """Close shared algorithms instance."""
    global _ALGO_INSTANCE
    if _ALGO_INSTANCE is not None:
        _ALGO_INSTANCE.close()
        _ALGO_INSTANCE = None
