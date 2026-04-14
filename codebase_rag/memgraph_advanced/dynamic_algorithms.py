"""Dynamic graph algorithms for real-time updates."""

from typing import Any, Optional
from loguru import logger

from ..services.graph_service import MemgraphIngestor
from ..config import settings


class DynamicGraphAlgorithms:
    """Wrapper for dynamic MAGE algorithms."""

    def __init__(self, use_dynamic: Optional[bool] = None):
        """
        Initialize dynamic algorithms wrapper.

        Args:
            use_dynamic: Explicitly enable/disable dynamic algorithms.
                        If None, auto-detect based on Memgraph edition and config.
        """
        if use_dynamic is not None:
            self.use_dynamic = use_dynamic
            return

        config_value = settings.MEMGRAPH_USE_DYNAMIC_ALGORITHMS
        if config_value is True:
            self.use_dynamic = True
            return

        # Auto-detect if not explicitly enabled/disabled
        try:
            # Create a temporary connection to detect edition
            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                username=settings.MEMGRAPH_USERNAME,
                password=settings.MEMGRAPH_PASSWORD,
            ) as ingestor:
                self.use_dynamic = ingestor.dynamic_algorithms_enabled
        except Exception as e:
            logger.debug(
                f"Could not auto-detect Memgraph edition, falling back to full algorithm runs: {e}"
            )
            self.use_dynamic = False

    def update_pagerank_dynamic(
        self,
        new_relationships: Optional[list[tuple[int, int]]],
        deleted_relationships: Optional[list[tuple[int, int]]],
    ) -> dict:
        """
        Incrementally update PageRank scores when graph changes.

        Args:
            new_relationships: List of (source_id, target_id) for new edges
            deleted_relationships: List of (source_id, target_id) for deleted edges

        Returns:
            Updated PageRank statistics
        """
        if not self.use_dynamic:
            # Fallback to full recalculation
            return self._full_pagerank_recalculation()

        try:
            cypher = """
            // Check if dynamic pagerank is available
            CALL dynamic.pagerank_online.update($new_edges, $deleted_edges)
            YIELD node, rank
            SET node.pagerank_score = rank
            RETURN count(node) AS updated
            """

            params = {
                "new_edges": [
                    {"from": s, "to": t} for s, t in (new_relationships or [])
                ],
                "deleted_edges": [
                    {"from": s, "to": t} for s, t in (deleted_relationships or [])
                ],
            }

            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                username=settings.MEMGRAPH_USERNAME,
                password=settings.MEMGRAPH_PASSWORD,
            ) as ingestor:
                results = ingestor.fetch_all(cypher, params)
                return {
                    "updated_nodes": results[0]["updated"] if results else 0,
                    "method": "dynamic",
                }

        except Exception as e:
            # If dynamic algorithm fails, fall back to full recalculation
            logger.debug(
                f"Dynamic PageRank update failed, falling back to full recalculation: {e}"
            )
            return self._full_pagerank_recalculation()

    def _full_pagerank_recalculation(self) -> dict:
        """Fallback: Full PageRank recalculation for non-enterprise instances."""
        cypher = """
        CALL pagerank.get()
        YIELD node, rank
        SET node.pagerank_score = rank
        RETURN count(node) AS updated
        """

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        ) as ingestor:
            results = ingestor.fetch_all(cypher)
            return {
                "updated_nodes": results[0]["updated"] if results else 0,
                "method": "full",
            }

    def update_communities_dynamic(self, changed_nodes: Optional[list[int]]) -> dict:
        """
        Incrementally update community assignments.

        Args:
            changed_nodes: List of node IDs that changed

        Returns:
            Updated community statistics
        """
        if not self.use_dynamic:
            # Fallback to full community recalculation
            return self._full_community_recalculation()

        try:
            cypher = """
            CALL dynamic.community_detection_online.update($changed_nodes)
            YIELD node, community_id
            SET node.community_id = community_id
            RETURN community_id, count(node) AS size
            ORDER BY size DESC
            """

            params = {"changed_nodes": changed_nodes or []}

            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                username=settings.MEMGRAPH_USERNAME,
                password=settings.MEMGRAPH_PASSWORD,
            ) as ingestor:
                results = ingestor.fetch_all(cypher, params)
                return {
                    "updated_communities": len(results),
                    "total_nodes_updated": sum(r["size"] for r in results),
                    "method": "dynamic",
                }

        except Exception as e:
            # If dynamic algorithm fails, fall back to full recalculation
            logger.debug(
                f"Dynamic community update failed, falling back to full recalculation: {e}"
            )
            return self._full_community_recalculation()

    def _full_community_recalculation(self) -> dict:
        """Fallback: Full community detection recalculation."""
        cypher = """
        CALL community_detection.leiden(
            "CALLS",
            "OUTGOING",
            { community_property: "community_id", weight_property: "weight" }
        ) YIELD node, community_id
        SET node.community_id = community_id
        RETURN community_id, count(node) AS size
        ORDER BY size DESC
        """

        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            username=settings.MEMGRAPH_USERNAME,
            password=settings.MEMGRAPH_PASSWORD,
        ) as ingestor:
            results = ingestor.fetch_all(cypher)
            return {
                "updated_communities": len(results),
                "total_nodes_updated": sum(r["size"] for r in results),
                "method": "full",
            }
