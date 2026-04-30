"""Dynamic graph algorithms for real-time updates."""

from collections.abc import Generator
from contextlib import contextmanager

from loguru import logger

from ..config import settings
from ..services import QueryProtocol
from ..services.graph_service import MemgraphIngestor


def _coerce_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


class DynamicGraphAlgorithms:
    """Wrapper for dynamic MAGE algorithms."""

    @staticmethod
    def _is_missing_procedure_error(error: Exception) -> bool:
        message = str(error).lower()
        return "there is no procedure named" in message or (
            "procedure" in message and "not found" in message
        )

    @staticmethod
    def _is_no_communities_error(error: Exception) -> bool:
        return "no communities detected" in str(error).lower()

    def __init__(
        self,
        use_dynamic: bool | None = None,
        ingestor: QueryProtocol | None = None,
    ):
        """
        Initialize dynamic algorithms wrapper.

        Args:
            use_dynamic: Explicitly enable/disable dynamic algorithms.
                        If None, auto-detect based on Memgraph edition and config.
            ingestor: Optional injected Memgraph connection for reuse.
        """
        self._ingestor = ingestor

        if use_dynamic is not None:
            self.use_dynamic = use_dynamic
            return

        config_value = settings.MEMGRAPH_USE_DYNAMIC_ALGORITHMS
        if config_value is True:
            self.use_dynamic = True
            return

        # Auto-detect if not explicitly enabled/disabled
        if self._ingestor is not None and hasattr(
            self._ingestor, "dynamic_algorithms_enabled"
        ):
            try:
                self.use_dynamic = self._ingestor.dynamic_algorithms_enabled
                return
            except Exception as e:
                logger.debug(
                    f"Could not auto-detect Memgraph edition from injected ingestor: {e}"
                )

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

    @contextmanager
    def _with_ingestor(self) -> Generator[QueryProtocol, None, None]:
        if self._ingestor is not None:
            yield self._ingestor
        else:
            with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                username=settings.MEMGRAPH_USERNAME,
                password=settings.MEMGRAPH_PASSWORD,
            ) as ingestor:
                yield ingestor

    def update_pagerank_dynamic(
        self,
        new_relationships: list[tuple[int, int]] | None,
        deleted_relationships: list[tuple[int, int]] | None,
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

            with self._with_ingestor() as ingestor:
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

        with self._with_ingestor() as ingestor:
            results = ingestor.fetch_all(cypher)
            return {
                "updated_nodes": results[0]["updated"] if results else 0,
                "method": "full",
            }

    def update_communities_dynamic(self, changed_nodes: list[int] | None) -> dict:
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

            with self._with_ingestor() as ingestor:
                results = ingestor.fetch_all(cypher, params)
                return {
                    "updated_communities": len(results),
                    "total_nodes_updated": sum(
                        _coerce_int(r.get("size"), 0) for r in results
                    ),
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
        CALL leiden_community_detection.get()
        YIELD node, community_id
        SET node.community_id = community_id
        RETURN community_id, count(node) AS size
        ORDER BY size DESC
        """

        with self._with_ingestor() as ingestor:
            try:
                results = ingestor.fetch_all(cypher)
                return {
                    "updated_communities": len(results),
                    "total_nodes_updated": sum(
                        _coerce_int(r.get("size"), 0) for r in results
                    ),
                    "method": "full",
                }
            except Exception as exc:
                if self._is_missing_procedure_error(exc):
                    logger.info(
                        "Skipping community recalculation (not supported in your Memgraph edition)"
                    )
                    return {
                        "updated_communities": 0,
                        "total_nodes_updated": 0,
                        "method": "unavailable",
                    }
                if self._is_no_communities_error(exc):
                    logger.info(
                        "No communities detected during recalculation"
                    )
                    return {
                        "updated_communities": 0,
                        "total_nodes_updated": 0,
                        "method": "no_communities",
                    }
                raise
