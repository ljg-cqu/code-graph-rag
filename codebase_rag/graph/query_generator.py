"""Memgraph query generator with capability detection.

This module detects Memgraph server capabilities and generates compatible
Cypher queries tailored to the detected version and feature support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..exceptions import MemgraphCompatibilityError

if TYPE_CHECKING:
    import mgclient


@dataclass
class MemgraphCapabilities:
    """Dataclass representing detected Memgraph capabilities."""

    version: str = ""
    supports_vector_search: bool = False
    supports_vector_index: bool = False
    supports_cosine_similarity: bool = False
    supports_l2_distance: bool = False
    supports_vector_search_procedure: bool = False
    supports_if_not_exists_index: bool = False
    vector_function_syntax: str = "cosine_similarity"  # Default for newer versions


class MemgraphQueryGenerator:
    """Generates Memgraph-compatible Cypher queries based on detected capabilities."""

    def __init__(self, connection: mgclient.Connection) -> None:
        """Initialize query generator with Memgraph connection.

        Args:
            connection: mgclient connection connected to Memgraph instance.
        """
        self._conn = connection
        self._capabilities: MemgraphCapabilities | None = None

    @property
    def capabilities(self) -> MemgraphCapabilities:
        """Get detected Memgraph capabilities (lazy loaded)."""
        if not self._capabilities:
            self._capabilities = self._detect_capabilities()
        return self._capabilities

    def _run_query(self, query: str, params: dict | None = None) -> list[dict]:
        """Run a query and return results as list of dicts."""
        params = params or {}
        cursor = self._conn.cursor()
        try:
            cursor.execute(query, params)
            if not cursor.description:
                return []
            columns = [desc.name for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()

    def _is_missing_procedure_error(self, error_message: str) -> bool:
        return "procedure" in error_message and (
            "doesn't exist" in error_message
            or "does not exist" in error_message
            or "not found" in error_message
            or "unknown procedure" in error_message
        )

    def _is_missing_vector_index_error(self, error_message: str) -> bool:
        missing_index_markers = (
            "vector index",
            "index",
        )
        missing_markers = (
            "doesn't exist",
            "does not exist",
            "not found",
            "no such",
            "missing",
        )
        return any(marker in error_message for marker in missing_index_markers) and any(
            marker in error_message for marker in missing_markers
        )

    def _detect_capabilities(self) -> MemgraphCapabilities:
        """Detect Memgraph version and supported features.

        Returns:
            MemgraphCapabilities object with detected features.
        """
        capabilities = MemgraphCapabilities()

        # Get Memgraph version
        try:
            result = self._run_query("SHOW VERSION")
            if result:
                capabilities.version = result[0]["version"]
                logger.debug(f"Detected Memgraph version: {capabilities.version}")
        except Exception as e:
            logger.warning(f"Failed to detect Memgraph version: {e}")
            capabilities.version = "unknown"

        # Check vector search procedure support (vector_search.search())
        try:
            test_query = """
                CALL vector_search.search("test_index", 1, [1.0, 2.0])
                YIELD node, similarity
            """
            self._run_query(test_query)
            capabilities.supports_vector_search_procedure = True
            capabilities.supports_vector_search = True
            logger.debug("Memgraph supports vector_search.search() procedure")
        except Exception as e:
            error_str = str(e).lower()
            if self._is_missing_vector_index_error(error_str):
                capabilities.supports_vector_search_procedure = True
                capabilities.supports_vector_search = True
                logger.debug(
                    "Memgraph supports vector_search.search() procedure (probe hit missing test index)"
                )
            elif self._is_missing_procedure_error(error_str):
                capabilities.supports_vector_search_procedure = False
            else:
                capabilities.supports_vector_search_procedure = False
                logger.debug(
                    f"Unable to confirm vector_search.search() procedure support: {e}"
                )

        # Check direct vector function support
        if not capabilities.supports_vector_search_procedure:
            try:
                # Test if vector similarity functions exist
                test_query = """
                    RETURN cosine_similarity([1.0, 2.0], [3.0, 4.0]) AS sim
                """
                self._run_query(test_query)
                capabilities.supports_vector_search = True
                capabilities.supports_cosine_similarity = True
                capabilities.vector_function_syntax = "cosine_similarity"
                logger.debug("Memgraph supports cosine similarity function")
            except Exception:
                # Check for older syntax (vector.cosine_similarity)
                try:
                    test_query = """
                        RETURN vector.cosine_similarity([1.0, 2.0], [3.0, 4.0]) AS sim
                    """
                    self._run_query(test_query)
                    capabilities.supports_vector_search = True
                    capabilities.supports_cosine_similarity = True
                    capabilities.vector_function_syntax = "vector.cosine_similarity"
                    logger.debug("Memgraph uses older vector.cosine_similarity syntax")
                except Exception as e:
                    logger.warning(f"Memgraph does not support vector search: {e}")
                    capabilities.supports_vector_search = False

        # Check L2 distance support
        if (
            capabilities.supports_vector_search
            and not capabilities.supports_vector_search_procedure
        ):
            try:
                test_query = f"""
                    RETURN {capabilities.vector_function_syntax.replace("cosine_similarity", "l2_distance")}([1.0, 2.0], [3.0, 4.0]) AS dist
                """
                self._run_query(test_query)
                capabilities.supports_l2_distance = True
            except Exception:
                capabilities.supports_l2_distance = False

        # Check vector index support
        try:
            self._run_query("SHOW INDEXES")
            capabilities.supports_vector_index = True
        except Exception:
            capabilities.supports_vector_index = False

        # Check IF NOT EXISTS support for index creation
        if capabilities.supports_vector_index:
            try:
                # Test with a dummy index, clean up after
                self._run_query(
                    "CREATE INDEX IF NOT EXISTS test_dummy_index ON :Test(prop)"
                )
                self._run_query("DROP INDEX test_dummy_index IF EXISTS")
                capabilities.supports_if_not_exists_index = True
                logger.debug("Memgraph supports IF NOT EXISTS for index creation")
            except Exception:
                capabilities.supports_if_not_exists_index = False
                logger.debug(
                    "Memgraph does not support IF NOT EXISTS for index creation"
                )

        return capabilities

    def generate_vector_search_query(
        self,
        node_label: str,
        vector_property: str,
        query_vector: list[float],
        top_k: int = 10,
        metric: str = "cosine",
        additional_filters: str = "",
        return_properties: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate compatible vector search query based on detected capabilities.

        Args:
            node_label: Label of nodes to search.
            vector_property: Property name containing vector embeddings.
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            metric: Similarity metric ("cosine" or "l2").
            additional_filters: Additional Cypher filter conditions (optional).
            return_properties: List of properties to return (defaults to all).

        Returns:
            Tuple of (cypher_query, parameters_dict)

        Raises:
            MemgraphCompatibilityError: If vector search is not supported.
        """
        if not self.capabilities.supports_vector_search:
            raise MemgraphCompatibilityError(
                "Connected Memgraph instance does not support vector search. "
                "Please upgrade to Memgraph 2.10 or newer."
            )

        # Select appropriate similarity function
        if metric == "cosine":
            if not self.capabilities.supports_cosine_similarity:
                raise MemgraphCompatibilityError(
                    "Cosine similarity is not supported by this Memgraph instance."
                )
            func_name = self.capabilities.vector_function_syntax
            # Cosine similarity: higher value = more similar, sort descending
            order_clause = "ORDER BY similarity DESC"
        elif metric == "l2":
            if not self.capabilities.supports_l2_distance:
                raise MemgraphCompatibilityError(
                    "L2 distance is not supported by this Memgraph instance."
                )
            func_name = self.capabilities.vector_function_syntax.replace(
                "cosine_similarity", "l2_distance"
            )
            # L2 distance: lower value = more similar, sort ascending
            order_clause = "ORDER BY distance ASC"
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'l2'.")

        # Build return clause
        if return_properties:
            return_clause = f"RETURN node, {metric} AS score"
            for prop in return_properties:
                return_clause += f", node.{prop} AS {prop}"
        else:
            return_clause = f"RETURN node, {metric} AS score"

        # Build filter clause
        filter_clause = ""
        if additional_filters.strip():
            filter_clause = f"WHERE {additional_filters.strip()}"

        # Build final query
        query = f"""
            MATCH (node:{node_label})
            {filter_clause}
            WITH node, {func_name}(node.{vector_property}, $query_vector) AS {metric}
            {order_clause}
            LIMIT $top_k
            {return_clause}
        """

        parameters = {
            "query_vector": query_vector,
            "top_k": top_k,
        }

        return query, parameters

    def generate_vector_index_creation_query(
        self,
        index_name: str,
        node_label: str,
        vector_property: str,
        vector_dim: int,
        metric: str = "cosine",
        capacity: int = 100000,
    ) -> tuple[str, dict[str, Any]]:
        """Generate compatible vector index creation query.

        Args:
            index_name: Name of the vector index.
            node_label: Label of nodes to index.
            vector_property: Property name containing vector embeddings.
            vector_dim: Dimension of the vectors.
            metric: Similarity metric ("cosine" or "l2").
            capacity: Maximum number of vectors the index can hold.

        Returns:
            Tuple of (cypher_query, parameters_dict)

        Raises:
            MemgraphCompatibilityError: If vector index is not supported.
        """
        if not self.capabilities.supports_vector_index:
            raise MemgraphCompatibilityError(
                "Connected Memgraph instance does not support vector indexes."
            )

        # Build query based on IF NOT EXISTS support (Memgraph doesn't support params in CONFIG)
        if self.capabilities.supports_if_not_exists_index:
            query = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                ON :{node_label}({vector_property})
                WITH CONFIG {{
                    "dimension": {vector_dim},
                    "similarity_metric": "{metric}",
                    "capacity": {capacity}
                }}
            """
        else:
            # Older Memgraph versions don't support IF NOT EXISTS
            query = f"""
                CREATE VECTOR INDEX {index_name}
                ON :{node_label}({vector_property})
                WITH CONFIG {{
                    "dimension": {vector_dim},
                    "similarity_metric": "{metric}",
                    "capacity": {capacity}
                }}
            """

        return query, {}


class QueryGenerator:
    """Auto-detects Memgraph version and license to generate compatible Cypher queries."""

    __slots__ = ("_memgraph_version", "_has_enterprise_license")

    def __init__(self) -> None:
        from .. import constants as cs

        self._memgraph_version: tuple[int, ...] = cs.QUERY_GEN_FALLBACK_VERSION
        self._has_enterprise_license: bool = False
        self._detect_memgraph_capabilities()

    def _detect_memgraph_capabilities(self) -> None:
        import mgclient as _mgclient

        from .. import constants as cs
        from ..config import settings

        conn = None
        cursor = None
        try:
            conn = _mgclient.connect(
                host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
            )
            cursor = conn.cursor()

            cursor.execute(cs.QUERY_GEN_SHOW_VERSION)
            row = cursor.fetchone()
            if row:
                parts = str(row[0]).split(".")
                self._memgraph_version = tuple(int(p) for p in parts if p.isdigit())

            cursor.execute(cs.QUERY_GEN_SHOW_LICENSE)
            row = cursor.fetchone()
            if row:
                self._has_enterprise_license = (
                    cs.QUERY_GEN_ENTERPRISE_KEYWORD in str(row[0]).lower()
                )
        except Exception:
            self._memgraph_version = cs.QUERY_GEN_FALLBACK_VERSION
            self._has_enterprise_license = False
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def get_disconnected_nodes_query(self) -> str:
        """Return a Cypher query for disconnected production nodes compatible with detected Memgraph capabilities."""
        from .. import constants as cs

        if (
            self._has_enterprise_license
            and self._memgraph_version >= cs.QUERY_GEN_MIN_ENTERPRISE_VERSION
        ):
            return cs.QUERY_GEN_DISCONNECTED_NODES_PARALLEL
        return cs.QUERY_GEN_DISCONNECTED_NODES
