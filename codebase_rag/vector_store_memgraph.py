"""Memgraph native vector storage backend.

Uses Memgraph's built-in vector index support (v3.0.0+) via vector_search module.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Literal, overload

from loguru import logger

import mgclient

from . import constants as cs
from . import logs as ls
from .config import settings
from .graph.query_generator import MemgraphQueryGenerator
from .types_defs import ResultRow
from .vector_backend import VectorBackend

# Model info
UNIXCODER_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_VERSION = 1


def _find_vector_index(
    rows: list[dict[str, str | int | None]],
    index_name: str,
) -> dict[str, str | int | None] | None:
    for row in rows:
        if str(row.get("index_name") or "") == index_name:
            return row
    return None


def _read_vector_index_dimension(
    index_info: dict[str, str | int | None] | None,
) -> int | None:
    if index_info is None:
        return None

    raw_dimension = index_info.get("dimension")
    if raw_dimension is None:
        return None

    try:
        return int(raw_dimension)
    except (TypeError, ValueError):
        return None


class MemgraphBackend(VectorBackend):
    """Memgraph native vector storage using vector_search module.

    Stores embeddings directly on node properties and uses Memgraph's
    built-in vector index for similarity search.

    Advantages:
    - Single database for graph structure and vectors
    - Hybrid queries: vector search + graph traversal in one Cypher query
    - Lower latency: no cross-database coordination
    """

    LABELS_TO_INDEX = cs.EMBEDDABLE_CODE_NODE_LABELS

    def __init__(self, is_document: bool = False) -> None:
        self.is_document = is_document
        self._conn: mgclient.Connection | None = None
        self._query_generator: MemgraphQueryGenerator | None = None

    def _create_connection(self) -> mgclient.Connection:
        """Create a new Memgraph connection."""
        if self.is_document:
            host = settings.DOC_MEMGRAPH_HOST
            port = settings.DOC_MEMGRAPH_PORT
            username = settings.DOC_MEMGRAPH_USERNAME
            password = settings.DOC_MEMGRAPH_PASSWORD
        else:
            host = settings.MEMGRAPH_HOST
            port = settings.MEMGRAPH_PORT
            username = settings.MEMGRAPH_USERNAME
            password = settings.MEMGRAPH_PASSWORD

        if username:
            conn = mgclient.connect(
                host=host,
                port=port,
                username=username,
                password=password,
            )
        else:
            conn = mgclient.connect(
                host=host,
                port=port,
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

    @property
    def query_generator(self) -> MemgraphQueryGenerator:
        """Get lazy-initialized query generator for Memgraph compatibility."""
        if self._query_generator is None:
            # Create a persistent connection for the query generator
            self._conn = self._create_connection()
            self._query_generator = MemgraphQueryGenerator(self._conn)
        return self._query_generator

    def initialize(self) -> None:
        """Create vector indexes for embeddable node types.

        Uses Memgraph's CREATE VECTOR INDEX syntax with required capacity.
        Index is created for each embeddable label (Function, Method, etc.)
        Automatically recreates indexes if dimension mismatch is detected.
        """
        logger.info(ls.MG_VECTOR_INIT.format(index=settings.MEMGRAPH_VECTOR_INDEX_NAME))
        effective_dim = settings.get_effective_vector_dim()
        capabilities = self.query_generator.capabilities

        # Get existing index info upfront
        existing_indexes: list[dict[str, str | int | None]] = []
        try:
            existing_indexes = self._execute_query("SHOW VECTOR INDEX INFO;")
        except Exception:
            pass  # Ignore if command fails (older Memgraph versions without vector support)

        for label in self.LABELS_TO_INDEX:
            index_name = f"{label.lower()}_embedding_index"

            # Check if index exists and has correct dimension
            existing_index = _find_vector_index(existing_indexes, index_name)
            needs_recreate = False

            if existing_index:
                current_dim = _read_vector_index_dimension(existing_index)
                if current_dim != effective_dim:
                    logger.warning(
                        f"Vector index {index_name} has dimension {current_dim}, but current embedding model "
                        f"requires {effective_dim}. Recreating index and clearing old incompatible embeddings..."
                    )
                    needs_recreate = True

            # Create index if it doesn't exist or needs recreation
            if not existing_index or needs_recreate:
                try:
                    if capabilities.supports_vector_index:
                        if needs_recreate:
                            # Clear embeddings for this label only when recreating due to dimension mismatch
                            try:
                                clear_cypher = f"MATCH (n:{label}) SET n.embedding = NULL, n.embedding_model = NULL, n.embedding_version = NULL"
                                self._execute_query(clear_cypher)
                                logger.debug(
                                    f"Cleared old embeddings for {label} nodes"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to clear old embeddings for {label} nodes: {e}"
                                )
                            # Drop existing index
                            try:
                                self._execute_query(f"DROP VECTOR INDEX {index_name};")
                            except Exception as e:
                                logger.debug(f"Failed to drop index {index_name}: {e}")

                        # Use query generator to create compatible index creation query
                        cypher, params = (
                            self.query_generator.generate_vector_index_creation_query(
                                index_name=index_name,
                                node_label=label,
                                vector_property="embedding",
                                vector_dim=effective_dim,
                                metric=settings.MEMGRAPH_VECTOR_METRIC,
                                capacity=settings.MEMGRAPH_VECTOR_CAPACITY,
                            )
                        )
                        self._execute_query(cypher, params)
                    else:
                        # Fallback: no vector index support, just proceed without indexes
                        logger.debug(
                            f"Vector index not supported for label {label}, skipping index creation"
                        )
                        continue

                    logger.info(
                        ls.MG_VECTOR_INDEX_CREATED.format(
                            index=index_name,
                            label=label,
                            dim=effective_dim,
                            capacity=settings.MEMGRAPH_VECTOR_CAPACITY,
                        )
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    if "already exists" in error_str or "duplicate" in error_str:
                        logger.info(ls.MG_VECTOR_INDEX_EXISTS.format(index=index_name))
                    else:
                        logger.error(
                            ls.MG_VECTOR_INDEX_FAILED.format(
                                index=index_name,
                                error=e,
                            )
                        )
                        raise
            else:
                logger.info(ls.MG_VECTOR_INDEX_EXISTS.format(index=index_name))

        # Check index info
        self._check_index_info()

    def _check_index_info(self) -> None:
        """Show vector index information."""
        try:
            cypher = "SHOW VECTOR INDEX INFO;"
            results = self._execute_query(cypher)
            if results:
                logger.debug(ls.MG_VECTOR_INDEX_INFO.format(info=results))
        except Exception:
            pass  # Non-critical

    def _store_single(
        self, node_id: int, embedding: list[float], qualified_name: str
    ) -> int:
        cypher = """
        MATCH (n) WHERE id(n) = $node_id
        SET n.embedding = $embedding,
            n.embedding_model = $model_name,
            n.embedding_version = $version
        RETURN count(n) AS stored;
        """

        params = {
            "node_id": node_id,
            "embedding": embedding,
            "model_name": settings.EMBEDDING_MODEL,
            "version": EMBEDDING_VERSION,
        }

        try:
            results = self._execute_query(cypher, params)
            return results[0].get("stored", 0) if results else 0
        except Exception as e:
            error_str = str(e).lower()
            extra_msg = ""
            if (
                "vector index property must have the same number of dimensions"
                in error_str
            ):
                extra_msg = " This is likely due to a dimension mismatch between your embedding model and existing vector indexes. The vector store will automatically fix this on next initialization, or you can run `cgr vector recreate-indexes --code` manually."
            logger.warning(
                ls.EMBEDDING_STORE_FAILED.format(
                    name=qualified_name, error=f"{e}{extra_msg}"
                )
            )
            return 0

    def _store_individually(
        self, points: Sequence[tuple[int, list[float], str]]
    ) -> int:
        return sum(
            self._store_single(node_id, embedding, qualified_name)
            for node_id, embedding, qualified_name in points
        )

    def _verified_count(self, node_ids: set[int], fallback_count: int) -> int:
        try:
            return len(self.verify_ids(node_ids))
        except Exception:
            return fallback_count

    def store_batch(self, points: Sequence[tuple[int, list[float], str]]) -> int:
        """Store embeddings as node properties.

        Updates nodes by their internal ID, setting embedding property.
        """
        if not points:
            return 0

        logger.debug(
            f"store_batch called with {len(points)} points, "
            f"node_ids: {[p[0] for p in points[:5]]}{'...' if len(points) > 5 else ''}"
        )

        # Batch update using UNWIND
        cypher = """
        UNWIND $points AS p
        MATCH (n) WHERE id(n) = p.node_id
        SET n.embedding = p.embedding,
            n.embedding_model = $model_name,
            n.embedding_version = $version
        RETURN count(n) AS stored;
        """

        params = {
            "points": [{"node_id": nid, "embedding": emb} for nid, emb, _ in points],
            "model_name": settings.EMBEDDING_MODEL,
            "version": EMBEDDING_VERSION,
        }
        node_ids = {node_id for node_id, _, _ in points}

        try:
            results = self._execute_query(cypher, params)
            logger.debug(f"store_batch query results: {results}")
            stored = results[0].get("stored", 0) if results else 0

            if stored < len(points):
                try:
                    found_ids = self.verify_ids(node_ids)
                    missing_points = [
                        point for point in points if point[0] not in found_ids
                    ]
                except Exception:
                    missing_points = list(points)

                if missing_points:
                    logger.warning(
                        f"Batch stored {stored} of {len(points)} embeddings, retrying {len(missing_points)} individually"
                    )
                    stored += self._store_individually(missing_points)
                    stored = self._verified_count(node_ids, stored)

            logger.debug(ls.EMBEDDING_BATCH_STORED.format(count=stored))
            return stored
        except Exception as e:
            logger.warning(ls.EMBEDDING_BATCH_FAILED.format(error=e))
            stored = self._store_individually(points)
            return self._verified_count(node_ids, stored)

    @overload
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        include_context: Literal[False] = False,
        max_context_depth: int = 2,
    ) -> list[tuple[int, float]]: ...

    @overload
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        include_context: Literal[True] = True,
        max_context_depth: int = 2,
    ) -> list[ResultRow]: ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        include_context: bool = False,
        max_context_depth: int = 2,
    ) -> list[tuple[int, float]] | list[ResultRow]:
        """Hybrid vector + graph retrieval using Memgraph native capabilities.

        Implements atomic retrieval pipeline:
        1. Vector similarity search (auto-detects Memgraph version and uses compatible syntax)
        2. BFS context expansion to get related nodes
        3. Ranking by combined similarity + PageRank score
        4. Optional context inclusion for richer results

        Args:
            query_embedding: Query vector
            top_k: Number of primary results to return
            filters: Optional filters (e.g., {"project_prefix": "myproject."})
            include_context: If True, returns full node details + context paths
            max_context_depth: Maximum BFS depth for context expansion

        Returns:
            List of (node_id, similarity) tuples if include_context=False,
            or list of full result dicts if include_context=True
        """
        effective_top_k = top_k if top_k > 0 else settings.VECTOR_SEARCH_TOP_K
        project_prefix = filters.get("project_prefix") if filters else None
        capabilities = self.query_generator.capabilities

        all_results = []
        seen_node_ids = set()

        # Skip vector search entirely if not supported
        if not capabilities.supports_vector_search:
            logger.debug(
                "Memgraph vector search not supported, skipping vector retrieval"
            )
            return all_results

        # Search across all label indexes
        for label in self.LABELS_TO_INDEX:
            try:
                if capabilities.supports_vector_search_procedure:
                    # Use optimized vector_search.search() procedure (newer Memgraph versions)
                    if include_context:
                        cypher = """
                        WITH $embedding AS query_vec, $top_k AS top_k, $project_prefix AS project_prefix, $max_depth AS max_depth
                        CALL vector_search.search($index_name, top_k * 3, query_vec)
                        YIELD node AS start_node, similarity AS sim
                        WITH start_node, sim, max_depth
                        WHERE ($project_prefix IS NULL OR start_node.qualified_name STARTS WITH $project_prefix)
                        MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS 1..max_depth]-(related)
                        WHERE related:Function OR related:Class OR related:Module

                        WITH
                            DISTINCT related,
                            sim,
                            COALESCE(related.pagerank_score, 0.1) AS pr_score,
                            collect(DISTINCT [n IN nodes(path) | n.qualified_name]) AS context_paths
                        ORDER BY (sim * 0.7) + (pr_score * 0.3) DESC
                        LIMIT $top_k

                        RETURN
                            id(related) AS node_id,
                            related.name AS name,
                            related.qualified_name AS qualified_name,
                            related.path AS file_path,
                            related.docstring AS docstring,
                            related.start_line AS start_line,
                            related.end_line AS end_line,
                            sim AS similarity,
                            pr_score AS pagerank_score,
                            context_paths
                        """
                    else:
                        cypher = """
                        WITH $embedding AS query_vec, $top_k AS top_k, $project_prefix AS project_prefix
                        CALL vector_search.search($index_name, top_k * 2, query_vec)
                        YIELD node AS n, similarity AS sim
                        WITH n, sim
                        WHERE ($project_prefix IS NULL OR n.qualified_name STARTS WITH $project_prefix)
                        WITH n, sim, COALESCE(n.pagerank_score, 0.1) AS pr_score
                        ORDER BY (sim * 0.7) + (pr_score * 0.3) DESC
                        LIMIT $top_k
                        RETURN id(n) AS node_id, sim AS similarity
                        """

                    params = {
                        "index_name": f"{label.lower()}_embedding_index",
                        "embedding": query_embedding,
                        "top_k": effective_top_k,
                        "project_prefix": project_prefix,
                        "max_depth": max_context_depth,
                    }

                    label_results = self._execute_query(cypher, params)
                else:
                    # Use compatibility mode: direct similarity function call (older Memgraph versions)
                    additional_filters = ""
                    if project_prefix:
                        additional_filters = (
                            f"node.qualified_name STARTS WITH '{project_prefix}'"
                        )

                    # Generate compatible vector search query
                    cypher_base, params_base = (
                        self.query_generator.generate_vector_search_query(
                            node_label=label,
                            vector_property="embedding",
                            query_vector=query_embedding,
                            top_k=effective_top_k * 3,
                            additional_filters=additional_filters,
                        )
                    )

                    if include_context:
                        # Add context expansion to generated query
                        cypher = f"""
                        WITH $embedding AS query_vec, $top_k AS top_k, $project_prefix AS project_prefix, $max_depth AS max_depth
                        MATCH (start_node:{label})
                        {f"WHERE {additional_filters}" if additional_filters else ""}
                        WITH start_node, {capabilities.vector_function_syntax}(start_node.embedding, query_vec) AS sim, max_depth
                        ORDER BY sim DESC
                        LIMIT $top_k * 3

                        MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS 1..max_depth]-(related)
                        WHERE related:Function OR related:Class OR related:Module

                        WITH
                            DISTINCT related,
                            sim,
                            COALESCE(related.pagerank_score, 0.1) AS pr_score,
                            collect(DISTINCT [n IN nodes(path) | n.qualified_name]) AS context_paths
                        ORDER BY (sim * 0.7) + (pr_score * 0.3) DESC
                        LIMIT $top_k

                        RETURN
                            id(related) AS node_id,
                            related.name AS name,
                            related.qualified_name AS qualified_name,
                            related.path AS file_path,
                            related.docstring AS docstring,
                            related.start_line AS start_line,
                            related.end_line AS end_line,
                            sim AS similarity,
                            pr_score AS pagerank_score,
                            context_paths
                        """
                        params = {
                            "embedding": query_embedding,
                            "top_k": effective_top_k,
                            "project_prefix": project_prefix,
                            "max_depth": max_context_depth,
                        }
                    else:
                        # Lightweight search without context
                        cypher = f"""
                        MATCH (n:{label})
                        {f"WHERE {additional_filters}" if additional_filters else ""}
                        WITH n, {capabilities.vector_function_syntax}(n.embedding, $query_vector) AS sim
                        WITH n, sim, COALESCE(n.pagerank_score, 0.1) AS pr_score
                        ORDER BY (sim * 0.7) + (pr_score * 0.3) DESC
                        LIMIT $top_k
                        RETURN id(n) AS node_id, sim AS similarity
                        """
                        params = {
                            "query_vector": query_embedding,
                            "top_k": effective_top_k,
                        }

                    label_results = self._execute_query(cypher, params)

                # Process results
                for res in label_results:
                    node_id = int(res["node_id"])
                    if node_id not in seen_node_ids:
                        seen_node_ids.add(node_id)
                        if include_context:
                            all_results.append(res)
                        else:
                            all_results.append((node_id, float(res["similarity"])))

            except Exception as e:
                logger.debug(f"Search failed for label {label}: {e}")
                continue

        # Sort final results and return top_k
        if include_context:
            all_results.sort(
                key=lambda x: (x["similarity"] * 0.7) + (x["pagerank_score"] * 0.3),
                reverse=True,
            )
            return all_results[:effective_top_k]
        else:
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[:effective_top_k]

    def delete_batch(self, node_ids: Sequence[int]) -> int:
        """Remove embeddings from nodes (set to NULL)."""
        if not node_ids:
            return 0

        cypher = """
        MATCH (n)
        WHERE id(n) IN $node_ids
        SET n.embedding = NULL,
            n.embedding_model = NULL,
            n.embedding_version = NULL
        RETURN count(n) AS deleted;
        """

        params = {"node_ids": list(node_ids)}

        try:
            results = self._execute_query(cypher, params)
            return results[0].get("deleted", 0) if results else 0
        except Exception as e:
            logger.warning(ls.VECTOR_DELETE_FAILED.format(error=e))
            return 0

    def verify_ids(self, expected_ids: set[int]) -> set[int]:
        """Check which IDs have embeddings stored."""
        if not expected_ids:
            return set()

        # Batch check in chunks to avoid large queries
        chunk_size = 1000
        found_ids: set[int] = set()
        ids_list = list(expected_ids)

        for i in range(0, len(ids_list), chunk_size):
            chunk = ids_list[i : i + chunk_size]
            cypher = """
            MATCH (n)
            WHERE id(n) IN $node_ids AND n.embedding IS NOT NULL
            RETURN collect(id(n)) AS found_ids;
            """
            results = self._execute_query(cypher, {"node_ids": chunk})
            if results and results[0].get("found_ids"):
                found_ids.update(results[0]["found_ids"])

        return found_ids

    def get_stats(self) -> dict:
        """Return embedding statistics."""
        capabilities = self.query_generator.capabilities
        base_stats = {
            "backend": "memgraph",
            "healthy": self.health_check(),
            "stats_available": False,
            "is_document_backend": self.is_document,
            "memgraph_version": capabilities.version,
            "vector_search_supported": capabilities.supports_vector_search,
            "vector_search_procedure_supported": capabilities.supports_vector_search_procedure,
            "vector_index_supported": capabilities.supports_vector_index,
            "total_embeddings": 0,
            "node_types": 0,
            "dimension": 0,
            "models": [],
        }
        cypher = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN
            count(n) AS total_embeddings,
            count(DISTINCT labels(n)[0]) AS node_types,
            min(size(n.embedding)) AS min_dimension,
            max(size(n.embedding)) AS max_dimension,
            collect(DISTINCT n.embedding_model) AS models;
        """
        try:
            results = self._execute_query(cypher)
            if results:
                return {
                    **base_stats,
                    "stats_available": True,
                    "total_embeddings": results[0].get("total_embeddings", 0),
                    "node_types": results[0].get("node_types", 0),
                    "dimension": results[0].get("max_dimension", 0),
                    "models": results[0].get("models", []),
                }
        except Exception as e:
            return {**base_stats, "stats_error": str(e)}
        return base_stats

    def health_check(self) -> bool:
        """Check if Memgraph connection is healthy."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("RETURN 1 AS health;")
                cursor.close()
                return True
        except Exception:
            return False

    def recreate_vector_indexes(
        self, new_dimension: int | None = None, clear_existing_embeddings: bool = True
    ) -> None:
        """Drop and recreate all vector indexes with new dimension.

        This is needed when switching embedding models with different dimensions.

        Args:
            new_dimension: New vector dimension for the indexes. Defaults to effective dimension from settings if not provided.
            clear_existing_embeddings: If True, removes all existing embedding properties from nodes before recreating indexes.
                This is required when changing dimensions to avoid index creation failures from incompatible old embeddings.
        """
        if new_dimension is None:
            new_dimension = settings.get_effective_vector_dim()
        logger.info(f"Recreating vector indexes with dimension {new_dimension}...")
        capabilities = self.query_generator.capabilities

        if clear_existing_embeddings:
            logger.info("Clearing all existing embedding properties from nodes...")
            for label in self.LABELS_TO_INDEX:
                try:
                    cypher = f"MATCH (n:{label}) SET n.embedding = NULL, n.embedding_model = NULL, n.embedding_version = NULL"
                    self._execute_query(cypher)
                    logger.debug(f"Cleared embeddings for {label} nodes")
                except Exception as e:
                    logger.warning(f"Failed to clear embeddings for {label} nodes: {e}")

        for label in self.LABELS_TO_INDEX:
            index_name = f"{label.lower()}_embedding_index"

            # Drop existing index
            try:
                self._execute_query(f"DROP VECTOR INDEX {index_name};")
                logger.debug(f"Dropped index {index_name}")
            except Exception as e:
                logger.debug(f"Could not drop index {index_name}: {e}")

            # Create new index using the same query generator logic as initialize for consistency
            try:
                if capabilities.supports_vector_index:
                    cypher, params = (
                        self.query_generator.generate_vector_index_creation_query(
                            index_name=index_name,
                            node_label=label,
                            vector_property="embedding",
                            vector_dim=new_dimension,
                            metric=settings.MEMGRAPH_VECTOR_METRIC,
                            capacity=settings.MEMGRAPH_VECTOR_CAPACITY,
                        )
                    )
                    self._execute_query(cypher, params)
                else:
                    logger.debug(
                        f"Vector index not supported for label {label}, skipping index creation"
                    )
                    continue

                logger.info(
                    ls.MG_VECTOR_INDEX_CREATED.format(
                        index=index_name,
                        label=label,
                        dim=new_dimension,
                        capacity=settings.MEMGRAPH_VECTOR_CAPACITY,
                    )
                )
            except Exception as e:
                logger.error(
                    ls.MG_VECTOR_INDEX_FAILED.format(
                        index=index_name,
                        error=e,
                    )
                )
                raise

        logger.info(
            f"Vector indexes recreated successfully with dimension {new_dimension}"
        )

    def add_item(self, id: int, embedding: list[float], metadata: dict) -> None:
        """Add a single entity embedding to the vector store.

        Args:
            id: Memgraph internal node ID
            embedding: Vector embedding
            metadata: Entity metadata
        """
        cypher = """
        MATCH (n) WHERE id(n) = $node_id
        SET n.embedding = $embedding,
            n.embedding_model = $model_name,
            n.embedding_version = $version
        SET n += $metadata
        """

        params = {
            "node_id": id,
            "embedding": embedding,
            "model_name": settings.EMBEDDING_MODEL,
            "version": EMBEDDING_VERSION,
            "metadata": metadata,
        }

        self._execute_query(cypher, params)

    def close(self) -> None:
        """Close Memgraph connection."""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info(ls.MG_VECTOR_CLOSED)
            except Exception:
                pass
            self._conn = None
