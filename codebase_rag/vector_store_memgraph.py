"""Memgraph native vector storage backend.

Uses Memgraph's built-in vector index support (v3.0.0+) via vector_search module.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from contextlib import contextmanager

import mgclient
from loguru import logger

from . import logs as ls
from .config import settings
from .graph.query_generator import MemgraphQueryGenerator
from .vector_backend import VectorBackend

# Model info
UNIXCODER_MODEL = "microsoft/unixcoder-base"
EMBEDDING_VERSION = 1


class MemgraphBackend(VectorBackend):
    """Memgraph native vector storage using vector_search module.

    Stores embeddings directly on node properties and uses Memgraph's
    built-in vector index for similarity search.

    Advantages:
    - Single database (no separate Qdrant container)
    - Hybrid queries: vector search + graph traversal in one Cypher query
    - Lower latency: no cross-database coordination
    """

    LABELS_TO_INDEX = (
        "Function",
        "Method",
        "Class",
        "Interface",
        "Contract",
        "Library",
        "Enum",
        "Type",
        "Union",
        "Event",
        "Modifier",
        "StateVariable",
        "CustomError",
        "Hotkey",
        "Hotstring",
        "Label",
        "AhkClass",
    )

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
        """
        logger.info(ls.MG_VECTOR_INIT.format(index=settings.MEMGRAPH_VECTOR_INDEX_NAME))

        capabilities = self.query_generator.capabilities
        for label in self.LABELS_TO_INDEX:
            index_name = f"{label.lower()}_embedding_index"

            try:
                if capabilities.supports_vector_index:
                    # Use query generator to create compatible index creation query
                    cypher, params = (
                        self.query_generator.generate_vector_index_creation_query(
                            index_name=index_name,
                            node_label=label,
                            vector_property="embedding",
                            vector_dim=settings.MEMGRAPH_VECTOR_DIM,
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
                        dim=settings.MEMGRAPH_VECTOR_DIM,
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

        try:
            results = self._execute_query(cypher, params)
            logger.debug(f"store_batch query results: {results}")
            stored = results[0].get("stored", 0) if results else 0
            logger.debug(ls.EMBEDDING_BATCH_STORED.format(count=stored))
            return stored
        except Exception as e:
            logger.warning(ls.EMBEDDING_BATCH_FAILED.format(error=e))
            return 0

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        include_context: bool = False,
        max_context_depth: int = 2,
    ) -> list[tuple[int, float]] | list[dict]:
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
                        WHERE ($project_prefix IS NULL OR start_node.qualified_name STARTS WITH $project_prefix)

                        MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS (1..max_depth)]-(related)
                        WHERE related:Function OR related:Class OR related:Module

                        WITH
                            DISTINCT related,
                            sim,
                            COALESCE(related.pagerank_score, 0.1) AS pr_score,
                            collect(DISTINCT [n IN nodes(path) | n.qualified_name]) AS context_paths
                        ORDER BY (sim * 0.7) + (pr_score * 0.3) DESC
                        LIMIT top_k

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
                        WHERE ($project_prefix IS NULL OR n.qualified_name STARTS WITH $project_prefix)
                        WITH n, sim, COALESCE(n.pagerank_score, 0.1) AS pr_score
                        ORDER BY (sim * 0.7) + (pr_score * 0.3) DESC
                        LIMIT top_k
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
                        WITH start_node, {capabilities.vector_function_syntax}(start_node.embedding, query_vec) AS sim
                        ORDER BY sim DESC
                        LIMIT top_k * 3

                        MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS *BFS (1..max_depth)]-(related)
                        WHERE related:Function OR related:Class OR related:Module

                        WITH
                            DISTINCT related,
                            sim,
                            COALESCE(related.pagerank_score, 0.1) AS pr_score,
                            collect(DISTINCT [n IN nodes(path) | n.qualified_name]) AS context_paths
                        ORDER BY (sim * 0.7) + (pr_score * 0.3) DESC
                        LIMIT top_k

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
                    "backend": "memgraph",
                    "total_embeddings": results[0].get("total_embeddings", 0),
                    "node_types": results[0].get("node_types", 0),
                    "dimension": results[0].get("max_dimension", 0),
                    "models": results[0].get("models", []),
                }
        except Exception:
            pass
        return {"backend": "memgraph", "total_embeddings": 0}

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

    def recreate_vector_indexes(self, new_dimension: int) -> None:
        """Drop and recreate all vector indexes with new dimension.

        This is needed when switching embedding models with different dimensions.
        WARNING: Existing embeddings will become incompatible after this operation.

        Args:
            new_dimension: New vector dimension for the indexes.
        """
        logger.info(f"Recreating vector indexes with dimension {new_dimension}...")

        for label in self.LABELS_TO_INDEX:
            index_name = f"{label.lower()}_embedding_index"

            # Drop existing index (Memgraph doesn't support IF EXISTS)
            drop_cypher = f"DROP VECTOR INDEX {index_name};"
            try:
                self._execute_query(drop_cypher)
                logger.debug(f"Dropped index {index_name}")
            except Exception as e:
                # Index may not exist, which is fine
                logger.debug(f"Could not drop index {index_name}: {e}")

            # Create new index with updated dimension
            create_cypher = f"""
            CREATE VECTOR INDEX {index_name}
            ON :{label}(embedding)
            WITH CONFIG {{
                "dimension": {new_dimension},
                "capacity": {settings.MEMGRAPH_VECTOR_CAPACITY},
                "metric": "{settings.MEMGRAPH_VECTOR_METRIC}"
            }};
            """
            try:
                self._execute_query(create_cypher)
                logger.info(
                    ls.MG_VECTOR_INDEX_CREATED.format(
                        index=index_name,
                        label=label,
                        dim=new_dimension,
                        capacity=settings.MEMGRAPH_VECTOR_CAPACITY,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to create index {index_name}: {e}")
                raise

        logger.info(f"Vector indexes recreated with dimension {new_dimension}")

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
