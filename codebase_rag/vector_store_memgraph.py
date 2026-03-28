"""Memgraph native vector storage backend.

Uses Memgraph's built-in vector index support (v3.0.0+) via vector_search module.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Generator

import mgclient
from loguru import logger

from . import logs as ls
from .config import settings
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

    LABELS_TO_INDEX = ("Function", "Method", "Class", "Interface", "Contract", "Library")

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

    def _execute_query(
        self, query: str, params: dict | None = None
    ) -> list[dict]:
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

    def initialize(self) -> None:
        """Create vector indexes for embeddable node types.

        Uses Memgraph's CREATE VECTOR INDEX syntax with required capacity.
        Index is created for each embeddable label (Function, Method, etc.)
        """
        logger.info(ls.MG_VECTOR_INIT.format(index=settings.MEMGRAPH_VECTOR_INDEX_NAME))

        for label in self.LABELS_TO_INDEX:
            index_name = f"{label.lower()}_embedding_index"
            cypher = f"""
            CREATE VECTOR INDEX {index_name}
            ON :{label}(embedding)
            WITH CONFIG {{
                "dimension": {settings.MEMGRAPH_VECTOR_DIM},
                "capacity": {settings.MEMGRAPH_VECTOR_CAPACITY},
                "metric": "{settings.MEMGRAPH_VECTOR_METRIC}"
            }};
            """

            try:
                self._execute_query(cypher)
                logger.info(
                    ls.MG_VECTOR_INDEX_CREATED.format(
                        index=f"{label.lower()}_embedding_index",
                        label=label,
                        dim=settings.MEMGRAPH_VECTOR_DIM,
                        capacity=settings.MEMGRAPH_VECTOR_CAPACITY,
                    )
                )
            except Exception as e:
                error_str = str(e).lower()
                if "already exists" in error_str or "duplicate" in error_str:
                    logger.info(
                        ls.MG_VECTOR_INDEX_EXISTS.format(
                            index=f"{label.lower()}_embedding_index"
                        )
                    )
                else:
                    logger.error(
                        ls.MG_VECTOR_INDEX_FAILED.format(
                            index=f"{label.lower()}_embedding_index",
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

    def store_batch(
        self, points: Sequence[tuple[int, list[float], str]]
    ) -> int:
        """Store embeddings as node properties.

        Updates nodes by their internal ID, setting embedding property.
        """
        if not points:
            return 0

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
            "points": [
                {"node_id": nid, "embedding": emb}
                for nid, emb, _ in points
            ],
            "model_name": UNIXCODER_MODEL,
            "version": EMBEDDING_VERSION,
        }

        try:
            results = self._execute_query(cypher, params)
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
    ) -> list[tuple[int, float]]:
        """Search using native vector_search.search procedure.

        Searches across all label-specific indexes (function_embedding_index,
        method_embedding_index, etc.) and combines results.

        Note: Parameter order is (index_name, limit, query_vector)
        Returns nodes sorted by distance (ascending).
        """
        # Use per-label index search directly (avoids fallback overhead)
        effective_top_k = top_k if top_k > 0 else settings.VECTOR_SEARCH_TOP_K
        return self._search_with_fallback(query_embedding, effective_top_k, filters)

    def _search_with_fallback(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[tuple[int, float]]:
        """Search across individual label indexes and combine results."""
        results: list[tuple[int, float]] = []
        project_prefix = filters.get("project_prefix") if filters else None

        for label in self.LABELS_TO_INDEX:
            index_name = f"{label.lower()}_embedding_index"

            # Fetch more results if filtering by project prefix
            fetch_count = top_k * 3 if project_prefix else top_k

            if project_prefix:
                cypher = """
                CALL vector_search.search($index_name, $fetch_count, $embedding)
                YIELD node, distance, similarity
                WHERE node.qualified_name STARTS WITH $project_prefix
                RETURN id(node) AS node_id, similarity
                ORDER BY distance ASC;
                """
            else:
                cypher = """
                CALL vector_search.search($index_name, $top_k, $embedding)
                YIELD node, distance, similarity
                RETURN id(node) AS node_id, similarity
                ORDER BY distance ASC;
                """

            params = {
                "index_name": index_name,
                "embedding": query_embedding,
                "top_k": top_k,
                "fetch_count": fetch_count,
                "project_prefix": project_prefix,
            }

            try:
                label_results = self._execute_query(cypher, params)
                for r in label_results:
                    results.append((int(r["node_id"]), float(r.get("similarity", 0.0))))
            except Exception:
                continue  # Skip if index doesn't exist

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

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

    def close(self) -> None:
        """Close Memgraph connection."""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info(ls.MG_VECTOR_CLOSED)
            except Exception:
                pass
            self._conn = None