"""Semantic search in documents tool.

Provides vector similarity search for document chunks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ...embeddings import get_embedding_provider
from ...config import settings

if TYPE_CHECKING:
    from ...services.graph_service import MemgraphIngestor
    from ...vector_backend import VectorBackend


def document_semantic_search(
    query: str,
    ingestor: MemgraphIngestor | None = None,
    vector_backend: VectorBackend | None = None,
    workspace: str = "default",
    limit: int = 10,
    min_similarity: float = 0.7,
) -> list[dict]:
    """
    Search documents using semantic similarity.

    Args:
        query: Natural language query
        ingestor: Graph service for retrieving results
        vector_backend: Vector backend for similarity search
        workspace: Workspace filter
        limit: Maximum results to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of matching chunks with similarity scores
    """
    # Get embedding for query using active embedding config
    config = settings.active_embedding_config
    provider = get_embedding_provider(
        provider=config.provider,
        model_id=config.model_id,
        api_key=config.api_key,
        endpoint=config.endpoint,
        keep_alive=config.keep_alive,
        project_id=config.project_id,
        region=config.region,
        provider_type=config.provider_type,
        service_account_file=config.service_account_file,
        device=config.device,
    )
    query_embedding = provider.embed(query)

    # Validate embedding dimension
    expected_dimension = provider.dimension
    if expected_dimension and len(query_embedding) != expected_dimension:
        raise ValueError(
            f"Query embedding dimension mismatch: expected {expected_dimension}, "
            f"got {len(query_embedding)}. Check EMBEDDING_MODEL configuration."
        )

    # Use vector backend for similarity search
    if vector_backend and ingestor:
        # Use vector backend for similarity, then fetch chunk details from graph
        backend_results = vector_backend.search(
            query_embedding=query_embedding,
            top_k=limit * 2,  # Fetch more to account for min_similarity filtering
            filters={"workspace": workspace},
        )
        # Filter by min_similarity and get node IDs
        filtered_results = [
            (nid, sim) for nid, sim in backend_results
            if sim >= min_similarity
        ][:limit]

        if filtered_results:
            # Fetch chunk details from graph
            node_ids = [nid for nid, _ in filtered_results]
            similarity_map = {nid: sim for nid, sim in filtered_results}

            # Query to get chunk details by node IDs
            chunk_query = """
            MATCH (d:Document)-[:CONTAINS_CHUNK]->(c:Chunk)
            WHERE id(c) IN $node_ids
            OPTIONAL MATCH (c)-[:BELONGS_TO_SECTION]->(s:Section)
            RETURN
                c.content as content,
                c.qualified_name as chunk_qn,
                c.start_line as chunk_start_line,
                s.title as section_title,
                s.qualified_name as section_qn,
                d.path as document_path,
                id(c) as node_id
            """
            chunks = ingestor.fetch_all(chunk_query, {"node_ids": node_ids})

            # Combine similarity scores with chunk data
            results = []
            for chunk in chunks:
                chunk["similarity"] = similarity_map.get(chunk.get("node_id", 0), 0.0)
                results.append(chunk)
        else:
            results = []
    elif ingestor:
        # Fallback to Memgraph native vector search
        results = _search_memgraph_native(
            ingestor,
            query_embedding,
            workspace,
            limit,
            min_similarity,
        )
    else:
        raise ValueError("Either ingestor or vector_backend must be provided")

    logger.debug(f"Semantic search found {len(results)} results for: {query[:50]}...")
    return results


def _search_memgraph_native(
    ingestor: MemgraphIngestor,
    embedding: list[float],
    workspace: str,
    limit: int,
    min_similarity: float,
) -> list[dict]:
    """Use Memgraph's native vector search.

    Output fields per Memgraph spec: node, distance, similarity

    Graph structure:
    - Document-[:CONTAINS_CHUNK]->Chunk
    - Chunk-[:BELONGS_TO_SECTION]->Section (optional)
    """
    query = """
    CALL vector_search.search(
        'doc_embeddings',
        $limit,
        $embedding
    ) YIELD node, distance, similarity
    WITH node, distance, similarity
    WHERE node.workspace = $workspace AND similarity >= $min_similarity
    MATCH (d:Document)-[:CONTAINS_CHUNK]->(node)
    OPTIONAL MATCH (node)-[:BELONGS_TO_SECTION]->(s:Section)
    RETURN
        node.content as content,
        node.qualified_name as chunk_qn,
        node.start_line as chunk_start_line,
        s.title as section_title,
        s.qualified_name as section_qn,
        d.path as document_path,
        similarity
    ORDER BY similarity DESC
    """

    return ingestor.fetch_all(
        query,
        params={
            "embedding": embedding,
            "workspace": workspace,
            "limit": limit,
            "min_similarity": min_similarity,
        },
    )


def search_documents_by_keywords(
    ingestor: MemgraphIngestor,
    keywords: list[str],
    workspace: str = "default",
    limit: int = 20,
) -> list[dict]:
    """
    Search documents by keyword matching (fallback when embeddings unavailable).

    Args:
        ingestor: Graph service
        keywords: List of keywords to match
        workspace: Workspace filter
        limit: Maximum results

    Returns:
        List of matching documents/sections

    Note:
        Graph structure:
        - Document-[:CONTAINS_CHUNK]->Chunk
        - Chunk-[:BELONGS_TO_SECTION]->Section (optional)
    """
    # Build keyword search pattern using parameterized query
    # Each keyword is passed as a separate parameter to prevent injection
    query = """
    MATCH (d:Document)-[:CONTAINS_CHUNK]->(c:Chunk)
    WHERE d.workspace = $workspace
    AND ANY(kw IN $keywords WHERE c.content CONTAINS kw)
    OPTIONAL MATCH (c)-[:BELONGS_TO_SECTION]->(s:Section)
    RETURN
        c.content as content,
        c.qualified_name as chunk_qn,
        s.title as section_title,
        d.path as document_path,
        d.file_type as file_type
    LIMIT $limit
    """

    return ingestor.fetch_all(
        query,
        params={"workspace": workspace, "keywords": keywords, "limit": limit},
    )


__all__ = [
    "document_semantic_search",
    "search_documents_by_keywords",
]