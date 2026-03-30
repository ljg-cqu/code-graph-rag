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
    # Get embedding for query
    provider = get_embedding_provider(
        provider=settings.EMBEDDING_PROVIDER,
        model_id=settings.EMBEDDING_MODEL,
    )
    query_embedding = provider.embed(query)

    # Use vector backend for similarity search
    if vector_backend:
        results = vector_backend.search(
            embedding=query_embedding,
            limit=limit,
            min_similarity=min_similarity,
            node_label="Chunk",
            filters={"workspace": workspace},
        )
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
    """Use Memgraph's native vector search."""
    # Convert embedding to string format for Cypher
    embedding_str = str(embedding)

    query = """
    CALL vector_search.search(
        'chunk_embeddings',
        $embedding,
        $limit
    ) YIELD node, score
    WHERE node.workspace = $workspace AND score >= $min_similarity
    MATCH (d:Document)-[:CONTAINS_SECTION]->(s:Section)-[:CONTAINS_CHUNK]->(node)
    RETURN
        node.content as content,
        node.qualified_name as chunk_qn,
        node.chunk_index as chunk_index,
        s.title as section_title,
        s.qualified_name as section_qn,
        d.path as document_path,
        score
    ORDER BY score DESC
    """

    return ingestor.execute_query(
        query,
        parameters={
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
    """
    # Build keyword search pattern
    keyword_pattern = " OR ".join(
        [f"c.content CONTAINS '{kw}'" for kw in keywords]
    )

    query = f"""
    MATCH (d:Document)-[:CONTAINS_SECTION]->(s:Section)-[:CONTAINS_CHUNK]->(c:Chunk)
    WHERE d.workspace = $workspace AND ({keyword_pattern})
    RETURN
        c.content as content,
        c.qualified_name as chunk_qn,
        s.title as section_title,
        d.path as document_path,
        d.file_type as file_type
    LIMIT $limit
    """

    return ingestor.execute_query(
        query,
        parameters={"workspace": workspace, "limit": limit},
    )


__all__ = [
    "document_semantic_search",
    "search_documents_by_keywords",
]