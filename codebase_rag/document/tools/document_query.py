"""Query document graph tool.

Provides Cypher queries for document graph retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ...services.graph_service import MemgraphIngestor


def query_document_graph(
    ingestor: MemgraphIngestor,
    query: str,
    workspace: str = "default",
) -> list[dict]:
    """
    Execute a query on the document graph.

    Args:
        ingestor: Graph service instance
        query: Cypher query string
        workspace: Workspace filter

    Returns:
        List of matching documents/sections/chunks
    """
    # Prepend workspace filter to query
    workspace_filter = f"workspace = '{workspace}'"

    # Add workspace filter if not present
    if "workspace" not in query:
        # Inject workspace filter into WHERE clause or add one
        if "WHERE" in query.upper():
            query = query.replace("WHERE", f"WHERE {workspace_filter} AND", 1)
        elif "MATCH" in query.upper():
            # Find position to insert WHERE after MATCH pattern
            match_end = query.upper().find("RETURN")
            if match_end != -1:
                query = (
                    query[:match_end] + f" WHERE {workspace_filter} " + query[match_end:]
                )

    try:
        results = ingestor.execute_query(query)
        logger.debug(f"Document graph query returned {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Document graph query failed: {e}")
        raise


def get_document_by_path(
    ingestor: MemgraphIngestor,
    document_path: str,
    workspace: str = "default",
) -> dict | None:
    """Get a document node by its path."""
    query = """
    MATCH (d:Document)
    WHERE d.path = $path AND d.workspace = $workspace
    RETURN d
    """
    results = ingestor.execute_query(
        query, parameters={"path": document_path, "workspace": workspace}
    )
    return results[0] if results else None


def get_document_sections(
    ingestor: MemgraphIngestor,
    document_path: str,
    workspace: str = "default",
) -> list[dict]:
    """Get all sections for a document."""
    query = """
    MATCH (d:Document)-[:CONTAINS_SECTION]->(s:Section)
    WHERE d.path = $path AND d.workspace = $workspace
    RETURN s
    ORDER BY s.start_line
    """
    return ingestor.execute_query(
        query, parameters={"path": document_path, "workspace": workspace}
    )


def get_section_chunks(
    ingestor: MemgraphIngestor,
    section_qn: str,
    workspace: str = "default",
) -> list[dict]:
    """Get all chunks for a section."""
    query = """
    MATCH (s:Section)-[:CONTAINS_CHUNK]->(c:Chunk)
    WHERE s.qualified_name = $qn AND s.workspace = $workspace
    RETURN c
    ORDER BY c.chunk_index
    """
    return ingestor.execute_query(
        query, parameters={"qn": section_qn, "workspace": workspace}
    )


__all__ = [
    "query_document_graph",
    "get_document_by_path",
    "get_document_sections",
    "get_section_chunks",
]