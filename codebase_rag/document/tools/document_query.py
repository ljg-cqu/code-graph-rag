"""Query document graph tool.

Provides Cypher queries for document graph retrieval.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ...services.graph_service import MemgraphIngestor

# Workspace must be a safe identifier (alphanumeric, underscore, hyphen)
WORKSPACE_PATTERN = re.compile(r"^[\w\-]+$")


def _validate_workspace(workspace: str) -> str:
    """Validate workspace identifier for safe Cypher injection.

    Args:
        workspace: Workspace name to validate

    Returns:
        Validated workspace name

    Raises:
        ValueError: If workspace contains unsafe characters
    """
    if not WORKSPACE_PATTERN.match(workspace):
        raise ValueError(
            f"Invalid workspace identifier: '{workspace}'. "
            "Must contain only alphanumeric characters, underscores, and hyphens."
        )
    return workspace


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
        workspace: Workspace filter (validated for safety)

    Returns:
        List of matching documents/sections/chunks

    Raises:
        ValueError: If workspace identifier is invalid
    """
    # Validate workspace to prevent Cypher injection
    workspace = _validate_workspace(workspace)

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
        results = ingestor.fetch_all(query, params={})
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
    results = ingestor.fetch_all(
        query, params={"path": document_path, "workspace": workspace}
    )
    return results[0] if results else None


def get_document_sections(
    ingestor: MemgraphIngestor,
    document_path: str,
    workspace: str = "default",
) -> list[dict]:
    """Get all sections for a document, including nested subsections.

    Traverses CONTAINS_SECTION for top-level sections and HAS_SUBSECTION* for nested ones.
    """
    query = """
    MATCH (d:Document)-[:CONTAINS_SECTION]->(s:Section)
    WHERE d.path = $path AND d.workspace = $workspace
    RETURN s
    UNION
    MATCH (d:Document)-[:CONTAINS_SECTION]->(:Section)-[:HAS_SUBSECTION*]->(s:Section)
    WHERE d.path = $path AND d.workspace = $workspace
    RETURN s
    ORDER BY s.start_line
    """
    return ingestor.fetch_all(
        query, params={"path": document_path, "workspace": workspace}
    )


def get_section_chunks(
    ingestor: MemgraphIngestor,
    section_qn: str,
    workspace: str = "default",
    include_subsections: bool = True,
) -> list[dict]:
    """Get all chunks for a section.

    Args:
        ingestor: Graph service
        section_qn: Section qualified name
        workspace: Workspace filter
        include_subsections: If True, include chunks from all nested subsections

    Returns:
        List of chunks ordered by start_line

    Note:
        Chunks have BELONGS_TO_SECTION relationship to their containing section.
        When include_subsections=True, also traverses HAS_SUBSECTION* to get
        chunks from the entire section subtree.
    """
    if include_subsections:
        query = """
        MATCH (c:Chunk)-[:BELONGS_TO_SECTION]->(s:Section)
        WHERE s.qualified_name = $qn AND s.workspace = $workspace
        RETURN c
        UNION
        MATCH (parent:Section)-[:HAS_SUBSECTION*]->(sub:Section)
        WHERE parent.qualified_name = $qn AND parent.workspace = $workspace
        MATCH (c:Chunk)-[:BELONGS_TO_SECTION]->(sub)
        RETURN c
        ORDER BY c.start_line
        """
    else:
        query = """
        MATCH (c:Chunk)-[:BELONGS_TO_SECTION]->(s:Section)
        WHERE s.qualified_name = $qn AND s.workspace = $workspace
        RETURN c
        ORDER BY c.start_line
        """
    return ingestor.fetch_all(
        query, params={"qn": section_qn, "workspace": workspace}
    )


__all__ = [
    "query_document_graph",
    "get_document_by_path",
    "get_document_sections",
    "get_section_chunks",
]