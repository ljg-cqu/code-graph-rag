"""Read document content tool.

Provides retrieval of document content from graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ...services.graph_service import MemgraphIngestor


def read_document_content(
    ingestor: MemgraphIngestor,
    document_path: str,
    workspace: str = "default",
    include_sections: bool = True,
    include_chunks: bool = False,
) -> dict:
    """
    Read document content from graph.

    Args:
        ingestor: Graph service instance
        document_path: Path to document
        workspace: Workspace filter
        include_sections: Include section content
        include_chunks: Include chunk content

    Returns:
        Document content with optional sections and chunks
    """
    query = """
    MATCH (d:Document)
    WHERE d.path = $path AND d.workspace = $workspace
    RETURN d
    """

    results = ingestor.execute_query(
        query, parameters={"path": document_path, "workspace": workspace}
    )

    if not results:
        logger.warning(f"Document not found: {document_path}")
        return {"error": "Document not found", "path": document_path}

    document = results[0]
    content = {
        "path": document_path,
        "file_type": document.get("file_type"),
        "word_count": document.get("word_count"),
        "section_count": document.get("section_count"),
        "modified_date": document.get("modified_date"),
    }

    if include_sections:
        content["sections"] = _get_sections_with_content(
            ingestor, document_path, workspace, include_chunks
        )

    return content


def _get_sections_with_content(
    ingestor: MemgraphIngestor,
    document_path: str,
    workspace: str,
    include_chunks: bool,
) -> list[dict]:
    """Get sections with their content."""
    query = """
    MATCH (d:Document)-[:CONTAINS_SECTION]->(s:Section)
    WHERE d.path = $path AND d.workspace = $workspace
    RETURN s
    ORDER BY s.start_line
    """

    sections = ingestor.execute_query(
        query, parameters={"path": document_path, "workspace": workspace}
    )

    section_list = []
    for section in sections:
        section_data = {
            "title": section.get("title"),
            "level": section.get("level"),
            "start_line": section.get("start_line"),
            "end_line": section.get("end_line"),
            "content": section.get("content_snippet"),  # Section stores content_snippet
        }

        if include_chunks:
            section_data["chunks"] = _get_chunks_for_section(
                ingestor, section.get("qualified_name"), workspace
            )

        section_list.append(section_data)

    return section_list


def _get_chunks_for_section(
    ingestor: MemgraphIngestor,
    section_qn: str,
    workspace: str,
) -> list[dict]:
    """Get chunks for a section.

    Note: Chunks have BELONGS_TO_SECTION relationship to their containing section.
    """
    query = """
    MATCH (c:Chunk)-[:BELONGS_TO_SECTION]->(s:Section)
    WHERE s.qualified_name = $qn AND s.workspace = $workspace
    RETURN c
    ORDER BY c.start_line
    """

    return ingestor.execute_query(
        query, parameters={"qn": section_qn, "workspace": workspace}
    )


def read_section_content(
    ingestor: MemgraphIngestor,
    section_qn: str,
    workspace: str = "default",
) -> dict | None:
    """
    Read a specific section's content.

    Args:
        ingestor: Graph service
        section_qn: Section qualified name (path#title)
        workspace: Workspace filter

    Returns:
        Section content or None if not found
    """
    query = """
    MATCH (s:Section)
    WHERE s.qualified_name = $qn AND s.workspace = $workspace
    RETURN s
    """

    results = ingestor.execute_query(
        query, parameters={"qn": section_qn, "workspace": workspace}
    )

    if not results:
        return None

    return results[0]


def read_chunk_content(
    ingestor: MemgraphIngestor,
    chunk_qn: str,
    workspace: str = "default",
) -> dict | None:
    """
    Read a specific chunk's content.

    Args:
        ingestor: Graph service
        chunk_qn: Chunk qualified name
        workspace: Workspace filter

    Returns:
        Chunk content or None if not found
    """
    query = """
    MATCH (c:Chunk)
    WHERE c.qualified_name = $qn AND c.workspace = $workspace
    RETURN c
    """

    results = ingestor.execute_query(
        query, parameters={"qn": chunk_qn, "workspace": workspace}
    )

    if not results:
        return None

    return results[0]


__all__ = [
    "read_document_content",
    "read_section_content",
    "read_chunk_content",
]