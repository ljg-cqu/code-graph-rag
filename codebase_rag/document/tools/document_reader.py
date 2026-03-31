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
    include_subsections: bool = True,
) -> dict:
    """
    Read document content from graph.

    Args:
        ingestor: Graph service instance
        document_path: Path to document
        workspace: Workspace filter
        include_sections: Include section content
        include_chunks: Include chunk content
        include_subsections: When True, include chunks from all nested subsections
            (traverses HAS_SUBSECTION*). When False, only include chunks from
            top-level sections.

    Returns:
        Document content with optional sections and chunks
    """
    query = """
    MATCH (d:Document)
    WHERE d.path = $path AND d.workspace = $workspace
    RETURN d
    """

    results = ingestor.fetch_all(
        query, params={"path": document_path, "workspace": workspace}
    )

    if not results:
        logger.warning(f"Document not found: {document_path}")
        return {"error": "Document not found", "path": document_path}

    document = results[0]
    # Backward compatibility: check both new and old property names
    # Old property name was 'section_count', new is 'total_section_count'
    total_section_count = document.get("total_section_count")
    if total_section_count is None:
        total_section_count = document.get("section_count")

    content = {
        "path": document_path,
        "file_type": document.get("file_type"),
        "word_count": document.get("word_count"),
        "total_section_count": total_section_count,
        "modified_date": document.get("modified_date"),
    }

    if include_sections:
        content["sections"] = _get_sections_with_content(
            ingestor, document_path, workspace, include_chunks, include_subsections
        )

    return content


def _get_sections_with_content(
    ingestor: MemgraphIngestor,
    document_path: str,
    workspace: str,
    include_chunks: bool,
    include_subsections: bool = True,
) -> list[dict]:
    """Get sections with their content.

    Args:
        ingestor: Graph service instance
        document_path: Path to document
        workspace: Workspace filter
        include_chunks: Include chunk content in sections
        include_subsections: When True, traverse HAS_SUBSECTION* to get all nested sections.
            When False, only return top-level sections (direct CONTAINS_SECTION children).

    Returns:
        ALL sections including nested subsections by traversing HAS_SUBSECTION relationships.
    """
    # Query traverses the full hierarchy: Document -> Section (via CONTAINS_SECTION)
    # and Section -> Section (via HAS_SUBSECTION*)
    if include_subsections:
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
    else:
        # Only top-level sections (direct CONTAINS_SECTION children)
        query = """
        MATCH (d:Document)-[:CONTAINS_SECTION]->(s:Section)
        WHERE d.path = $path AND d.workspace = $workspace
        RETURN s
        ORDER BY s.start_line
        """

    sections = ingestor.fetch_all(
        query, params={"path": document_path, "workspace": workspace}
    )

    section_list = []
    for section in sections:
        section_data = {
            "title": section.get("title"),
            "level": section.get("level"),
            "start_line": section.get("start_line"),
            "end_line": section.get("end_line"),
            "content": section.get("content_snippet"),  # Section stores content_snippet
            "qualified_name": section.get("qualified_name"),
        }

        if include_chunks:
            section_data["chunks"] = _get_chunks_for_section(
                ingestor, section.get("qualified_name"), workspace, include_subsections
            )

        section_list.append(section_data)

    return section_list


def _get_chunks_for_section(
    ingestor: MemgraphIngestor,
    section_qn: str,
    workspace: str,
    include_subsections: bool = True,
) -> list[dict]:
    """Get chunks for a section.

    Args:
        ingestor: Graph service instance
        section_qn: Section qualified name
        workspace: Workspace filter
        include_subsections: When True, traverse HAS_SUBSECTION* to collect chunks from
            the entire section subtree. When False, only get chunks directly belonging
            to this section.

    Returns:
        Chunks for the section, optionally including nested subsection chunks.
    """
    if include_subsections:
        # Get chunks directly belonging to this section OR any of its subsections
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
        # Only chunks directly belonging to this section
        query = """
        MATCH (c:Chunk)-[:BELONGS_TO_SECTION]->(s:Section)
        WHERE s.qualified_name = $qn AND s.workspace = $workspace
        RETURN c
        ORDER BY c.start_line
        """

    return ingestor.fetch_all(
        query, params={"qn": section_qn, "workspace": workspace}
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

    results = ingestor.fetch_all(
        query, params={"qn": section_qn, "workspace": workspace}
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

    results = ingestor.fetch_all(
        query, params={"qn": chunk_qn, "workspace": workspace}
    )

    if not results:
        return None

    return results[0]


__all__ = [
    "read_document_content",
    "read_section_content",
    "read_chunk_content",
]