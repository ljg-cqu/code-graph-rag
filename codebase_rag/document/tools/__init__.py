"""Document-specific MCP tools.

Tools for document graph queries and search.
"""

from .document_query import (
    get_document_by_path,
    get_document_sections,
    get_section_chunks,
    query_document_graph,
)
from .document_reader import (
    read_chunk_content,
    read_document_content,
    read_section_content,
)
from .document_search import (
    document_semantic_search,
    search_documents_by_keywords,
)

__all__ = [
    "query_document_graph",
    "get_document_by_path",
    "get_document_sections",
    "get_section_chunks",
    "document_semantic_search",
    "search_documents_by_keywords",
    "read_document_content",
    "read_section_content",
    "read_chunk_content",
]
