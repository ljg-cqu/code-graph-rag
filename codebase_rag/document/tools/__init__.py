"""Document-specific MCP tools.

Tools for document graph queries and search.
"""

from .document_query import (
    query_document_graph,
    get_document_by_path,
    get_document_sections,
    get_section_chunks,
)
from .document_search import (
    document_semantic_search,
    search_documents_by_keywords,
)
from .document_reader import (
    read_document_content,
    read_section_content,
    read_chunk_content,
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