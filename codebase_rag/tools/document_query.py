"""Document GraphRAG tools for interactive agent.

Provides Pydantic AI Tool factories for querying the document graph
and merged queries across code and document graphs.
"""

from __future__ import annotations

from loguru import logger
from pydantic_ai import Tool

from .. import constants as cs
from ..config import settings
from ..shared.query_router import QueryMode, QueryRequest, QueryRouter
from . import tool_descriptions as td


def create_query_document_graph_tool(
    query_router: QueryRouter | None = None,
) -> Tool:
    """Create query_document_graph tool for Pydantic AI agent.

    Usage: Query the DOCUMENT graph/vector ONLY.
    Example: "How do I use the authentication API?"
    """

    async def query_document_graph(
        natural_language_query: str,
        top_k: int = 5,
    ) -> str:
        """Query document graph with natural language.

        Args:
            natural_language_query: Question about documentation
            top_k: Maximum results to return

        Returns:
            Formatted results from document graph
        """
        logger.info(f"Querying document graph: {natural_language_query[:50]}...")

        # Create router if not provided (standalone usage)
        router = query_router
        if router is None:
            router = _create_document_query_router()
            if router is None:
                return cs.MSG_SEMANTIC_NO_RESULTS.format(query=natural_language_query)

        request = QueryRequest(
            question=natural_language_query,
            mode=QueryMode.DOCUMENT_ONLY,
            top_k=top_k,
        )

        try:
            response = router.query(request)

            if not response.sources:
                return f"No relevant documents found for: {natural_language_query}"

            # Format response for agent
            result_lines = [f"**Document Query Results:**\n"]
            for i, source in enumerate(response.sources, 1):
                result_lines.append(
                    f"{i}. **{source.qualified_name or source.path}** "
                    f"({source.node_type or 'Section'})"
                )
                if source.line_range:
                    result_lines.append(f"   Lines: {source.line_range[0]}-{source.line_range[1]}")

            result_lines.append(f"\n\n**Answer:**\n{response.answer}")

            return "\n".join(result_lines)

        except Exception as e:
            logger.error(f"Document query failed: {e}")
            return f"Document query failed: {e}"

    return Tool(
        query_document_graph,
        name=td.AgenticToolName.QUERY_DOCUMENT_GRAPH,
        description=td.QUERY_DOCUMENT_GRAPH,
    )


def create_query_both_graphs_tool(
    query_router: QueryRouter | None = None,
) -> Tool:
    """Create query_both_graphs tool for Pydantic AI agent.

    Usage: Query BOTH code and document graphs, merge results.
    Example: "Tell me everything about authentication"
    """

    async def query_both_graphs(
        natural_language_query: str,
        top_k: int = 5,
    ) -> str:
        """Query both code and document graphs with merged results.

        Args:
            natural_language_query: Question spanning code and docs
            top_k: Maximum results per graph

        Returns:
            Merged results from both graphs
        """
        logger.info(f"Querying both graphs: {natural_language_query[:50]}...")

        # Create router if not provided (standalone usage)
        router = query_router
        if router is None:
            router = _create_merged_query_router()
            if router is None:
                return f"Graph connections not available for: {natural_language_query}"

        request = QueryRequest(
            question=natural_language_query,
            mode=QueryMode.BOTH_MERGED,
            top_k=top_k,
        )

        try:
            response = router.query(request)

            # Separate sources by type
            code_sources = [s for s in response.sources if s.type == "code"]
            doc_sources = [s for s in response.sources if s.type == "document"]

            result_lines = ["**Merged Query Results:**\n"]

            if code_sources:
                result_lines.append("**Code Results:**")
                for i, source in enumerate(code_sources, 1):
                    result_lines.append(
                        f"{i}. `{source.qualified_name or source.path}` "
                        f"({source.node_type or 'Function'})"
                    )

            if doc_sources:
                result_lines.append("\n**Document Results:**")
                for i, source in enumerate(doc_sources, 1):
                    result_lines.append(
                        f"{i}. **{source.qualified_name or source.path}**"
                    )

            result_lines.append(f"\n\n**Answer:**\n{response.answer}")

            if response.warnings:
                result_lines.append(f"\n\n**Warnings:** {', '.join(response.warnings)}")

            return "\n".join(result_lines)

        except Exception as e:
            logger.error(f"Merged query failed: {e}")
            return f"Merged query failed: {e}"

    return Tool(
        query_both_graphs,
        name=td.AgenticToolName.QUERY_BOTH_GRAPHS,
        description=td.QUERY_BOTH_GRAPHS,
    )


def _create_document_query_router() -> QueryRouter | None:
    """Create QueryRouter with document graph connection.

    Used for standalone tool usage when router not injected.
    """
    try:
        from ..services.graph_service import MemgraphIngestor

        # Connect to document graph
        doc_ingestor = MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
            batch_size=100,
        )

        return QueryRouter(
            code_graph=None,  # Not needed for document-only queries
            doc_graph=doc_ingestor,
        )

    except Exception as e:
        logger.warning(f"Could not create document query router: {e}")
        return None


def _create_merged_query_router() -> QueryRouter | None:
    """Create QueryRouter with both graph connections.

    Used for standalone tool usage when router not injected.
    """
    try:
        from ..services.graph_service import MemgraphIngestor

        # Connect to both graphs
        code_ingestor = MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
            batch_size=100,
        )

        doc_ingestor = MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
            batch_size=100,
        )

        return QueryRouter(
            code_graph=code_ingestor,
            doc_graph=doc_ingestor,
        )

    except Exception as e:
        logger.warning(f"Could not create merged query router: {e}")
        return None


__all__ = [
    "create_query_document_graph_tool",
    "create_query_both_graphs_tool",
]