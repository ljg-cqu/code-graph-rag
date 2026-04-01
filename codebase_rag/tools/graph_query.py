"""Graph query tool for dual-graph (code + document) queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from pydantic_ai import Tool

if TYPE_CHECKING:
    from ..shared.query_router import QueryRouter


def create_graph_query_tool(query_router: QueryRouter) -> Tool:
    """Create a tool for querying code and document graphs.

    This tool allows the agent to route queries to appropriate graphs
    based on the current session mode.

    Args:
        query_router: QueryRouter instance with code and document graph connections

    Returns:
        PydanticAI Tool instance
    """
    from ..shared.query_router import QueryMode, QueryRequest

    def query_graphs(
        question: str,
        mode: str | None = None,
        top_k: int = 5,
    ) -> str:
        """Query code and/or document graphs.

        Args:
            question: Natural language query
            mode: Override session mode (optional). Options: code_only, document_only,
                  both_merged, code_vs_doc, doc_vs_code
            top_k: Maximum results per graph

        Returns:
            Formatted query results with source attribution
        """
        # Parse mode override
        effective_mode = query_router.current_mode
        if mode:
            try:
                effective_mode = QueryMode(mode.lower())
            except ValueError:
                valid_modes = [m.value for m in QueryMode]
                return f"Invalid mode: {mode}. Use one of: {valid_modes}"

        # Create query request
        request = QueryRequest(
            question=question,
            mode=effective_mode,
            top_k=top_k,
        )

        # Execute query
        response = query_router.query(request)

        # Format response with source attribution
        result_parts = [response.answer]

        if response.sources:
            code_sources = [s for s in response.sources if s.type == "code"]
            doc_sources = [s for s in response.sources if s.type == "document"]

            if code_sources:
                result_parts.append(f"\n\n**Code Sources ({len(code_sources)}):**")
                for source in code_sources:
                    result_parts.append(
                        f"- `{source.qualified_name}` in {source.path}"
                    )

            if doc_sources:
                result_parts.append(f"\n\n**Document Sources ({len(doc_sources)}):**")
                for source in doc_sources:
                    result_parts.append(
                        f"- `{source.qualified_name}` in {source.path}"
                    )

        if response.warnings:
            result_parts.append(f"\n\n⚠️ **Warnings:** {', '.join(response.warnings)}")

        if response.validation_report:
            report = response.validation_report.to_dict()
            result_parts.append(
                f"\n\n**Validation Report:** {report['passed']}/{report['total']} "
                f"({report['accuracy_score']:.1%} accurate)"
            )

        return "\n".join(result_parts)

    return Tool(
        function=query_graphs,
        name="query_graphs",
        description=(
            "Query code and/or document graphs based on the current mode. "
            "Use this for comprehensive questions that may span both code and documentation. "
            "Results include clear source attribution (code vs. document). "
            "Modes: code_only, document_only, both_merged, code_vs_doc, doc_vs_code."
        ),
    )


__all__ = ["create_graph_query_tool"]