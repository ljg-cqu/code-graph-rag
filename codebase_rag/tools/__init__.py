"""Tools for code analysis, retrieval, and manipulation."""

from pydantic_ai import Tool

from ..shared.query_router import QueryMode
from .graph_algorithms_tools import (
    analyze_path,
    community_summary,
    create_analyze_path_tool,
    create_community_summary_tool,
    create_expand_context_tool,
    create_find_bottlenecks_tool,
    create_find_similar_functions_tool,
    expand_context,
    find_bottlenecks,
    find_similar_functions,
)

# Tools that are mode-specific
_CODE_ONLY_TOOL_NAMES = {
    "validate_code_against_spec",
}
_DOCUMENT_ONLY_TOOL_NAMES = {
    "query_document_graph",
}
_BOTH_MERGED_TOOL_NAMES = {
    "query_both_graphs",
    "validate_doc_against_code",
}


def get_tools_for_mode(mode: QueryMode, tools: list[Tool]) -> list[Tool]:
    """Filter tools based on query mode.

    Args:
        mode: Current query mode
        tools: Full list of available tools

    Returns:
        Filtered list of tools appropriate for the mode
    """
    if mode == QueryMode.BOTH_MERGED:
        return tools

    filtered: list[Tool] = []
    for tool in tools:
        name = getattr(tool, "name", "")
        if name in _BOTH_MERGED_TOOL_NAMES:
            continue
        if mode == QueryMode.DOCUMENT_ONLY and name in _CODE_ONLY_TOOL_NAMES:
            continue
        if mode == QueryMode.CODE_ONLY and name in _DOCUMENT_ONLY_TOOL_NAMES:
            continue
        filtered.append(tool)

    return filtered


__all__ = [
    "community_summary",
    "analyze_path",
    "expand_context",
    "find_similar_functions",
    "find_bottlenecks",
    "create_community_summary_tool",
    "create_analyze_path_tool",
    "create_expand_context_tool",
    "create_find_similar_functions_tool",
    "create_find_bottlenecks_tool",
    "get_tools_for_mode",
]
