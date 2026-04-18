from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from codebase_rag.tools.tool_descriptions import AgenticToolName

FILE_READ_TOOLS = frozenset({
    AgenticToolName.READ_FILE,
    AgenticToolName.GET_CODE_SNIPPET,
    AgenticToolName.GET_FUNCTION_SOURCE,
})


@dataclass
class InvestigationState:
    rounds_completed: int = 0
    tools_used: set[str] = field(default_factory=set)
    files_read: list[str] = field(default_factory=list)
    graph_queries_run: int = 0
    tool_failures: set[str] = field(default_factory=set)
    _recent_calls: list[tuple[str, str]] = field(default_factory=list, repr=False)

    def is_duplicate(self, tool_name: str, query_arg: str) -> bool:
        return (tool_name, query_arg) in self._recent_calls

    def record_tool(
        self, tool_name: str, query_arg: str = "", results_count: int = -1
    ) -> None:
        """Record tool usage and track failures separately from usage.

        Tools that return empty results are still counted as "used" for
        sufficiency purposes, but are also tracked as failures for diagnostics.
        """
        # ALWAYS record tool usage, even if it returned 0 results
        self.tools_used.add(tool_name)
        self._recent_calls.append((tool_name, query_arg))

        if tool_name == AgenticToolName.QUERY_GRAPH:
            self.graph_queries_run += 1

        if tool_name in FILE_READ_TOOLS and query_arg:
            self.files_read.append(query_arg)

        # Track failures separately, but don't exclude from usage
        if results_count == 0 and tool_name in {
            AgenticToolName.SEMANTIC_SEARCH,
            AgenticToolName.QUERY_GRAPH,
        }:
            self.tool_failures.add(tool_name)
            logger.debug(
                f"Tool {tool_name} returned empty results — tracked as failure but still counts as usage"
            )

    @classmethod
    def from_parallel_worker(cls, worker_id: int) -> InvestigationState:
        return cls()
