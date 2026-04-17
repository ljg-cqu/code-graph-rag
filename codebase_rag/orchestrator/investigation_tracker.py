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

    def record_tool(
        self, tool_name: str, query_arg: str = "", results_count: int = -1
    ) -> None:
        self.tools_used.add(tool_name)

        if tool_name == AgenticToolName.QUERY_GRAPH:
            self.graph_queries_run += 1

        if tool_name in FILE_READ_TOOLS and query_arg:
            self.files_read.append(query_arg)

        if results_count == 0 and tool_name in {
            AgenticToolName.SEMANTIC_SEARCH,
            AgenticToolName.QUERY_GRAPH,
        }:
            self.tool_failures.add(tool_name)
            logger.debug(
                f"Tool {tool_name} returned empty results — will be excluded from sufficiency checks"
            )

    @classmethod
    def from_parallel_worker(cls, worker_id: int) -> InvestigationState:
        return cls()
