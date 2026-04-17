from __future__ import annotations

from typing import TypedDict

from codebase_rag.tools.tool_descriptions import AgenticToolName

from .investigation_tracker import InvestigationState
from .sufficiency_analyzer import InvestigationRequirements

MAX_REJECTION_LIMIT = 3


class SufficiencyMetadata(TypedDict):
    passed: bool
    warning: str | None
    tools_used: list[str]
    files_read: list[str]


class SubtaskResult(TypedDict):
    content: str
    sufficiency: SufficiencyMetadata


def evaluate_sufficiency(
    state: InvestigationState,
    reqs: InvestigationRequirements,
    rejection_count: int = 0,
) -> tuple[bool, str | None]:
    if rejection_count >= MAX_REJECTION_LIMIT:
        return True, None

    if state.rounds_completed < reqs.min_rounds:
        return False, (
            f"Investigation too shallow. You need at least {reqs.min_rounds} rounds of querying "
            f"(currently at round {state.rounds_completed})."
        )

    if reqs.requires_vector:
        if AgenticToolName.SEMANTIC_SEARCH not in state.tool_failures:
            if AgenticToolName.SEMANTIC_SEARCH not in state.tools_used:
                return False, (
                    "CRITICAL: You must use `semantic_search` first to find relevant candidates by intent. "
                    "This is the recommended entry point for functional and structural queries."
                )

    if reqs.requires_graph:
        if AgenticToolName.QUERY_GRAPH not in state.tool_failures:
            if AgenticToolName.QUERY_GRAPH not in state.tools_used:
                return False, (
                    "CRITICAL: You must use `query_graph` to understand structural relationships "
                    "between code elements."
                )

    if reqs.requires_file_read:
        all_file_tools_failed = (
            AgenticToolName.READ_FILE in state.tool_failures
            and AgenticToolName.GET_CODE_SNIPPET in state.tool_failures
            and AgenticToolName.GET_FUNCTION_SOURCE in state.tool_failures
        )
        if not all_file_tools_failed and not state.files_read:
            return False, (
                "CRITICAL: You MUST read the actual source files using `read_file`, "
                "`get_code_snippet`, or `get_function_source`. Graph data alone is insufficient "
                "for implementation-level questions."
            )

    return True, None


def evaluate_parallel_worker_sufficiency(
    state: InvestigationState,
    reqs: InvestigationRequirements,
    worker_id: int,
) -> tuple[bool, str | None]:
    warnings: list[str] = []

    if reqs.requires_file_read and not state.files_read:
        warnings.append(
            f"[Worker {worker_id}] Subtask requires file-level evidence but no files were read."
        )

    if reqs.requires_graph and AgenticToolName.QUERY_GRAPH not in state.tools_used:
        if AgenticToolName.QUERY_GRAPH not in state.tool_failures:
            warnings.append(
                f"[Worker {worker_id}] Subtask did not query the code graph for structural context."
            )

    if warnings:
        return False, " | ".join(warnings)

    return True, None
