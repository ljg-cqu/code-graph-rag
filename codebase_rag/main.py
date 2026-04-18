from __future__ import annotations

import asyncio
import difflib
import json
import os
import re
import shlex
import shutil
import signal
import sys
import uuid
from collections import deque
from collections.abc import Coroutine, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import print_formatted_text
from pydantic_ai import DeferredToolRequests, DeferredToolResults, Tool, ToolDenied
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from . import constants as cs
from . import exceptions as ex
from . import logs as ls
from .config import ModelConfig, load_cgrignore_patterns, settings
from .utils.shutdown_manager import shutdown_manager
from .context_compressor import ContextCompressor
from .models import AppContext
from .orchestrator import (
    ConcurrencyEligibilityClassifier,
    SubAgentOrchestrator,
    TaskSplitter,
)
from .prompts import OPTIMIZATION_PROMPT, OPTIMIZATION_PROMPT_WITH_REFERENCE
from .providers.base import get_provider_from_config
from .services import QueryProtocol
from .services.graph_service import MemgraphIngestor
from .services.llm import CypherGenerator, create_rag_orchestrator
from .shared.query_router import QueryMode, QueryRouter
from .tools.code_retrieval import CodeRetriever, create_code_retrieval_tool
from .tools.codebase_query import create_query_tool
from .tools.directory_lister import DirectoryLister, create_directory_lister_tool
from .tools.document_analyzer import DocumentAnalyzer, create_document_analyzer_tool
from .tools.document_index import create_index_documents_tool
from .tools.document_query import (
    create_query_both_graphs_tool,
    create_query_document_graph_tool,
)
from .tools.document_validation import (
    create_validate_code_against_spec_tool,
    create_validate_doc_against_code_tool,
)
from .tools.file_editor import FileEditor, create_file_editor_tool
from .tools.file_reader import FileReader, create_file_reader_tool
from .tools.file_writer import FileWriter, create_file_writer_tool
from .tools.graph_navigation import (
    GraphNavigator,
    create_find_implementations_tool,
    create_find_references_tool,
    create_get_call_hierarchy_tool,
    create_get_import_dependencies_tool,
    create_get_project_structure_tool,
)
from .tools.graph_query import create_graph_query_tool
from .tools.python_inspector import PythonObjectInspector, create_inspect_python_object_tool
from .tools.semantic_search import (
    create_get_function_source_tool,
    create_semantic_search_tool,
)
from .tools.shell_command import ShellCommander, create_shell_command_tool
from .types_defs import (
    CHAT_LOOP_UI,
    OPTIMIZATION_LOOP_UI,
    ORANGE_STYLE,
    AgentLoopUI,
    CancelledResult,
    ConfirmationToolNames,
    CreateFileArgs,
    GraphData,
    ModelInfo,
    RawToolArgs,
    ReplaceCodeArgs,
    ShellCommandArgs,
    ToolArgs,
)

if TYPE_CHECKING:
    from prompt_toolkit.key_binding import KeyPressEvent
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models import Model


def style(
    text: str, color: cs.Color, modifier: cs.StyleModifier = cs.StyleModifier.BOLD
) -> str:
    # Escape Rich markup patterns in the text to prevent MarkupError
    # when content contains [/? or [word] patterns that Rich would interpret as markup
    escaped_text = text.replace("[", "\\[")
    if modifier == cs.StyleModifier.NONE:
        return f"[{color}]{escaped_text}[/{color}]"
    return f"[{modifier} {color}]{escaped_text}[/{modifier} {color}]"


def dim(text: str) -> str:
    # Escape Rich markup patterns to prevent MarkupError
    escaped_text = text.replace("[", "\\[")
    return f"[{cs.StyleModifier.DIM}]{escaped_text}[/{cs.StyleModifier.DIM}]"


def _stringify_message_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts = [_stringify_message_content(item) for item in content]
        return "\n".join(part for part in parts if part)
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except TypeError:
        return str(content)


def _message_history_to_context(message_history: list[ModelMessage]) -> list[dict[str, str]]:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        UserPromptPart,
    )

    context: list[dict[str, str]] = []
    for message in message_history:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    context.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    context.append(
                        {
                            "role": "user",
                            "content": _stringify_message_content(part.content),
                        }
                    )
        elif isinstance(message, ModelResponse):
            response_parts = [
                part.content
                for part in message.parts
                if isinstance(part, TextPart) and part.content
            ]
            if response_parts:
                context.append(
                    {"role": "assistant", "content": "\n".join(response_parts)}
                )
    return context


def _context_to_message_history(context: list[dict[str, Any]]) -> list[ModelMessage]:
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        SystemPromptPart,
        TextPart,
        UserPromptPart,
    )

    message_history: list[ModelMessage] = []
    for entry in context:
        role = str(entry.get("role") or "user")
        content = _stringify_message_content(entry.get("content", ""))
        if not content:
            continue
        if role == "assistant":
            message_history.append(ModelResponse(parts=[TextPart(content)]))
        elif role == "system":
            message_history.append(ModelRequest(parts=[SystemPromptPart(content)]))
        else:
            message_history.append(ModelRequest(parts=[UserPromptPart(content)]))
    return message_history


# Yolo mode warning banner
YOLO_WARNING = """
[bold red]YOLO MODE ENABLED[/bold red]
All tool operations will be auto-approved without confirmation.
Use with caution on production codebases.
"""


@dataclass(frozen=True)
class RealtimeConfig:
    """Configuration for realtime file watching."""

    enabled: bool = False
    debounce: float = cs.DEFAULT_DEBOUNCE_SECONDS
    max_wait: float = cs.DEFAULT_MAX_WAIT_SECONDS
    enable_code: bool = True
    enable_docs: bool = False
    enable_json: bool = False


@dataclass(frozen=True)
class ParallelExecutionConfig:
    worker_count: int | None = None
    auto_split: bool = settings.CGR_AUTO_SPLIT_ENABLED
    no_parallel: bool = False
    dry_run: bool = False
    scheduling_strategy: str = "fifo"
    doc_workspace: str = "default"
    force_parallel: bool = False  # NEW: Explicit user override


def _display_yolo_warning() -> None:
    """Display yolo mode warning banner if yolo mode is enabled."""
    if app_context.session.yolo_mode:
        app_context.console.print(Panel(YOLO_WARNING, border_style=cs.Color.RED))


app_context = AppContext()


def init_session_log(project_root: Path) -> Path:
    # If project_root is a file, use its parent directory for .tmp
    if project_root.is_file():
        project_root = project_root.parent
    log_dir = project_root / cs.TMP_DIR
    log_dir.mkdir(exist_ok=True)
    app_context.session.log_file = (
        log_dir / f"{cs.SESSION_LOG_PREFIX}{uuid.uuid4().hex[:8]}{cs.SESSION_LOG_EXT}"
    )
    with open(app_context.session.log_file, "w") as f:
        f.write(cs.SESSION_LOG_HEADER)
    return app_context.session.log_file


def log_session_event(event: str) -> None:
    if app_context.session.log_file:
        with open(app_context.session.log_file, "a") as f:
            f.write(f"{event}\n")


def get_session_context() -> str:
    if app_context.session.log_file and app_context.session.log_file.exists():
        content = app_context.session.log_file.read_text(encoding="utf-8")
        return f"{cs.SESSION_CONTEXT_START}{content}{cs.SESSION_CONTEXT_END}"
    return ""


def _print_unified_diff(target: str, replacement: str, path: str) -> None:
    separator = dim(cs.HORIZONTAL_SEPARATOR)
    app_context.console.print(f"\n{cs.UI_DIFF_FILE_HEADER.format(path=path)}")
    app_context.console.print(separator)

    diff = difflib.unified_diff(
        target.splitlines(keepends=True),
        replacement.splitlines(keepends=True),
        fromfile=cs.DIFF_LABEL_BEFORE,
        tofile=cs.DIFF_LABEL_AFTER,
        lineterm="",
    )

    for line in diff:
        line = line.rstrip("\n")
        match line[:1]:
            case cs.DiffMarker.ADD | cs.DiffMarker.DEL if line.startswith(
                cs.DiffMarker.HEADER_ADD
            ) or line.startswith(cs.DiffMarker.HEADER_DEL):
                app_context.console.print(dim(line))
            case cs.DiffMarker.HUNK:
                app_context.console.print(
                    style(line, cs.Color.CYAN, cs.StyleModifier.NONE)
                )
            case cs.DiffMarker.ADD:
                app_context.console.print(
                    style(line, cs.Color.GREEN, cs.StyleModifier.NONE)
                )
            case cs.DiffMarker.DEL:
                app_context.console.print(
                    style(line, cs.Color.RED, cs.StyleModifier.NONE)
                )
            case _:
                app_context.console.print(line)

    app_context.console.print(separator)


def _print_new_file_content(path: str, content: str) -> None:
    separator = dim(cs.HORIZONTAL_SEPARATOR)
    app_context.console.print(f"\n{cs.UI_NEW_FILE_HEADER.format(path=path)}")
    app_context.console.print(separator)

    for line in content.splitlines():
        app_context.console.print(
            style(f"{cs.DiffMarker.ADD} {line}", cs.Color.GREEN, cs.StyleModifier.NONE)
        )

    app_context.console.print(separator)


def _to_tool_args(
    tool_name: str, raw_args: RawToolArgs, tool_names: ConfirmationToolNames
) -> ToolArgs:
    match tool_name:
        case tool_names.replace_code:
            return ReplaceCodeArgs(
                file_path=raw_args.file_path,
                target_code=raw_args.target_code,
                replacement_code=raw_args.replacement_code,
            )
        case tool_names.create_file:
            return CreateFileArgs(
                file_path=raw_args.file_path,
                content=raw_args.content,
            )
        case tool_names.shell_command:
            return ShellCommandArgs(command=raw_args.command)
        case _:
            return ShellCommandArgs()


def _display_tool_call_diff(
    tool_name: str,
    tool_args: ToolArgs,
    tool_names: ConfirmationToolNames,
    file_path: str | None = None,
) -> None:
    match tool_name:
        case tool_names.replace_code:
            target = str(tool_args.get(cs.ARG_TARGET_CODE, ""))
            replacement = str(tool_args.get(cs.ARG_REPLACEMENT_CODE, ""))
            path = str(
                tool_args.get(cs.ARG_FILE_PATH, file_path or cs.DIFF_FALLBACK_PATH)
            )
            _print_unified_diff(target, replacement, path)

        case tool_names.create_file:
            path = str(tool_args.get(cs.ARG_FILE_PATH, ""))
            content = str(tool_args.get(cs.ARG_CONTENT, ""))
            _print_new_file_content(path, content)

        case tool_names.shell_command:
            command = tool_args.get(cs.ARG_COMMAND, "")
            app_context.console.print(f"\n{cs.UI_SHELL_COMMAND_HEADER}")
            app_context.console.print(
                style(f"$ {command}", cs.Color.YELLOW, cs.StyleModifier.NONE)
            )

        case _:
            app_context.console.print(
                cs.UI_TOOL_ARGS_FORMAT.format(
                    args=json.dumps(tool_args, indent=cs.JSON_INDENT)
                )
            )


def _process_tool_approvals(
    requests: DeferredToolRequests,
    approval_prompt: str,
    denial_default: str,
    tool_names: ConfirmationToolNames,
) -> DeferredToolResults:
    deferred_results = DeferredToolResults()

    for call in requests.approvals:
        tool_args = _to_tool_args(
            call.tool_name, RawToolArgs(**call.args_as_dict()), tool_names
        )

        # YOLO MODE: Skip UI, auto-approve
        if app_context.session.yolo_mode:
            logger.info(f"YOLO: Auto-approving {call.tool_name}")
            deferred_results.approvals[call.tool_call_id] = True
            continue

        # Normal confirmation flow
        app_context.console.print(
            f"\n{cs.UI_TOOL_APPROVAL.format(tool_name=call.tool_name)}"
        )
        _display_tool_call_diff(call.tool_name, tool_args, tool_names)

        if app_context.session.confirm_edits:
            if Confirm.ask(style(approval_prompt, cs.Color.CYAN)):
                deferred_results.approvals[call.tool_call_id] = True
            else:
                feedback = Prompt.ask(
                    cs.UI_FEEDBACK_PROMPT,
                    default="",
                )
                denial_msg = feedback.strip() or denial_default
                deferred_results.approvals[call.tool_call_id] = ToolDenied(denial_msg)
        else:
            deferred_results.approvals[call.tool_call_id] = True

    return deferred_results


def _setup_common_initialization(repo_path: str) -> Path:
    project_root = Path(repo_path).resolve()
    if not project_root.exists():
        raise FileNotFoundError(
            f"Repository path '{repo_path}' does not exist.\nHINT: If you used --repo-path, make sure to provide a valid path after it, e.g. --repo-path . --clean"
        )
    # Use parent directory if repo_path is a single file
    tmp_base = project_root.parent if project_root.is_file() else project_root
    tmp_dir = tmp_base / cs.TMP_DIR
    if tmp_dir.exists():
        if tmp_dir.is_dir():
            shutil.rmtree(tmp_dir)
        else:
            tmp_dir.unlink()
    tmp_dir.mkdir()

    return project_root


def _create_configuration_table(
    repo_path: str,
    title: str = cs.DEFAULT_TABLE_TITLE,
    language: str | None = None,
    doc_graph_connected: bool = False,
    query_mode: QueryMode | None = None,
    doc_workspace: str = "default",
) -> Table:
    from .shared.query_router import QueryMode

    # Default to CODE_ONLY if not specified
    if query_mode is None:
        query_mode = QueryMode.CODE_ONLY

    table = Table(title=style(title, cs.Color.GREEN))
    table.add_column(cs.TABLE_COL_CONFIGURATION, style=cs.Color.CYAN)
    table.add_column(cs.TABLE_COL_VALUE, style=cs.Color.MAGENTA)

    if language:
        table.add_row(cs.TABLE_ROW_TARGET_LANGUAGE, language)

    orchestrator_config = settings.active_orchestrator_config
    table.add_row(
        cs.TABLE_ROW_ORCHESTRATOR_MODEL,
        f"{orchestrator_config.model_id} ({orchestrator_config.provider})",
    )

    cypher_config = settings.active_cypher_config
    table.add_row(
        cs.TABLE_ROW_CYPHER_MODEL,
        f"{cypher_config.model_id} ({cypher_config.provider})",
    )

    orch_endpoint = (
        orchestrator_config.endpoint
        if orchestrator_config.provider == cs.Provider.OLLAMA
        else None
    )
    cypher_endpoint = (
        cypher_config.endpoint if cypher_config.provider == cs.Provider.OLLAMA else None
    )

    if orch_endpoint and cypher_endpoint and orch_endpoint == cypher_endpoint:
        table.add_row(cs.TABLE_ROW_OLLAMA_ENDPOINT, orch_endpoint)
    else:
        if orch_endpoint:
            table.add_row(cs.TABLE_ROW_OLLAMA_ORCHESTRATOR, orch_endpoint)
        if cypher_endpoint:
            table.add_row(cs.TABLE_ROW_OLLAMA_CYPHER, cypher_endpoint)

    # Code Graph connection
    table.add_row(
        cs.TABLE_ROW_CODE_GRAPH,
        f"{settings.MEMGRAPH_HOST}:{settings.MEMGRAPH_PORT}",
    )

    # Document Graph connection
    if doc_graph_connected:
        table.add_row(
            cs.TABLE_ROW_DOCUMENT_GRAPH,
            f"{settings.DOC_MEMGRAPH_HOST}:{settings.DOC_MEMGRAPH_PORT} (workspace: {doc_workspace})",
        )
    else:
        table.add_row(
            cs.TABLE_ROW_DOCUMENT_GRAPH,
            cs.TABLE_ROW_DOC_GRAPH_NOT_CONNECTED,
        )

    # Query mode
    table.add_row(cs.TABLE_ROW_QUERY_MODE, query_mode)

    # Yolo mode indicator
    yolo_status = cs.YOLO_ENABLED if app_context.session.yolo_mode else cs.YOLO_DISABLED
    table.add_row(cs.TABLE_ROW_YOLO_MODE, yolo_status)

    confirmation_status = (
        cs.CONFIRM_ENABLED if app_context.session.confirm_edits else cs.CONFIRM_DISABLED
    )
    table.add_row(cs.TABLE_ROW_EDIT_CONFIRMATION, confirmation_status)
    table.add_row(cs.TABLE_ROW_TARGET_REPOSITORY, repo_path)

    return table


async def run_optimization_loop(
    rag_agent: Agent[None, str | DeferredToolRequests],
    message_history: list[ModelMessage],
    project_root: Path,
    language: str,
    tool_names: ConfirmationToolNames,
    reference_document: str | None = None,
    parallel_config: ParallelExecutionConfig | None = None,
) -> None:
    app_context.console.print(cs.UI_OPTIMIZATION_START.format(language=language))
    document_info = (
        cs.UI_REFERENCE_DOC_INFO.format(reference_document=reference_document)
        if reference_document
        else ""
    )
    app_context.console.print(
        Panel(
            cs.UI_OPTIMIZATION_PANEL.format(document_info=document_info),
            border_style=cs.Color.YELLOW,
        )
    )

    initial_question = (
        OPTIMIZATION_PROMPT_WITH_REFERENCE.format(
            language=language, reference_document=reference_document
        )
        if reference_document
        else OPTIMIZATION_PROMPT.format(language=language)
    )

    await _run_interactive_loop(
        rag_agent,
        message_history,
        project_root,
        OPTIMIZATION_LOOP_UI,
        style(cs.PROMPT_YOUR_RESPONSE, cs.Color.CYAN),
        tool_names,
        initial_question,
        parallel_config=parallel_config,
    )


async def run_with_cancellation[T](
    coro: Coroutine[None, None, T], timeout: float | None = None
) -> T | CancelledResult:
    task = asyncio.create_task(coro)

    try:
        return await asyncio.wait_for(task, timeout=timeout) if timeout else await task
    except TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        app_context.console.print(
            f"\n{style(cs.MSG_TIMEOUT_FORMAT.format(timeout=timeout), cs.Color.YELLOW)}"
        )
        return CancelledResult(cancelled=True)
    except asyncio.CancelledError:
        # Cancelled from outside (e.g., signal handler) - don't print,
        # the signal handler already printed the message
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return CancelledResult(cancelled=True)
    except KeyboardInterrupt:
        # Fallback for Windows or when signal handlers not installed
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        app_context.console.print(
            f"\n{style(cs.MSG_THINKING_CANCELLED, cs.Color.YELLOW)}"
        )
        return CancelledResult(cancelled=True)


async def _run_agent_response_loop(
    rag_agent: Agent[None, str | DeferredToolRequests],
    message_history: list[ModelMessage],
    question_with_context: str,
    config: AgentLoopUI,
    tool_names: ConfirmationToolNames,
    model_override: Model | None = None,
    model_override_config: ModelConfig | None = None,
) -> None:
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from .orchestrator.investigation_tracker import InvestigationState
    from .orchestrator.sufficiency_analyzer import analyze_requirements
    from .orchestrator.sufficiency_gatekeeper import evaluate_sufficiency

    requirements = analyze_requirements(question_with_context)
    state = InvestigationState()
    rejection_count = 0
    deferred_results: DeferredToolResults | None = None

    while True:
        # === AUTOMATIC CONTEXT COMPRESSION HOOK ===
        if (
            settings.CONTEXT_COMPRESSION_ENABLED
            and message_history
            and not deferred_results
        ):
            from .utils.token_utils import count_tokens

            # Estimate total tokens (history + new question)
            # Avoid circular references by stringifying directly instead of asdict
            serialized = "\n".join(
                [str(m) for m in message_history] + [question_with_context]
            )
            total_tokens = count_tokens(serialized)
            # Get max context window from model config (default to 256k if not specified)
            max_context = settings.DEFAULT_CONTEXT_WINDOW
            try:
                # Check for role-specific override first (highest precedence)
                if settings.ORCHESTRATOR_CONTEXT_WINDOW:
                    max_context = settings.ORCHESTRATOR_CONTEXT_WINDOW
                elif model_override_config:
                    # Get from override if set
                    provider = get_provider_from_config(model_override_config)
                    max_context = provider.get_model_context_window(
                        model_override_config.model_id
                    )
                else:
                    # Get from default orchestrator config
                    provider = get_provider_from_config(
                        settings.active_orchestrator_config
                    )
                    max_context = provider.get_model_context_window(
                        settings.active_orchestrator_config.model_id
                    )
            except Exception as e:
                # Fallback to default
                logger.debug(
                    f"Failed to retrieve model context window, using default {settings.DEFAULT_CONTEXT_WINDOW:,}: {e}"
                )

            trigger_threshold = int(
                max_context * settings.CONTEXT_COMPRESSION_AUTO_TRIGGER_PCT / 100
            )

            if total_tokens >= trigger_threshold:
                app_context.console.print(
                    style(
                        f"⚠️ Context approaching limit: {total_tokens}/{max_context} tokens, running automatic compression...",
                        cs.Color.YELLOW,
                    )
                )

                # Convert history to compressor format
                context = _message_history_to_context(message_history)

                compressor = ContextCompressor(
                    context=context,
                    max_context=max_context,
                    aggressive_mode=False,
                    worker_count=settings.CONTEXT_COMPRESSION_PARALLEL_WORKERS,
                )
                result = compressor.compress_sync()

                if not result.was_rolled_back and result.compressed_context:
                    compressed_history = _context_to_message_history(
                        result.compressed_context
                    )
                    if compressed_history:
                        message_history[:] = compressed_history
                        app_context.console.print(
                            style(
                                f"✅ Compressed to {result.compressed_tokens:,} tokens ({result.reduction_pct:.1%} reduction, {result.retention_score:.1%} retention)",
                                cs.Color.GREEN,
                            )
                        )
                    else:
                        app_context.console.print(
                            style(
                                "⚠️ Compression returned no usable message history, proceeding with original",
                                cs.Color.YELLOW,
                            )
                        )
                elif not result.was_rolled_back and not result.compressed_context:
                    # Skip compression if result is empty
                    app_context.console.print(
                        style(
                            "⚠️ Compression returned empty context, proceeding with original",
                            cs.Color.YELLOW,
                        )
                    )
                else:
                    app_context.console.print(
                        style(
                            "⚠️ Compression rolled back, proceeding with original context",
                            cs.Color.YELLOW,
                        )
                    )

        with app_context.console.status(config.status_message):
            response = await run_with_cancellation(
                rag_agent.run(
                    question_with_context,
                    message_history=message_history,
                    deferred_tool_results=deferred_results,
                    model=model_override,
                ),
            )

        if isinstance(response, CancelledResult):
            log_session_event(config.cancelled_log)
            app_context.session.cancelled = True
            break

        if isinstance(response.output, DeferredToolRequests):
            for call in response.output.approvals:
                args = call.args_as_dict()
                tool_name = call.tool_name
                query_arg = _extract_tool_query_arg(tool_name, args)
                state.record_tool(tool_name, query_arg)

            state.rounds_completed += 1

            deferred_results = _process_tool_approvals(
                response.output,
                config.approval_prompt,
                config.denial_default,
                tool_names,
            )
            new_msgs = response.new_messages()
            message_history.extend(new_msgs)
            _update_state_from_tool_returns(new_msgs, state)
            continue

        output_text = response.output
        if not isinstance(output_text, str):
            continue

        if not app_context.session.yolo_mode:
            is_sufficient, feedback = evaluate_sufficiency(
                state, requirements, rejection_count
            )
            if not is_sufficient:
                rejection_count += 1
                if rejection_count >= 3:
                    app_context.console.print(
                        Panel(
                            "⚠️ Max rejections reached (3). Accepting response with incomplete investigation.",
                            border_style=cs.Color.YELLOW,
                        )
                    )
                else:
                    feedback_msg = (
                        f"\n**SYSTEM CORRECTION (attempt {rejection_count}/3):** {feedback}\n\n"
                        f"You are not allowed to answer yet. Please use the required tools "
                        f"to gather more information before generating a final response."
                    )
                    message_history.extend(response.new_messages())
                    message_history.append(
                        ModelRequest(parts=[UserPromptPart(feedback_msg)])
                    )
                    app_context.console.print(
                        Panel(
                            f"⚠️ Investigation Incomplete: {feedback}",
                            border_style=cs.Color.YELLOW,
                        )
                    )
                    continue
        markdown_response = Markdown(output_text)
        app_context.console.print(
            Panel(
                markdown_response,
                title=config.panel_title,
                border_style=cs.Color.GREEN,
            )
        )

        log_session_event(f"{cs.SESSION_PREFIX_ASSISTANT}{output_text}")
        message_history.extend(response.new_messages())
        break


def _extract_tool_query_arg(tool_name: str, args: dict[str, object]) -> str:
    from .tools.tool_descriptions import AgenticToolName

    if tool_name == AgenticToolName.QUERY_GRAPH:
        return str(args.get("natural_language_query", ""))
    if tool_name == AgenticToolName.SEMANTIC_SEARCH:
        return str(args.get("query", ""))
    if tool_name == AgenticToolName.READ_FILE:
        return str(args.get("file_path", ""))
    if tool_name == AgenticToolName.GET_CODE_SNIPPET:
        return str(args.get("qualified_name", ""))
    if tool_name == AgenticToolName.GET_FUNCTION_SOURCE:
        return str(args.get("node_id", ""))
    return str(args.get("query", args.get("command", "")))


def _update_state_from_tool_returns(
    new_messages: list[ModelMessage],
    state: object,
) -> None:
    from pydantic_ai.messages import ModelRequest, ToolReturnPart

    from .orchestrator.investigation_tracker import InvestigationState

    if not isinstance(state, InvestigationState):
        return

    for msg in new_messages:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolReturnPart):
                continue
            content = str(part.content) if part.content is not None else ""
            if not content or "no results" in content.lower() or "not found" in content.lower():
                state.tool_failures.add(part.tool_name)


def _find_image_paths(question: str) -> list[Path]:
    try:
        if os.name == "nt":
            # (H) On Windows, shlex.split with posix=False to preserve backslashes
            tokens = shlex.split(question, posix=False)
        else:
            tokens = shlex.split(question)
    except ValueError:
        tokens = question.split()

    image_paths: list[Path] = []
    for token in tokens:
        # (H) Strip quotes if they remain (shlex with posix=False might keep some)
        token = token.strip("'\"")
        # (H) Check if it looks like an image path
        if token.lower().endswith(cs.IMAGE_EXTENSIONS):
            # (H) On Windows, could be C:\... or \...
            # (H) On POSIX, starts with /
            p = Path(token)
            if p.is_absolute() or token.startswith("/") or token.startswith("\\"):
                image_paths.append(p)
    return image_paths


def _get_path_variants(path_str: str) -> tuple[str, ...]:
    return (
        path_str.replace(" ", r"\ "),
        f"'{path_str}'",
        f'"{path_str}"',
        path_str,
    )


def _replace_path_in_question(question: str, old_path: str, new_path: str) -> str:
    for variant in _get_path_variants(old_path):
        if variant in question:
            return question.replace(variant, new_path)
    logger.warning(ls.PATH_NOT_IN_QUESTION.format(path=old_path))
    return question


def _handle_chat_images(question: str, project_root: Path) -> str:
    image_files = _find_image_paths(question)
    if not image_files:
        return question

    tmp_dir = project_root / cs.TMP_DIR
    tmp_dir.mkdir(exist_ok=True)
    updated_question = question

    for original_path in image_files:
        if not original_path.exists() or not original_path.is_file():
            logger.warning(ls.IMAGE_NOT_FOUND.format(path=original_path))
            continue

        try:
            new_path = tmp_dir / f"{uuid.uuid4()}-{original_path.name}"
            shutil.copy(original_path, new_path)
            new_relative = str(new_path.relative_to(project_root))
            updated_question = _replace_path_in_question(
                updated_question, str(original_path), new_relative
            )
            logger.info(ls.IMAGE_COPIED.format(path=new_relative))
        except Exception as e:
            logger.error(ls.IMAGE_COPY_FAILED.format(error=e))

    return updated_question


def get_multiline_input(prompt_text: str = cs.PROMPT_ASK_QUESTION) -> str:
    bindings = KeyBindings()

    @bindings.add(cs.KeyBinding.CTRL_J)
    def submit(event: KeyPressEvent) -> None:
        event.app.exit(result=event.app.current_buffer.text)

    @bindings.add(cs.KeyBinding.ENTER)
    def new_line(event: KeyPressEvent) -> None:
        event.current_buffer.insert_text("\n")

    @bindings.add(cs.KeyBinding.CTRL_C)
    def keyboard_interrupt(event: KeyPressEvent) -> None:
        event.app.exit(exception=KeyboardInterrupt)

    command_completer = WordCompleter(
        [
            cs.MODELS_COMMAND_PREFIX,
            cs.MODEL_COMMAND_PREFIX,
            cs.MODE_COMMAND_PREFIX,
            cs.HELP_COMMAND,
            cs.COMPRESS_COMMAND_PREFIX,
        ],
        ignore_case=True,
    )

    clean_prompt = Text.from_markup(prompt_text).plain

    print_formatted_text(
        HTML(
            cs.UI_INPUT_PROMPT_HTML.format(
                prompt=clean_prompt, hint=cs.MULTILINE_INPUT_HINT
            )
        )
    )

    result = prompt(
        "",
        multiline=True,
        key_bindings=bindings,
        completer=command_completer,
        wrap_lines=True,
        style=ORANGE_STYLE,
    )
    if result is None:
        raise EOFError
    stripped: str = result.strip()
    return stripped


def _handle_models_command(
    command: str, current_model_config: ModelConfig | None = None
) -> None:
    """Handle /models command to display available models."""
    from .models_dynamic import build_dynamic_model_catalog

    parts = command.strip().split(maxsplit=1)
    arg = parts[1].strip().lower() if len(parts) > 1 else None

    if arg == cs.HELP_ARG:
        app_context.console.print(cs.UI_MODELS_USAGE)
        return

    catalog = build_dynamic_model_catalog()

    if arg == "debug":
        _display_models_debug(catalog, current_model_config)
        return

    if arg is None:
        _display_models_table(catalog, current_model_config)
        return

    if arg in catalog:
        provider_models = {arg: catalog[arg]}
        _display_models_table(provider_models, current_model_config)
    else:
        valid_providers = ", ".join(catalog.keys())
        app_context.console.print(
            cs.UI_MODELS_INVALID_PROVIDER.format(provider=arg, available=valid_providers)
        )


def _display_models_table(
    catalog: dict[str, list[DynamicModelInfo]],
    current_model_config: ModelConfig | None = None,
) -> None:
    """Display formatted model table using Rich Text for safe markup.

    Shows configured (.env) models first with indicators:
      - configured/working models
      - static models that may need API key configuration
    """
    from .models_catalog import PROVIDER_DISPLAY_NAMES

    if not catalog:
        app_context.console.print("No models available.")
        return

    current_config = current_model_config or settings.active_orchestrator_config
    current_provider = current_config.provider
    current_model_id = current_config.model_id

    for provider, models in catalog.items():
        display_name = PROVIDER_DISPLAY_NAMES.get(provider, provider.title())
        app_context.console.print(Text(f"  {display_name}", style="bold cyan"))

        # Sort: configured models first, then by model_id
        sorted_models = sorted(
            models, key=lambda m: (0 if m.is_configured else 1, m.model_id)
        )

        for model_info in sorted_models:
            is_current = (
                provider == current_provider
                and model_info.model_id == current_model_id
            )

            # Status indicator
            if model_info.is_configured:
                status_icon = "\u2705"  # ✅
            elif model_info.requires_api_key and not model_info.is_local:
                status_icon = "\u26a0\ufe0f"  # ⚠️
            else:
                status_icon = "\u2022"

            marker = status_icon

            ctx = model_info.context_window
            if ctx >= 1_000_000:
                ctx_str = f"{ctx // 1_000_000}M"
            elif ctx >= 1_000:
                ctx_str = f"{ctx // 1_000}K"
            else:
                ctx_str = str(ctx)

            line = Text(f"  {marker} ")
            line.append(model_info.model_id, style="bold")
            if model_info.description:
                line.append(f" (Context: {ctx_str} tokens) - {model_info.description}")
            else:
                line.append(f" (Context: {ctx_str} tokens)")

            # Source attribution for env-configured models
            if model_info.source != "static":
                line.append(
                    f" - Configured from .env", style="green"
                )

            if is_current:
                line.append(" ✓", style="bold green")
                line.append(" [Active]", style="bold green")

            app_context.console.print(line)

        app_context.console.print("")

    current_model_str = f"{current_provider}{cs.CHAR_COLON}{current_model_id}"
    app_context.console.print(
        style(f"Current Model: {current_model_str}", cs.Color.CYAN)
    )
    app_context.console.print(
        style("Usage: /model <provider>:<model_id> to switch", cs.Color.YELLOW, cs.StyleModifier.NONE)
    )


def _display_models_debug(
    catalog: dict[str, list[DynamicModelInfo]],
    current_model_config: ModelConfig | None = None,
) -> None:
    """Display debugging information about model discovery and configuration."""
    from .providers.base import PROVIDER_REGISTRY
    from .config import API_KEY_INFO, LOCAL_PROVIDERS
    import os

    console = app_context.console
    console.print("[bold yellow]Model Debugging Information[/bold yellow]")
    console.print()

    # Provider registry
    console.print("[bold cyan]Provider Registry:[/bold cyan]")
    for provider, cls in PROVIDER_REGISTRY.items():
        console.print(f"  {provider}: {cls.__name__}")
    console.print()

    # API key info
    console.print("[bold cyan]API Key Environment Variables:[/bold cyan]")
    for provider, info in API_KEY_INFO.items():
        env_var = info['env_var']
        has_key = os.environ.get(env_var) is not None
        console.print(f"  {provider}: {env_var} {'✅' if has_key else '❌'}")
    console.print()

    # Local providers
    console.print(f"[bold cyan]Local Providers:[/bold cyan] {', '.join(LOCAL_PROVIDERS)}")
    console.print()

    # Environment variables for custom providers
    console.print("[bold cyan]Custom Provider API Keys:[/bold cyan]")
    custom_keys = []
    for key in os.environ:
        if key.endswith('_API_KEY') and key not in {info['env_var'] for info in API_KEY_INFO.values()}:
            custom_keys.append(key)
    if custom_keys:
        for key in sorted(custom_keys):
            # Hide actual key values
            value = os.environ[key]
            masked = '****' + value[-4:] if len(value) > 4 else '****'
            console.print(f"  {key}: {masked}")
    else:
        console.print("  None")
    console.print()

    # Catalog statistics
    console.print("[bold cyan]Catalog Statistics:[/bold cyan]")
    total_models = sum(len(models) for models in catalog.values())
    console.print(f"  Total providers: {len(catalog)}")
    console.print(f"  Total models: {total_models}")
    configured = sum(1 for models in catalog.values() for m in models if m.is_configured)
    console.print(f"  Configured models: {configured}")
    console.print()

    # Current model configuration
    current_config = current_model_config or settings.active_orchestrator_config
    console.print(f"[bold cyan]Current Model Config:[/bold cyan]")
    console.print(f"  Provider: {current_config.provider}")
    console.print(f"  Model ID: {current_config.model_id}")
    console.print(f"  Endpoint: {current_config.endpoint or '(default)'}")
    console.print(f"  API Key present: {'✅' if current_config.api_key and current_config.api_key != cs.DEFAULT_API_KEY else '❌'}")
    console.print()

    # Detailed model list
    console.print("[bold cyan]Detailed Model List:[/bold cyan]")
    for provider, models in catalog.items():
        console.print(f"  [bold]{provider}[/bold]:")
        for model in models:
            status = '✅' if model.is_configured else '❌'
            source = model.source
            ctx = model.context_window
            endpoint = model.endpoint or '(default)'
            console.print(f"    {status} {model.model_id} (ctx={ctx}, source={source}, endpoint={endpoint})")
        console.print()


def _create_model_from_string(
    model_string: str, current_override_config: ModelConfig | None = None
) -> tuple[Model, str, ModelConfig]:
    base_config = current_override_config or settings.active_orchestrator_config

    if cs.CHAR_COLON not in model_string:
        raise ValueError(ex.MODEL_FORMAT_INVALID)
    provider_name, model_id = (
        p.strip() for p in settings.parse_model_string(model_string)
    )
    if not model_id:
        raise ValueError(ex.MODEL_ID_EMPTY)
    if not provider_name:
        raise ValueError(ex.PROVIDER_EMPTY)

    # Look up dynamic catalog for endpoint and configuration info
    model_info = _get_dynamic_model_info(provider_name, model_id)
    dynamic_endpoint = model_info.endpoint if model_info else None

    # Warn if model is not configured (missing API key)
    if model_info and not model_info.is_configured:
        logger.warning(
            f"Model {provider_name}:{model_id} is not configured (missing API key). "
            "It may fail at runtime."
        )

    if provider_name == base_config.provider:
        config = replace(
            base_config,
            model_id=model_id,
            endpoint=dynamic_endpoint or base_config.endpoint,
        )
    elif provider_name == cs.Provider.OLLAMA:
        config = ModelConfig(
            provider=provider_name,
            model_id=model_id,
            endpoint=dynamic_endpoint or settings.ollama_endpoint,
            api_key=cs.DEFAULT_API_KEY,
        )
    else:
        config = ModelConfig(
            provider=provider_name,
            model_id=model_id,
            endpoint=dynamic_endpoint,
        )

    canonical_string = f"{provider_name}{cs.CHAR_COLON}{model_id}"
    provider = get_provider_from_config(config)
    return provider.create_model(model_id), canonical_string, config


def _get_dynamic_model_info(provider: str, model_id: str) -> DynamicModelInfo | None:
    """Look up model info from the dynamic catalog.

    Returns the DynamicModelInfo if found in catalog, or None.
    """
    from .models_dynamic import build_dynamic_model_catalog, DynamicModelInfo

    catalog = build_dynamic_model_catalog()
    models = catalog.get(provider, [])
    for m in models:
        if m.model_id == model_id:
            return m
    return None


def _find_dynamic_endpoint(provider: str, model_id: str) -> str | None:
    """Look up a model's endpoint from the dynamic catalog.

    Returns the endpoint if the model was configured via .env with a custom
    endpoint, or None if not found / no custom endpoint.
    """
    model_info = _get_dynamic_model_info(provider, model_id)
    if model_info and model_info.endpoint:
        return model_info.endpoint
    return None


def _handle_model_command(
    command: str,
    current_model: Model | None,
    current_model_string: str | None,
    current_config: ModelConfig | None,
) -> tuple[Model | None, str | None, ModelConfig | None]:
    parts = command.strip().split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else None

    if not arg:
        if current_model_string:
            display_model = current_model_string
        else:
            config = settings.active_orchestrator_config
            display_model = f"{config.provider}{cs.CHAR_COLON}{config.model_id}"
        app_context.console.print(cs.UI_MODEL_CURRENT.format(model=display_model))
        return current_model, current_model_string, current_config

    if arg.lower() == cs.HELP_ARG:
        app_context.console.print(cs.UI_MODEL_USAGE)
        return current_model, current_model_string, current_config

    try:
        new_model, canonical_model_string, new_config = _create_model_from_string(
            arg, current_config
        )
        logger.info(ls.MODEL_SWITCHED.format(model=canonical_model_string))
        app_context.console.print(
            cs.UI_MODEL_SWITCHED.format(model=canonical_model_string)
        )
        # Warn if model is not configured
        model_info = _get_dynamic_model_info(new_config.provider, new_config.model_id)
        if model_info and not model_info.is_configured:
            app_context.console.print(
                style(
                    f"⚠️ Warning: Model {new_config.provider}:{new_config.model_id} is not configured (missing API key). It may fail at runtime.",
                    cs.Color.YELLOW,
                )
            )
        return new_model, canonical_model_string, new_config
    except (ValueError, AssertionError) as e:
        logger.error(ls.MODEL_SWITCH_FAILED.format(error=e))
        app_context.console.print(cs.UI_MODEL_SWITCH_ERROR.format(error=e))
        return current_model, current_model_string, current_config


def _handle_mode_command(
    command: str,
    query_router: QueryRouter | None,
    current_mode: QueryMode,
) -> tuple[QueryMode, str]:
    """Handle /mode command in chat session.

    Args:
        command: Full command string (e.g., "/mode both_merged")
        query_router: Active QueryRouter instance (None if code-only)
        current_mode: Current query mode

    Returns:
        Tuple of (new_mode, status_message)
    """
    from .shared.query_router import QueryMode

    parts = command.strip().split(maxsplit=1)
    arg = parts[1].strip().lower() if len(parts) > 1 else None

    if not arg:
        return current_mode, f"Current mode: {current_mode.value}"

    if arg == cs.HELP_ARG:
        return (
            current_mode,
            """
Available modes:
  /mode code_only       - Query code graph only
  /mode document_only   - Query document graph only
  /mode both_merged     - Query both, merge results
  /mode code_vs_doc     - Validate code against docs
  /mode doc_vs_code     - Validate docs against code
  /mode                 - Show current mode
""",
        )

    try:
        new_mode = QueryMode(arg)

        # Validate mode is available
        if new_mode != QueryMode.CODE_ONLY and (
            query_router is None or query_router.doc_graph is None
        ):
            return current_mode, (
                f"Mode '{new_mode.value}' requires document graph. "
                "Restart with --with-docs flag."
            )

        # Update mode in router if available
        if query_router:
            query_router.current_mode = new_mode

        logger.info(f"Mode switched to: {new_mode.value}")
        return new_mode, f"Mode switched to: {new_mode.value}"

    except ValueError:
        return current_mode, f"Invalid mode: {arg}. Use /mode help for options."


def _has_write_intent(prompt: str) -> bool:
    """
    Enhanced write operation detection with contextual understanding.
    The current implementation uses r"\\b(create|write|edit|modify|...)\\b" which
    matches write-related words ANYWHERE in the prompt, causing false positives
    on read-only queries like "show me how the code creates objects" or
    "explain what the write method does".

    This replacement:
    1. Checks for explicit imperative write commands (verb + code target noun)
    2. Excludes queries in known read-only phrasing patterns (question/explanation)
    3. Returns False for ambiguous cases where write words appear but no imperative
       command structure is detected (instead of the current blanket True)
    """
    # Use strict mode if configured (backward compatibility)
    if settings.CGR_PARALLEL_WRITE_DETECTION_STRICT:
        write_patterns = [
            r"\b(create|write|edit|modify|update|delete|remove|refactor|rename|move|implement|fix|patch)\b",
            r"\badd\b.{0,40}\b(file|files|code|test|tests|function|class|method|doc|docs|documentation|config)\b",
        ]
        lowered_prompt = prompt.lower()
        return any(re.search(pattern, lowered_prompt) for pattern in write_patterns)

    lowered_prompt = prompt.lower().strip()

    # ── Layer 1: Explicit imperative write command patterns ──
    # These match when a write verb is used as an imperative/instruction
    # directly targeting a code entity (verb followed by a code noun object).
    # This eliminates false positives where "write"/"create" appear as nouns
    # or in descriptive/analytical contexts.
    explicit_write_patterns = [
        # Imperative: "create a file", "modify the code", "fix the following test"
        r"\b(create|write|edit|modify|update|delete|remove|refactor|rename|move|implement|fix|patch)\s+(the\s+)?(following\s+)?(file|files|code|test|tests|function|class|method|doc|docs|documentation|config|configuration|module|package|script|component)\b",
        # "add/insert a function/class/test" — add requires a direct object
        r"\b(add|insert)\s+(the\s+)?(following\s+)?(code|function|class|test|tests|documentation|module|package|file|files)\b",
        # "generate and save", "produce to file" — explicit save intent
        r"\b(generate|produce)\s+(and\s+)?(save|persist|store|write|output\s+to)\b",
        # "save/store/persist the result/output/code" — explicit persistence
        r"\b(save|store|persist)\s+(the\s+)?(result|output|code|file|changes|modification)\b",
        # "replace X with Y", "overwrite the file" — destructive operations
        r"\b(replace|overwrite)\s+.*\b(with|by)\b",
        # "remove/delete the file/function" (without a question context)
        r"\b(remove|delete)\s+(the\s+)?(file|files|directory|folder|code|function|class|method|module)\b",
    ]

    for pattern in explicit_write_patterns:
        if re.search(pattern, lowered_prompt, re.IGNORECASE):
            return True

    # ── Layer 2: Read-only context detection ──
    # If the prompt is phrased as a question, explanation request, or
    # analytical query, any write-related words are being used descriptively,
    # not as instructions to modify code.
    read_only_context_patterns = [
        # Questions about how to do something (learning, not doing)
        r"how\s+(to|do|can\s+i|does|should\s+i)\s+(write|create|modify|update|delete|remove|edit|implement|fix)",
        # Requests for examples or demonstrations
        r"(example|demonstration|sample|illustration)\s+(of|for|showing)\s+(writing|creating|modifying|updating|deleting|removing|how\s+to)",
        # Best practices / guidelines (knowledge, not action)
        r"(best\s+practice|guideline|recommendation|pattern|convention)\s+(for|about|on)\s+(writing|creating|modifying|updating|deleting|removing)",
        # Explanation requests
        r"(explain|describe|what\s+(does|is|are)|tell\s+me\s+about|show\s+me\s+how)\s+.*(write|create|modify|update|delete|remove|wrote|created|writes|creates)",
        # Documentation/reference queries
        r"(documentation|docs|reference|api)\s+(for|about|on)\s+(write|create|modify|update|delete|remove)",
        # Analytical/review queries: "analyze how X creates Y", "review the update logic"
        r"(analyze|review|examine|investigate|compare|find|search|list|count|check|verify|understand)\s+.*(write|create|modify|update|delete|remove)",
        # Past-tense or third-person: "where the code writes to disk", "how the factory creates objects"
        r"(where|how|when|why)\s+.*(writes|creates|modifies|updates|deletes|removes|wrote|created|modified|updated|deleted|removed)",
        # "the write method", "the create function" — referring to named entities
        r"(the\s+)?(write|create|modify|update|delete|remove)\s+(method|function|handler|callback|operation|routine|procedure|class|module|interface|trait|decorator)",
    ]

    # If ANY read-only context pattern matches, treat the entire prompt as read-only
    # regardless of whether individual write words appear. This is the key fix:
    # queries like "explain what the write method does" or "show me how the code
    # creates objects" will match read-only patterns and return False.
    for pattern in read_only_context_patterns:
        if re.search(pattern, lowered_prompt, re.IGNORECASE):
            return False

    # ── Layer 3: Ambiguous case handling ──
    # If write-related words appear but no explicit imperative command matched
    # (Layer 1) and no read-only context matched (Layer 2), we have an
    # ambiguous case. The current code returns True for ALL such cases
    # (any word in lowered_prompt from write_words → True), which is overly
    # conservative. Instead, we only return True if the write word appears
    # in a syntactic position suggesting an instruction (followed by a direct
    # object within 40 chars, similar to the current second pattern but broader).
    ambiguous_write_patterns = [
        # "write something", "create something" with a nearby target
        r"\b(create|write|edit|modify|update|delete|remove|refactor|rename|move|implement)\b.{0,40}\b(file|files|code|test|tests|function|class|method|doc|docs|documentation|config|module|package|component|script)\b",
        # Standalone imperative without explicit target but with instruction cues
        r"\b(please|kindly|make\s+sure|ensure)\s+.*\b(create|write|edit|modify|update|delete|remove)\b",
    ]

    for pattern in ambiguous_write_patterns:
        if re.search(pattern, lowered_prompt, re.IGNORECASE):
            return True

    # ── Layer 4: Default safe ──
    # If nothing matched, default to False (read-only). This is a deliberate
    # change from the current behavior which defaults to True when write words
    # appear. The rationale: the explicit and ambiguous patterns above already
    # catch genuine write intents; remaining cases are likely read-only queries
    # that happen to contain write-related words in passing. The LLM eligibility
    # classifier provides a second safety net for truly ambiguous edge cases.
    return False


def _normalize_parallel_config(
    parallel_config: ParallelExecutionConfig | None,
) -> ParallelExecutionConfig:
    normalized = parallel_config or ParallelExecutionConfig()
    scheduling_strategy = normalized.scheduling_strategy.lower()
    if scheduling_strategy not in {"fifo", "round-robin"}:
        raise ValueError("Invalid scheduling strategy. Use 'fifo' or 'round-robin'.")
    return ParallelExecutionConfig(
        worker_count=normalized.worker_count,
        auto_split=normalized.auto_split,
        no_parallel=normalized.no_parallel,
        dry_run=normalized.dry_run,
        scheduling_strategy=scheduling_strategy,
        doc_workspace=normalized.doc_workspace,
        force_parallel=normalized.force_parallel,  # NEW: Must be passed through
    )


async def _run_interactive_loop(
    rag_agent: Agent[None, str | DeferredToolRequests],
    message_history: list[ModelMessage],
    project_root: Path,
    config: AgentLoopUI,
    input_prompt: str,
    tool_names: ConfirmationToolNames,
    initial_question: str | None = None,
    query_router: QueryRouter | None = None,
    current_mode: QueryMode | None = None,
    parallel_config: ParallelExecutionConfig | None = None,
) -> None:
    from .shared.query_router import QueryMode

    # Default to CODE_ONLY if not specified
    if current_mode is None:
        current_mode = QueryMode.CODE_ONLY

    normalized_parallel_config = _normalize_parallel_config(parallel_config)
    concurrency_classifier = ConcurrencyEligibilityClassifier()
    subagent_orchestrator = SubAgentOrchestrator(
        worker_count=normalized_parallel_config.worker_count,
        scheduling_strategy=normalized_parallel_config.scheduling_strategy,
        repo_path=str(project_root),
        enable_document_graph=query_router is not None
        and query_router.doc_graph is not None,
        query_mode=current_mode,
        doc_workspace=normalized_parallel_config.doc_workspace,
    )
    task_splitter = TaskSplitter(repo_path=str(project_root))

    # Set up signal handlers for graceful Ctrl+C handling
    # Note: We use a local flag and nested function because the processing task
    # changes dynamically per user input iteration.
    loop = asyncio.get_running_loop()
    _shutdown_requested = False
    _current_processing_task: asyncio.Task | None = None

    def _handle_sigint() -> None:
        """Handle SIGINT by cancelling current processing or exiting."""
        nonlocal _shutdown_requested

        if _shutdown_requested:
            # Second interrupt - force exit with cleanup
            app_context.console.print(f"\n{style(cs.MSG_FORCE_EXIT, cs.Color.RED)}")
            shutdown_manager.initiate_shutdown(signal.SIGINT)
            # initiate_shutdown will call sys.exit, but just in case:
            return
        _shutdown_requested = True

        # Cancel processing task if active
        if _current_processing_task and not _current_processing_task.done():
            app_context.console.print(
                f"\n{style(cs.MSG_THINKING_CANCELLED, cs.Color.YELLOW)}"
            )
            _current_processing_task.cancel()
        # During input (no processing task), let prompt_toolkit's Ctrl+C
        # binding handle the first interrupt, but track that shutdown was
        # requested so second Ctrl+C forces exit

    def _handle_sigterm() -> None:
        """Handle SIGTERM by initiating graceful shutdown."""
        shutdown_manager.initiate_shutdown(signal.SIGTERM)

    # Install signal handlers
    try:
        loop.add_signal_handler(signal.SIGINT, _handle_sigint)
        loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)
    except (NotImplementedError, RuntimeError):
        # Windows or loop already closed
        pass

    try:
        init_session_log(project_root)
        app_context.session.history = message_history
        question = initial_question or ""
        model_override: Model | None = None
        model_override_string: str | None = None
        model_override_config: ModelConfig | None = None

        while True:
            try:
                _shutdown_requested = False  # Reset for each iteration

                if not initial_question or question != initial_question:
                    question = await asyncio.to_thread(
                        get_multiline_input, input_prompt
                    )

                stripped_question = question.strip()
                stripped_lower = stripped_question.lower()

                if stripped_lower in cs.EXIT_COMMANDS:
                    break

                if not stripped_question:
                    initial_question = None
                    continue

                command_parts = stripped_lower.split(maxsplit=1)
                if command_parts[0] == cs.MODELS_COMMAND_PREFIX:
                    _handle_models_command(stripped_question, model_override_config)
                    initial_question = None
                    continue
                if command_parts[0] == cs.MODEL_COMMAND_PREFIX:
                    model_override, model_override_string, model_override_config = (
                        _handle_model_command(
                            stripped_question,
                            model_override,
                            model_override_string,
                            model_override_config,
                        )
                    )
                    initial_question = None
                    continue
                if command_parts[0] == cs.MODE_COMMAND_PREFIX:
                    current_mode, status_message = _handle_mode_command(
                        stripped_question,
                        query_router,
                        current_mode,
                    )
                    app_context.console.print(status_message)
                    initial_question = None
                    continue
                if command_parts[0] == cs.HELP_COMMAND:
                    app_context.console.print(cs.UI_HELP_COMMANDS)
                    initial_question = None
                    continue
                if command_parts[0] == cs.COMPRESS_COMMAND_PREFIX:
                    # Manual /compress command handler
                    from rich.table import Table

                    aggressive = "--aggressive" in stripped_lower
                    preserve_match = re.search(
                        r"--preserve\s*([^\s]*)", stripped_question
                    )
                    preserve_pattern = (
                        preserve_match.group(1).strip("'\"") if preserve_match else None
                    )
                    workers_match = re.search(r"--workers\s*(\d+)", stripped_question)
                    workers = (
                        int(workers_match.group(1))
                        if workers_match
                        else settings.CONTEXT_COMPRESSION_PARALLEL_WORKERS
                    )

                    if not app_context.session.history:
                        app_context.console.print(
                            style("⚠️ No existing context to compress", cs.Color.YELLOW)
                        )
                        initial_question = None
                        continue

                    app_context.console.print(
                        style(
                            f"🔄 Running context compression with {workers} parallel round-robin workers...",
                            cs.Color.CYAN,
                        )
                    )

                    # Convert session history to format expected by compressor
                    context = _message_history_to_context(app_context.session.history)

                    # Get max context window for current model
                    max_context = settings.DEFAULT_CONTEXT_WINDOW
                    try:
                        # Check for role-specific override first (highest precedence)
                        if settings.ORCHESTRATOR_CONTEXT_WINDOW:
                            max_context = settings.ORCHESTRATOR_CONTEXT_WINDOW
                        elif model_override_config:
                            # Get from model override if set
                            provider = get_provider_from_config(model_override_config)
                            max_context = provider.get_model_context_window(
                                model_override_config.model_id
                            )
                        else:
                            # Get from default orchestrator config
                            provider = get_provider_from_config(
                                settings.active_orchestrator_config
                            )
                            max_context = provider.get_model_context_window(
                                settings.active_orchestrator_config.model_id
                            )
                    except Exception as e:
                        logger.debug(
                            f"Failed to retrieve model context window for /compress command, using default {settings.DEFAULT_CONTEXT_WINDOW:,}: {e}"
                        )

                    compressor = ContextCompressor(
                        context=context,
                        max_context=max_context,
                        aggressive_mode=aggressive,
                        preserve_pattern=preserve_pattern,
                        worker_count=workers,
                    )
                    result = compressor.compress_sync()

                    # Display results
                    table = Table(
                        title=style("Context Compression Results", cs.Color.GREEN),
                        show_header=True,
                        header_style=f"{cs.StyleModifier.BOLD} {cs.Color.MAGENTA}",
                    )
                    table.add_column("Metric", style=cs.Color.CYAN)
                    table.add_column("Value", style=cs.Color.YELLOW)

                    table.add_row("Original tokens", f"{result.original_tokens:,}")
                    table.add_row("Compressed tokens", f"{result.compressed_tokens:,}")
                    table.add_row("Reduction", f"{result.reduction_pct:.1%}")
                    table.add_row("Semantic retention", f"{result.retention_score:.1%}")
                    table.add_row("Strategy used", result.strategy_used)
                    table.add_row(
                        "Execution time", f"{result.execution_time * 1000:.0f}ms"
                    )
                    if result.archive_id:
                        table.add_row(
                            "Archive ID (restore available)", result.archive_id
                        )

                    app_context.console.print(table)

                    if result.was_rolled_back:
                        app_context.console.print(
                            style(
                                "⚠️ Compression rolled back: retention too low, no changes made",
                                cs.Color.YELLOW,
                            )
                        )
                    else:
                        compressed_history = _context_to_message_history(
                            result.compressed_context
                        )
                        if compressed_history:
                            app_context.session.history[:] = compressed_history
                            app_context.console.print(
                                style(
                                    "✅ Context compressed successfully! Session continues with reduced token usage",
                                    cs.Color.GREEN,
                                )
                            )
                        else:
                            app_context.console.print(
                                style(
                                    "⚠️ Compression returned no usable message history, keeping original context",
                                    cs.Color.YELLOW,
                                )
                            )

                    initial_question = None
                    continue

                log_session_event(f"{cs.SESSION_PREFIX_USER}{question}")

                if app_context.session.cancelled:
                    question_with_context = question + get_session_context()
                    app_context.session.reset_cancelled()
                else:
                    question_with_context = question

                question_with_context = _handle_chat_images(
                    question_with_context, project_root
                )

                subagent_orchestrator.query_mode = current_mode

                has_write_operations = _has_write_intent(question_with_context)
                preview_subtasks: list[dict[str, Any]] = []
                preview_count: int | None = None

                if normalized_parallel_config.auto_split:
                    preview_subtasks = task_splitter.split_task(question_with_context)
                    preview_count = len(preview_subtasks)

                if normalized_parallel_config.no_parallel:
                    logger.info(
                        "Parallel execution skipped due to explicit sequential override"
                    )
                elif has_write_operations:
                    # SAFETY: Write operations ALWAYS block parallel execution.
                    # force_parallel does NOT override this — it only bypasses the
                    # LLM eligibility classifier's threshold, not the write-safety gate.
                    # Attempting to parallelize write operations risks data corruption
                    # from concurrent file modifications.
                    logger.info(
                        "Parallel execution skipped: contains write operations "
                        "(cannot be overridden by --force-parallel for safety)"
                    )
                elif normalized_parallel_config.force_parallel:
                    # force_parallel bypasses the LLM eligibility classifier's threshold.
                    # It should only be used when the user knows their task is read-only
                    # but the classifier incorrectly rejects it (e.g., low confidence).
                    logger.warning(
                        "Parallel execution forced by user override (--force-parallel). "
                        "Write safety checks are still enforced; this only bypasses "
                        "the LLM eligibility threshold."
                    )
                    eligible, task_type, confidence = True, "user_forced", 1.0
                    if preview_count is None or preview_count < settings.CGR_PARALLEL_MIN_SUBTASKS:
                        logger.info(
                            f"Parallel execution downgraded to sequential because only {preview_count or 0} safe subtasks were found"
                        )
                    else:
                        app_context.console.print(
                            style(
                                f"\n✅ Auto-activating parallel execution: {task_type} (confidence: {confidence:.2f})",
                                cs.Color.GREEN,
                            )
                        )
                        app_context.console.print(
                            style(
                                f"🔄 Using {subagent_orchestrator.dynamic_controller.get_effective_worker_count(normalized_parallel_config.worker_count, preview_count)} workers with {normalized_parallel_config.scheduling_strategy} scheduling",
                                cs.Color.CYAN,
                            )
                        )
                        app_context.console.print(
                            style(
                                f"📋 Split into {preview_count} independent subtasks",
                                cs.Color.CYAN,
                            )
                        )

                        aggregator = subagent_orchestrator.execute_tasks(
                            preview_subtasks,
                            dry_run=normalized_parallel_config.dry_run,
                        )
                        parallel_result = aggregator.consolidate()
                        
                        # RECORD: Successful forced parallel execution
                        concurrency_classifier.record_execution_result(task_type, success=True)
                        
                        summary_label = (
                            "plan generated"
                            if normalized_parallel_config.dry_run
                            else "completed"
                        )
                        app_context.console.print(
                            style(
                                f"⚡ Parallel execution {summary_label} in {aggregator.metadata['total_execution_time']:.2f}s",
                                cs.Color.GREEN,
                            )
                        )
                        context_header = (
                            "### Parallel Execution Plan"
                            if normalized_parallel_config.dry_run
                            else "### Parallel Execution Results"
                        )
                        question_with_context += (
                            f"\n\n{context_header}:\n{parallel_result}"
                        )
                elif (
                    preview_count is not None
                    and preview_count > settings.CGR_PARALLEL_MAX_QUEUE_SIZE
                ):
                    logger.info(
                        f"Parallel execution skipped because preview split exceeded queue limit ({preview_count} > {settings.CGR_PARALLEL_MAX_QUEUE_SIZE})"
                    )
                else:
                    (
                        eligible,
                        task_type,
                        confidence,
                    ) = await concurrency_classifier.is_eligible(
                        question_with_context,
                        subtask_count=preview_count,
                        has_write_operations=has_write_operations,
                    )

                    if not eligible:
                        logger.info(
                            f"Parallel execution skipped: task_type={task_type}, confidence={confidence:.2f}"
                        )
                        # RECORD: LLM-driven rejection for calibration
                        # Only record non-deterministic rejection types (LLM decisions).
                        # Deterministic rejections (write, safety rules) are always correct
                        # and don't benefit from threshold adjustment.
                        if task_type not in (
                            "write_operation",
                            "user_requested_sequential",
                            "safety_rule_blocked",
                            "insufficient_subtasks",
                            "concurrency_disabled",
                        ):
                            concurrency_classifier.record_execution_result(task_type, success=False)
                    elif not normalized_parallel_config.auto_split:
                        logger.info(
                            "Parallel execution skipped because auto-splitting is disabled"
                        )
                    elif preview_count is None or preview_count < settings.CGR_PARALLEL_MIN_SUBTASKS:
                        logger.info(
                            f"Parallel execution downgraded to sequential because only {preview_count or 0} safe subtasks were found"
                        )
                    else:
                        app_context.console.print(
                            style(
                                f"\n✅ Auto-activating parallel execution: {task_type} (confidence: {confidence:.2f})",
                                cs.Color.GREEN,
                            )
                        )
                        app_context.console.print(
                            style(
                                f"🔄 Using {subagent_orchestrator.dynamic_controller.get_effective_worker_count(normalized_parallel_config.worker_count, preview_count)} workers with {normalized_parallel_config.scheduling_strategy} scheduling",
                                cs.Color.CYAN,
                            )
                        )
                        app_context.console.print(
                            style(
                                f"📋 Split into {preview_count} independent subtasks",
                                cs.Color.CYAN,
                            )
                        )

                        aggregator = subagent_orchestrator.execute_tasks(
                            preview_subtasks,
                            dry_run=normalized_parallel_config.dry_run,
                        )
                        parallel_result = aggregator.consolidate()
                        
                        # RECORD: Successful classifier-approved parallel execution
                        concurrency_classifier.record_execution_result(task_type, success=True)
                        
                        summary_label = (
                            "plan generated"
                            if normalized_parallel_config.dry_run
                            else "completed"
                        )
                        app_context.console.print(
                            style(
                                f"⚡ Parallel execution {summary_label} in {aggregator.metadata['total_execution_time']:.2f}s",
                                cs.Color.GREEN,
                            )
                        )
                        context_header = (
                            "### Parallel Execution Plan"
                            if normalized_parallel_config.dry_run
                            else "### Parallel Execution Results"
                        )
                        question_with_context += (
                            f"\n\n{context_header}:\n{parallel_result}"
                        )

                # Create a task for the agent response loop so it can be cancelled
                _current_processing_task = asyncio.create_task(
                    _run_agent_response_loop(
                        rag_agent,
                        message_history,
                        question_with_context,
                        config,
                        tool_names,
                        model_override,
                        model_override_config,
                    )
                )
                try:
                    await _current_processing_task
                except asyncio.CancelledError:
                    # Defensive: CancelledError propagates if run_with_cancellation
                    # doesn't catch it (shouldn't happen in current design)
                    break
                finally:
                    _current_processing_task = None

                initial_question = None

            except KeyboardInterrupt:
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(ls.UNEXPECTED.format(error=e))
                app_context.console.print(cs.UI_ERR_UNEXPECTED.format(error=e))

    finally:
        # Clean up signal handlers
        try:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
        except (NotImplementedError, RuntimeError):
            pass


async def run_chat_loop(
    rag_agent: Agent[None, str | DeferredToolRequests],
    message_history: list[ModelMessage],
    project_root: Path,
    tool_names: ConfirmationToolNames,
    query_router: QueryRouter | None = None,
    current_mode: QueryMode | None = None,
    parallel_config: ParallelExecutionConfig | None = None,
) -> None:
    await _run_interactive_loop(
        rag_agent,
        message_history,
        project_root,
        CHAT_LOOP_UI,
        style(cs.PROMPT_ASK_QUESTION, cs.Color.CYAN),
        tool_names,
        query_router=query_router,
        current_mode=current_mode,
        parallel_config=parallel_config,
    )


def _update_single_model_setting(role: cs.ModelRole, model_string: str) -> None:
    provider, model = settings.parse_model_string(model_string)

    match role:
        case cs.ModelRole.ORCHESTRATOR:
            current_config = settings.active_orchestrator_config
            set_method = settings.set_orchestrator
        case cs.ModelRole.CYPHER:
            current_config = settings.active_cypher_config
            set_method = settings.set_cypher

    kwargs = current_config.to_update_kwargs()

    if provider == cs.Provider.OLLAMA and not kwargs[cs.FIELD_ENDPOINT]:
        kwargs[cs.FIELD_ENDPOINT] = settings.ollama_endpoint
        kwargs[cs.FIELD_API_KEY] = cs.DEFAULT_API_KEY

    set_method(provider, model, **kwargs)


def update_model_settings(
    orchestrator: str | None,
    cypher: str | None,
) -> None:
    if orchestrator:
        _update_single_model_setting(cs.ModelRole.ORCHESTRATOR, orchestrator)
    if cypher:
        _update_single_model_setting(cs.ModelRole.CYPHER, cypher)


def _write_graph_json(ingestor: MemgraphIngestor, output_path: Path) -> GraphData:
    graph_data: GraphData = ingestor.export_graph_to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding=cs.ENCODING_UTF8) as f:
        json.dump(graph_data, f, indent=cs.JSON_INDENT, ensure_ascii=False)

    return graph_data


def connect_memgraph(batch_size: int) -> MemgraphIngestor:
    return MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=batch_size,
        username=settings.MEMGRAPH_USERNAME,
        password=settings.MEMGRAPH_PASSWORD,
    )


def connect_doc_memgraph(batch_size: int = 1000) -> MemgraphIngestor:
    """Connect to DOCUMENT graph backend (DOC_MEMGRAPH_HOST:DOC_MEMGRAPH_PORT).

    Args:
        batch_size: Batch size for bulk operations

    Returns:
        MemgraphIngestor instance for document graph
    """
    return MemgraphIngestor(
        host=settings.DOC_MEMGRAPH_HOST,
        port=settings.DOC_MEMGRAPH_PORT,
        batch_size=batch_size,
        username=settings.DOC_MEMGRAPH_USERNAME,
        password=settings.DOC_MEMGRAPH_PASSWORD,
    )


@contextmanager
def connect_both_graphs(
    batch_size: int,
    doc_workspace: str = "default",
) -> Generator[tuple[MemgraphIngestor, MemgraphIngestor], None, None]:
    """Connect to both code and document graphs with proper context management.

    Args:
        batch_size: Batch size for bulk operations
        doc_workspace: Workspace identifier for document graph (used for logging)

    Yields:
        Tuple of (code_graph, doc_graph) ingestors

    Raises:
        Exception: If connection fails, properly cleans up partial connections

    Note:
        Uses manual __enter__/__exit__ calls to manage both connections within
        a single context manager. This is necessary because we need to yield
        both connections together and ensure both are cleaned up on error.
        Safety guarantees: both connections open before yield, both closed on
        any exception, partial cleanup on mid-connection failure, no connection leaks.
    """
    logger.info(f"Connecting to dual graphs with doc_workspace={doc_workspace}")

    code_graph = MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=batch_size,
        username=settings.MEMGRAPH_USERNAME,
        password=settings.MEMGRAPH_PASSWORD,
    )
    doc_graph = MemgraphIngestor(
        host=settings.DOC_MEMGRAPH_HOST,
        port=settings.DOC_MEMGRAPH_PORT,
        batch_size=batch_size,
        username=settings.DOC_MEMGRAPH_USERNAME,
        password=settings.DOC_MEMGRAPH_PASSWORD,
    )

    # Enter code_graph first, with proper cleanup on failure
    code_graph.__enter__()
    try:
        doc_graph.__enter__()
    except Exception:
        # doc_graph failed, cleanup code_graph before raising
        code_graph.__exit__(*sys.exc_info())
        raise

    try:
        yield (code_graph, doc_graph)
    except Exception:
        # Exit both on error with proper exception info
        doc_graph.__exit__(*sys.exc_info())
        code_graph.__exit__(*sys.exc_info())
        raise
    else:
        # Exit both on success
        doc_graph.__exit__(None, None, None)
        code_graph.__exit__(None, None, None)


def _check_graph_freshness(
    repo_path: Path,
    with_docs: bool,
    doc_workspace: str = "default",
) -> tuple[bool, bool, list[str]]:
    """Check if code and document graphs are up-to-date.

    Args:
        repo_path: Repository path
        with_docs: Whether to check document graph
        doc_workspace: Document workspace identifier

    Returns:
        Tuple of (code_fresh, docs_fresh, warnings)

    Note: This is a best-effort check. For comprehensive freshness validation,
    use file hash comparison.
    """
    warnings: list[str] = []
    code_fresh = True
    docs_fresh = True

    # === Check Code Graph Freshness ===
    try:
        with connect_memgraph(batch_size=1) as ingestor:
            # Check if graph has nodes
            result = ingestor.fetch_all("MATCH (n) RETURN count(n) as count")
            if not result or result[0].get("count", 0) == 0:
                code_fresh = False
                warnings.append("Code graph is empty")
            else:
                # Check hash cache exists (basic check)
                cache_path = repo_path / cs.HASH_CACHE_FILENAME
                if not cache_path.exists():
                    warnings.append("Code hash cache not found (may be stale)")
    except Exception as e:
        logger.warning(f"Could not check code graph freshness: {e}")
        warnings.append(f"Code graph check failed: {e}")

    # === Check Document Graph Freshness ===
    if with_docs:
        try:
            with connect_doc_memgraph(batch_size=1) as ingestor:
                # Check if document graph has nodes for this workspace
                result = ingestor.fetch_all(
                    "MATCH (d:Document {workspace: $ws}) RETURN count(d) as count",
                    {"ws": doc_workspace},
                )
                if not result or result[0].get("count", 0) == 0:
                    docs_fresh = False
                    warnings.append(
                        f"No documents indexed for workspace '{doc_workspace}'"
                    )
                else:
                    # Check version cache exists
                    cgr_dir = repo_path / ".cgr"
                    version_cache_path = cgr_dir / "doc_versions.json"
                    if not version_cache_path.exists():
                        docs_fresh = False
                        warnings.append("Document version cache not found")
        except Exception as e:
            logger.warning(f"Could not check document graph freshness: {e}")
            warnings.append(f"Document graph check failed: {e}")

    return (code_fresh, docs_fresh, warnings)


def _prompt_for_reindex(
    code_fresh: bool,
    docs_fresh: bool,
    warnings: list[str],
) -> tuple[bool, bool]:
    """Prompt user to re-index if graphs are stale.

    Args:
        code_fresh: Is code graph up-to-date
        docs_fresh: Is document graph up-to-date
        warnings: List of freshness warnings

    Returns:
        Tuple of (should_index_code, should_index_docs)
    """
    should_index_code = False
    should_index_docs = False

    # Display warnings
    if warnings:
        app_context.console.print(
            style("\n⚠️  Graph Freshness Warnings:", cs.Color.YELLOW)
        )
        for warning in warnings:
            app_context.console.print(f"  - {warning}")

    # Prompt for code indexing
    if not code_fresh:
        if Confirm.ask("\nCode graph appears stale. Index now?"):
            should_index_code = True

    # Prompt for document indexing
    if not docs_fresh:
        if Confirm.ask("Document graph appears stale. Index now?"):
            should_index_docs = True

    return (should_index_code, should_index_docs)


def export_graph_to_file(ingestor: MemgraphIngestor, output: str) -> bool:
    output_path = Path(output)

    try:
        graph_data = _write_graph_json(ingestor, output_path)
        metadata = graph_data[cs.KEY_METADATA]
        app_context.console.print(
            cs.UI_GRAPH_EXPORT_SUCCESS.format(path=output_path.absolute())
        )
        app_context.console.print(
            cs.UI_GRAPH_EXPORT_STATS.format(
                nodes=metadata[cs.KEY_TOTAL_NODES],
                relationships=metadata[cs.KEY_TOTAL_RELATIONSHIPS],
            )
        )
        return True

    except Exception as e:
        app_context.console.print(cs.UI_ERR_EXPORT_FAILED.format(error=e))
        logger.exception(ls.EXPORT_ERROR.format(error=e))
        return False


def detect_excludable_directories(repo_path: Path) -> set[str]:
    detected: set[str] = set()
    queue: deque[tuple[Path, int]] = deque([(repo_path, 0)])
    while queue:
        current, depth = queue.popleft()
        if depth > cs.INTERACTIVE_BFS_MAX_DEPTH:
            continue
        try:
            entries = list(current.iterdir())
        except PermissionError:
            continue
        for path in entries:
            if not path.is_dir():
                continue
            if path.name in cs.IGNORE_PATTERNS:
                detected.add(path.relative_to(repo_path).as_posix())
            else:
                queue.append((path, depth + 1))
    return detected


def _get_grouping_key(path: str) -> str:
    parts = Path(path).parts
    if not parts:
        return cs.INTERACTIVE_DEFAULT_GROUP
    for part in parts:
        if part in cs.IGNORE_PATTERNS:
            return part
    return parts[0]


def _group_paths_by_pattern(paths: set[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for path in paths:
        key = _get_grouping_key(path)
        if key not in groups:
            groups[key] = []
        groups[key].append(path)
    for group_paths in groups.values():
        group_paths.sort()
    return groups


def _format_nested_count(count: int) -> str:
    template = (
        cs.INTERACTIVE_NESTED_SINGULAR if count == 1 else cs.INTERACTIVE_NESTED_PLURAL
    )
    return template.format(count=count)


def _display_grouped_table(groups: dict[str, list[str]]) -> list[str]:
    sorted_roots = sorted(groups.keys())
    table = Table(title=style(cs.INTERACTIVE_TITLE_GROUPED, cs.Color.CYAN))
    table.add_column(cs.INTERACTIVE_COL_NUM, style=cs.Color.YELLOW, width=4)
    table.add_column(cs.INTERACTIVE_COL_PATTERN)
    table.add_column(cs.INTERACTIVE_COL_NESTED, style=cs.INTERACTIVE_STYLE_DIM)

    for i, root in enumerate(sorted_roots, 1):
        nested_count = len(groups[root])
        table.add_row(str(i), root, _format_nested_count(nested_count))

    app_context.console.print(table)
    app_context.console.print(
        style(
            cs.INTERACTIVE_INSTRUCTIONS_GROUPED, cs.Color.YELLOW, cs.StyleModifier.NONE
        )
    )
    return sorted_roots


def _display_nested_table(pattern: str, paths: list[str]) -> None:
    title = cs.INTERACTIVE_TITLE_NESTED.format(pattern=pattern)
    table = Table(title=style(title, cs.Color.CYAN))
    table.add_column(cs.INTERACTIVE_COL_NUM, style=cs.Color.YELLOW, width=4)
    table.add_column(cs.INTERACTIVE_COL_PATH)

    for i, path in enumerate(paths, 1):
        table.add_row(str(i), path)

    app_context.console.print(table)
    app_context.console.print(
        style(
            cs.INTERACTIVE_INSTRUCTIONS_NESTED.format(pattern=pattern),
            cs.Color.YELLOW,
            cs.StyleModifier.NONE,
        )
    )


def _prompt_nested_selection(pattern: str, paths: list[str]) -> set[str]:
    _display_nested_table(pattern, paths)

    response = Prompt.ask(
        style(cs.INTERACTIVE_PROMPT_KEEP, cs.Color.CYAN),
        default=cs.INTERACTIVE_KEEP_NONE,
    )

    if response.lower() == cs.INTERACTIVE_KEEP_ALL:
        return set(paths)
    if response.lower() == cs.INTERACTIVE_KEEP_NONE:
        return set()

    selected: set[str] = set()
    for part in response.split(","):
        part = part.strip()
        if not part:
            continue
        if part.isdigit():
            idx = int(part) - 1
            if 0 <= idx < len(paths):
                selected.add(paths[idx])
            else:
                logger.warning(ls.EXCLUDE_INVALID_INDEX.format(index=part))
        else:
            logger.warning(ls.EXCLUDE_INVALID_INPUT.format(input=part))

    return selected


def prompt_for_unignored_directories(
    repo_path: Path,
    cli_excludes: list[str] | None = None,
) -> frozenset[str]:
    detected = detect_excludable_directories(repo_path)
    cgrignore = load_cgrignore_patterns(repo_path)
    cli_patterns = frozenset(cli_excludes) if cli_excludes else frozenset()
    pre_excluded = cli_patterns | cgrignore.exclude

    if not detected and not pre_excluded:
        return cgrignore.unignore

    all_candidates = detected | pre_excluded
    groups = _group_paths_by_pattern(all_candidates)
    sorted_roots = _display_grouped_table(groups)

    response = Prompt.ask(
        style(cs.INTERACTIVE_PROMPT_KEEP, cs.Color.CYAN),
        default=cs.INTERACTIVE_KEEP_NONE,
    )

    if response.lower() == cs.INTERACTIVE_KEEP_ALL:
        return frozenset(all_candidates) | cgrignore.unignore

    if response.lower() == cs.INTERACTIVE_KEEP_NONE:
        return cgrignore.unignore

    selected: set[str] = set()
    expand_requests: list[int] = []
    regular_selections: list[int] = []

    for part in response.split(","):
        part = part.strip().lower()
        if not part:
            continue

        if part.endswith(cs.INTERACTIVE_EXPAND_SUFFIX) and part[:-1].isdigit():
            expand_requests.append(int(part[:-1]) - 1)
        elif part.isdigit():
            regular_selections.append(int(part) - 1)
        else:
            logger.warning(ls.EXCLUDE_INVALID_INPUT.format(input=part))

    for idx in expand_requests:
        if 0 <= idx < len(sorted_roots):
            root = sorted_roots[idx]
            nested_selected = _prompt_nested_selection(root, groups[root])
            selected.update(nested_selected)
        else:
            logger.warning(ls.EXCLUDE_INVALID_INDEX.format(index=idx + 1))

    for idx in regular_selections:
        if 0 <= idx < len(sorted_roots):
            root = sorted_roots[idx]
            selected.update(groups[root])
        else:
            logger.warning(ls.EXCLUDE_INVALID_INDEX.format(index=idx + 1))

    return frozenset(selected) | cgrignore.unignore


def _validate_provider_config(role: cs.ModelRole, config: ModelConfig) -> None:
    from .providers import get_provider_from_config

    try:
        provider = get_provider_from_config(config)
        provider.validate_config()
    except Exception as e:
        raise ValueError(ex.CONFIG.format(role=role.value.title(), error=e)) from e


def _initialize_services_and_agent(
    repo_path: str,
    ingestor: QueryProtocol,
    doc_ingestor: MemgraphIngestor | None = None,
    query_mode: QueryMode | None = None,
    doc_workspace: str = "default",
) -> tuple[
    Agent[None, str | DeferredToolRequests],
    ConfirmationToolNames,
    QueryRouter | None,
]:
    """Initialize services and agent with optional document graph support.

    Args:
        repo_path: Repository path
        ingestor: Code graph ingestor
        doc_ingestor: Document graph ingestor (optional)
        query_mode: Initial query mode (defaults to CODE_ONLY)
        doc_workspace: Document workspace identifier

    Returns:
        Tuple of (rag_agent, confirmation_tool_names, query_router)
    """
    # Default to CODE_ONLY if not specified
    if query_mode is None:
        query_mode = QueryMode.CODE_ONLY

    _validate_provider_config(
        cs.ModelRole.ORCHESTRATOR, settings.active_orchestrator_config
    )
    _validate_provider_config(cs.ModelRole.CYPHER, settings.active_cypher_config)

    cypher_generator = CypherGenerator()
    code_retriever = CodeRetriever(project_root=repo_path, ingestor=ingestor)
    file_reader = FileReader(project_root=repo_path)
    file_writer = FileWriter(project_root=repo_path)
    file_editor = FileEditor(project_root=repo_path)
    shell_commander = ShellCommander(
        project_root=repo_path, timeout=settings.SHELL_COMMAND_TIMEOUT
    )
    directory_lister = DirectoryLister(project_root=repo_path)

    # === Introspection & navigation tools ===
    python_inspector = PythonObjectInspector(project_root=repo_path)
    graph_navigator = GraphNavigator(project_root=repo_path, ingestor=ingestor)

    # === Document-aware services ===
    if doc_ingestor:
        document_analyzer = DocumentAnalyzer(
            project_root=repo_path,
            doc_graph=doc_ingestor,
        )

        # Create QueryRouter for dual-graph queries
        query_router = QueryRouter(
            code_graph=ingestor,
            doc_graph=doc_ingestor,
        )
        # Store mode in router instance
        query_router.current_mode = query_mode
    else:
        document_analyzer = DocumentAnalyzer(project_root=repo_path)
        query_router = None

    query_tool = create_query_tool(ingestor, cypher_generator, app_context.console)
    code_tool = create_code_retrieval_tool(code_retriever)
    file_reader_tool = create_file_reader_tool(file_reader)
    file_writer_tool = create_file_writer_tool(file_writer)
    file_editor_tool = create_file_editor_tool(file_editor)
    shell_command_tool = create_shell_command_tool(shell_commander)
    directory_lister_tool = create_directory_lister_tool(directory_lister)

    # Enhanced document analyzer tool (returns list[Tool])
    document_analyzer_tools = create_document_analyzer_tool(
        document_analyzer,
        enable_graph_queries=doc_ingestor is not None,
        workspace=doc_workspace,
    )

    semantic_search_tool = create_semantic_search_tool()
    function_source_tool = create_get_function_source_tool()

    # Introspection & navigation tools
    inspect_python_tool = create_inspect_python_object_tool(python_inspector)
    find_references_tool = create_find_references_tool(graph_navigator)
    call_hierarchy_tool = create_get_call_hierarchy_tool(graph_navigator)
    find_impl_tool = create_find_implementations_tool(graph_navigator)
    project_structure_tool = create_get_project_structure_tool(graph_navigator)
    import_deps_tool = create_get_import_dependencies_tool(graph_navigator)

    # Build tools list
    tools: list[Tool] = [
        query_tool,
        code_tool,
        file_reader_tool,
        file_writer_tool,
        file_editor_tool,
        shell_command_tool,
        directory_lister_tool,
        *document_analyzer_tools,
        semantic_search_tool,
        function_source_tool,
        inspect_python_tool,
        find_references_tool,
        call_hierarchy_tool,
        find_impl_tool,
        project_structure_tool,
        import_deps_tool,
    ]

    # Add Document GraphRAG tools only if query_router is available
    if query_router:
        query_document_graph_tool = create_query_document_graph_tool(query_router)
        query_both_graphs_tool = create_query_both_graphs_tool(query_router)
        validate_code_tool = create_validate_code_against_spec_tool(query_router)
        validate_doc_tool = create_validate_doc_against_code_tool(query_router)
        index_docs_tool = create_index_documents_tool()
        graph_query_tool = create_graph_query_tool(query_router)

        tools.extend(
            [
                query_document_graph_tool,
                query_both_graphs_tool,
                validate_code_tool,
                validate_doc_tool,
                index_docs_tool,
                graph_query_tool,
            ]
        )

    confirmation_tool_names = ConfirmationToolNames(
        replace_code=file_editor_tool.name,
        create_file=file_writer_tool.name,
        shell_command=shell_command_tool.name,
    )

    rag_agent = create_rag_orchestrator(tools=tools)
    return rag_agent, confirmation_tool_names, query_router


def main_single_query(repo_path: str, batch_size: int, question: str) -> None:
    _setup_common_initialization(repo_path)
    # (H) Override logger to stderr so stdout is clean for scripted output
    logger.remove()
    # Add console handler to stderr only, error level
    logger.add(sys.stderr, level=cs.LOG_LEVEL_ERROR, format=cs.LOG_FORMAT)

    with connect_memgraph(batch_size) as ingestor:
        rag_agent, _, _ = _initialize_services_and_agent(repo_path, ingestor)
        response = asyncio.run(rag_agent.run(question, message_history=[]))
        print(response.output)  # noqa: T201


async def main_async(
    repo_path: str,
    batch_size: int,
    parallel_config: ParallelExecutionConfig | None = None,
    realtime_config: RealtimeConfig | None = None,
) -> None:
    """Original main_async - unchanged for backward compatibility.

    Calls main_unified_async with default parameters.
    """
    await main_unified_async(
        repo_path=repo_path,
        batch_size=batch_size,
        with_docs=False,
        query_mode=QueryMode.CODE_ONLY,
        parallel_config=parallel_config,
        realtime_config=realtime_config,
    )


async def main_unified_async(
    repo_path: str,
    batch_size: int,
    with_docs: bool = False,
    query_mode: QueryMode | None = None,
    doc_workspace: str = "default",
    parallel_config: ParallelExecutionConfig | None = None,
    realtime_config: RealtimeConfig | None = None,
    _fallback_attempted: bool = False,
) -> None:
    """Main async entry point with dual-graph support.

    Args:
        repo_path: Repository path
        batch_size: Batch size for graph operations
        with_docs: Enable document graph
        query_mode: Initial query mode (defaults to CODE_ONLY)
        doc_workspace: Document workspace identifier
        realtime_config: Optional realtime file watcher configuration
        _fallback_attempted: Internal flag to prevent infinite recursion on fallback
    """
    # Default to CODE_ONLY if not specified
    if query_mode is None:
        query_mode = QueryMode.CODE_ONLY

    project_root = _setup_common_initialization(repo_path)

    # Display configuration table
    table = _create_configuration_table(
        repo_path,
        doc_graph_connected=with_docs,
        query_mode=query_mode,
        doc_workspace=doc_workspace,
    )
    app_context.console.print(table)

    # Display yolo mode warning if enabled
    _display_yolo_warning()

    if with_docs:
        # Connect to both graphs
        try:
            with connect_both_graphs(batch_size, doc_workspace) as (
                code_graph,
                doc_graph,
            ):
                app_context.console.print(
                    style("✅ Connected to code graph", cs.Color.GREEN)
                )
                app_context.console.print(
                    style(
                        f"✅ Connected to document graph (workspace: {doc_workspace})",
                        cs.Color.GREEN,
                    )
                )

                app_context.console.print(
                    Panel(
                        style(cs.MSG_CHAT_INSTRUCTIONS, cs.Color.YELLOW),
                        border_style=cs.Color.YELLOW,
                    )
                )

                # Initialize agent with both graphs
                rag_agent, tool_names, query_router = _initialize_services_and_agent(
                    repo_path,
                    code_graph,
                    doc_ingestor=doc_graph,
                    query_mode=query_mode,
                    doc_workspace=doc_workspace,
                )

                watcher_manager = None
                if realtime_config and realtime_config.enabled:
                    watcher_manager = _create_watcher_manager(
                        project_root,
                        code_graph,
                        realtime_config,
                        doc_ingestor=doc_graph,
                    )
                    watcher_manager.start()

                try:
                    await run_chat_loop(
                        rag_agent,
                        [],
                        project_root,
                        tool_names,
                        query_router=query_router,
                        current_mode=query_mode,
                        parallel_config=parallel_config,
                    )
                finally:
                    if watcher_manager:
                        watcher_manager.stop()
                        watcher_manager.join(timeout=5.0)

        except Exception as e:
            # Fallback to code-only if document graph fails
            app_context.console.print(
                style(f"⚠️  Document graph unavailable: {e}", cs.Color.YELLOW)
            )
            app_context.console.print(
                style("Continuing with code graph only...", cs.Color.YELLOW)
            )
            # Retry with code-only (with recursion guard to prevent infinite loop)
            if not _fallback_attempted:
                await main_unified_async(
                    repo_path,
                    batch_size,
                    with_docs=False,
                    parallel_config=parallel_config,
                    realtime_config=realtime_config,
                    _fallback_attempted=True,
                )
            else:
                # Already tried fallback, re-raise to avoid infinite recursion
                raise
    else:
        # Code graph only (existing behavior)
        async with connect_memgraph(batch_size) as ingestor:
            app_context.console.print(style(cs.MSG_CONNECTED_MEMGRAPH, cs.Color.GREEN))
            app_context.console.print(
                Panel(
                    style(cs.MSG_CHAT_INSTRUCTIONS, cs.Color.YELLOW),
                    border_style=cs.Color.YELLOW,
                )
            )

            rag_agent, tool_names, query_router = _initialize_services_and_agent(
                repo_path, ingestor
            )

            watcher_manager = None
            if realtime_config and realtime_config.enabled:
                watcher_manager = _create_watcher_manager(
                    project_root,
                    ingestor,
                    realtime_config,
                )
                watcher_manager.start()

            try:
                await run_chat_loop(
                    rag_agent,
                    [],
                    project_root,
                    tool_names,
                    query_router=query_router,
                    parallel_config=parallel_config,
                )
            finally:
                if watcher_manager:
                    watcher_manager.stop()
                    watcher_manager.join(timeout=5.0)


def _create_watcher_manager(
    project_root: Path,
    code_ingestor: QueryProtocol,
    realtime_config: RealtimeConfig,
    doc_ingestor: MemgraphIngestor | None = None,
):
    """Create a UnifiedWatcherManager with the appropriate handlers.

    Args:
        project_root: Repository root path
        code_ingestor: Shared code graph ingestor (thread-safe)
        realtime_config: Realtime watcher configuration
        doc_ingestor: Optional shared document graph ingestor

    Returns:
        Configured UnifiedWatcherManager instance
    """
    from .parser_loader import load_parsers

    from realtime_updater import (
        JSONChangeEventHandler,
        UnifiedWatcherManager,
    )

    parsers, queries = load_parsers()

    # Create GraphUpdater for the watcher (shares the same ingestor)
    code_updater = GraphUpdater(
        ingestor=code_ingestor,
        repo_path=project_root,
        parsers=parsers,
        queries=queries,
    )

    # Create doc updater if docs enabled
    doc_updater = None
    if realtime_config.enable_docs and doc_ingestor:
        from .document.document_updater import DocumentGraphUpdater

        doc_updater = DocumentGraphUpdater(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
            repo_path=project_root,
        )

    # Create JSON handler if JSON enabled
    json_handler = None
    if realtime_config.enable_json:
        json_handler = JSONChangeEventHandler(
            repo_path=project_root,
            debounce_seconds=realtime_config.debounce,
            max_wait_seconds=realtime_config.max_wait,
        )

    return UnifiedWatcherManager(
        repo_path=project_root,
        code_updater=code_updater,
        doc_updater=doc_updater,
        json_handler=json_handler,
        debounce_seconds=realtime_config.debounce,
        max_wait_seconds=realtime_config.max_wait,
    )


async def main_optimize_async(
    language: str,
    target_repo_path: str,
    reference_document: str | None = None,
    orchestrator: str | None = None,
    cypher: str | None = None,
    batch_size: int | None = None,
) -> None:
    project_root = _setup_common_initialization(target_repo_path)

    update_model_settings(orchestrator, cypher)

    app_context.console.print(
        cs.UI_OPTIMIZATION_INIT.format(language=language, path=project_root)
    )

    table = _create_configuration_table(
        str(project_root), cs.OPTIMIZATION_TABLE_TITLE, language
    )
    app_context.console.print(table)

    # Display yolo mode warning if enabled
    _display_yolo_warning()

    effective_batch_size = settings.resolve_batch_size(batch_size)

    async with connect_memgraph(effective_batch_size) as ingestor:
        app_context.console.print(style(cs.MSG_CONNECTED_MEMGRAPH, cs.Color.GREEN))

        rag_agent, tool_names, _ = _initialize_services_and_agent(
            target_repo_path, ingestor
        )
        await run_optimization_loop(
            rag_agent, [], project_root, language, tool_names, reference_document
        )
