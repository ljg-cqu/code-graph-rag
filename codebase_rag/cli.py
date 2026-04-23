import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version as get_version
from pathlib import Path
from typing import Literal, cast

import typer
from loguru import logger
from rich.panel import Panel
from rich.table import Table

from . import cli_help as ch
from . import constants as cs
from . import logs as ls
from .config import load_cgrignore_patterns, settings
from .graph_updater import GraphUpdater
from .json_ingestion import are_json_embeddings_available, recreate_json_vector_index
from .main import (
    ParallelExecutionConfig,
    RealtimeConfig,
    _check_graph_freshness,
    _prompt_for_reindex,
    app_context,
    connect_doc_memgraph,
    connect_memgraph,
    export_graph_to_file,
    main_async,
    main_optimize_async,
    main_single_query,
    main_unified_async,
    prompt_for_unignored_directories,
    style,
    update_model_settings,
)
from .parser_loader import load_parsers
from .services.protobuf_service import ProtobufFileIngestor
from .tools.health_checker import HealthChecker
from .tools.language import cli as language_cli
from .types_defs import ResultRow
from .vector_store_memgraph import MemgraphBackend

type ValidationScope = Literal["all", "sections", "claims"]


@dataclass
class EmbeddingAvailabilityInfo:
    """Information about embedding provider availability for startup display."""

    status: Literal["available", "fallback", "unavailable"]
    provider: str | None = None
    model: str | None = None
    original_provider: str | None = None
    error: str | None = None
    install_command: str | None = None


def _validate_embedding_provider() -> EmbeddingAvailabilityInfo:
    """Validate embedding provider at startup.

    LLM-First Design: This is deterministic infrastructure validation.
    Uses existing get_best_available_provider_with_status() for availability checks.

    Returns:
        EmbeddingAvailabilityInfo with provider status for display.
    """
    from .embeddings.provider_availability import (
        get_best_available_provider_with_status,
    )

    provider, status = get_best_available_provider_with_status()

    if provider and status and status.is_available:
        if status.fallback_available:
            # Using fallback
            logger.warning(
                f"Primary embedding provider unavailable: {status.reason}. "
                f"Falling back to {provider}."
            )
            return EmbeddingAvailabilityInfo(
                status="fallback",
                provider=provider,
                original_provider=status.name,
                error=status.reason,
            )
        return EmbeddingAvailabilityInfo(
            status="available",
            provider=provider,
            model=settings.EMBEDDING_MODEL,
        )

    # No provider available
    return EmbeddingAvailabilityInfo(
        status="unavailable",
        provider=settings.EMBEDDING_PROVIDER,
        error=status.reason if status else "Unknown error",
        install_command=status.install_command if status else None,
    )


def _display_embedding_status(info: EmbeddingAvailabilityInfo) -> None:
    """Display embedding provider status banner at startup.

    LLM-First Design: Banner content is deterministic based on status.
    User-facing messaging can be enhanced by LLM later if needed.
    """
    if info.status == "available":
        return  # Silent success

    if info.status == "fallback":
        _warning(
            f"Embedding provider: Using fallback '{info.provider}' "
            f"(primary '{info.original_provider}' unavailable: {info.error})"
        )
        return

    # Unavailable - show helpful banner
    lines = [
        f"Provider: {info.provider} ({settings.EMBEDDING_MODEL})",
        "Status:   UNAVAILABLE",
        f"Reason:   {info.error}",
        "",
        "Document indexing will run in structural-only mode.",
    ]

    if info.install_command:
        lines.append(f"Fix:      {info.install_command}")

    lines.append("Run 'cgr doctor' for more diagnostics.")

    panel = Panel(
        "\n".join(lines),
        title="⚠ Embedding Provider Status",
        border_style="yellow",
    )
    app_context.console.print(panel)


def _check_large_excluded_documents(
    repo_path: Path,
    workspace: str = "default",
) -> None:
    """Check for large excluded documents in the graph at startup.

    LLM-First Design: Deterministic check using existing .cgrignore patterns.
    """
    try:
        from .document.document_updater import DocumentGraphUpdater

        updater = DocumentGraphUpdater(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
            repo_path=repo_path,
            workspace=workspace,
        )

        excluded_large = updater.check_for_large_excluded_documents()

        if not excluded_large:
            return

        lines = [
            "The following documents are in the graph but should be excluded:",
            "",
        ]
        for doc_path, chunk_count in excluded_large:
            lines.append(f"  {doc_path} ({chunk_count:,} chunks)")

        lines.extend([
            "",
            "These documents consume significant graph space and may slow queries.",
            "",
            "To remove them, run:",
            "  cgr clean-docs",
            "",
            "Or run indexing with --clean to rebuild the entire graph.",
        ])

        panel = Panel(
            "\n".join(lines),
            title="⚠ Large Excluded Documents Detected",
            border_style="yellow",
        )
        app_context.console.print(panel)

    except Exception as e:
        logger.debug(f"Failed to check for large excluded documents: {e}")


def _normalize_validation_scope(scope: str) -> ValidationScope:
    if scope in {"all", "sections", "claims"}:
        return cast(ValidationScope, scope)
    raise typer.BadParameter("--scope must be one of: all, sections, claims")

app = typer.Typer(
    name=cs.PACKAGE_NAME,
    help=ch.APP_DESCRIPTION,
    no_args_is_help=True,
    add_completion=False,
)

# Vector subcommand group
vector_app = typer.Typer(help="Vector index management commands")
app.add_typer(vector_app, name="vector")


@vector_app.command(
    "recreate-indexes",
    help="Recreate vector indexes (use after switching embedding models)",
)
def vector_recreate_indexes(
    dimension: int | None = typer.Option(
        None,
        help="New vector dimension (defaults to current embedding model dimension)",
    ),
    clear_embeddings: bool = typer.Option(
        True, help="Clear existing incompatible embeddings during recreation"
    ),
    code: bool = typer.Option(True, help="Apply to code graph"),
    docs: bool = typer.Option(False, help="Apply to document graph"),
    json: bool = typer.Option(False, help="Apply to JSON graph"),
):
    """Recreate vector indexes with correct dimension for your current embedding model."""
    if not any([code, docs, json]):
        _error(
            "You must select at least one graph type to apply to (--code, --docs, --json)"
        )
        raise typer.Exit(1)

    if code:
        _info("Recreating vector indexes for code graph...")
        vector_store = MemgraphBackend(is_document=False)
        vector_store.recreate_vector_indexes(
            new_dimension=dimension, clear_existing_embeddings=clear_embeddings
        )
        vector_store.close()

    if docs:
        _info("Recreating vector indexes for document graph...")
        doc_backend = MemgraphBackend(is_document=True)
        doc_backend.recreate_vector_indexes(
            new_dimension=dimension, clear_existing_embeddings=clear_embeddings
        )
        doc_backend.close()

    if json:
        available, reason = are_json_embeddings_available()
        if not available:
            _warning(f"Skipping JSON vector index recreation: {reason}")
        else:
            _info("Recreating vector indexes for JSON graph...")
            recreate_json_vector_index(
                batch_size=settings.JSON_MEMGRAPH_BATCH_SIZE,
                dimension=dimension or settings.get_effective_vector_dim("json"),
                clear_existing_embeddings=clear_embeddings,
                force_recreate=True,
            )

    _success(
        "Vector indexes recreated successfully! Reindex your data to generate new compatible embeddings."
    )


def _version_callback(value: bool) -> None:
    if value:
        app_context.console.print(
            cs.CLI_MSG_VERSION.format(
                package=cs.PACKAGE_NAME, version=get_version(cs.PACKAGE_NAME)
            ),
            highlight=False,
        )
        raise typer.Exit()


def validate_models_early() -> None:
    try:
        orchestrator_config = settings.active_orchestrator_config
        orchestrator_config.validate_api_key(cs.ModelRole.ORCHESTRATOR)

        cypher_config = settings.active_cypher_config
        cypher_config.validate_api_key(cs.ModelRole.CYPHER)
    except ValueError as e:
        app_context.console.print(style(str(e), cs.Color.RED))
        raise typer.Exit(1) from e


def _update_and_validate_models(orchestrator: str | None, cypher: str | None) -> None:
    try:
        update_model_settings(orchestrator, cypher)
    except ValueError as e:
        app_context.console.print(style(str(e), cs.Color.RED))
        raise typer.Exit(1) from e

    validate_models_early()


def _resolve_exclude_settings(
    repo_path: Path,
    exclude: list[str] | None,
    interactive_setup: bool,
    should_prompt: bool,
) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    cgrignore = load_cgrignore_patterns(repo_path)
    cli_excludes = frozenset(exclude) if exclude else frozenset()
    exclude_paths = cli_excludes | cgrignore.exclude or None

    if interactive_setup and should_prompt:
        unignore_paths = prompt_for_unignored_directories(repo_path, exclude)
    else:
        unignore_paths = cgrignore.unignore or None

    return exclude_paths, unignore_paths


def _handle_indexing(
    repo_path: Path,
    index_code: bool,
    index_docs: bool,
    index_all: bool,
    with_docs: bool,
    clean: bool,
    batch_size: int,
    project_name: str | None,
    exclude: list[str] | None,
    interactive_setup: bool,
    doc_workspace: str = "default",
    output: str | None = None,
    index_timeout: int = 3600,
    # New JSON ingestion parameters (backward compatible defaults)
    ingest_json: bool = False,
    json_path: str | None = None,
    json_skip_invalid: bool = True,
    json_parallel_workers: int = 10,
    json_exclude: list[str] | None = None,
) -> tuple[bool, bool, bool]:
    """Handle code and document indexing before chat.

    Args:
        repo_path: Repository path
        index_code: --index-code flag
        index_docs: --index-docs flag
        index_all: --index-all flag
        with_docs: --with-docs flag
        clean: --clean flag
        batch_size: Batch size
        project_name: Project name override
        exclude: Exclude patterns
        interactive_setup: Interactive setup flag
        doc_workspace: Document workspace identifier (default: 'default')
        output: Output path for graph export (optional)
        index_timeout: Maximum seconds for indexing operations (default: 300s)
        ingest_json: Whether to enable JSON ingestion during indexing
        json_path: Path to JSON file/directory to ingest
        json_skip_invalid: Whether to skip invalid JSON files (or fail)
        json_parallel_workers: Number of parallel workers for JSON ingestion
        json_exclude: Patterns of JSON files to exclude from ingestion

    Returns:
        Tuple of (code_indexed, docs_indexed, effective_with_docs)

    Raises:
        typer.Exit: If code indexing fails with --index-code flag (blocking)
        Note: Document indexing failures are non-blocking, return docs_indexed=False
    """
    code_indexed = False
    docs_indexed = False
    effective_with_docs = with_docs

    # Resolve effective flags
    effective_index_code = index_code or index_all
    effective_index_docs = index_docs or index_all

    # Implied --with-docs if indexing docs
    if effective_index_docs:
        effective_with_docs = True

    exclude_paths, unignore_paths = _resolve_exclude_settings(
        repo_path,
        exclude,
        interactive_setup,
        should_prompt=effective_index_code or effective_index_docs,
    )

    # === Code Indexing ===
    if effective_index_code:
        _info(style(cs.CLI_MSG_UPDATING_GRAPH.format(path=repo_path), cs.Color.GREEN))

        if not interactive_setup:
            _info(style(cs.CLI_MSG_AUTO_EXCLUDE, cs.Color.YELLOW))

        with connect_memgraph(batch_size) as ingestor:
            if clean:
                _info(style(cs.CLI_MSG_CLEANING_DB, cs.Color.YELLOW))
                ingestor.clean_database()
                _delete_hash_cache(repo_path)

            ingestor.ensure_constraints()
            parsers, queries = load_parsers()

            updater = GraphUpdater(
                ingestor=ingestor,
                repo_path=repo_path,
                parsers=parsers,
                queries=queries,
                unignore_paths=unignore_paths,
                exclude_paths=exclude_paths,
                project_name=project_name,
            )
            updater.run(force=clean)

            if output:
                _info(style(cs.CLI_MSG_EXPORTING_TO.format(path=output), cs.Color.CYAN))
                if not export_graph_to_file(ingestor, output):
                    raise typer.Exit(1)

        _info(style(cs.CLI_MSG_GRAPH_UPDATED, cs.Color.GREEN))
        code_indexed = True

    # === Document Indexing ===
    if effective_index_docs:
        _info(
            style(
                f"Indexing documents in: {repo_path} (workspace: {doc_workspace})",
                cs.Color.CYAN,
            )
        )

        try:
            from codebase_rag.document.document_updater import DocumentGraphUpdater

            # Clean document database if requested
            if clean:
                with connect_doc_memgraph(batch_size) as doc_ingestor:
                    _info(style("Cleaning document database...", cs.Color.YELLOW))
                    doc_ingestor.clean_database()
                    _info(style("Document database cleaned.", cs.Color.GREEN))

            updater = DocumentGraphUpdater(
                host=settings.DOC_MEMGRAPH_HOST,
                port=settings.DOC_MEMGRAPH_PORT,
                repo_path=repo_path,
                batch_size=batch_size,
                workspace=doc_workspace,
                exclude_paths=exclude_paths,
                unignore_paths=unignore_paths,
            )
            # DocumentGraphUpdater.run() returns dict with keys:
            # documents_indexed, sections_created, chunks_created, errors
            stats = updater.run(force=clean)

            # Display indexing stats
            table = Table(
                title=style("Document Indexing Results", cs.Color.GREEN),
                show_header=True,
                header_style=f"{cs.StyleModifier.BOLD} {cs.Color.MAGENTA}",
            )
            table.add_column("Metric", style=cs.Color.CYAN)
            table.add_column("Count", style=cs.Color.YELLOW, justify="right")

            for key, value in stats.items():
                table.add_row(key.replace("_", " ").title(), str(value))

            app_context.console.print(table)

            docs_indexed = True

        except Exception as e:
            _info(style(f"Document indexing failed: {e}", cs.Color.RED))
            logger.exception("Document indexing failed")
            # Don't block chat - continue with code only
            docs_indexed = False
            # Per spec: disable --with-docs when document indexing fails
            effective_with_docs = False

    # JSON ingestion runs independently of document indexing
    if ingest_json:
        from codebase_rag.json_ingestion import ingest_json_data

        # Clean JSON database if requested, consistent with other databases
        if clean:
            _info(style("Cleaning JSON database...", cs.Color.YELLOW))
            from codebase_rag.json_ingestion import _create_json_ingestor

            with _create_json_ingestor(batch_size) as json_ingestor:
                json_ingestor.clean_database()
            _info(style("JSON database cleaned.", cs.Color.GREEN))

        _info(
            style(
                f"Running JSON ingestion with {json_parallel_workers} parallel workers...",
                cs.Color.CYAN,
            )
        )

        # Resolve target JSON path (user-provided or repo root)
        target_json_path = json_path or str(repo_path)

        try:
            ingest_result = ingest_json_data(
                input_path=target_json_path,
                skip_existing=True,
                batch_size=batch_size,
                incremental=True,
                dry_run=False,
                parallel_workers=json_parallel_workers,
                exclude_patterns=json_exclude,
                # Link ingested data to active document workspace for isolation
                metadata_override={"workspace": doc_workspace},
            )

            # Display ingestion results
            json_table = Table(
                title=style("JSON Ingestion Results", cs.Color.GREEN),
                show_header=True,
                header_style=f"{cs.StyleModifier.BOLD} {cs.Color.MAGENTA}",
            )
            json_table.add_column("Metric", style=cs.Color.CYAN)
            json_table.add_column("Count", style=cs.Color.YELLOW, justify="right")

            json_table.add_row(
                "JSON files processed",
                str(getattr(ingest_result, "files_processed", 0)),
            )
            json_table.add_row(
                "Invalid JSON files skipped",
                str(getattr(ingest_result, "files_skipped", 0)),
            )
            json_table.add_row(
                "Entities ingested", str(ingest_result.entities_ingested)
            )
            json_table.add_row(
                "Relationships ingested",
                str(ingest_result.relationships_ingested),
            )

            app_context.console.print(json_table)

            # Handle errors if fail-on-invalid is enabled
            if ingest_result.errors:
                _info(
                    style(
                        f"Ingestion completed with {len(ingest_result.errors)} errors:",
                        cs.Color.YELLOW,
                    )
                )
                for error in ingest_result.errors[:15]:
                    _info(style(f"  - {error}", cs.Color.RED))
                if len(ingest_result.errors) > 15:
                    _info(
                        style(
                            f"  ... and {len(ingest_result.errors) - 15} more errors",
                            cs.Color.RED,
                        )
                    )

                if not json_skip_invalid:
                    raise ValueError(
                        "JSON ingestion failed (--json-fail-on-invalid enabled)"
                    )

        except Exception as e:
            _info(style(f"JSON ingestion failed: {e}", cs.Color.RED))
            if not json_skip_invalid:
                raise typer.Exit(1) from e

    return (code_indexed, docs_indexed, effective_with_docs)


@app.callback()
def _global_options(
    version: bool | None = typer.Option(
        None,
        "--version",
        "-v",
        help=ch.HELP_VERSION,
        callback=_version_callback,
        is_eager=True,
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress non-essential output (progress messages, banners, informational logs).",
        is_eager=True,
    ),
    log_level: str | None = typer.Option(
        None,
        "--log-level",
        help="Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        is_eager=True,
    ),
) -> None:
    settings.QUIET = quiet

    # Apply logging CLI flag overrides
    if log_level:
        # Validate and set log level
        settings.LOG_LEVEL = log_level

    # Remove all existing logger handlers to reconfigure from scratch
    logger.remove()

    # Add console handler
    # In quiet mode, only show ERROR level logs on console
    console_log_level = "ERROR" if quiet else settings.LOG_LEVEL
    logger.add(
        lambda msg: app_context.console.print(msg, end=""),
        level=console_log_level
    )


def _info(msg: str) -> None:
    if not settings.QUIET:
        app_context.console.print(msg)


def _error(msg: str) -> None:
    if not settings.QUIET:
        app_context.console.print(style(msg, cs.Color.RED))


def _warning(msg: str) -> None:
    if not settings.QUIET:
        app_context.console.print(style(msg, cs.Color.YELLOW))


def _success(msg: str) -> None:
    if not settings.QUIET:
        app_context.console.print(style(msg, cs.Color.GREEN))


def _delete_hash_cache(repo_path: Path) -> None:
    """Delete hash cache file from repo directory.

    If repo_path is a file, uses its parent directory.
    """
    if repo_path.is_file():
        repo_path = repo_path.parent
    cache_path = repo_path / cs.HASH_CACHE_FILENAME
    if cache_path.exists():
        _info(
            style(
                cs.CLI_MSG_CLEANING_HASH_CACHE.format(path=cache_path),
                cs.Color.YELLOW,
            )
        )
        cache_path.unlink(missing_ok=True)


@app.command(help=ch.CMD_START)
def start(
    repo_path: str | None = typer.Option(
        None, "-r", "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
    index_code: bool = typer.Option(
        False,
        "--index-code",
        help=ch.HELP_INDEX_CODE,
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help=ch.HELP_CLEAN_DB,
    ),
    output: str | None = typer.Option(
        None,
        "-o",
        "--output",
        help=ch.HELP_OUTPUT_GRAPH,
    ),
    orchestrator: str | None = typer.Option(
        None,
        "--orchestrator",
        help=ch.HELP_ORCHESTRATOR,
    ),
    cypher: str | None = typer.Option(
        None,
        "--cypher",
        help=ch.HELP_CYPHER_MODEL,
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help=ch.HELP_NO_CONFIRM,
    ),
    yolo: bool = typer.Option(
        False,
        "--yolo",
        "-y",
        help=ch.HELP_YOLO,
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help=ch.HELP_BATCH_SIZE,
    ),
    project_name: str | None = typer.Option(
        None,
        "--project-name",
        help=ch.HELP_PROJECT_NAME,
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        help=ch.HELP_EXCLUDE_PATTERNS,
    ),
    interactive_setup: bool = typer.Option(
        False,
        "--interactive-setup",
        help=ch.HELP_INTERACTIVE_SETUP,
    ),
    ask_agent: str | None = typer.Option(
        None,
        "-a",
        "--ask-agent",
        help=ch.HELP_ASK_AGENT,
    ),
    # === NEW: Document Graph Support ===
    with_docs: bool = typer.Option(
        False,
        "--with-docs",
        help=ch.HELP_WITH_DOCS,
    ),
    index_docs: bool = typer.Option(
        False,
        "--index-docs",
        help=ch.HELP_INDEX_DOCS,
    ),
    index_all: bool = typer.Option(
        False,
        "--index-all",
        help=ch.HELP_INDEX_ALL,
    ),
    doc_workspace: str = typer.Option(
        "default",
        "--doc-workspace",
        help=ch.HELP_DOC_WORKSPACE,
    ),
    check_freshness: bool = typer.Option(
        True,
        "--check-freshness/--no-check-freshness",
        help=ch.HELP_CHECK_FRESHNESS,
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help=ch.HELP_MODE,
    ),
    index_timeout: int = typer.Option(
        300,
        "--index-timeout",
        help=ch.HELP_INDEX_TIMEOUT,
    ),
    # === NEW: Parallel Sub-Agent Flags ===
    parallel_workers: int | None = typer.Option(
        None,
        "--parallel-workers",
        "-p",
        help=ch.HELP_PARALLEL_WORKERS,
        min=1,
    ),
    auto_split: bool = typer.Option(
        settings.CGR_AUTO_SPLIT_ENABLED,
        "--auto-split/--no-auto-split",
        help=ch.HELP_AUTO_SPLIT,
    ),
    no_parallel: bool = typer.Option(
        False,
        "--no-parallel",
        help=ch.HELP_NO_PARALLEL,
    ),
    parallel_dry_run: bool = typer.Option(
        False,
        "--parallel-dry-run",
        help=ch.HELP_PARALLEL_DRY_RUN,
    ),
    scheduling_strategy: str = typer.Option(
        "fifo",
        "--scheduling-strategy",
        help=ch.HELP_SCHEDULING_STRATEGY,
    ),
    force_parallel: bool = typer.Option(
        False,
        "--force-parallel",
        help="Force parallel execution bypassing LLM eligibility threshold "
             "(write-safety checks are still enforced)",
    ),
    # New JSON ingestion flags (disabled by default, backward compatible)
    ingest_json: bool = typer.Option(
        False,
        "--ingest-json",
        help="Enable automatic JSON ingestion (runs independently of document/code indexing, validates against ingestion_schema.json)",
    ),
    json_path: str | None = typer.Option(
        None,
        "--json-path",
        help="Path to specific JSON file or directory to ingest (defaults to repo root scanning for *.json if not provided)",
    ),
    json_skip_invalid: bool = typer.Option(
        True,
        "--json-skip-invalid/--json-fail-on-invalid",
        help="Skip invalid JSON files (default) or fail ingestion if any JSON is invalid",
    ),
    json_parallel_workers: int = typer.Option(
        10,
        "--json-workers",
        min=1,
        max=32,
        help="Number of parallel workers for JSON ingestion (default: 10, max: 32)",
    ),
    json_exclude: list[str] | None = typer.Option(
        None,
        "--json-exclude",
        help="Patterns of JSON files to exclude from ingestion (supports glob patterns)",
    ),
    # === Realtime Updater Flags ===
    realtime_updater: bool = typer.Option(
        settings.REALTIME_UPDATER_ENABLED,
        "--realtime-updater/--no-realtime-updater",
        help="Enable real-time file system monitoring and automatic graph updates",
    ),
    realtime_debounce: float = typer.Option(
        settings.REALTIME_DEBOUNCE_SECONDS,
        "--realtime-debounce",
        "-rd",
        help="Debounce delay in seconds for real-time updates (0 to disable)",
    ),
    realtime_max_wait: float = typer.Option(
        settings.REALTIME_MAX_WAIT_SECONDS,
        "--realtime-max-wait",
        "-rm",
        help="Maximum wait time in seconds before processing changes",
    ),
    realtime_code: bool = typer.Option(
        settings.REALTIME_CODE_ENABLED,
        "--realtime-code/--no-realtime-code",
        help="Enable real-time updates for code files",
    ),
    realtime_docs: bool = typer.Option(
        settings.REALTIME_DOCS_ENABLED,
        "--realtime-docs/--no-realtime-docs",
        help="Enable real-time updates for document files",
    ),
    realtime_json: bool = typer.Option(
        settings.REALTIME_JSON_ENABLED,
        "--realtime-json/--no-realtime-json",
        help="Enable real-time updates for JSON files",
    ),
) -> None:
    import re

    from codebase_rag.shared.query_router import QueryMode

    # Yolo mode: --yolo or --no-confirm both enable it (CLI flags take precedence)
    yolo_enabled = yolo or no_confirm or settings.CGR_YOLO_MODE
    if yolo_enabled:
        app_context.session.yolo_mode = True
        app_context.session.confirm_edits = False
        # SECURITY WARNING: Yolo mode enables auto-approval of all operations
        from rich.panel import Panel
        from rich.text import Text

        warning = Text("⚠️  YOLO MODE ENABLED ⚠️\n", style="bold red")
        warning.append(
            "All file edits, shell commands, and external API calls will be auto-approved without confirmation.\n"
        )
        warning.append(
            "This is intended for testing and trusted environments only. Use at your own risk.\n",
            style="yellow",
        )
        app_context.console.print(Panel(warning, style="bold red"))
    else:
        app_context.session.yolo_mode = False
        app_context.session.confirm_edits = True

    # === CLI Flag Validation ===
    # Calculate effective_with_docs
    effective_with_docs = with_docs or index_docs or index_all

    # Mode validation: non-code_only and non-auto modes require --with-docs
    # "auto" is always allowed since it resolves after graph connection
    if mode.lower() != "auto" and mode != "code_only" and not effective_with_docs:
        typer.echo(
            f"ERROR: Mode '{mode}' requires document graph. "
            f"Add --with-docs, --index-docs, or --index-all flag.",
            err=True,
        )
        raise typer.Exit(1)

    # Parse and validate mode
    try:
        if mode.lower() == "auto":
            query_mode = None  # Triggers auto-detection in main.py
        else:
            query_mode = QueryMode(mode.lower())
    except ValueError:
        typer.echo(
            f"ERROR: Invalid mode '{mode}'. "
            f"Valid modes: auto, code_only, document_only, both_merged, code_vs_doc, doc_vs_code",
            err=True,
        )
        raise typer.Exit(1)

    # Workspace validation: valid identifier pattern
    if not re.match(r"^[a-zA-Z0-9_-]{1,64}$", doc_workspace):
        typer.echo(
            f"ERROR: Invalid workspace '{doc_workspace}'. "
            f"Must be 1-64 chars: letters, numbers, underscore, hyphen only.",
            err=True,
        )
        raise typer.Exit(1)

    normalized_scheduling_strategy = scheduling_strategy.lower()
    if normalized_scheduling_strategy not in {"fifo", "round-robin"}:
        typer.echo(
            "ERROR: Invalid scheduling strategy. Use 'fifo' or 'round-robin'.",
            err=True,
        )
        raise typer.Exit(1)

    target_repo_path = repo_path or settings.TARGET_REPO_PATH

    # Validate repo path exists
    if not Path(target_repo_path).exists():
        typer.echo(
            f"ERROR: Repository path '{target_repo_path}' does not exist.\n"
            f"HINT: If you used --repo-path, make sure to provide a valid path after it, e.g. --repo-path . --clean",
            err=True,
        )
        raise typer.Exit(1)

    # --output requires --index-code or --index-all (which triggers code update)
    if output and not (index_code or index_all):
        app_context.console.print(
            style(cs.CLI_ERR_OUTPUT_REQUIRES_UPDATE, cs.Color.RED)
        )
        raise typer.Exit(1)

    # === Handle --clean alone (no indexing) ===
    # Preserves backward compatibility: clean database and return immediately
    # without model validation, freshness check, or chat session
    if clean and not (index_code or index_docs or index_all):
        effective_batch_size = settings.resolve_batch_size(batch_size)
        _info(style(cs.CLI_MSG_CLEANING_DB, cs.Color.YELLOW))
        # Clean main code database
        with connect_memgraph(effective_batch_size) as ingestor:
            ingestor.clean_database()
        # Clean document database
        _info(style("Cleaning document database...", cs.Color.YELLOW))
        with connect_doc_memgraph(effective_batch_size) as doc_ingestor:
            doc_ingestor.clean_database()
        # Clean JSON database
        _info(style("Cleaning JSON database...", cs.Color.YELLOW))
        from codebase_rag.json_ingestion import _create_json_ingestor

        with _create_json_ingestor(effective_batch_size) as json_ingestor:
            json_ingestor.clean_database()
        _delete_hash_cache(Path(target_repo_path))
        _info(style(cs.CLI_MSG_CLEAN_DONE, cs.Color.GREEN))
        return

    effective_batch_size = settings.resolve_batch_size(batch_size)

    _update_and_validate_models(orchestrator, cypher)

    # === Validate Embedding Provider ===
    # Show warning if embedding provider is unavailable
    embedding_info = _validate_embedding_provider()
    _display_embedding_status(embedding_info)

    # === Check for Large Excluded Documents ===
    # Warn if large documents that should be excluded are in the graph
    if effective_with_docs:
        _check_large_excluded_documents(Path(target_repo_path), doc_workspace)

    # === Handle indexing with new flags ===
    code_indexed, docs_indexed, effective_with_docs = _handle_indexing(
        repo_path=Path(target_repo_path),
        index_code=index_code,
        index_docs=index_docs,
        index_all=index_all,
        with_docs=with_docs,
        clean=clean,
        batch_size=effective_batch_size,
        project_name=project_name,
        exclude=exclude,
        interactive_setup=interactive_setup,
        doc_workspace=doc_workspace,
        output=output,
        index_timeout=index_timeout,
        # Pass new JSON ingestion parameters
        ingest_json=ingest_json,
        json_path=json_path,
        json_skip_invalid=json_skip_invalid,
        json_parallel_workers=json_parallel_workers,
        json_exclude=json_exclude,
    )

    # If only updating graph (no chat), return
    if code_indexed and not effective_with_docs and not ask_agent:
        return

    # === Freshness Check ===
    if check_freshness and not code_indexed and not docs_indexed:
        # Only check freshness if we haven't just indexed
        repo_to_check = Path(target_repo_path)
        code_fresh, docs_fresh, warnings = _check_graph_freshness(
            repo_to_check, effective_with_docs, doc_workspace
        )

        if warnings:
            should_index_code, should_index_docs = _prompt_for_reindex(
                code_fresh, docs_fresh, warnings
            )
            exclude_paths, unignore_paths = _resolve_exclude_settings(
                repo_to_check,
                exclude,
                interactive_setup,
                should_prompt=should_index_code or should_index_docs,
            )

            # Handle re-indexing if user confirmed
            if should_index_code:
                _info(
                    style(
                        cs.CLI_MSG_UPDATING_GRAPH.format(path=repo_to_check),
                        cs.Color.GREEN,
                    )
                )
                with connect_memgraph(effective_batch_size) as ingestor:
                    ingestor.ensure_constraints()
                    parsers, queries = load_parsers()
                    updater = GraphUpdater(
                        ingestor=ingestor,
                        repo_path=repo_to_check,
                        parsers=parsers,
                        queries=queries,
                    )
                    updater.run(force=False)
                    _info(style(cs.CLI_MSG_GRAPH_UPDATED, cs.Color.GREEN))

            if should_index_docs:
                _info(
                    style(
                        f"Indexing documents in: {repo_to_check}",
                        cs.Color.CYAN,
                    )
                )
                try:
                    from codebase_rag.document.document_updater import (
                        DocumentGraphUpdater,
                    )

                    updater = DocumentGraphUpdater(
                        host=settings.DOC_MEMGRAPH_HOST,
                        port=settings.DOC_MEMGRAPH_PORT,
                        repo_path=repo_to_check,
                        workspace=doc_workspace,
                        exclude_paths=exclude_paths,
                        unignore_paths=unignore_paths,
                    )
                    stats = updater.run(force=False)
                    _info(style(f"Documents indexed: {stats}", cs.Color.GREEN))
                    effective_with_docs = True
                except Exception as e:
                    _info(style(f"Document indexing failed: {e}", cs.Color.RED))
                    effective_with_docs = False

    # === Start chat session ===
    parallel_config = ParallelExecutionConfig(
        worker_count=parallel_workers,
        auto_split=auto_split,
        no_parallel=no_parallel,
        dry_run=parallel_dry_run,
        scheduling_strategy=normalized_scheduling_strategy,
        doc_workspace=doc_workspace,
        force_parallel=force_parallel,  # NEW
    )

    # Build realtime config if enabled
    rt_config = RealtimeConfig(
        enabled=realtime_updater,
        debounce=realtime_debounce,
        max_wait=realtime_max_wait,
        enable_code=realtime_code,
        enable_docs=realtime_docs,
        enable_json=realtime_json,
    )

    try:
        if ask_agent:
            main_single_query(target_repo_path, effective_batch_size, ask_agent)
        elif effective_with_docs:
            # Use unified async with document graph support
            asyncio.run(
                main_unified_async(
                    target_repo_path,
                    effective_batch_size,
                    with_docs=effective_with_docs,
                    query_mode=query_mode,
                    doc_workspace=doc_workspace,
                    parallel_config=parallel_config,
                    realtime_config=rt_config,
                )
            )
        else:
            asyncio.run(
                main_async(
                    target_repo_path,
                    effective_batch_size,
                    parallel_config=parallel_config,
                    realtime_config=rt_config,
                )
            )
    except KeyboardInterrupt:
        app_context.console.print(style(cs.CLI_MSG_APP_TERMINATED, cs.Color.RED))
    except ValueError as e:
        app_context.console.print(
            style(cs.CLI_ERR_STARTUP.format(error=e), cs.Color.RED)
        )


@app.command(help=ch.CMD_INDEX)
def index(
    repo_path: str | None = typer.Option(
        None, "-r", "--repo-path", help=ch.HELP_REPO_PATH_INDEX
    ),
    output_proto_dir: str = typer.Option(
        ...,
        "-o",
        "--output-proto-dir",
        help=ch.HELP_OUTPUT_PROTO_DIR,
    ),
    split_index: bool = typer.Option(
        False,
        "--split-index",
        help=ch.HELP_SPLIT_INDEX,
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        help=ch.HELP_EXCLUDE_PATTERNS,
    ),
    interactive_setup: bool = typer.Option(
        False,
        "--interactive-setup",
        help=ch.HELP_INTERACTIVE_SETUP,
    ),
) -> None:
    target_repo_path = repo_path or settings.TARGET_REPO_PATH
    # Validate repo path exists
    if not Path(target_repo_path).exists():
        typer.echo(
            f"ERROR: Repository path '{target_repo_path}' does not exist.",
            err=True,
        )
        raise typer.Exit(1)
    repo_to_index = Path(target_repo_path)
    _info(style(cs.CLI_MSG_INDEXING_AT.format(path=repo_to_index), cs.Color.GREEN))

    _info(style(cs.CLI_MSG_OUTPUT_TO.format(path=output_proto_dir), cs.Color.CYAN))

    exclude_paths, unignore_paths = _resolve_exclude_settings(
        repo_to_index,
        exclude,
        interactive_setup,
        should_prompt=interactive_setup,
    )
    if not interactive_setup:
        _info(style(cs.CLI_MSG_AUTO_EXCLUDE, cs.Color.YELLOW))

    try:
        ingestor = ProtobufFileIngestor(
            output_path=output_proto_dir, split_index=split_index
        )
        parsers, queries = load_parsers()
        updater = GraphUpdater(
            ingestor=ingestor,
            repo_path=repo_to_index,
            parsers=parsers,
            queries=queries,
            unignore_paths=unignore_paths,
            exclude_paths=exclude_paths,
        )

        updater.run()
        _info(style(cs.CLI_MSG_INDEXING_DONE, cs.Color.GREEN))

    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_INDEXING.format(error=e), cs.Color.RED)
        )
        logger.exception(ls.INDEXING_FAILED)
        raise typer.Exit(1) from e


@app.command(help=ch.CMD_EXPORT)
def export(
    output: str = typer.Option(..., "-o", "--output", help=ch.HELP_OUTPUT_PATH),
    format_json: bool = typer.Option(
        True, "--json/--no-json", help=ch.HELP_FORMAT_JSON
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help=ch.HELP_BATCH_SIZE,
    ),
) -> None:
    if not format_json:
        app_context.console.print(style(cs.CLI_ERR_ONLY_JSON, cs.Color.RED))
        raise typer.Exit(1)

    _info(style(cs.CLI_MSG_CONNECTING_MEMGRAPH, cs.Color.CYAN))

    effective_batch_size = settings.resolve_batch_size(batch_size)

    try:
        with connect_memgraph(effective_batch_size) as ingestor:
            _info(style(cs.CLI_MSG_EXPORTING_DATA, cs.Color.CYAN))

            if not export_graph_to_file(ingestor, output):
                raise typer.Exit(1)

    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_EXPORT_FAILED.format(error=e), cs.Color.RED)
        )
        logger.exception(ls.EXPORT_ERROR.format(error=e))
        raise typer.Exit(1) from e


@app.command(help=ch.CMD_OPTIMIZE)
def optimize(
    language: str = typer.Argument(
        ...,
        help=ch.HELP_LANGUAGE_ARG,
    ),
    repo_path: str | None = typer.Option(
        None, "-r", "--repo-path", help=ch.HELP_REPO_PATH_OPTIMIZE
    ),
    reference_document: str | None = typer.Option(
        None,
        "--reference-document",
        help=ch.HELP_REFERENCE_DOC,
    ),
    orchestrator: str | None = typer.Option(
        None,
        "--orchestrator",
        help=ch.HELP_ORCHESTRATOR,
    ),
    cypher: str | None = typer.Option(
        None,
        "--cypher",
        help=ch.HELP_CYPHER_MODEL,
    ),
    no_confirm: bool = typer.Option(
        False,
        "--no-confirm",
        help=ch.HELP_NO_CONFIRM,
    ),
    yolo: bool = typer.Option(
        False,
        "--yolo",
        "-y",
        help=ch.HELP_YOLO,
    ),
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help=ch.HELP_BATCH_SIZE,
    ),
) -> None:
    # Yolo mode: --yolo or --no-confirm both enable it (CLI flags take precedence)
    yolo_enabled = yolo or no_confirm or settings.CGR_YOLO_MODE
    if yolo_enabled:
        app_context.session.yolo_mode = True
        app_context.session.confirm_edits = False
        # SECURITY WARNING: Yolo mode enables auto-approval of all operations
        from rich.panel import Panel
        from rich.text import Text

        warning = Text("⚠️  YOLO MODE ENABLED ⚠️\n", style="bold red")
        warning.append(
            "All file edits, shell commands, and external API calls will be auto-approved without confirmation.\n"
        )
        warning.append(
            "This is intended for testing and trusted environments only. Use at your own risk.\n",
            style="yellow",
        )
        app_context.console.print(Panel(warning, style="bold red"))
    else:
        app_context.session.yolo_mode = False
        app_context.session.confirm_edits = True

    target_repo_path = repo_path or settings.TARGET_REPO_PATH
    # Validate repo path exists
    if not Path(target_repo_path).exists():
        typer.echo(
            f"ERROR: Repository path '{target_repo_path}' does not exist.",
            err=True,
        )
        raise typer.Exit(1)

    _update_and_validate_models(orchestrator, cypher)

    try:
        asyncio.run(
            main_optimize_async(
                language,
                target_repo_path,
                reference_document,
                orchestrator,
                cypher,
                batch_size,
            )
        )
    except KeyboardInterrupt:
        app_context.console.print(style(cs.CLI_MSG_APP_TERMINATED, cs.Color.RED))
    except ValueError as e:
        app_context.console.print(
            style(cs.CLI_ERR_STARTUP.format(error=e), cs.Color.RED)
        )


@app.command(name=ch.CLICommandName.MCP_SERVER, help=ch.CMD_MCP_SERVER)
def mcp_server(
    transport: cs.MCPTransport = typer.Option(
        cs.MCPTransport.STDIO, help=ch.HELP_MCP_TRANSPORT
    ),
    host: str = typer.Option(None, help=ch.HELP_MCP_HTTP_HOST),
    port: int = typer.Option(None, help=ch.HELP_MCP_HTTP_PORT),
) -> None:
    try:
        if transport == cs.MCPTransport.HTTP:
            from codebase_rag.mcp import serve_http

            resolved_host = host or settings.MCP_HTTP_HOST
            resolved_port = port or settings.MCP_HTTP_PORT
            asyncio.run(serve_http(host=resolved_host, port=resolved_port))
        else:
            from codebase_rag.mcp import serve_stdio

            asyncio.run(serve_stdio())
    except KeyboardInterrupt:
        app_context.console.print(style(cs.CLI_MSG_APP_TERMINATED, cs.Color.RED))
    except ValueError as e:
        app_context.console.print(
            style(cs.CLI_ERR_CONFIG.format(error=e), cs.Color.RED)
        )
        _info(style(cs.CLI_MSG_HINT_TARGET_REPO, cs.Color.YELLOW))
    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_MCP_SERVER.format(error=e), cs.Color.RED)
        )


@app.command(name=ch.CLICommandName.GRAPH_LOADER, help=ch.CMD_GRAPH_LOADER)
def graph_loader_command(
    graph_file: str = typer.Argument(..., help=ch.HELP_GRAPH_FILE),
) -> None:
    from .graph_loader import load_graph

    try:
        graph = load_graph(graph_file)
        summary = graph.summary()

        app_context.console.print(style(cs.CLI_MSG_GRAPH_SUMMARY, cs.Color.GREEN))
        app_context.console.print(f"  Total nodes: {summary['total_nodes']}")
        app_context.console.print(
            f"  Total relationships: {summary['total_relationships']}"
        )
        app_context.console.print(
            f"  Node types: {list(summary['node_labels'].keys())}"
        )
        app_context.console.print(
            f"  Relationship types: {list(summary['relationship_types'].keys())}"
        )
        app_context.console.print(
            f"  Exported at: {summary['metadata']['exported_at']}"
        )

    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_LOAD_GRAPH.format(error=e), cs.Color.RED)
        )
        raise typer.Exit(1) from e


@app.command(
    name=ch.CLICommandName.LANGUAGE,
    help=ch.CMD_LANGUAGE,
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def language_command(ctx: typer.Context) -> None:
    language_cli(ctx.args, standalone_mode=False)


@app.command(name=ch.CLICommandName.DOCTOR, help=ch.CMD_DOCTOR)
def doctor() -> None:
    checker = HealthChecker()
    results = checker.run_all_checks()

    passed, total = checker.get_summary()

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="cyan", no_wrap=False)

    for result in results:
        status = "✓" if result.passed else "✗"
        status_color = cs.Color.GREEN if result.passed else cs.Color.RED
        status_text = style(status, status_color, cs.StyleModifier.NONE)

        check_name = f"{status_text} {result.name}"
        table.add_row(check_name)

    panel = Panel(
        table,
        title="Health Check",
        border_style="dim",
        padding=(1, 2),
    )

    app_context.console.print(panel)

    app_context.console.print()
    summary_text = f"{passed}/{total} checks passed"
    if passed == total:
        app_context.console.print(style(summary_text, cs.Color.GREEN))
    else:
        app_context.console.print(style(summary_text, cs.Color.YELLOW))

    failed_checks = [r for r in results if not r.passed and r.error]
    if failed_checks:
        app_context.console.print()
        app_context.console.print(style("Failed checks details:", cs.Color.YELLOW))
        for result in failed_checks:
            error_msg = f"  {result.name}: {result.error}"
            app_context.console.print(
                style(error_msg, cs.Color.YELLOW, cs.StyleModifier.NONE)
            )

    if passed < total:
        raise typer.Exit(1)


@app.command(name=ch.CLICommandName.MIGRATE_DATA, help=ch.CMD_MIGRATE_DATA)
def migrate_data(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Report changes without applying them. Use --no-dry-run to execute.",
    ),
) -> None:
    """Run data model migrations for existing graph data.

    Defaults to dry-run mode for safety. Pass --no-dry-run to apply changes.
    """
    from .migrations.data_model_migrations import run_migrations

    results = run_migrations(dry_run=dry_run)

    if dry_run:
        app_context.console.print("[yellow]Dry run mode - no changes made[/yellow]")

    app_context.console.print("\nMigration Results:")
    for name, count in results.items():
        status = "would migrate" if dry_run else "migrated"
        app_context.console.print(f"  {name}: {count} nodes {status}")


def _build_stats_table(
    title: str,
    col_label: str,
    rows: list[ResultRow],
    get_label: Callable[[ResultRow], str],
    total_label: str,
) -> Table:
    table = Table(
        title=style(title, cs.Color.GREEN),
        show_header=True,
        header_style=f"{cs.StyleModifier.BOLD} {cs.Color.MAGENTA}",
    )
    table.add_column(col_label, style=cs.Color.CYAN)
    table.add_column(cs.CLI_STATS_COL_COUNT, style=cs.Color.YELLOW, justify="right")
    total = 0
    for row in rows:
        raw_count = row.get("count", 0)
        count = int(raw_count) if isinstance(raw_count, (int, float)) else 0
        total += count
        table.add_row(get_label(row), f"{count:,}")
    table.add_section()
    table.add_row(
        style(total_label, cs.Color.GREEN),
        style(f"{total:,}", cs.Color.GREEN),
    )
    return table


@app.command(name=ch.CLICommandName.STATS, help=ch.CMD_STATS)
def stats() -> None:
    from .cypher_queries import (
        CYPHER_STATS_NODE_COUNTS,
        CYPHER_STATS_RELATIONSHIP_COUNTS,
    )

    app_context.console.print(style(cs.CLI_MSG_CONNECTING_STATS, cs.Color.CYAN))

    try:
        with connect_memgraph(batch_size=1) as ingestor:
            node_results = ingestor.fetch_all(CYPHER_STATS_NODE_COUNTS)
            rel_results = ingestor.fetch_all(CYPHER_STATS_RELATIONSHIP_COUNTS)

            app_context.console.print(
                _build_stats_table(
                    cs.CLI_STATS_NODE_TITLE,
                    cs.CLI_STATS_COL_NODE_TYPE,
                    node_results,
                    lambda r: ":".join(r.get("labels", [])) or cs.CLI_STATS_UNKNOWN,
                    cs.CLI_STATS_TOTAL_NODES,
                )
            )
            app_context.console.print()
            app_context.console.print(
                _build_stats_table(
                    cs.CLI_STATS_REL_TITLE,
                    cs.CLI_STATS_COL_REL_TYPE,
                    rel_results,
                    lambda r: str(r.get("type", cs.CLI_STATS_UNKNOWN)),
                    cs.CLI_STATS_TOTAL_RELS,
                )
            )

    except Exception as e:
        app_context.console.print(
            style(cs.CLI_ERR_STATS_FAILED.format(error=e), cs.Color.RED)
        )
        logger.exception(ls.STATS_ERROR.format(error=e))
        raise typer.Exit(1) from e


@app.command(name=ch.CLICommandName.QUOTA, help=ch.CMD_QUOTA)
def quota() -> None:
    """Display LLM provider quota status and usage information."""
    from .quota_status_reporter import QuotaStatusReporter

    reporter = QuotaStatusReporter(app_context.console)
    reporter.print_status(settings)


# Document GraphRAG CLI commands


@app.command(name=ch.CLICommandName.QUERY_DOCS, help=ch.CMD_QUERY_DOCS)
def query_docs(
    repo_path: str | None = typer.Option(
        None, "-r", "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
    query: str = typer.Argument(..., help=ch.HELP_QUERY),
    top_k: int = typer.Option(5, "--top-k", "-k", help=ch.HELP_TOP_K),
) -> None:
    """Query the document graph using natural language."""
    from dataclasses import asdict

    from .services.graph_service import MemgraphIngestor
    from .shared.query_router import QueryMode, QueryRequest, QueryRouter

    _info(style(f"Querying document graph: {query}", cs.Color.CYAN))

    try:
        with MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
        ) as doc_graph:
            query_router = QueryRouter(doc_graph=doc_graph)

            request = QueryRequest(
                question=query,
                mode=QueryMode.DOCUMENT_ONLY,
                top_k=top_k,
            )
            response = query_router.query(request)

            result = asdict(response)
            app_context.console.print(
                Panel(
                    result.get("answer", "No results found"),
                    title=f"Document Query (Mode: {response.mode.value})",
                    border_style="cyan",
                )
            )

            if result.get("sources"):
                app_context.console.print(
                    f"\n[bold]Sources:[/bold] {len(result['sources'])} document(s)"
                )

            if result.get("warnings"):
                app_context.console.print(
                    f"[yellow]Warnings:[/yellow] {', '.join(result['warnings'])}"
                )

    except Exception as e:
        app_context.console.print(style(f"Query failed: {e}", cs.Color.RED))
        raise typer.Exit(1) from e


@app.command(name=ch.CLICommandName.QUERY_ALL, help=ch.CMD_QUERY_ALL)
def query_all(
    repo_path: str | None = typer.Option(
        None, "-r", "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
    query: str = typer.Argument(..., help=ch.HELP_QUERY),
    top_k: int = typer.Option(5, "--top-k", "-k", help=ch.HELP_TOP_K),
) -> None:
    """Query both code and document graphs, merge results."""
    from dataclasses import asdict

    from .services.graph_service import MemgraphIngestor
    from .shared.query_router import QueryMode, QueryRequest, QueryRouter

    _info(style(f"Querying both graphs: {query}", cs.Color.CYAN))

    try:
        # Connect to both graphs
        with (
            MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            ) as code_graph,
            MemgraphIngestor(
                host=settings.DOC_MEMGRAPH_HOST,
                port=settings.DOC_MEMGRAPH_PORT,
            ) as doc_graph,
        ):
            query_router = QueryRouter(code_graph=code_graph, doc_graph=doc_graph)

            request = QueryRequest(
                question=query,
                mode=QueryMode.BOTH_MERGED,
                top_k=top_k,
            )
            response = query_router.query(request)

            result = asdict(response)
            app_context.console.print(
                Panel(
                    result.get("answer", "No results found"),
                    title=f"Merged Query (Mode: {response.mode.value})",
                    border_style="green",
                )
            )

            # Show source breakdown
            code_sources = [s for s in response.sources if s.type == "code"]
            doc_sources = [s for s in response.sources if s.type == "document"]

            if code_sources:
                app_context.console.print(
                    f"\n[bold cyan]Code Sources:[/bold cyan] {len(code_sources)}"
                )
            if doc_sources:
                app_context.console.print(
                    f"[bold magenta]Document Sources:[/bold magenta] {len(doc_sources)}"
                )

            if result.get("warnings"):
                app_context.console.print(
                    f"[yellow]Warnings:[/yellow] {', '.join(result['warnings'])}"
                )

    except Exception as e:
        app_context.console.print(style(f"Query failed: {e}", cs.Color.RED))
        raise typer.Exit(1) from e


@app.command(name=ch.CLICommandName.VALIDATE_SPEC, help=ch.CMD_VALIDATE_SPEC)
def validate_spec(
    repo_path: str | None = typer.Option(
        None, "-r", "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
    spec_path: str = typer.Argument(..., help=ch.HELP_SPEC_PATH),
    scope: str = typer.Option("all", "--scope", "-s", help=ch.HELP_SCOPE),
    max_cost: float = typer.Option(0.50, "--max-cost", "-c", help=ch.HELP_MAX_COST),
    dry_run: bool = typer.Option(False, "--dry-run", help=ch.HELP_DRY_RUN),
) -> None:
    """Validate code against a specification document."""
    from dataclasses import asdict

    from .services.graph_service import MemgraphIngestor
    from .shared.query_router import QueryMode, QueryRequest, QueryRouter
    from .shared.validation.api import ValidationRequest, ValidationTriggerAPI

    _info(style(f"Validating code against spec: {spec_path}", cs.Color.CYAN))

    try:
        # Connect to both graphs
        with (
            MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            ) as code_graph,
            MemgraphIngestor(
                host=settings.DOC_MEMGRAPH_HOST,
                port=settings.DOC_MEMGRAPH_PORT,
            ) as doc_graph,
        ):
            query_router = QueryRouter(code_graph=code_graph, doc_graph=doc_graph)
            validation_api = ValidationTriggerAPI(llm_provider="google")
            normalized_scope = _normalize_validation_scope(scope)

            # Step 1: Cost estimation
            validation_request = ValidationRequest(
                document_path=spec_path,
                mode="CODE_VS_DOC",
                scope=normalized_scope,
                max_cost_usd=max_cost,
                dry_run=dry_run,
            )

            trigger_result = asyncio.run(
                validation_api.request_validation(validation_request)
            )

            if not trigger_result.accepted:
                app_context.console.print(
                    Panel(
                        trigger_result.message,
                        title="Validation Not Executed",
                        border_style="yellow",
                    )
                )
                if trigger_result.cost_estimate:
                    app_context.console.print(
                        f"\n[yellow]Estimated cost:[/yellow] ${trigger_result.cost_estimate.estimated_cost_usd:.2f}"
                    )
                return

            # Show cost estimate
            if trigger_result.cost_estimate:
                app_context.console.print(
                    f"[green]Estimated cost:[/green] ${trigger_result.cost_estimate.estimated_cost_usd:.2f} "
                    f"({trigger_result.cost_estimate.estimated_llm_calls} LLM calls)"
                )

            # Step 2: Execute validation
            request = QueryRequest(
                question=f"Validate code against {spec_path}",
                mode=QueryMode.CODE_VS_DOC,
                scope=normalized_scope,
            )
            response = query_router.query(request)

            result = asdict(response)
            report = result.get("validation_report")

            if report:
                border_style = (
                    "green" if report.get("accuracy_score", 0) > 0.8 else "yellow"
                )
                app_context.console.print(
                    Panel(
                        result.get("answer", "Validation complete"),
                        title=(
                            f"Validation Report: {report.get('passed', 0)}/{report.get('total', 0)} "
                            f"({report.get('accuracy_score', 0):.1%} accurate)"
                        ),
                        border_style=border_style,
                    )
                )
            else:
                app_context.console.print(
                    Panel(
                        result.get("answer", "Validation complete"),
                        title="Validation Report",
                        border_style="cyan",
                    )
                )

            if result.get("warnings"):
                app_context.console.print(
                    f"[yellow]Warnings:[/yellow] {', '.join(result['warnings'])}"
                )

    except Exception as e:
        app_context.console.print(style(f"Validation failed: {e}", cs.Color.RED))
        raise typer.Exit(1) from e


@app.command(name=ch.CLICommandName.VALIDATE_DOC, help=ch.CMD_VALIDATE_DOC)
def validate_doc(
    repo_path: str | None = typer.Option(
        None, "-r", "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
    doc_path: str = typer.Argument(..., help=ch.HELP_DOC_PATH),
    scope: str = typer.Option("all", "--scope", "-s", help=ch.HELP_SCOPE),
    max_cost: float = typer.Option(0.50, "--max-cost", "-c", help=ch.HELP_MAX_COST),
    dry_run: bool = typer.Option(False, "--dry-run", help=ch.HELP_DRY_RUN),
) -> None:
    """Validate documentation against actual code."""
    from dataclasses import asdict

    from .services.graph_service import MemgraphIngestor
    from .shared.query_router import QueryMode, QueryRequest, QueryRouter
    from .shared.validation.api import ValidationRequest, ValidationTriggerAPI

    _info(style(f"Validating doc against code: {doc_path}", cs.Color.CYAN))

    try:
        # Connect to both graphs
        with (
            MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
            ) as code_graph,
            MemgraphIngestor(
                host=settings.DOC_MEMGRAPH_HOST,
                port=settings.DOC_MEMGRAPH_PORT,
            ) as doc_graph,
        ):
            query_router = QueryRouter(code_graph=code_graph, doc_graph=doc_graph)
            validation_api = ValidationTriggerAPI(llm_provider="google")
            normalized_scope = _normalize_validation_scope(scope)

            # Step 1: Cost estimation
            validation_request = ValidationRequest(
                document_path=doc_path,
                mode="DOC_VS_CODE",
                scope=normalized_scope,
                max_cost_usd=max_cost,
                dry_run=dry_run,
            )

            trigger_result = asyncio.run(
                validation_api.request_validation(validation_request)
            )

            if not trigger_result.accepted:
                app_context.console.print(
                    Panel(
                        trigger_result.message,
                        title="Validation Not Executed",
                        border_style="yellow",
                    )
                )
                if trigger_result.cost_estimate:
                    app_context.console.print(
                        f"\n[yellow]Estimated cost:[/yellow] ${trigger_result.cost_estimate.estimated_cost_usd:.2f}"
                    )
                return

            # Show cost estimate
            if trigger_result.cost_estimate:
                app_context.console.print(
                    f"[green]Estimated cost:[/green] ${trigger_result.cost_estimate.estimated_cost_usd:.2f} "
                    f"({trigger_result.cost_estimate.estimated_llm_calls} LLM calls)"
                )

            # Step 2: Execute validation
            request = QueryRequest(
                question=f"Validate {doc_path} against code",
                mode=QueryMode.DOC_VS_CODE,
                scope=normalized_scope,
            )
            response = query_router.query(request)

            result = asdict(response)
            report = result.get("validation_report")

            if report:
                border_style = (
                    "green" if report.get("accuracy_score", 0) > 0.8 else "yellow"
                )
                app_context.console.print(
                    Panel(
                        result.get("answer", "Validation complete"),
                        title=(
                            f"Validation Report: {report.get('passed', 0)}/{report.get('total', 0)} "
                            f"({report.get('accuracy_score', 0):.1%} accurate)"
                        ),
                        border_style=border_style,
                    )
                )
            else:
                app_context.console.print(
                    Panel(
                        result.get("answer", "Validation complete"),
                        title="Validation Report",
                        border_style="cyan",
                    )
                )

            if result.get("warnings"):
                app_context.console.print(
                    f"[yellow]Warnings:[/yellow] {', '.join(result['warnings'])}"
                )

    except Exception as e:
        app_context.console.print(style(f"Validation failed: {e}", cs.Color.RED))
        raise typer.Exit(1) from e


@app.command(name=ch.CLICommandName.INDEX_DOCS, help=ch.CMD_INDEX_DOCS)
def index_docs(
    repo_path: str | None = typer.Option(
        None, "-r", "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
    clean: bool = typer.Option(
        False,
        "--clean",
        help=ch.HELP_CLEAN_DOC_DB,
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force re-indexing (ignore version cache)"
    ),
) -> None:
    """Index documents into the document graph."""
    from .document.document_updater import DocumentGraphUpdater
    from .services.graph_service import MemgraphIngestor

    target_repo_path = repo_path or settings.TARGET_REPO_PATH
    # Validate repo path exists
    if not Path(target_repo_path).exists():
        typer.echo(
            f"ERROR: Repository path '{target_repo_path}' does not exist.",
            err=True,
        )
        raise typer.Exit(1)
    repo_to_index = Path(target_repo_path)

    _info(style(f"Indexing documents in: {repo_to_index}", cs.Color.CYAN))

    try:
        with MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
        ) as ingestor:
            if clean:
                _info(style("Cleaning document database...", cs.Color.YELLOW))
                ingestor.clean_database()
                _info(style("Document database cleaned.", cs.Color.GREEN))

        updater = DocumentGraphUpdater(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
            repo_path=repo_to_index,
        )
        stats = updater.run(force=force)

        table = Table(
            title=style("Document Indexing Results", cs.Color.GREEN),
            show_header=True,
            header_style=f"{cs.StyleModifier.BOLD} {cs.Color.MAGENTA}",
        )
        table.add_column("Metric", style=cs.Color.CYAN)
        table.add_column("Count", style=cs.Color.YELLOW, justify="right")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        app_context.console.print(table)

    except Exception as e:
        app_context.console.print(style(f"Document indexing failed: {e}", cs.Color.RED))
        logger.exception("Document indexing failed")
        raise typer.Exit(1) from e


@app.command(name=ch.CLICommandName.CLEAN_DOCS, help=ch.CMD_CLEAN_DOCS)
def clean_docs(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be removed without actually removing.",
    ),
    workspace: str = typer.Option(
        "default",
        "--workspace",
        help="Workspace to clean (default: 'default').",
    ),
    repo_path: str | None = typer.Option(
        None, "-r", "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
) -> None:
    """Remove documents from graph that match .cgrignore patterns.

    This command is useful when you've updated .cgrignore and want to
    remove previously indexed documents that should now be excluded.

    Examples:
        # Preview what would be removed
        cgr clean-docs --dry-run

        # Remove excluded documents
        cgr clean-docs
    """
    from .document.document_updater import DocumentGraphUpdater

    target_repo_path = repo_path or settings.TARGET_REPO_PATH
    repo_to_clean = Path(target_repo_path)

    if not repo_to_clean.exists():
        typer.echo(
            f"ERROR: Repository path '{target_repo_path}' does not exist.",
            err=True,
        )
        raise typer.Exit(1)

    updater = DocumentGraphUpdater(
        host=settings.DOC_MEMGRAPH_HOST,
        port=settings.DOC_MEMGRAPH_PORT,
        repo_path=repo_to_clean,
        workspace=workspace,
    )

    if dry_run:
        # Preview mode
        excluded = updater.preview_excluded_documents()
        if not excluded:
            _info("No documents would be removed.")
            return

        table = Table(title=style("Documents that would be removed", cs.Color.YELLOW))
        table.add_column("Path", style=cs.Color.CYAN)
        table.add_column("Reason", style=cs.Color.YELLOW)

        for doc_path, reason in excluded:
            table.add_row(str(doc_path), reason or "Excluded by .cgrignore")

        app_context.console.print(table)
        _info(f"Total: {len(excluded)} documents would be removed.")
    else:
        # Actual cleanup
        removed = updater.cleanup_excluded_documents()
        _success(f"Removed {removed} excluded documents from graph.")


@app.command(
    name=ch.CLICommandName.INGEST_JSON,
    help=ch.CMD_INGEST_JSON,
)
def ingest_json(
    input_path: str = typer.Argument(
        ..., help="Path to JSON file or directory containing JSON files."
    ),
    dataset_id: str | None = typer.Option(
        None,
        "--dataset-id",
        help="Override dataset ID (defaults to value in JSON metadata).",
    ),
    skip_existing: bool = typer.Option(
        False,
        "--skip-existing",
        help="Skip entities/relationships that already exist in the graph.",
    ),
    batch_size: int = typer.Option(
        100, "--batch-size", min=1, help="Batch size for database operations."
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Run incremental update, only process changed entities/relationships.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate input and calculate changes without writing to databases.",
    ),
    conflict_resolution: str = typer.Option(
        "last-write-wins",
        "--conflict-resolution",
        help="Conflict resolution strategy: last-write-wins (default), highest-confidence-wins, manual-review.",
    ),
    json_filter: str = typer.Option(
        "lenient",
        "--json-filter",
        help="JSON file filtering preset: lenient (default), strict (only .cgr.json files), none (no filtering).",
    ),
    exclude: list[str] | None = typer.Option(
        None,
        "--exclude",
        help="Patterns of JSON files to exclude from ingestion (supports glob patterns).",
    ),
) -> None:
    from .json_ingestion import ingest_json_data

    _info(style(f"Ingesting JSON data from: {input_path}", cs.Color.CYAN))
    if dataset_id:
        _info(style(f"Using dataset ID: {dataset_id}", cs.Color.CYAN))

    try:
        result = ingest_json_data(
            input_path=input_path,
            dataset_id=dataset_id,
            skip_existing=skip_existing,
            batch_size=batch_size,
            incremental=incremental,
            dry_run=dry_run,
            conflict_resolution=conflict_resolution,
            exclude_patterns=exclude,
            filter_preset=json_filter,
        )

        # Display results
        table = Table(
            title=style(
                f"Ingestion Results {'(Dry Run)' if dry_run else ''}", cs.Color.GREEN
            ),
            show_header=True,
            header_style=f"{cs.StyleModifier.BOLD} {cs.Color.MAGENTA}",
        )
        table.add_column("Metric", style=cs.Color.CYAN)
        table.add_column("Count", style=cs.Color.YELLOW, justify="right")

        table.add_row("Entities Processed", str(result.entities_processed))
        table.add_row("Entities Ingested", str(result.entities_ingested))
        table.add_row("Entities Updated", str(result.entities_updated))
        table.add_row("Entities Skipped", str(result.entities_skipped))
        table.add_row("Entities Failed", str(result.entities_failed))
        table.add_section()
        table.add_row("Relationships Processed", str(result.relationships_processed))
        table.add_row("Relationships Ingested", str(result.relationships_ingested))
        table.add_row("Relationships Updated", str(result.relationships_updated))
        table.add_row("Relationships Skipped", str(result.relationships_skipped))
        table.add_row("Relationships Failed", str(result.relationships_failed))

        app_context.console.print(table)

        if result.errors:
            app_context.console.print()
            app_context.console.print(
                style(
                    f"Ingestion completed with {len(result.errors)} errors:",
                    cs.Color.YELLOW,
                )
            )
            for error in result.errors[:10]:  # Show up to 10 errors
                app_context.console.print(style(f"  - {error}", cs.Color.RED))
            if len(result.errors) > 10:
                app_context.console.print(
                    style(
                        f"  ... and {len(result.errors) - 10} more errors", cs.Color.RED
                    )
                )

    except Exception as e:
        app_context.console.print(style(f"Ingestion failed: {e}", cs.Color.RED))
        logger.exception("JSON ingestion failed")
        raise typer.Exit(1) from e


@app.command(
    name=ch.CLICommandName.DELETE_DATASET,
    help=ch.CMD_DELETE_DATASET,
)
def delete_dataset_command(
    dataset_id: str = typer.Argument(..., help="ID of the dataset to delete."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be deleted without making changes."
    ),
) -> None:
    from .json_ingestion import delete_dataset

    _info(
        style(
            f"Deleting dataset: {dataset_id}{' (Dry Run)' if dry_run else ''}",
            cs.Color.YELLOW,
        )
    )

    try:
        success, nodes_deleted, rels_deleted, errors = delete_dataset(
            dataset_id, dry_run=dry_run
        )

        if success:
            if dry_run:
                _info(
                    style(
                        f"Dry run: Would delete {nodes_deleted} nodes and {rels_deleted} relationships for dataset {dataset_id}",
                        cs.Color.GREEN,
                    )
                )
            else:
                _info(
                    style(
                        f"Successfully deleted dataset {dataset_id}: {nodes_deleted} nodes, {rels_deleted} relationships",
                        cs.Color.GREEN,
                    )
                )
        else:
            app_context.console.print(
                style(f"Failed to delete dataset {dataset_id}", cs.Color.RED)
            )
            for error in errors:
                app_context.console.print(style(f"  - {error}", cs.Color.RED))
            raise typer.Exit(1)

    except Exception as e:
        app_context.console.print(style(f"Delete dataset failed: {e}", cs.Color.RED))
        logger.exception("Delete dataset failed")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
