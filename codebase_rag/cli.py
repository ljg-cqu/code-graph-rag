import asyncio
from collections.abc import Callable
from importlib.metadata import version as get_version
from pathlib import Path

import typer
from loguru import logger
from rich.panel import Panel
from rich.table import Table

from . import cli_help as ch
from . import constants as cs
from . import logs as ls
from .config import load_cgrignore_patterns, settings
from .graph_updater import GraphUpdater
from .main import (
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

app = typer.Typer(
    name=cs.PACKAGE_NAME,
    help=ch.APP_DESCRIPTION,
    no_args_is_help=True,
    add_completion=False,
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


def _handle_indexing(
    repo_path: Path,
    update_graph: bool,
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
    index_timeout: int = 300,
) -> tuple[bool, bool, bool]:
    """Handle code and document indexing before chat.

    Args:
        repo_path: Repository path
        update_graph: --update-graph flag
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

    Returns:
        Tuple of (code_indexed, docs_indexed, effective_with_docs)

    Raises:
        typer.Exit: If code indexing fails with --update-graph flag (blocking)
        Note: Document indexing failures are non-blocking, return docs_indexed=False
    """
    code_indexed = False
    docs_indexed = False
    effective_with_docs = with_docs

    # Resolve effective flags
    effective_update_graph = update_graph or index_all
    effective_index_docs = index_docs or index_all

    # Implied --with-docs if indexing docs
    if effective_index_docs:
        effective_with_docs = True

    # === Code Indexing ===
    if effective_update_graph:
        _info(style(cs.CLI_MSG_UPDATING_GRAPH.format(path=repo_path), cs.Color.GREEN))

        cgrignore = load_cgrignore_patterns(repo_path)
        cli_excludes = frozenset(exclude) if exclude else frozenset()
        exclude_paths = cli_excludes | cgrignore.exclude or None
        unignore_paths: frozenset[str] | None = None

        if interactive_setup:
            unignore_paths = prompt_for_unignored_directories(repo_path, exclude)
        else:
            _info(style(cs.CLI_MSG_AUTO_EXCLUDE, cs.Color.YELLOW))
            unignore_paths = cgrignore.unignore or None

        with connect_memgraph(batch_size) as ingestor:
            if clean:
                _info(style(cs.CLI_MSG_CLEANING_DB, cs.Color.YELLOW))
                ingestor.clean_database()
                _delete_hash_cache(repo_path)

            ingestor.ensure_constraints()
            parsers, queries = load_parsers()

            updater = GraphUpdater(
                ingestor=ingestor,
                repo_path=str(repo_path),
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
            # Preserve user's explicit --with-docs flag; only disable if indexing was the only reason
            effective_with_docs = with_docs

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
) -> None:
    settings.QUIET = quiet
    if quiet:
        logger.remove()
        logger.add(lambda msg: app_context.console.print(msg, end=""), level="ERROR")


def _info(msg: str) -> None:
    if not settings.QUIET:
        app_context.console.print(msg)


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
        None, "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
    update_graph: bool = typer.Option(
        False,
        "--update-graph",
        help=ch.HELP_UPDATE_GRAPH,
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
        "code_only",
        "--mode",
        help=ch.HELP_MODE,
    ),
    index_timeout: int = typer.Option(
        300,
        "--index-timeout",
        help=ch.HELP_INDEX_TIMEOUT,
    ),
) -> None:
    import re

    from codebase_rag.shared.query_router import QueryMode

    app_context.session.confirm_edits = not no_confirm

    # === CLI Flag Validation ===
    # Calculate effective_with_docs
    effective_with_docs = with_docs or index_docs or index_all

    # Mode validation: non-code_only modes require --with-docs
    if mode != "code_only" and not effective_with_docs:
        typer.echo(
            f"ERROR: Mode '{mode}' requires document graph. "
            f"Add --with-docs, --index-docs, or --index-all flag.",
            err=True,
        )
        raise typer.Exit(1)

    # Parse and validate mode
    try:
        query_mode = QueryMode(mode.lower())
    except ValueError:
        typer.echo(
            f"ERROR: Invalid mode '{mode}'. "
            f"Valid modes: code_only, document_only, both_merged, code_vs_doc, doc_vs_code",
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

    target_repo_path = repo_path or settings.TARGET_REPO_PATH

    # --output requires --update-graph or --index-all (which triggers code update)
    if output and not (update_graph or index_all):
        app_context.console.print(
            style(cs.CLI_ERR_OUTPUT_REQUIRES_UPDATE, cs.Color.RED)
        )
        raise typer.Exit(1)

    # --clean requires --update-graph, --index-docs, or --index-all
    if clean and not (update_graph or index_docs or index_all):
        typer.echo(
            "ERROR: --clean requires --update-graph, --index-docs, or --index-all.",
            err=True,
        )
        raise typer.Exit(1)

    effective_batch_size = settings.resolve_batch_size(batch_size)

    _update_and_validate_models(orchestrator, cypher)

    # === Handle indexing with new flags ===
    code_indexed, docs_indexed, effective_with_docs = _handle_indexing(
        repo_path=Path(target_repo_path),
        update_graph=update_graph,
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
                    from codebase_rag.document.document_updater import DocumentGraphUpdater

                    updater = DocumentGraphUpdater(
                        host=settings.DOC_MEMGRAPH_HOST,
                        port=settings.DOC_MEMGRAPH_PORT,
                        repo_path=repo_to_check,
                        workspace=doc_workspace,
                    )
                    stats = updater.run(force=False)
                    _info(style(f"Documents indexed: {stats}", cs.Color.GREEN))
                    effective_with_docs = True
                except Exception as e:
                    _info(style(f"Document indexing failed: {e}", cs.Color.RED))
                    effective_with_docs = False

    # === Start chat session ===
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
                )
            )
        else:
            asyncio.run(main_async(target_repo_path, effective_batch_size))
    except KeyboardInterrupt:
        app_context.console.print(style(cs.CLI_MSG_APP_TERMINATED, cs.Color.RED))
    except ValueError as e:
        app_context.console.print(
            style(cs.CLI_ERR_STARTUP.format(error=e), cs.Color.RED)
        )


@app.command(help=ch.CMD_INDEX)
def index(
    repo_path: str | None = typer.Option(
        None, "--repo-path", help=ch.HELP_REPO_PATH_INDEX
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
    repo_to_index = Path(target_repo_path)
    _info(style(cs.CLI_MSG_INDEXING_AT.format(path=repo_to_index), cs.Color.GREEN))

    _info(style(cs.CLI_MSG_OUTPUT_TO.format(path=output_proto_dir), cs.Color.CYAN))

    cgrignore = load_cgrignore_patterns(repo_to_index)
    cli_excludes = frozenset(exclude) if exclude else frozenset()
    exclude_paths = cli_excludes | cgrignore.exclude or None
    unignore_paths: frozenset[str] | None = None
    if interactive_setup:
        unignore_paths = prompt_for_unignored_directories(repo_to_index, exclude)
    else:
        _info(style(cs.CLI_MSG_AUTO_EXCLUDE, cs.Color.YELLOW))
        unignore_paths = cgrignore.unignore or None

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
        None, "--repo-path", help=ch.HELP_REPO_PATH_OPTIMIZE
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
    batch_size: int | None = typer.Option(
        None,
        "--batch-size",
        min=1,
        help=ch.HELP_BATCH_SIZE,
    ),
) -> None:
    app_context.session.confirm_edits = not no_confirm

    target_repo_path = repo_path or settings.TARGET_REPO_PATH

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


# Document GraphRAG CLI commands

@app.command(name=ch.CLICommandName.QUERY_DOCS, help=ch.CMD_QUERY_DOCS)
def query_docs(
    query: str = typer.Argument(..., help=ch.HELP_QUERY),
    top_k: int = typer.Option(5, "--top-k", "-k", help=ch.HELP_TOP_K),
) -> None:
    """Query the document graph using natural language."""
    from .shared.query_router import QueryRequest, QueryMode, QueryRouter
    from .services.graph_service import MemgraphIngestor
    from dataclasses import asdict

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
    query: str = typer.Argument(..., help=ch.HELP_QUERY),
    top_k: int = typer.Option(5, "--top-k", "-k", help=ch.HELP_TOP_K),
) -> None:
    """Query both code and document graphs, merge results."""
    from .shared.query_router import QueryRequest, QueryMode, QueryRouter
    from .services.graph_service import MemgraphIngestor
    from dataclasses import asdict

    _info(style(f"Querying both graphs: {query}", cs.Color.CYAN))

    try:
        # Connect to both graphs
        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
        ) as code_graph, MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
        ) as doc_graph:
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
    spec_path: str = typer.Argument(..., help=ch.HELP_SPEC_PATH),
    scope: str = typer.Option("all", "--scope", "-s", help=ch.HELP_SCOPE),
    max_cost: float = typer.Option(0.50, "--max-cost", "-c", help=ch.HELP_MAX_COST),
    dry_run: bool = typer.Option(False, "--dry-run", help=ch.HELP_DRY_RUN),
) -> None:
    """Validate code against a specification document."""
    from .shared.query_router import QueryRequest, QueryMode, QueryRouter
    from .shared.validation.api import ValidationTriggerAPI, ValidationRequest
    from .services.graph_service import MemgraphIngestor
    from dataclasses import asdict

    _info(style(f"Validating code against spec: {spec_path}", cs.Color.CYAN))

    try:
        # Connect to both graphs
        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
        ) as code_graph, MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
        ) as doc_graph:
            query_router = QueryRouter(code_graph=code_graph, doc_graph=doc_graph)
            validation_api = ValidationTriggerAPI(llm_provider="google")

            # Step 1: Cost estimation
            validation_request = ValidationRequest(
                document_path=spec_path,
                mode="CODE_VS_DOC",
                scope=scope,
                max_cost_usd=max_cost,
                dry_run=dry_run,
            )

            trigger_result = asyncio.run(validation_api.request_validation(validation_request))

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
                scope=scope,
            )
            response = query_router.query(request)

            result = asdict(response)
            report = result.get("validation_report")

            if report:
                border_style = "green" if report.get("accuracy_score", 0) > 0.8 else "yellow"
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
    doc_path: str = typer.Argument(..., help=ch.HELP_DOC_PATH),
    scope: str = typer.Option("all", "--scope", "-s", help=ch.HELP_SCOPE),
    max_cost: float = typer.Option(0.50, "--max-cost", "-c", help=ch.HELP_MAX_COST),
    dry_run: bool = typer.Option(False, "--dry-run", help=ch.HELP_DRY_RUN),
) -> None:
    """Validate documentation against actual code."""
    from .shared.query_router import QueryRequest, QueryMode, QueryRouter
    from .shared.validation.api import ValidationTriggerAPI, ValidationRequest
    from .services.graph_service import MemgraphIngestor
    from dataclasses import asdict

    _info(style(f"Validating doc against code: {doc_path}", cs.Color.CYAN))

    try:
        # Connect to both graphs
        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
        ) as code_graph, MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
        ) as doc_graph:
            query_router = QueryRouter(code_graph=code_graph, doc_graph=doc_graph)
            validation_api = ValidationTriggerAPI(llm_provider="google")

            # Step 1: Cost estimation
            validation_request = ValidationRequest(
                document_path=doc_path,
                mode="DOC_VS_CODE",
                scope=scope,
                max_cost_usd=max_cost,
                dry_run=dry_run,
            )

            trigger_result = asyncio.run(validation_api.request_validation(validation_request))

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
                scope=scope,
            )
            response = query_router.query(request)

            result = asdict(response)
            report = result.get("validation_report")

            if report:
                border_style = "green" if report.get("accuracy_score", 0) > 0.8 else "yellow"
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
        None, "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
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
        app_context.console.print(
            style(f"Document indexing failed: {e}", cs.Color.RED)
        )
        logger.exception("Document indexing failed")
        raise typer.Exit(1) from e


if __name__ == "__main__":
    app()
