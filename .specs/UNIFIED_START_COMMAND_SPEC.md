# Unified `cgr start` Command Specification

**Version:** 2.2.0
**Status:** Implementation-Ready
**Created:** 2025-01-15
**Updated:** 2025-01-18 (Addressed 22 high-severity review issues, added CLI validation, fixed resource leaks, expanded test coverage)
**Author:** System Architecture Team
**Reviewers:** 10 parallel workers (crush, opencode, qwen, goose, kimi, claude)  

---

## 1. Executive Summary

### 1.1 Problem Statement

The current `cgr start` command has two limitations:

1. **Query Limitation**: Only queries the **code GraphRAG**, requiring separate commands (`query-docs`, `query-all`, `validate-spec`) for document GraphRAG access.

2. **Indexing Limitation**: Does not support document indexing, requiring users to run `cgr index-docs` as a separate step before querying documents.

This creates a fragmented workflow:
```bash
# Current workflow (3-4 commands) - document querying requires separate commands
cgr index --update-graph              # Index code
cgr index-docs                        # Index documents
cgr query-docs "question"             # Query documents (separate command)
cgr start                             # Query code only (no document support)
```

### 1.2 Solution Overview

Enhance `cgr start` to be a **unified entry point** for both **indexing** and **querying** across **code and document graphs**:

| Feature | Implementation |
|---------|---------------|
| **Dual-Graph Querying** | `--with-docs` flag + `--mode` selection |
| **Document Indexing** | `--index-docs` flag (opt-in) |
| **Code Indexing** | `--update-graph` (existing, enhanced) |
| **Full Indexing** | `--index-all` convenience flag |
| **Freshness Checking** | `--check-freshness` (default on) |
| **Backend Routing** | Explicit connection management per graph type |

### 1.3 Design Principles

1. **Explicit Backend Routing**: Code graph → `MEMGRAPH_HOST:7687`, Document graph → `DOC_MEMGRAPH_HOST:7688`
2. **Opt-In Indexing**: No automatic indexing by default (performance, user control)
3. **Explicit Query Modes**: No automatic query routing (correctness, predictability)
4. **Backward Compatibility**: Default behavior unchanged for existing users
5. **Error Isolation**: Document issues don't block code queries and vice versa
6. **Clear Attribution**: Every result indicates its source (code vs. document)

---

## 2. Current Architecture Analysis

### 2.1 Memgraph Backend Configuration

```python
# codebase_rag/config.py

# Code Graph Backend (Port 7687)
MEMGRAPH_HOST: str = "localhost"
MEMGRAPH_PORT: int = 7687
MEMGRAPH_HTTP_PORT: int = 7444
MEMGRAPH_USERNAME: str | None = None
MEMGRAPH_PASSWORD: str | None = None

# Document Graph Backend (Port 7688)
DOC_MEMGRAPH_HOST: str = "localhost"
DOC_MEMGRAPH_PORT: int = 7688
DOC_MEMGRAPH_HTTP_PORT: int = 7445  # HTTP port for document graph monitoring
DOC_MEMGRAPH_USERNAME: str | None = None
DOC_MEMGRAPH_PASSWORD: str | None = None

# QueryMode enum values (defined in codebase_rag/shared/query_router.py):
# - CODE_ONLY: Query code graph only (default)
# - DOCUMENT_ONLY: Query document graph only
# - BOTH_MERGED: Query both graphs, merge results with attribution
# - CODE_VS_DOC: Validate code against documentation (docs = truth)
# - DOC_VS_CODE: Validate documentation against code (code = truth)
```

### 2.2 Existing Commands

| Command | Backend | Indexing | Querying | Mode |
|---------|---------|----------|----------|------|
| `cgr start` | Code (7687) | Optional (`--update-graph`) | ✅ Yes | N/A |
| `cgr index` | Code (7687) | ✅ Yes | ❌ No | N/A |
| `cgr index-docs` | Doc (7688) | ✅ Yes | ❌ No | N/A |
| `cgr query-docs` | Doc (7688) | ❌ No | ✅ Yes | DOCUMENT_ONLY |
| `cgr query-all` | Both (7687+7688) | ❌ No | ✅ Yes | BOTH_MERGED |
| `cgr validate-spec` | Both (7687+7688) | ❌ No | ✅ Yes | CODE_VS_DOC |
| `cgr validate-doc` | Both (7687+7688) | ❌ No | ✅ Yes | DOC_VS_CODE |

### 2.3 Existing Components for Reuse

| Component | Location | Backend | Reusability |
|-----------|----------|---------|-------------|
| `QueryRouter` | `codebase_rag/shared/query_router.py` | Both | High — routing logic |
| `QueryMode` enum | `codebase_rag/shared/query_router.py` | Both | High — mode definitions |
| `MemgraphIngestor` | `codebase_rag/services/graph_service.py` | Both | High — connection management |
| `DocumentGraphUpdater` | `codebase_rag/document/document_updater.py` | Doc (7688) | High — document indexing |
| `GraphUpdater` | `codebase_rag/graph_updater.py` | Code (7687) | High — code indexing |
| `document_semantic_search` | `codebase_rag/document/tools/document_search.py` | Doc (7688) | High — document search |
| `/model` command | `codebase_rag/main.py` | N/A | High — pattern for `/mode` |
| `connect_memgraph()` | `codebase_rag/main.py` | Code (7687) | Medium — needs doc equivalent |

---

## 3. Proposed Design

### 3.1 Command-Line Interface Changes

#### 3.1.1 Enhanced `start` Command Signature

```python
# In codebase_rag/cli.py

from typing import Annotated
from codebase_rag.shared.query_router import QueryMode

@app.command(help=ch.CMD_START)
def start(
    # === Existing flags (unchanged) ===
    repo_path: str | None = typer.Option(
        None, "--repo-path", help=ch.HELP_REPO_PATH_RETRIEVAL
    ),
    update_graph: bool = typer.Option(
        False, "--update-graph", help=ch.HELP_UPDATE_GRAPH,
    ),
    clean: bool = typer.Option(
        False, "--clean", help=ch.HELP_CLEAN_DB,
    ),
    output: str | None = typer.Option(
        None, "-o", "--output", help=ch.HELP_OUTPUT_GRAPH,
    ),
    orchestrator: str | None = typer.Option(
        None, "--orchestrator", help=ch.HELP_ORCHESTRATOR,
    ),
    cypher: str | None = typer.Option(
        None, "--cypher", help=ch.HELP_CYPHER_MODEL,
    ),
    no_confirm: bool = typer.Option(
        False, "--no-confirm", help=ch.HELP_NO_CONFIRM,
    ),
    batch_size: int | None = typer.Option(
        None, "--batch-size", min=1, help=ch.HELP_BATCH_SIZE,
    ),
    project_name: str | None = typer.Option(
        None, "--project-name", help=ch.HELP_PROJECT_NAME,
    ),
    exclude: list[str] | None = typer.Option(
        None, "--exclude", help=ch.HELP_EXCLUDE_PATTERNS,
    ),
    interactive_setup: bool = typer.Option(
        False, "--interactive-setup", help=ch.HELP_INTERACTIVE_SETUP,
    ),
    ask_agent: str | None = typer.Option(
        None, "-a", "--ask-agent", help=ch.HELP_ASK_AGENT,
    ),
    
    # === NEW: Document Graph Support ===
    with_docs: bool = typer.Option(
        False,
        "--with-docs",
        help="Connect to document graph for querying (default: code graph only)",
    ),
    
    # === NEW: Document Indexing Flags ===
    index_docs: bool = typer.Option(
        False,
        "--index-docs",
        help="Index documents before starting chat (requires --with-docs or implies it)",
    ),
    
    index_all: bool = typer.Option(
        False,
        "--index-all",
        help="Index both code and documents before starting chat",
    ),
    
    doc_workspace: str = typer.Option(
        "default",
        "--doc-workspace",
        help="Document graph workspace identifier for multi-tenant isolation (default: 'default')",
    ),
    
    # === NEW: Freshness Checking ===
    check_freshness: bool = typer.Option(
        True,
        "--check-freshness/--no-check-freshness",
        help="Check if graphs are up-to-date before starting (default: check enabled)",
    ),
    
    # === NEW: Query Mode ===
    mode: str = typer.Option(
        "code_only",
        "--mode",
        help="Query routing mode: code_only (default), document_only, both_merged, code_vs_doc, doc_vs_code",
    ),

    # === NEW: Indexing Timeout ===
    index_timeout: int = typer.Option(
        300,
        "--index-timeout",
        help="Maximum seconds for indexing operations (default: 300s)",
    ),
) -> None:
    """Start interactive chat session with your codebase.

    Supports both code and document GraphRAG with explicit indexing and querying options.
    """
    # === CLI Flag Validation ===
    # Mode validation: non-code_only modes require --with-docs (or implied by --index-docs)
    effective_with_docs = with_docs or index_docs or index_all
    if mode != "code_only" and not effective_with_docs:
        typer.echo(
            f"ERROR: Mode '{mode}' requires document graph. "
            f"Add --with-docs, --index-docs, or --index-all flag.",
            err=True
        )
        raise typer.Exit(1)

    # Parse and validate mode
    try:
        query_mode = QueryMode(mode.lower())
    except ValueError:
        typer.echo(
            f"ERROR: Invalid mode '{mode}'. "
            f"Valid modes: code_only, document_only, both_merged, code_vs_doc, doc_vs_code",
            err=True
        )
        raise typer.Exit(1)

    # Workspace validation: valid identifier pattern
    import re
    if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', doc_workspace):
        typer.echo(
            f"ERROR: Invalid workspace '{doc_workspace}'. "
            f"Must be 1-64 chars: letters, numbers, underscore, hyphen only.",
            err=True
        )
        raise typer.Exit(1)
```

#### 3.1.2 Help Text Updates

```python
# In codebase_rag/cli_help.py

CMD_START = (
    "Start interactive chat session with your codebase. "
    "Supports both code and document GraphRAG. "
    "Use --with-docs to enable document queries. "
    "Use --index-docs or --index-all to index before chatting. "
    "Use --mode to specify query mode (default: code_only)."
)

HELP_WITH_DOCS = (
    "Connect to document graph (DOC_MEMGRAPH_HOST:DOC_MEMGRAPH_PORT) "
    "in addition to code graph. Required for document queries."
)

HELP_INDEX_DOCS = (
    "Index documents from --repo-path into document graph before starting chat. "
    "Implies --with-docs. Uses DocumentGraphUpdater with version caching."
)

HELP_INDEX_ALL = (
    "Index both code (--update-graph) and documents (--index-docs) before starting. "
    "Convenience flag for first-time setup or major updates."
)

HELP_DOC_WORKSPACE = (
    "Workspace identifier for multi-tenant document graphs. "
    "Isolates document data between projects. Must match workspace used during indexing."
)

HELP_MODE = (
    "Query routing mode. Options:\n"
    "  - code_only: Query code graph only (default)\n"
    "  - document_only: Query document graph only\n"
    "  - both_merged: Query both graphs, merge results with attribution\n"
    "  - code_vs_doc: Validate code against documentation (doc is truth)\n"
    "  - doc_vs_code: Validate documentation against code (code is truth)\n"
    "\nNote: Document graph requires --with-docs flag."
)

HELP_CHECK_FRESHNESS = (
    "Check if indexed graphs are up-to-date with repository. "
    "If stale, prompts to re-index. Disable with --no-check-freshness for faster startup."
)
```

### 3.2 Backend Connection Management

#### 3.2.1 Connection Helper Functions

```python
# In codebase_rag/main.py

from contextlib import contextmanager
from typing import Generator, Any

from .services.graph_service import MemgraphIngestor
from .config import settings


def connect_memgraph(batch_size: int) -> MemgraphIngestor:
    """Connect to CODE graph backend (MEMGRAPH_HOST:MEMGRAPH_PORT).
    
    Args:
        batch_size: Batch size for bulk operations
        
    Returns:
        MemgraphIngestor instance for code graph
    """
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
    import sys
    from loguru import logger

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
```

#### 3.2.2 Backend Routing Table

| Operation | Backend | Host/Port | Ingestor | Updater |
|-----------|---------|-----------|----------|---------|
| Code Indexing | Code Graph | `MEMGRAPH_HOST:7687` | `MemgraphIngestor` | `GraphUpdater` |
| Code Querying | Code Graph | `MEMGRAPH_HOST:7687` | `MemgraphIngestor` | `QueryRouter` |
| Doc Indexing | Doc Graph | `DOC_MEMGRAPH_HOST:7688` | `MemgraphIngestor` | `DocumentGraphUpdater` |
| Doc Querying | Doc Graph | `DOC_MEMGRAPH_HOST:7688` | `MemgraphIngestor` | `QueryRouter` |

---

### 3.3 Indexing Logic

#### 3.3.1 Indexing Decision Flow

```python
# In codebase_rag/cli.py - start command

from pathlib import Path
from rich.table import Table
from codebase_rag.document.document_updater import DocumentGraphUpdater
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers
from codebase_rag.config import load_cgrignore_patterns

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
    from codebase_rag.main import _info, style, _delete_hash_cache
    from codebase_rag import constants as cs
    from loguru import logger
    import typer

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
            from codebase_rag.main import prompt_for_unignored_directories
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
                from codebase_rag.main import export_graph_to_file
                _info(style(cs.CLI_MSG_EXPORTING_TO.format(path=output), cs.Color.CYAN))
                if not export_graph_to_file(ingestor, output):
                    raise typer.Exit(1)

        _info(style(cs.CLI_MSG_GRAPH_UPDATED, cs.Color.GREEN))
        code_indexed = True

    # === Document Indexing ===
    if effective_index_docs:
        _info(style(f"Indexing documents in: {repo_path} (workspace: {doc_workspace})", cs.Color.CYAN))

        try:
            with connect_doc_memgraph(batch_size) as ingestor:
                if clean:
                    _info(style("Cleaning document database...", cs.Color.YELLOW))
                    ingestor.clean_database()
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

            # Use stdout print instead of app_context
            from rich.console import Console
            Console().print(table)
            docs_indexed = True

        except Exception as e:
            _info(style(f"Document indexing failed: {e}", cs.Color.RED))
            logger.exception("Document indexing failed")
            # Don't block chat - continue with code only
            docs_indexed = False
            effective_with_docs = False  # Disable doc queries if indexing failed

    return (code_indexed, docs_indexed, effective_with_docs)
```

#### 3.3.2 Freshness Checking

```python
# In codebase_rag/main.py

from pathlib import Path
import json
from loguru import logger

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
    use file hash comparison (not implemented in v2.1).
    """
    warnings: list[str] = []
    code_fresh = True
    docs_fresh = True
    
    # === Check Code Graph Freshness ===
    try:
        with connect_memgraph(batch_size=1) as ingestor:
            # Check if graph has nodes
            result = ingestor.fetch_all(
                "MATCH (n) RETURN count(n) as count"
            )
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
                    {"ws": doc_workspace}
                )
                if not result or result[0].get("count", 0) == 0:
                    docs_fresh = False
                    warnings.append(f"No documents indexed for workspace '{doc_workspace}'")
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
    from rich.prompt import Confirm
    from rich.console import Console
    from codebase_rag.main import style
    from codebase_rag import constants as cs

    console = Console()
    should_index_code = False
    should_index_docs = False

    # Display warnings
    if warnings:
        console.print(
            style("\n⚠️  Graph Freshness Warnings:", cs.Color.YELLOW)
        )
        for warning in warnings:
            console.print(f"  - {warning}")

    # Prompt for code indexing
    if not code_fresh:
        if Confirm.ask("\nCode graph appears stale. Index now?"):
            should_index_code = True

    # Prompt for document indexing
    if not docs_fresh:
        if Confirm.ask("Document graph appears stale. Index now?"):
            should_index_docs = True
    
    return (should_index_code, should_index_docs)
```

---

### 3.4 Querying Logic

#### 3.4.1 Session Initialization with Dual Graphs

```python
# In codebase_rag/main.py

from typing import Any
from codebase_rag.shared.query_router import QueryMode

async def main_unified_async(
    repo_path: str,
    batch_size: int,
    with_docs: bool = False,
    query_mode: QueryMode = QueryMode.CODE_ONLY,
    doc_workspace: str = "default",
    _fallback_attempted: bool = False,  # Internal: prevents infinite recursion
) -> None:
    """Main async entry point with dual-graph support.
    
    Args:
        repo_path: Repository path
        batch_size: Batch size for graph operations
        with_docs: Enable document graph
        query_mode: Initial query mode
        doc_workspace: Document workspace identifier
        _fallback_attempted: Internal flag to prevent infinite recursion on fallback
    """
    project_root = _setup_common_initialization(repo_path)
    
    # Display configuration table
    table = _create_configuration_table(
        repo_path,
        doc_graph_connected=with_docs,
        query_mode=query_mode,
        doc_workspace=doc_workspace,
    )
    app_context.console.print(table)
    
    if with_docs:
        # Connect to both graphs
        try:
            with connect_both_graphs(batch_size, doc_workspace) as (code_graph, doc_graph):
                app_context.console.print(
                    style("✅ Connected to code graph", cs.Color.GREEN)
                )
                app_context.console.print(
                    style(f"✅ Connected to document graph (workspace: {doc_workspace})", cs.Color.GREEN)
                )
                
                app_context.console.print(
                    Panel(
                        style(cs.MSG_CHAT_INSTRUCTIONS, cs.Color.YELLOW),
                        border_style=cs.Color.YELLOW,
                    )
                )
                
                # Initialize agent with both graphs
                rag_agent, tool_names = _initialize_services_and_agent(
                    repo_path,
                    code_graph,
                    doc_ingestor=doc_graph,
                    query_mode=query_mode,
                    doc_workspace=doc_workspace,
                )
                
                await run_chat_loop(rag_agent, [], project_root, tool_names)
                
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
                    repo_path, batch_size, with_docs=False,
                    _fallback_attempted=True
                )
            else:
                # Already tried fallback, re-raise to avoid infinite recursion
                raise
    else:
        # Code graph only (existing behavior)
        with connect_memgraph(batch_size) as ingestor:
            app_context.console.print(
                style(cs.MSG_CONNECTED_MEMGRAPH, cs.Color.GREEN)
            )
            app_context.console.print(
                Panel(
                    style(cs.MSG_CHAT_INSTRUCTIONS, cs.Color.YELLOW),
                    border_style=cs.Color.YELLOW,
                )
            )
            
            rag_agent, tool_names = _initialize_services_and_agent(
                repo_path, ingestor
            )
            await run_chat_loop(rag_agent, [], project_root, tool_names)


# Keep existing main_async for backward compatibility
async def main_async(repo_path: str, batch_size: int) -> None:
    """Original main_async - unchanged for backward compatibility.
    
    Calls main_unified_async with default parameters.
    """
    await main_unified_async(
        repo_path=repo_path,
        batch_size=batch_size,
        with_docs=False,
        query_mode=QueryMode.CODE_ONLY,
    )
```

#### 3.4.2 Enhanced Configuration Table

```python
# In codebase_rag/main.py

from rich.table import Table
from codebase_rag.shared.query_router import QueryMode

def _create_configuration_table(
    repo_path: str,
    title: str = cs.DEFAULT_TABLE_TITLE,
    language: str | None = None,
    doc_graph_connected: bool = False,
    query_mode: QueryMode = QueryMode.CODE_ONLY,
    doc_workspace: str = "default",
) -> Table:
    """Create startup configuration table with document graph info."""
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
    
    # NEW: Code graph connection
    table.add_row(
        "Code Graph",
        f"{settings.MEMGRAPH_HOST}:{settings.MEMGRAPH_PORT}"
    )
    
    # NEW: Document graph connection
    if doc_graph_connected:
        table.add_row(
            "Document Graph",
            f"{settings.DOC_MEMGRAPH_HOST}:{settings.DOC_MEMGRAPH_PORT} (workspace: {doc_workspace})"
        )
    else:
        table.add_row(
            "Document Graph",
            "NOT CONNECTED (use --with-docs)"
        )
    
    # NEW: Query mode
    table.add_row("Query Mode", query_mode.value)
    
    confirmation_status = (
        cs.CONFIRM_ENABLED if app_context.session.confirm_edits else cs.CONFIRM_DISABLED
    )
    table.add_row(cs.TABLE_ROW_EDIT_CONFIRMATION, confirmation_status)
    table.add_row(cs.TABLE_ROW_TARGET_REPOSITORY, repo_path)
    
    return table
```

#### 3.4.3 QueryRouter Integration

**NOTE:** Requires adding `current_mode` attribute to `QueryRouter` class in `codebase_rag/shared/query_router.py`:

```python
# In codebase_rag/shared/query_router.py (ADD TO __init__)
class QueryRouter:
    def __init__(
        self,
        code_graph: MemgraphIngestor | None = None,
        doc_graph: MemgraphIngestor | None = None,
        code_vector: VectorBackend | None = None,
        doc_vector: VectorBackend | None = None,
    ):
        self.code_graph = code_graph
        self.doc_graph = doc_graph
        self.code_vector = code_vector
        self.doc_vector = doc_vector
        self.current_mode: QueryMode = QueryMode.CODE_ONLY  # NEW: for in-chat mode switching
```

```python
# In codebase_rag/main.py

from typing import Any
from pydantic_ai import Agent, DeferredToolRequests, Tool
from codebase_rag.shared.query_router import QueryRouter, QueryMode
from codebase_rag.services import QueryProtocol
from codebase_rag.services.graph_service import MemgraphIngestor
from codebase_rag.types_defs import ConfirmationToolNames

def _initialize_services_and_agent(
    repo_path: str,
    ingestor: QueryProtocol,
    doc_ingestor: MemgraphIngestor | None = None,
    query_mode: QueryMode = QueryMode.CODE_ONLY,
    doc_workspace: str = "default",
) -> tuple[Agent[None, str | DeferredToolRequests], ConfirmationToolNames]:
    """Initialize services and agent with optional document graph support.
    
    Args:
        repo_path: Repository path
        ingestor: Code graph ingestor
        doc_ingestor: Document graph ingestor (optional)
        query_mode: Initial query mode
        doc_workspace: Document workspace identifier
        
    Returns:
        Tuple of (rag_agent, confirmation_tool_names)
    """
    _validate_provider_config(
        cs.ModelRole.ORCHESTRATOR, settings.active_orchestrator_config
    )
    _validate_provider_config(cs.ModelRole.CYPHER, settings.active_cypher_config)
    
    # Initialize core services
    cypher_generator = CypherGenerator()
    code_retriever = CodeRetriever(project_root=repo_path, ingestor=ingestor)
    file_reader = FileReader(project_root=repo_path)
    file_writer = FileWriter(project_root=repo_path)
    file_editor = FileEditor(project_root=repo_path)
    shell_commander = ShellCommander(
        project_root=repo_path, timeout=settings.SHELL_COMMAND_TIMEOUT
    )
    directory_lister = DirectoryLister(project_root=repo_path)
    
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
    
    # Create tools
    query_tool = create_query_tool(ingestor, cypher_generator, app_context.console)
    code_tool = create_code_retrieval_tool(code_retriever)
    file_reader_tool = create_file_reader_tool(file_reader)
    file_writer_tool = create_file_writer_tool(file_writer)
    file_editor_tool = create_file_editor_tool(file_editor)
    shell_command_tool = create_shell_command_tool(shell_commander)
    directory_lister_tool = create_directory_lister_tool(directory_lister)
    
    # Enhanced document analyzer tool (returns list[Tool] in v2.1)
    document_analyzer_tool_list = create_document_analyzer_tool(
        document_analyzer,
        enable_graph_queries=(doc_ingestor is not None),
    )
    
    # === Build tools list ===
    tools: list[Tool] = [
        query_tool,
        code_tool,
        file_reader_tool,
        file_writer_tool,
        file_editor_tool,
        shell_command_tool,
        directory_lister_tool,
    ]
    
    # Add document analyzer tool(s)
    if isinstance(document_analyzer_tool_list, list):
        tools.extend(document_analyzer_tool_list)
    else:
        tools.append(document_analyzer_tool_list)
    
    # === Graph query tool (dual-graph) ===
    if query_router:
        from codebase_rag.tools.graph_query import create_graph_query_tool
        tools.append(create_graph_query_tool(query_router))
    
    tools.append(semantic_search_tool)
    tools.append(function_source_tool)
    
    confirmation_tool_names = ConfirmationToolNames(
        replace_code=file_editor_tool.name,
        create_file=file_writer_tool.name,
        shell_command=shell_command_tool.name,
    )
    
    rag_agent = create_rag_orchestrator(tools=tools)
    return rag_agent, confirmation_tool_names
```

---

### 3.5 In-Chat Command System

#### 3.5.1 `/mode` Command Implementation

```python
# In codebase_rag/main.py

from typing import Any
from codebase_rag.shared.query_router import QueryMode

def _handle_mode_command(
    command: str,
    query_router: Any | None,
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
    parts = command.strip().split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else None
    
    if not arg:
        # Show current mode
        return current_mode, f"Current mode: {current_mode.value}"
    
    if arg.lower() == "help":
        return current_mode, """
Available modes:
  /mode code_only       - Query code graph only
  /mode document_only   - Query document graph only
  /mode both_merged     - Query both, merge results
  /mode code_vs_doc     - Validate code against docs
  /mode doc_vs_code     - Validate docs against code
  /mode                 - Show current mode
"""
    
    try:
        new_mode = QueryMode(arg.lower())
        
        # Validate mode is available
        if new_mode != QueryMode.CODE_ONLY and query_router is None:
            return current_mode, (
                f"Mode '{new_mode.value}' requires document graph. "
                "Restart with --with-docs flag."
            )
        
        # Update mode in router if available
        if query_router:
            query_router.current_mode = new_mode  # type: ignore
        
        return new_mode, f"Mode switched to: {new_mode.value}"
        
    except ValueError:
        return current_mode, f"Invalid mode: {arg}. Use /mode help for options."
```

#### 3.5.2 Updated Help Commands

```python
# In codebase_rag/constants.py

UI_HELP_COMMANDS = """[bold cyan]Available commands:[/bold cyan]
  /model <provider:model> - Switch to a different model
  /model                  - Show current model
  /mode <mode>            - Switch query mode (code_only, document_only, both_merged, etc.)
  /mode                   - Show current mode
  /help                   - Show this help
  exit, quit              - Exit the session"""
```

---

### 3.6 Graph Query Tool

```python
# In codebase_rag/tools/graph_query.py (NEW FILE)

"""Graph query tool for dual-graph (code + document) queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic_ai import Tool

if TYPE_CHECKING:
    from codebase_rag.shared.query_router import QueryRouter, QueryMode


def create_graph_query_tool(query_router: QueryRouter) -> Tool:
    """Create a tool for querying code and document graphs.
    
    This tool allows the agent to route queries to appropriate graphs
    based on the current session mode.
    
    Args:
        query_router: QueryRouter instance with code and document graph connections
        
    Returns:
        PydanticAI Tool instance
    """
    from codebase_rag.shared.query_router import QueryRequest, QueryMode
    
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
        effective_mode = getattr(query_router, 'current_mode', QueryMode.CODE_ONLY)
        if mode:
            try:
                effective_mode = QueryMode(mode.lower())
            except ValueError:
                return f"Invalid mode: {mode}. Use one of: {[m.value for m in QueryMode]}"
        
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
```

### 3.7 Enhanced Document Analyzer Tool

**NOTE:** This requires refactoring `DocumentAnalyzer` class and updating all callers of `create_document_analyzer_tool()`.

```python
# In codebase_rag/tools/document_analyzer.py (UPDATED)

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from loguru import logger
from pydantic_ai import Tool

if TYPE_CHECKING:
    from codebase_rag.services.graph_service import MemgraphIngestor


class DocumentAnalyzer:
    """Document analyzer with optional document graph support."""
    
    def __init__(
        self,
        project_root: str,
        doc_graph: MemgraphIngestor | None = None,
    ) -> None:
        """Initialize document analyzer.
        
        Args:
            project_root: Project root path
            doc_graph: Optional document graph ingestor for graph queries
        """
        self.project_root = Path(project_root).resolve()
        self.doc_graph = doc_graph
        # ... existing Google GenAI client initialization ...
    
    def analyze_from_graph(
        self,
        question: str,
        top_k: int = 5,
    ) -> str:
        """Query document graph for relevant documentation.
        
        Args:
            question: Natural language query
            top_k: Maximum results
            
        Returns:
            Formatted document search results
        """
        if not self.doc_graph:
            return "Document graph is not available."
        
        from codebase_rag.document.tools.document_search import document_semantic_search
        
        results = document_semantic_search(
            query=question,
            ingestor=self.doc_graph,
            vector_backend=None,
            workspace="default",
            limit=top_k,
        )
        
        if not results:
            return f"No relevant documents found for: {question}"
        
        # Format results
        parts = ["**Relevant Documentation:**\n"]
        for i, result in enumerate(results, 1):
            parts.append(
                f"{i}. **{result['section_title']}** ({result['document_path']})\n"
                f"   {result['content'][:200]}...\n"
            )
        
        return "\n".join(parts)


def create_document_analyzer_tool(
    analyzer: DocumentAnalyzer,
    enable_graph_queries: bool = False,
) -> list[Tool]:
    """Create document analyzer tool(s).
    
    Args:
        analyzer: DocumentAnalyzer instance
        enable_graph_queries: If True, add graph query capability
        
    Returns:
        List of Tool instances
        
    Note:
        BREAKING CHANGE: Return type changed from Tool to list[Tool].
        Update all callers to handle list return type.
    """
    from codebase_rag import tool_descriptions as td
    
    tools: list[Tool] = []
    
    def analyze_document(file_path: str, question: str) -> str:
        """Analyze a document file."""
        try:
            result = analyzer.analyze(file_path, question)
            return result
        except Exception as e:
            logger.exception(f"Document analysis failed: {e}")
            return f"Error analyzing document: {e}"
    
    # Always add base analyze_document tool
    tools.append(
        Tool(
            function=analyze_document,
            name=td.AgenticToolName.ANALYZE_DOCUMENT,
            description=td.ANALYZE_DOCUMENT,
        )
    )
    
    # Add graph query tool if enabled
    if enable_graph_queries and analyzer.doc_graph:
        def analyze_docs(question: str, top_k: int = 5) -> str:
            """Search document graph for relevant documentation."""
            return analyzer.analyze_from_graph(question, top_k)
        
        tools.append(
            Tool(
                function=analyze_docs,
                name="analyze_docs",
                description="Search the document graph for relevant documentation.",
            )
        )
    
    return tools
```

**Migration Note:** Update existing callers in `main.py`:

```python
# OLD (single tool):
document_analyzer_tool = create_document_analyzer_tool(document_analyzer)
tools.append(document_analyzer_tool)

# NEW (list of tools):
document_analyzer_tools = create_document_analyzer_tool(
    document_analyzer,
    enable_graph_queries=(doc_ingestor is not None),
)
tools.extend(document_analyzer_tools)  # Use extend() not append()
```

---

## 4. Implementation Plan

### 4.0 Phase 0: Prerequisites Verification (Pre-Week 1)

**Prerequisites to Verify:**
1. `MemgraphIngestor` implements `__enter__`/`__exit__` context manager protocol (verified in `graph_service.py`)
2. `DocumentGraphUpdater.run()` returns dict with keys: `documents_indexed`, `sections_created`, `chunks_created`, `errors`
3. `QueryRouter` class exists in `codebase_rag/shared/query_router.py`
4. Test infrastructure: mock/real dual Memgraph instances on ports 7687/7688

**Task:** Run verification script before starting Phase 1.

### 4.1 Phase 1: Backend Connection Infrastructure (Week 1)

**Prerequisites:** Phase 0 verification complete

**Tasks:**
1. Create `connect_doc_memgraph()` function in `main.py`
2. Create `connect_both_graphs()` context manager (with credentials + proper cleanup)
3. Update `_create_configuration_table()` to show both graphs
4. Add backend routing constants to `constants.py`

**Files Modified:**
- `codebase_rag/main.py` (connection functions, table)
- `codebase_rag/constants.py` (backend messages)
- `codebase_rag/cli_help.py` (help text)

**Acceptance Criteria:**
- [ ] `connect_doc_memgraph()` connects to `DOC_MEMGRAPH_HOST:DOC_MEMGRAPH_PORT` with credentials
- [ ] `connect_both_graphs()` manages both connections safely:
  - Both connections open before yield
  - Both closed on any exception
  - Partial cleanup on mid-connection failure (if doc fails, code cleaned up)
  - No connection leaks
- [ ] Configuration table shows both backends with ports from config (not hardcoded)
- [ ] No regression in existing code graph connections

### 4.2 Phase 2a: CLI Flag Implementation (Week 2a)

**Prerequisites:** Phase 1 complete (connect_* functions available)

**Tasks:**
1. Add new flags to `start` command signature
2. Add CLI flag validation (mode requires --with-docs, workspace validation)
3. Implement `_handle_indexing()` function
4. Add workspace validation logic

**Files Modified:**
- `codebase_rag/cli.py` (start command, indexing logic)

**Acceptance Criteria:**
- [ ] `--with-docs` flag enables document graph
- [ ] `--index-docs` triggers document indexing (implies --with-docs)
- [ ] `--index-all` triggers both indexing operations
- [ ] `--mode` validation rejects non-code_only modes without --with-docs
- [ ] Workspace validation rejects invalid patterns (regex: `^[a-zA-Z0-9_-]{1,64}$`)
- [ ] Document indexing failures don't block chat (returns docs_indexed=False)

### 4.2b Phase 2b: Freshness & Error Handling (Week 2b)

**Prerequisites:** Phase 2a complete

**Tasks:**
1. Implement freshness checking logic (`_check_graph_freshness`)
2. Implement re-index prompting (`_prompt_for_reindex`)
3. Add error handling for document graph failures
4. Add indexing timeout enforcement

**Files Modified:**
- `codebase_rag/main.py` (freshness checking)

**Acceptance Criteria:**
- [ ] `--check-freshness` prompts for re-index if stale (warn + prompt, not "assume fresh")
- [ ] Specific failure scenarios tested: connection timeout, authentication failure, graph creation error, file parsing error
- [ ] Indexing timeout enforced (--index-timeout flag)

### 4.3 Phase 3: QueryRouter Integration (Week 3)

**Prerequisites:** Phase 1 + 2a + 2b complete

**Tasks:**
1. Add `current_mode` attribute to `QueryRouter.__init__` (CRITICAL: must be done first)
2. Create `graph_query.py` with `create_graph_query_tool()`
3. Update `_initialize_services_and_agent()` for dual graphs
4. Implement `/mode` in-chat command with chat loop integration
5. Update help system
6. **BREAKING CHANGE MIGRATION:** Update all `create_document_analyzer_tool()` callers

**Files Modified:**
- `codebase_rag/shared/query_router.py` (add current_mode attribute)
- `codebase_rag/tools/graph_query.py` (new file)
- `codebase_rag/main.py` (agent initialization, mode command)
- `codebase_rag/constants.py` (help messages)
- `codebase_rag/tools/document_analyzer.py` (return type fix)

**BREAKING CHANGE CALL SITES (must update all):**
1. `codebase_rag/main.py:~1019` - change `append()` to `extend()`
2. `codebase_rag/mcp/tools.py:~98-100` - change `append()` to `extend()`
3. `codebase_rag/tests/test_document_analyzer.py` - update 3 call sites
4. `codebase_rag/tests/integration/test_document_analyzer_integration.py` - update 2 call sites

**Acceptance Criteria:**
- [ ] `query_graphs` tool available when `--with-docs` enabled
- [ ] `/mode` command switches query modes (with chat loop integration shown)
- [ ] Mode persists through session (query_router.current_mode used by agent tools)
- [ ] Source attribution in all responses: every query response in both_merged mode includes distinct "Code Sources" and "Document Sources" sections with file paths

### 4.4 Phase 4: Testing & Documentation (Week 4)

**Prerequisites:** All phases complete

**Tasks:**
1. Write unit tests for backend routing
2. Write integration tests for dual-graph queries
3. Test indexing workflows with mock fixtures
4. Add performance benchmark tests
5. Update user documentation

**Files Created:**
- `codebase_rag/tests/test_unified_start.py`
- `docs/unified-start-guide.md`

**Test Infrastructure Setup:**
- Mock `MemgraphIngestor` or test instances on ports 7687/7688

**Acceptance Criteria:**
- [ ] ≥80% line coverage for new modules (measured with pytest-cov)
- [ ] Backend routing verified (7687 vs 7688) - tests mock both backends
- [ ] Indexing workflows tested end-to-end
- [ ] Performance benchmark: code graph connection <100ms, both_merged query <4s
- [ ] Documentation complete
- `codebase_rag/tests/test_unified_start.py`
- `docs/unified-start-guide.md`

**Acceptance Criteria:**
- [ ] All new code has test coverage
- [ ] Backend routing verified (7687 vs 7688)
- [ ] Indexing workflows tested end-to-end
- [ ] Documentation complete

---

## 5. Technical Specifications

### 5.1 Backend Routing Summary

| Component | Code Graph | Document Graph |
|-----------|-----------|----------------|
| **Host Env Var** | `MEMGRAPH_HOST` | `DOC_MEMGRAPH_HOST` |
| **Port Env Var** | `MEMGRAPH_PORT` (default: 7687) | `DOC_MEMGRAPH_PORT` (default: 7688) |
| **Username Env Var** | `MEMGRAPH_USERNAME` | `DOC_MEMGRAPH_USERNAME` |
| **Password Env Var** | `MEMGRAPH_PASSWORD` | `DOC_MEMGRAPH_PASSWORD` |
| **Connection Function** | `connect_memgraph()` | `connect_doc_memgraph()` |
| **Indexing Updater** | `GraphUpdater` | `DocumentGraphUpdater` |
| **Query Tool** | `create_query_tool()` (code-only mode) | `create_graph_query_tool(query_router)` (dual-graph mode) |
| **Vector Index** | `code_embeddings` (configurable) | `doc_embeddings` (configurable via `DOC_MEMGRAPH_VECTOR_INDEX_NAME`) |
| **Node Labels** | `Function`, `Class`, `Module`, etc. | `Document`, `Section`, `Chunk` |

### 5.2 Connection Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                     cgr start                               │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Indexing Phase (Optional)                      │
│  ┌─────────────────────┐    ┌─────────────────────────┐    │
│  │  --update-graph     │    │  --index-docs           │    │
│  │  GraphUpdater       │    │  DocumentGraphUpdater   │    │
│  │  MEMGRAPH:7687      │    │  DOC_MEMGRAPH:7688      │    │
│  └─────────────────────┘    └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Connection Phase                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  connect_both_graphs() (if --with-docs)             │   │
│  │    → yields (code_graph, doc_graph) atomically      │   │
│  │  OR connect_memgraph() (if code-only)               │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Query Phase (Interactive Chat)                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Tools: query_graphs (if --with-docs), query,       │   │
│  │         document_analyzer, code_retrieval, etc.     │   │
│  │  QueryRouter.current_mode controls routing          │   │
│  │  - Routes to code_graph (7687) or doc_graph (7688) │   │
│  │  - Mode: code_only, document_only, both_merged,    │   │
│  │            code_vs_doc, doc_vs_code                 │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Error Handling Matrix

| Error Scenario | Code Graph | Document Graph | User Impact |
|---------------|-----------|----------------|-------------|
| Connection failed | ❌ Block startup | ⚠️ Warn, continue code-only | Chat starts (code only) |
| Indexing failed | ❌ Block if `--update-graph` | ⚠️ Warn, disable `--with-docs` | Chat starts (code only) |
| Query failed | ⚠️ Show error, continue | ⚠️ Show error, continue | Session continues |
| Mode invalid | ⚠️ Show error | ⚠️ Show error | Mode unchanged |
| Freshness check failed | ⚠️ Warn, prompt for re-index | ⚠️ Warn, prompt for re-index | User decides |
| Workspace invalid | N/A | ❌ Block startup | Error message shown |
| Permission denied | ❌ Block startup | ⚠️ Warn, fallback to code-only | Chat starts (code only) |
| Partial indexing (--index-all) | Code succeeds | Docs fail | ⚠️ Continue with code only |

### 5.4 Security Considerations

1. **Workspace Isolation**: Document graph uses `workspace` parameter for multi-tenancy (validated: regex `^[a-zA-Z0-9_-]{1,64}$`)
2. **Path Traversal**: Both indexers validate paths are within repo boundaries
3. **Authentication**: Separate credentials for code vs. document graphs
4. **Resource Limits**: Indexing has timeouts (--index-timeout) and batch size limits
5. **Symlink Protection**: Both indexers reject symlinks pointing outside repo

---

## 6. Migration Guide for Existing Users

### 6.1 Breaking Changes (API Only)

**Note:** CLI commands remain unchanged. Breaking change affects only custom integrations.

| Change | Impact | Migration |
|--------|--------|-----------|
| `create_document_analyzer_tool()` returns `list[Tool]` instead of `Tool` | Custom callers using `tools.append(tool)` will fail | Change `append()` to `extend()` |

**Affected Files (internal):**
- `codebase_rag/main.py` - updated in Phase 3
- `codebase_rag/mcp/tools.py` - updated in Phase 3
- Test files - updated in Phase 3

**CLI Commands:** All existing CLI commands work unchanged:

```bash
# Existing workflows (all still work)
cgr start
cgr start --update-graph
cgr start --repo-path /path/to/repo
cgr index-docs --repo-path /path/to/repo
cgr query-docs "How do I use the API?"
```

### 6.2 New Unified Workflows

```bash
# First-time setup (was 3 commands, now 1) - --index-all implies --with-docs
cgr start --index-all

# Quick chat with docs (no indexing) - requires docs already indexed
cgr start --with-docs --mode both_merged

# Index docs and chat (no need for explicit --with-docs)
cgr start --index-docs

# Check freshness, prompt if stale
cgr start --with-docs --check-freshness

# Skip freshness check for speed
cgr start --with-docs --no-check-freshness

# Multi-tenant workspace example
cgr start --index-all --doc-workspace my-project
```

**Note on `--check-freshness`:** This is a basic check (graph has nodes, cache files exist). For comprehensive validation, run explicit indexing commands.

### 6.3 In-Chat Usage

```
> /mode both_merged
Mode switched to: both_merged

> /mode
Current mode: both_merged

> /mode help
Available modes:
  /mode code_only       - Query code graph only
  /mode document_only   - Query document graph only
  /mode both_merged     - Query both, merge results
  /mode code_vs_doc     - Validate code against docs
  /mode doc_vs_code     - Validate docs against code
  /mode                 - Show current mode

Note: Document-related modes (document_only, both_merged, code_vs_doc, doc_vs_code)
require the session to have been started with --with-docs, --index-docs, or --index-all.
Attempting to switch to these modes without document graph will show an error message.
```

---

## 7. Testing Strategy

### 7.1 Backend Routing Tests

```python
# test_unified_start.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from codebase_rag.main import connect_memgraph, connect_doc_memgraph, connect_both_graphs
from codebase_rag.config import settings

class TestBackendRouting:
    def test_connect_memgraph_uses_code_backend(self):
        """Verify connect_memgraph uses MEMGRAPH_HOST:MEMGRAPH_PORT with credentials."""
        with connect_memgraph(1000) as ingestor:
            assert ingestor._host == settings.MEMGRAPH_HOST
            assert ingestor._port == settings.MEMGRAPH_PORT
            assert ingestor._username == settings.MEMGRAPH_USERNAME
            assert ingestor._password == settings.MEMGRAPH_PASSWORD

    def test_connect_doc_memgraph_uses_doc_backend(self):
        """Verify connect_doc_memgraph uses DOC_MEMGRAPH_HOST:DOC_MEMGRAPH_PORT."""
        with connect_doc_memgraph(1000) as ingestor:
            assert ingestor._host == settings.DOC_MEMGRAPH_HOST
            assert ingestor._port == settings.DOC_MEMGRAPH_PORT

    def test_connect_both_graphs_returns_both(self):
        """Verify connect_both_graphs returns both ingestors with correct ports."""
        with connect_both_graphs(1000) as (code, doc):
            assert code._port == settings.MEMGRAPH_PORT
            assert doc._port == settings.DOC_MEMGRAPH_PORT

    def test_connect_both_graphs_cleanup_on_doc_failure(self):
        """Verify code_graph is cleaned up if doc_graph connection fails."""
        with patch('codebase_rag.main.MemgraphIngestor') as mock_ingestor:
            # Simulate doc_graph.__enter__ failing
            mock_code = MagicMock()
            mock_doc = MagicMock()
            mock_doc.__enter__.side_effect = ConnectionError("Doc graph unavailable")
            mock_ingestor.side_effect = [mock_code, mock_doc]

            with pytest.raises(ConnectionError):
                with connect_both_graphs(1000):
                    pass

            # Verify code_graph.__exit__ was called for cleanup
            mock_code.__exit__.assert_called_once()
```

### 7.2 Indexing Tests

```python
class TestIndexing:
    @patch('codebase_rag.cli.DocumentGraphUpdater')
    def test_index_docs_routes_to_doc_backend(self, mock_updater_class):
        """Verify --index-docs indexes to document graph."""
        mock_updater = MagicMock()
        mock_updater.run.return_value = {"documents_indexed": 10}
        mock_updater_class.return_value = mock_updater

        result = _handle_indexing(
            repo_path=Path("/test"),
            index_docs=True,
            # ... other params
        )

        mock_updater_class.assert_called_with(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
            # ... other args
        )

    @patch('codebase_rag.cli.GraphUpdater')
    def test_update_graph_routes_to_code_backend(self, mock_updater_class):
        """Verify --update-graph indexes to code graph."""
        # Verify GraphUpdater is called with MEMGRAPH_HOST:MEMGRAPH_PORT
        pass

    def test_index_all_indexes_both(self):
        """Verify --index-all triggers both indexers."""
        pass

    def test_workspace_validation_rejects_invalid(self):
        """Verify workspace validation rejects invalid patterns."""
        import re
        pattern = r'^[a-zA-Z0-9_-]{1,64}$'

        # Valid patterns
        assert re.match(pattern, "default")
        assert re.match(pattern, "my-project-123")
        assert re.match(pattern, "test_workspace")

        # Invalid patterns
        assert not re.match(pattern, "has spaces")
        assert not re.match(pattern, "has/slash")
        assert not re.match(pattern, "")
        assert not re.match(pattern, "x" * 65)
```

### 7.3 Query Routing Tests

```python
class TestQueryRouting:
    def test_code_only_mode_uses_code_backend(self):
        """Verify code_only mode queries only code graph."""
        mock_code = MagicMock()
        mock_doc = MagicMock()
        query_router = QueryRouter(code_graph=mock_code, doc_graph=mock_doc)
        query_router.current_mode = QueryMode.CODE_ONLY

        response = query_router.query(QueryRequest(question="test"))

        # Verify only code_graph was queried
        mock_code.query.assert_called()
        mock_doc.query.assert_not_called()

    def test_document_only_mode_uses_doc_backend(self):
        """Verify document_only mode queries only document graph."""
        pass

    def test_both_merged_mode_queries_both(self):
        """Verify both_merged mode queries both graphs."""
        pass

    def test_source_attribution_in_both_merged(self):
        """Verify responses include distinct Code Sources and Document Sources sections."""
        # Test that both_merged mode response includes:
        # - "**Code Sources (N):**" section
        # - "**Document Sources (M):**" section
        pass
```

### 7.4 Integration Tests

```python
class TestEndToEnd:
    def test_full_workflow_index_and_query(self):
        """Test: index code+docs, then query both."""
        # 1. Index code (mock)
        # 2. Index docs (mock)
        # 3. Start chat with --with-docs
        # 4. Query in both_merged mode
        # 5. Verify results from both graphs
        pass

    def test_freshness_check_prompts_reindex(self):
        """Test: stale docs trigger re-index prompt."""
        pass

    def test_doc_graph_failure_fallback_to_code(self):
        """Test: doc graph failure doesn't block code chat."""
        pass

    def test_workspace_isolation(self):
        """Test: workspace parameter isolates document queries."""
        pass

    def test_authentication_failure_graceful_error(self):
        """Test: authentication failure shows clear error message."""
        pass

    def test_network_timeout_handled(self):
        """Test: network timeout during query is handled gracefully."""
        pass
```

### 7.5 Manual Testing Checklist

- [ ] `cgr start` works without changes
- [ ] `cgr start --with-docs` connects to both graphs
- [ ] `cgr start --index-docs` indexes documents to correct port
- [ ] `cgr start --update-graph` indexes code to correct port
- [ ] `cgr start --index-all` indexes both to correct ports
- [ ] `/mode` command works in chat
- [ ] Mode switches affect query routing
- [ ] Source attribution is clear in both_merged mode
- [ ] Error handling for missing doc graph
- [ ] Backward compatibility maintained
- [ ] Workspace isolation works correctly
- [ ] Invalid workspace name rejected with clear error
- [ ] Hash cache corruption handled gracefully
- [ ] Index timeout enforced
    
    def test_both_merged_mode_queries_both(self):
        """Verify both_merged mode queries both graphs."""
        pass
```

### 7.4 Integration Tests

```python
class TestEndToEnd:
    def test_full_workflow_index_and_query(self):
        """Test: index code+docs, then query both."""
        # 1. Index code
        # 2. Index docs
        # 3. Start chat with --with-docs
        # 4. Query in both_merged mode
        # 5. Verify results from both graphs
        pass
    
    def test_freshness_check_prompts_reindex(self):
        """Test: stale docs trigger re-index prompt."""
        pass
    
    def test_doc_graph_failure_fallback_to_code(self):
        """Test: doc graph failure doesn't block code chat."""
        pass
```

### 7.5 Manual Testing Checklist

- [ ] `cgr start` works without changes
- [ ] `cgr start --with-docs` connects to both graphs
- [ ] `cgr start --index-docs` indexes documents to port 7688
- [ ] `cgr start --update-graph` indexes code to port 7687
- [ ] `cgr start --index-all` indexes both to correct ports
- [ ] `/mode` command works in chat
- [ ] Mode switches affect query routing
- [ ] Source attribution is clear
- [ ] Error handling for missing doc graph
- [ ] Backward compatibility maintained
- [ ] Workspace isolation works correctly

---

## 8. Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Wrong backend routing | High | Low | Explicit connection functions, tests verify ports |
| Document graph not available | Medium | Low | Graceful fallback to code-only with warning |
| Indexing blocks chat startup | Medium | Medium | Opt-in flags, `--index-timeout` (default 300s), progress bar |
| Query mode confusion | Low | Medium | Clear help text, CLI validation (mode requires --with-docs) |
| Performance degradation | Medium | Low | Optional features, default unchanged, benchmark tests |
| Workspace isolation breach | High | Very Low | Regex validation (`^[a-zA-Z0-9_-]{1,64}$`), tests |
| Breaking existing workflows | High | Very Low | Opt-in design, no defaults changed, migration guide |
| Timeout exceeded during indexing | Medium | Low | `--index-timeout` flag, graceful cancellation, warning message |

---

## 9. Success Metrics

### 9.1 Quality Metrics (Testable)

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Backend Routing Accuracy** | 100% (code→7687, doc→7688) | Unit tests mock both backends, verify port usage |
| **Query Success Rate** | >95% for dual-graph queries | Structured logging for query outcomes (success/failure with error type) |
| **Test Coverage** | ≥80% line coverage | pytest-cov measurement |

### 9.2 Performance Metrics (Benchmark Tests)

| Metric | Target | Benchmark Method |
|--------|--------|-----------------|
| **Code Graph Connection** | <100ms | pytest-benchmark in test_unified_start.py |
| **Document Graph Connection** | <100ms | pytest-benchmark in test_unified_start.py |
| **Query Latency (code_only)** | <2s | Benchmark with mock queries |
| **Query Latency (both_merged)** | <4s | Benchmark with mock dual queries |
| **Indexing Speed** | No regression | Baseline comparison: new unified time ≤ legacy `index-docs` time + tolerance |

### 9.3 Manual Testing Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Manual Checklist Pass Rate** | 100% items pass | Section 7.5 checklist completion |

**Note on Adoption Metrics:** Adoption metrics (usage percentages) require opt-in telemetry infrastructure. These are aspirational targets for future implementation when telemetry is added. Current focus is on quality and performance metrics which are directly measurable via tests and benchmarks.

---

## 10. Appendix

### 10.1 Related Documentation

- [QueryRouter API](../codebase_rag/shared/query_router.py) - routing implementation
- [Document GraphRAG Architecture](../docs/architecture.md#document-graphrag) - system design
- [Memgraph Backend Configuration](../codebase_rag/config.py) - config settings

### 10.2 Glossary

| Term | Definition |
|------|------------|
| Code Graph | Knowledge graph of codebase structure (Memgraph port from `MEMGRAPH_PORT`, default 7687) |
| Document Graph | Knowledge graph of documentation (Memgraph port from `DOC_MEMGRAPH_PORT`, default 7688) |
| Backend Routing | Directing operations to correct Memgraph instance based on graph type |
| Query Mode | Explicit routing mode for graph queries (code_only, document_only, both_merged, etc.) |
| Workspace | Multi-tenant isolation identifier for document graphs (validated: `^[a-zA-Z0-9_-]{1,64}$`) |
| Freshness Check | Basic validation that indexed graphs have nodes and cache files exist |
| Index Timeout | Maximum seconds for indexing operations (`--index-timeout`, default 300s) |

### 10.3 Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-15 | Opt-in indexing | Performance, user control |
| 2025-01-15 | Explicit backend routing | Correctness, prevent data corruption |
| 2025-01-16 | Workspace isolation with regex validation | Multi-tenant support, security |
| 2025-01-16 | Freshness checking default on with prompt | Prevent stale query results |
| 2025-01-16 | Error isolation | Doc issues don't block code queries |
| 2025-01-17 | Keep main_async signature | Backward compatibility |
| 2025-01-17 | Index timeout with graceful cancellation | User control, prevent indefinite blocking |
| 2025-01-18 | CLI flag validation for mode/workspace | Prevent runtime errors, clear error messages |

---

**END OF SPECIFICATION**
