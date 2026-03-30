import sys
import threading
import time
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from codebase_rag import cli_help as ch
from codebase_rag import logs
from codebase_rag import tool_errors as te
from codebase_rag.config import settings
from codebase_rag.constants import (
    CYPHER_DELETE_CALLS,
    CYPHER_DELETE_FILE,
    CYPHER_DELETE_MODULE,
    DEFAULT_DEBOUNCE_SECONDS,
    DEFAULT_MAX_WAIT_SECONDS,
    IGNORE_PATTERNS,
    IGNORE_SUFFIXES,
    KEY_PATH,
    LOG_LEVEL_INFO,
    REALTIME_LOGGER_FORMAT,
    WATCHER_SLEEP_INTERVAL,
    EventType,
    SupportedLanguage,
)
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.language_spec import get_language_spec
from codebase_rag.parser_loader import load_parsers
from codebase_rag.services import QueryProtocol
from codebase_rag.services.graph_service import MemgraphIngestor
from codebase_rag.shared.utils.file_classifier import classify_file, FileType


class CodeChangeEventHandler(FileSystemEventHandler):
    """
    Handles file system events with debouncing to prevent redundant graph updates.

    The handler implements a hybrid debounce strategy:
    - Debounce: Waits for a quiet period after the last change before processing
    - Max wait: Ensures updates happen within a maximum time window, even during
                continuous editing

    This prevents the graph update process from running repeatedly when a file
    is saved multiple times in quick succession (common during active development).
    """

    def __init__(
        self,
        updater: GraphUpdater,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    ):
        self.updater = updater
        self.ignore_patterns = IGNORE_PATTERNS
        self.ignore_suffixes = IGNORE_SUFFIXES

        # (H) Debounce configuration
        self.debounce_seconds = debounce_seconds
        self.max_wait_seconds = max_wait_seconds
        self.debounce_enabled = debounce_seconds > 0

        # (H) Thread-safe state for tracking pending changes
        self.timers: dict[str, threading.Timer] = {}
        self.first_event_time: dict[str, float] = {}
        self.pending_events: dict[str, FileSystemEvent] = {}
        self.lock = threading.Lock()

        if self.debounce_enabled:
            logger.info(
                logs.WATCHER_DEBOUNCE_ACTIVE.format(
                    debounce=debounce_seconds, max_wait=max_wait_seconds
                )
            )
        else:
            logger.info(logs.WATCHER_ACTIVE)

    def _is_relevant(self, path_str: str) -> bool:
        path = Path(path_str)
        if any(path.name.endswith(suffix) for suffix in self.ignore_suffixes):
            return False
        return all(part not in self.ignore_patterns for part in path.parts)

    def dispatch(self, event: FileSystemEvent) -> None:
        # (H) ┌─────────────────────────────────────────────────────────────────────┐
        # (H) │                      Real-Time Graph Update Steps                   │
        # (H) ├─────────────────────────────────────────────────────────────────────┤
        # (H) │ Step 1: Delete all old data from the graph for this file           │
        # (H) │         Provides a clean slate for the updated information         │
        # (H) │ Step 2: Clear the specific in-memory state for the file            │
        # (H) │         Prevents stale in-memory representations                   │
        # (H) │ Step 3: Re-parse the file if it was modified or created            │
        # (H) │         Rebuilds in-memory state (AST, function registry)          │
        # (H) │ Step 4: Re-process all function calls across the entire codebase   │
        # (H) │         Fixes "island" problem - changes reflect in all relations  │
        # (H) │ Step 5: Flush all collected changes to the database                │
        # (H) └─────────────────────────────────────────────────────────────────────┘
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()

        if event.is_directory or not self._is_relevant(src_path):
            return

        if not self.debounce_enabled:
            # (H) No debouncing - process immediately (legacy behavior)
            self._process_change(event)
            return

        # (H) Debounced processing with hybrid approach
        path = Path(src_path)
        relative_path_str = str(path.relative_to(self.updater.repo_path))
        current_time = time.time()

        with self.lock:
            # (H) Track the first event time for max-wait calculation
            if relative_path_str not in self.first_event_time:
                self.first_event_time[relative_path_str] = current_time
                logger.info(
                    logs.CHANGE_DEBOUNCING.format(
                        event_type=event.event_type,
                        name=path.name,
                        debounce=self.debounce_seconds,
                    )
                )

            # (H) Always store the latest event for this file
            self.pending_events[relative_path_str] = event

            # (H) Cancel any existing timer for this file
            if relative_path_str in self.timers:
                self.timers[relative_path_str].cancel()
                logger.debug(logs.DEBOUNCE_RESET.format(path=relative_path_str))

            # (H) Check if max wait time has been exceeded
            time_since_first = current_time - self.first_event_time[relative_path_str]

            if time_since_first >= self.max_wait_seconds:
                # (H) Max wait exceeded - process immediately
                logger.info(
                    logs.DEBOUNCE_MAX_WAIT.format(
                        max_wait=self.max_wait_seconds, path=relative_path_str
                    )
                )
                self._schedule_immediate_processing(relative_path_str)
            else:
                # (H) Schedule debounced processing
                remaining_wait = self.max_wait_seconds - time_since_first
                effective_delay = min(self.debounce_seconds, remaining_wait)
                timer = threading.Timer(
                    effective_delay,
                    self._process_debounced_change,
                    args=[relative_path_str],
                )
                timer.daemon = True
                self.timers[relative_path_str] = timer
                timer.start()

                logger.debug(
                    logs.DEBOUNCE_SCHEDULED.format(
                        path=relative_path_str,
                        debounce=self.debounce_seconds,
                        remaining=f"{remaining_wait:.1f}",
                    )
                )

    def _schedule_immediate_processing(self, relative_path_str: str) -> None:
        """Process a file change immediately (called when max wait is exceeded)."""
        # (H) Use a zero-delay timer to process in the timer thread
        timer = threading.Timer(
            0, self._process_debounced_change, args=[relative_path_str]
        )
        timer.daemon = True
        self.timers[relative_path_str] = timer
        timer.start()

    def _process_debounced_change(self, relative_path_str: str) -> None:
        """Process a debounced file change after the timer fires."""
        with self.lock:
            # (H) Retrieve and clear pending state for this file
            event = self.pending_events.pop(relative_path_str, None)
            self.first_event_time.pop(relative_path_str, None)
            self.timers.pop(relative_path_str, None)

        if event is None:
            logger.warning(logs.DEBOUNCE_NO_EVENT.format(path=relative_path_str))
            return

        logger.info(logs.DEBOUNCE_PROCESSING.format(path=relative_path_str))
        self._process_change(event)

    def _process_change(self, event: FileSystemEvent) -> None:
        """Execute the actual graph update for a file change."""
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()

        ingestor = self.updater.ingestor
        if not isinstance(ingestor, QueryProtocol):
            logger.warning(logs.WATCHER_SKIP_NO_QUERY)
            return

        path = Path(src_path)
        relative_path_str = str(path.relative_to(self.updater.repo_path))

        # (H) Only process events that actually change file content
        # (H) Skip read-only events like "opened", "closed_no_write" that don't modify the file
        relevant_events = {
            EventType.MODIFIED,
            EventType.CREATED,
            EventType.DELETED,  # (H) watchdog deletion event
        }
        if event.event_type not in relevant_events:
            return

        logger.warning(
            logs.CHANGE_DETECTED.format(event_type=event.event_type, path=path)
        )

        # (H) Step 1: Delete existing nodes for this file path
        # (H) Delete Module node and its children (for code files)
        ingestor.execute_write(CYPHER_DELETE_MODULE, {KEY_PATH: relative_path_str})
        # (H) Delete File node (for all files including non-code like .md, .json)
        ingestor.execute_write(CYPHER_DELETE_FILE, {KEY_PATH: relative_path_str})
        logger.debug(logs.DELETION_QUERY.format(path=relative_path_str))

        # (H) Step 2: Clear in-memory state
        self.updater.remove_file_from_state(path)

        # (H) Step 3: Re-parse code files and create File nodes for ALL files
        if event.event_type in (EventType.MODIFIED, EventType.CREATED):
            lang_config = get_language_spec(path.suffix)
            if (
                lang_config
                and isinstance(lang_config.language, SupportedLanguage)
                and lang_config.language in self.updater.parsers
            ):
                if result := self.updater.factory.definition_processor.process_file(
                    path,
                    lang_config.language,
                    self.updater.queries,
                    self.updater.factory.structure_processor.structural_elements,
                ):
                    root_node, language = result
                    self.updater.ast_cache[path] = (root_node, language)

            # (H) Create File node for ALL files (code and non-code like .md, .json, etc.)
            self.updater.factory.structure_processor.process_generic_file(
                path, path.name
            )

        # (H) Step 4
        logger.info(logs.RECALC_CALLS)
        ingestor.execute_write(CYPHER_DELETE_CALLS)
        self.updater._process_function_calls()

        # (H) Step 5: Flush changes to database
        self.updater.ingestor.flush_all()
        logger.success(logs.GRAPH_UPDATED.format(name=path.name))


def start_watcher(
    repo_path: str,
    host: str,
    port: int,
    batch_size: int | None = None,
    debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
) -> None:
    repo_path_obj = Path(repo_path).resolve()
    parsers, queries = load_parsers()

    effective_batch_size = settings.resolve_batch_size(batch_size)

    with MemgraphIngestor(
        host=host,
        port=port,
        batch_size=effective_batch_size,
        username=settings.MEMGRAPH_USERNAME,
        password=settings.MEMGRAPH_PASSWORD,
    ) as ingestor:
        _run_watcher_loop(
            ingestor,
            repo_path_obj,
            parsers,
            queries,
            debounce_seconds,
            max_wait_seconds,
        )


def _run_watcher_loop(
    ingestor,
    repo_path_obj,
    parsers,
    queries,
    debounce_seconds: float,
    max_wait_seconds: float,
):
    updater = GraphUpdater(ingestor, repo_path_obj, parsers, queries)

    # (H) Initial full scan builds the complete context for real-time updates
    logger.info(logs.INITIAL_SCAN)
    updater.run()
    logger.success(logs.INITIAL_SCAN_DONE)

    event_handler = CodeChangeEventHandler(
        updater,
        debounce_seconds=debounce_seconds,
        max_wait_seconds=max_wait_seconds,
    )
    observer = Observer()
    observer.schedule(event_handler, str(repo_path_obj), recursive=True)
    observer.start()
    logger.info(logs.WATCHING.format(path=repo_path_obj))

    try:
        while True:
            time.sleep(WATCHER_SLEEP_INTERVAL)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


def _validate_positive_int(value: int | None) -> int | None:
    if value is None:
        return None
    if value < 1:
        raise typer.BadParameter(te.INVALID_POSITIVE_INT.format(value=value))
    return value


def _validate_non_negative_float(value: float) -> float:
    if value < 0:
        raise typer.BadParameter(te.INVALID_NON_NEGATIVE_FLOAT.format(value=value))
    return value


def main(
    repo_path: Annotated[str, typer.Argument(help=ch.HELP_REPO_PATH_WATCH)],
    host: Annotated[
        str, typer.Option(help=ch.HELP_MEMGRAPH_HOST)
    ] = settings.MEMGRAPH_HOST,
    port: Annotated[
        int, typer.Option(help=ch.HELP_MEMGRAPH_PORT)
    ] = settings.MEMGRAPH_PORT,
    batch_size: Annotated[
        int | None,
        typer.Option(
            help=ch.HELP_BATCH_SIZE,
            callback=_validate_positive_int,
        ),
    ] = None,
    debounce: Annotated[
        float,
        typer.Option(
            "--debounce",
            "-d",
            help=ch.HELP_DEBOUNCE,
            callback=_validate_non_negative_float,
        ),
    ] = DEFAULT_DEBOUNCE_SECONDS,
    max_wait: Annotated[
        float,
        typer.Option(
            "--max-wait",
            "-m",
            help=ch.HELP_MAX_WAIT,
            callback=_validate_non_negative_float,
        ),
    ] = DEFAULT_MAX_WAIT_SECONDS,
) -> None:
    """
    Watch a repository for file changes and update the knowledge graph in real-time.

    The watcher uses a hybrid debouncing strategy to efficiently handle rapid file saves:

    - DEBOUNCE: After a file change, waits for a quiet period before processing.
      This batches rapid saves into a single update.

    - MAX_WAIT: Ensures updates happen within a maximum time window, even during
      continuous editing. Prevents indefinite delays.

    Examples:

        # Default settings (5s debounce, 30s max wait)
        python realtime_updater.py /path/to/repo

        # More aggressive batching for background monitoring
        python realtime_updater.py /path/to/repo --debounce 10 --max-wait 60

        # Quick feedback for demos
        python realtime_updater.py /path/to/repo --debounce 2 --max-wait 10

        # Disable debouncing (legacy behavior)
        python realtime_updater.py /path/to/repo --debounce 0
    """
    logger.remove()
    logger.add(sys.stdout, format=REALTIME_LOGGER_FORMAT, level=LOG_LEVEL_INFO)
    logger.info(logs.LOGGER_CONFIGURED)

    # (H) Validate max_wait is greater than debounce when both are enabled
    if debounce > 0 and max_wait > 0 and max_wait < debounce:
        logger.warning(
            logs.DEBOUNCE_MAX_WAIT_ADJUSTED.format(max_wait=max_wait, debounce=debounce)
        )
        max_wait = debounce

    start_watcher(repo_path, host, port, batch_size, debounce, max_wait)


if __name__ == "__main__":
    typer.run(main)


# Document GraphRAG Extension

# Document extensions to watch
DOC_EXTENSIONS = {".md", ".rst", ".txt", ".pdf", ".docx"}


class DocumentChangeEventHandler(FileSystemEventHandler):
    """
    Handles document file system events for the document graph.

    Pattern follows CodeChangeEventHandler but for documents.
    """

    def __init__(
        self,
        doc_updater,  # DocumentGraphUpdater
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    ):
        self.doc_updater = doc_updater
        self.ignore_patterns = IGNORE_PATTERNS
        self.doc_extensions = DOC_EXTENSIONS

        # Debounce configuration
        self.debounce_seconds = debounce_seconds
        self.max_wait_seconds = max_wait_seconds
        self.debounce_enabled = debounce_seconds > 0

        # Thread-safe state
        self.timers: dict[str, threading.Timer] = {}
        self.first_event_time: dict[str, float] = {}
        self.pending_events: dict[str, FileSystemEvent] = {}
        self.lock = threading.Lock()

    def _is_relevant(self, path_str: str) -> bool:
        """Check if file is a document we should process."""
        path = Path(path_str)
        if path.suffix.lower() not in self.doc_extensions:
            return False
        return all(part not in self.ignore_patterns for part in path.parts)

    def dispatch(self, event: FileSystemEvent) -> None:
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()

        if event.is_directory or not self._is_relevant(src_path):
            return

        if not self.debounce_enabled:
            self._process_change(event)
            return

        # Debounced processing
        path = Path(src_path)
        relative_path_str = str(path.relative_to(self.doc_updater.repo_path))
        current_time = time.time()

        with self.lock:
            if relative_path_str not in self.first_event_time:
                self.first_event_time[relative_path_str] = current_time
                logger.info(
                    f"Document change debouncing: {event.event_type} on {path.name}"
                )

            self.pending_events[relative_path_str] = event

            if relative_path_str in self.timers:
                self.timers[relative_path_str].cancel()

            time_since_first = current_time - self.first_event_time[relative_path_str]

            if time_since_first >= self.max_wait_seconds:
                self._schedule_immediate_processing(relative_path_str)
            else:
                remaining_wait = self.max_wait_seconds - time_since_first
                effective_delay = min(self.debounce_seconds, remaining_wait)
                timer = threading.Timer(
                    effective_delay,
                    self._process_debounced_change,
                    args=[relative_path_str],
                )
                timer.daemon = True
                self.timers[relative_path_str] = timer
                timer.start()

    def _schedule_immediate_processing(self, relative_path_str: str) -> None:
        timer = threading.Timer(
            0, self._process_debounced_change, args=[relative_path_str]
        )
        timer.daemon = True
        self.timers[relative_path_str] = timer
        timer.start()

    def _process_debounced_change(self, relative_path_str: str) -> None:
        with self.lock:
            event = self.pending_events.pop(relative_path_str, None)
            self.first_event_time.pop(relative_path_str, None)
            self.timers.pop(relative_path_str, None)

        if event is None:
            return

        self._process_change(event)

    def _process_change(self, event: FileSystemEvent) -> None:
        """Process document file change."""
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()

        path = Path(src_path)
        logger.info(f"Processing document change: {path.name}")

        relevant_events = {
            EventType.MODIFIED,
            EventType.CREATED,
            EventType.DELETED,
        }
        if event.event_type not in relevant_events:
            return

        try:
            if event.event_type == EventType.DELETED:
                # Remove from document graph
                logger.info(f"Document deleted: {path.name}")
                # TODO: Delete from document graph
            else:
                # Update document in graph
                result = self.doc_updater.update_file(path)
                logger.success(f"Document updated: {path.name} ({result})")
        except Exception as e:
            logger.error(f"Failed to process document {path.name}: {e}")


class UnifiedChangeEventHandler(FileSystemEventHandler):
    """
    Unified handler that routes file changes to appropriate updater.

    Routes:
    - Code files (py, js, ts, etc.) → CodeChangeEventHandler (code graph)
    - Document files (md, pdf, docx) → DocumentChangeEventHandler (doc graph)
    """

    def __init__(
        self,
        code_updater: GraphUpdater,
        doc_updater,
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    ):
        self.code_updater = code_updater
        self.doc_updater = doc_updater
        self.debounce_seconds = debounce_seconds
        self.max_wait_seconds = max_wait_seconds

        # Create delegate handlers
        self.code_handler = CodeChangeEventHandler(
            code_updater, debounce_seconds, max_wait_seconds
        )
        self.doc_handler = DocumentChangeEventHandler(
            doc_updater, debounce_seconds, max_wait_seconds
        )

    def dispatch(self, event: FileSystemEvent) -> None:
        """Route event to appropriate handler based on file type."""
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()

        if event.is_directory:
            return

        path = Path(src_path)

        # Skip files in ignored directories
        if any(part in IGNORE_PATTERNS for part in path.parts):
            return

        # Classify file
        classification = classify_file(path)

        if classification.file_type == FileType.CODE:
            self.code_handler.dispatch(event)
        elif classification.file_type == FileType.DOCUMENT:
            self.doc_handler.dispatch(event)
        # else: skip unknown file types


def start_unified_watcher(
    repo_path: str,
    code_host: str = settings.MEMGRAPH_HOST,
    code_port: int = settings.MEMGRAPH_PORT,
    doc_host: str = settings.DOC_MEMGRAPH_HOST,
    doc_port: int = settings.DOC_MEMGRAPH_PORT,
    batch_size: int | None = None,
    debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
    max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
) -> None:
    """
    Start unified watcher for both code and document graphs.

    This is the recommended way to watch a repository with both
    code and document support.
    """
    from codebase_rag.document.document_updater import DocumentGraphUpdater

    repo_path_obj = Path(repo_path).resolve()
    parsers, queries = load_parsers()

    effective_batch_size = settings.resolve_batch_size(batch_size)

    logger.info(f"Starting unified watcher for: {repo_path_obj}")

    # Initialize code graph updater
    with MemgraphIngestor(
        host=code_host,
        port=code_port,
        batch_size=effective_batch_size,
        username=settings.MEMGRAPH_USERNAME,
        password=settings.MEMGRAPH_PASSWORD,
    ) as code_ingestor:
        code_updater = GraphUpdater(
            code_ingestor, repo_path_obj, parsers, queries
        )

        # Initialize document graph updater
        doc_updater = DocumentGraphUpdater(
            host=doc_host,
            port=doc_port,
            repo_path=repo_path_obj,
        )

        # Initial scan
        logger.info("Running initial code graph scan...")
        code_updater.run()
        logger.success("Initial code graph scan complete")

        logger.info("Running initial document graph scan...")
        doc_updater.run()
        logger.success("Initial document graph scan complete")

        # Set up unified event handler
        event_handler = UnifiedChangeEventHandler(
            code_updater,
            doc_updater,
            debounce_seconds,
            max_wait_seconds,
        )

        observer = Observer()
        observer.schedule(event_handler, str(repo_path_obj), recursive=True)
        observer.start()

        logger.info(f"Watching {repo_path_obj} for code and document changes...")

        try:
            while True:
                time.sleep(WATCHER_SLEEP_INTERVAL)
        except KeyboardInterrupt:
            logger.info("Stopping watcher...")
            observer.stop()
        observer.join()
