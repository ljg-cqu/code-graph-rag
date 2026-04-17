# Realtime Updater Integration Design Specification

## 1. Overview

This specification outlines the implementation of a `--realtime_updater` parameter for the main `cgr start` command that enables real-time file system monitoring and graph updates without requiring users to run `realtime_updater.py` in a separate terminal. This feature will support automatic updates for code files, documentation files, JSON files, and other supported file types.

### 1.1 Problem Statement

Currently, users must run two separate processes:
- `cgr start [options]` for the interactive chat session
- `python realtime_updater.py [repo_path] [options]` for real-time file monitoring

This creates friction in the user experience and increases cognitive load. Users often forget to run the updater or run it with inconsistent configuration.

### 1.2 Goals

- **Seamless Integration**: Enable real-time updates with a single CLI flag
- **Resource Efficiency**: Share database connections and avoid duplicate work
- **Consistency**: Ensure the same configuration applies to both chat and updater
- **Reliability**: Handle edge cases gracefully without breaking the main session
- **Backward Compatibility**: Maintain existing behavior when flag is not used

## 2. Architecture

### 2.1 High-Level Design

The solution leverages the existing unified watcher infrastructure from `realtime_updater.py` but integrates it into the main application lifecycle as a background thread rather than a separate process.

```
Main Application (cgr start)
├── Main Thread: Interactive Chat Session
└── Background Thread: Realtime File Watcher
    ├── CodeChangeEventHandler → GraphUpdater (code graph)
    ├── DocumentChangeEventHandler → DocumentGraphUpdater (doc graph)
    └── JSONChangeEventHandler → handle_json_update_event() (json graph)
```

**Note on JSON Ingestion:** Unlike code and document graphs, JSON ingestion uses
`handle_json_update_event()` from `json_ingestion.py` which creates temporary
connections rather than a persistent ingestor. The `JSONChangeEventHandler` wraps
this function and does not require a shared `MemgraphIngestor` instance.

### 2.2 Key Components

#### 2.2.1 UnifiedWatcherManager

A new manager class that encapsulates the realtime watching functionality and provides lifecycle management. This manager wraps the existing `UnifiedChangeEventHandler` (defined in `realtime_updater.py:555`) and adds lifecycle methods for integration with the main application.

**Relationship to Existing Code:**
- `UnifiedChangeEventHandler` already routes file events to appropriate handlers based on file type
- `UnifiedWatcherManager` wraps this handler and manages the `Observer` lifecycle
- This avoids duplicating the event routing logic

```python
@dataclass
class RealtimeConfig:
    """Configuration for realtime file watching."""
    enabled: bool = False
    debounce: float = DEFAULT_DEBOUNCE_SECONDS
    max_wait: float = DEFAULT_MAX_WAIT_SECONDS
    enable_code: bool = True
    enable_docs: bool = False
    enable_json: bool = False


class UnifiedWatcherManager:
    """Manages lifecycle of realtime file watching integrated with main application.

    Wraps existing UnifiedChangeEventHandler and Observer to provide start/stop/join
    lifecycle management suitable for background thread operation.
    """

    def __init__(
        self,
        repo_path: Path,
        code_ingestor: MemgraphIngestor,
        doc_ingestor: MemgraphIngestor | None = None,
        dataset_id: str = "default",  # For JSON updates
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
        enable_code: bool = True,
        enable_docs: bool = False,
        enable_json: bool = False,
    ):
        # Initialize updaters and event handlers based on enabled features
        # Note: JSON handling uses handle_json_update_event() directly, no ingestor needed
        pass

    def start(self) -> None:
        # Start background observer thread
        pass

    def stop(self) -> None:
        # Gracefully stop observer and cleanup resources
        pass

    def join(self, timeout: float | None = None) -> None:
        # Wait for observer to finish
        pass
```

#### 2.2.2 JSON Change Handler

Extends the existing pattern to handle JSON files. Integrates with the existing `handle_json_update_event()` function from `json_ingestion.py:1422`.

```python
class JSONChangeEventHandler(FileSystemEventHandler):
    """Handles JSON file changes using existing handle_json_update_event()."""

    def __init__(
        self,
        repo_path: Path,
        dataset_id: str = "default",
        debounce_seconds: float = DEFAULT_DEBOUNCE_SECONDS,
        max_wait_seconds: float = DEFAULT_MAX_WAIT_SECONDS,
    ):
        self.repo_path = repo_path
        self.dataset_id = dataset_id
        # Debounce state (same pattern as CodeChangeEventHandler)
        self.lock = threading.Lock()
        self.timers: dict[str, threading.Timer] = {}
        self.first_event_time: dict[str, float] = {}
        self.pending_events: dict[str, FileSystemEvent] = {}

    def _process_change(self, event: FileSystemEvent) -> None:
        """Process JSON file change via handle_json_update_event()."""
        path = Path(event.src_path)

        # Load JSON file and create update event
        with open(path) as f:
            data = json.load(f)

        update_event = {
            "operation": "add" if event.event_type == "created" else "update",
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }

        # Use existing handler from json_ingestion.py
        result = handle_json_update_event(
            event=update_event,
            dataset_id=self.dataset_id,
        )
```

## 3. Implementation Details

### 3.1 CLI Integration

Add new parameters to the existing `start` command in `codebase_rag/cli.py`:

```python
@app.command(help=ch.CMD_START)
def start(
    # ... existing parameters ...
    realtime_updater: bool = typer.Option(
        False,
        "--realtime-updater",
        help="Enable real-time file system monitoring and automatic graph updates",
    ),
    realtime_debounce: float = typer.Option(
        DEFAULT_DEBOUNCE_SECONDS,
        "--realtime-debounce",
        "-rd",
        help="Debounce delay in seconds for real-time updates (0 to disable)",
    ),
    realtime_max_wait: float = typer.Option(
        DEFAULT_MAX_WAIT_SECONDS,
        "--realtime-max-wait",
        "-rm",
        help="Maximum wait time in seconds before processing changes",
    ),
    realtime_code: bool = typer.Option(
        True,
        "--realtime-code/--no-realtime-code",
        help="Enable real-time updates for code files",
    ),
    realtime_docs: bool = typer.Option(
        False,
        "--realtime-docs/--no-realtime-docs", 
        help="Enable real-time updates for document files",
    ),
    realtime_json: bool = typer.Option(
        False,
        "--realtime-json/--no-realtime-json",
        help="Enable real-time updates for JSON files",
    ),
):
```

### 3.2 Main Application Flow

Modify `main_unified_async` and `main_async` in `codebase_rag/main.py`:

```python
async def main_unified_async(
    repo_path: str,
    batch_size: int,
    with_docs: bool = False,
    query_mode: QueryMode | None = None,
    doc_workspace: str = "default",
    parallel_config: ParallelExecutionConfig | None = None,
    realtime_config: RealtimeConfig | None = None,  # NEW
):
    # ... existing setup ...

    if realtime_config and realtime_config.enabled:
        # Determine if we need dual-graph connections
        needs_doc_graph = with_docs or realtime_config.enable_docs

        if needs_doc_graph:
            # Create shared ingestors for both code and doc graphs
            with connect_both_graphs(batch_size, doc_workspace) as (code_graph, doc_graph):
                watcher_manager = UnifiedWatcherManager(
                    repo_path=Path(repo_path),
                    code_ingestor=code_graph,
                    doc_ingestor=doc_graph,
                    debounce_seconds=realtime_config.debounce,
                    max_wait_seconds=realtime_config.max_wait,
                    enable_code=realtime_config.enable_code,
                    enable_docs=realtime_config.enable_docs,
                    enable_json=realtime_config.enable_json,
                )
                watcher_manager.start()
                try:
                    await run_chat_loop(...)
                finally:
                    watcher_manager.stop()
                    watcher_manager.join(timeout=5.0)
        else:
            # Code-only mode: use single connection
            with connect_memgraph(batch_size) as code_graph:
                watcher_manager = UnifiedWatcherManager(
                    repo_path=Path(repo_path),
                    code_ingestor=code_graph,
                    doc_ingestor=None,
                    debounce_seconds=realtime_config.debounce,
                    max_wait_seconds=realtime_config.max_wait,
                    enable_code=realtime_config.enable_code,
                    enable_docs=False,
                    enable_json=realtime_config.enable_json,
                )
                watcher_manager.start()
                try:
                    await run_chat_loop(...)
                finally:
                    watcher_manager.stop()
                    watcher_manager.join(timeout=5.0)
    else:
        # Existing behavior - no realtime updates
        if with_docs:
            with connect_both_graphs(batch_size, doc_workspace) as (code_graph, doc_graph):
                # ... existing chat logic with dual graphs ...
        else:
            with connect_memgraph(batch_size) as ingestor:
                # ... existing chat logic ...
```

### 3.3 Resource Sharing Strategy

**Critical Design Decision**: Share database connections between chat session and realtime updater to avoid connection conflicts and ensure consistency.

- **Single Connection Pool**: Both chat tools and updater use the same `MemgraphIngestor` instances
- **Thread Safety**: `MemgraphIngestor` already supports thread-safe operations via internal locking
- **Batch Processing**: Realtime updates use immediate flushing (`flush_all()`) while chat operations can use batching

### 3.4 Error Handling and Recovery

- **Isolated Failures**: File processing errors in the updater should not crash the main chat session
- **Graceful Degradation**: If realtime monitoring fails, continue with chat-only mode
- **Logging**: Separate logger instance for realtime updates to avoid polluting chat output
- **Retry Logic**: Automatic retry for transient database connection issues

### 3.5 Performance Considerations

#### 3.5.1 Debouncing Strategy
Maintain the existing hybrid debouncing approach:
- **Debounce Period**: Wait for quiet period after last change (configurable)
- **Max Wait Time**: Ensure updates happen within maximum time window (configurable)
- **Immediate Processing**: Disable debouncing for critical operations (optional)

#### 3.5.2 Resource Limits
- **File System Events**: Use efficient file system monitoring (watchdog Observer)
- **Memory Usage**: Cache ASTs and file states with LRU eviction
- **CPU Usage**: Background thread with low priority scheduling

#### 3.5.3 Database Operations
- **Batch Updates**: Group related changes into single transactions
- **Index Maintenance**: Ensure vector indexes are updated incrementally
- **Conflict Resolution**: Handle concurrent modifications gracefully

## 4. Configuration and Defaults

### 4.1 Default Behavior
- `--realtime-updater`: Disabled by default (backward compatible)
- When enabled: Only code files are monitored by default (`--realtime-code=true`, `--realtime-docs=false`, `--realtime-json=false`)
- Debounce: 5 seconds (same as current `realtime_updater.py`)
- Max wait: 30 seconds (same as current `realtime_updater.py`)

### 4.2 Configuration Precedence
1. Explicit CLI flags (`--realtime-*`)
2. Environment variables (`CGR_REALTIME_*`)
3. Default values

### 4.3 Environment Variables
```bash
CGR_REALTIME_UPDATER=false
CGR_REALTIME_DEBOUNCE=5.0
CGR_REALTIME_MAX_WAIT=30.0
CGR_REALTIME_CODE=true
CGR_REALTIME_DOCS=false
CGR_REALTIME_JSON=false
```

## 5. User Experience

### 5.1 Status Indicators
When realtime updater is active, display status in chat interface:

```
[Realtime Updater: Active] Monitoring /path/to/repo
- Code files: ✓ Enabled (debounce: 5s, max wait: 30s)
- Document files: ✗ Disabled
- JSON files: ✗ Disabled
```

### 5.2 Update Notifications
Display non-intrusive notifications when updates occur:

```
[✓ Graph Updated] main.py modified (2 functions, 1 file node)
[✓ Graph Updated] README.md modified (1 document section)
```

### 5.3 Control Commands
Add chat commands to control realtime behavior:
- `/realtime status` - Show current realtime updater status
- `/realtime pause` - Temporarily pause monitoring
- `/realtime resume` - Resume monitoring
- `/realtime config` - Show/change configuration

## 6. Testing Strategy

### 6.1 Unit Tests
- `UnifiedWatcherManager` lifecycle management
- Event handler logic for different file types
- Debouncing and max-wait logic
- Error handling and recovery

### 6.2 Integration Tests
- End-to-end realtime updates with chat session
- Concurrent file modifications during active chat
- Resource cleanup on application exit
- Configuration validation and defaults
- **Concurrent Access Tests**:
  - Simultaneous writes from chat tools and watcher updates
  - Read during write operations (verify no partial data)
  - Connection sharing between main thread and watcher thread
  - Verify `flush_all()` from watcher doesn't affect batched chat operations

### 6.3 Performance Tests
- Large repository monitoring (10k+ files)
- Rapid file modification scenarios
- Memory usage over extended periods
- Database performance under load

## 7. Security Considerations

### 7.1 File System Access
- Respect existing `.cgrignore` patterns
- Validate file paths to prevent directory traversal
- Limit monitoring to repository root and subdirectories

### 7.2 Resource Exhaustion
- Maximum concurrent file processing limits
- Memory usage caps for file content caching
- Timeout limits for individual file processing

### 7.3 Database Security
- Use existing authenticated database connections
- No additional privileges required
- Respect existing database constraints and validation

## 8. Backward Compatibility

### 8.1 CLI Compatibility
- All existing CLI flags continue to work unchanged
- New flags are optional with sensible defaults
- No breaking changes to existing workflows

### 8.2 API Compatibility
- Existing `realtime_updater.py` script remains functional
- Can run both integrated and standalone modes simultaneously (with caution)
- Shared configuration between modes

### 8.3 Data Compatibility
- Same graph schema and update logic as standalone updater
- Consistent hash caching and version tracking
- Compatible with existing export/import functionality

## 9. Known Limitations and Future Work

### 9.1 Current Limitations
- **Single Repository**: Only monitors the primary repository specified in `--repo-path`
- **Local Files Only**: Does not handle remote file system events
- **Synchronous Processing**: File processing blocks the watcher thread (mitigated by debouncing)

### 9.2 Future Enhancements
- **Multiple Repository Support**: Monitor multiple repositories simultaneously
- **Remote File Systems**: Support network-mounted or cloud storage
- **Asynchronous Processing**: Non-blocking file processing with worker pools
- **Custom Event Handlers**: Plugin architecture for custom file types
- **Webhook Integration**: Notify external services of graph changes

## 10. Existing Components (Already Implemented)

The following components from `realtime_updater.py` can be reused directly:

| Component | Location | Status |
|-----------|----------|--------|
| `CodeChangeEventHandler` | `realtime_updater.py:39` | ✅ Ready to use |
| `DocumentChangeEventHandler` | `realtime_updater.py:426` | ✅ Ready to use |
| `UnifiedChangeEventHandler` | `realtime_updater.py:555` | ✅ Ready to use |
| `start_watcher()` | `realtime_updater.py:266` | Reference implementation |
| `start_unified_watcher()` | `realtime_updater.py:609` | Reference implementation |
| `GraphUpdater` | `codebase_rag/graph_updater.py` | ✅ Ready to use |
| `DocumentGraphUpdater` | `codebase_rag/document/document_updater.py` | ✅ Ready to use |
| `handle_json_update_event()` | `codebase_rag/json_ingestion.py:1422` | ✅ Ready to use |

### Key Integration Points

1. **Event Routing**: `UnifiedChangeEventHandler.dispatch()` already uses `classify_file()` to route events to the correct handler.

2. **Debouncing**: Both `CodeChangeEventHandler` and `DocumentChangeEventHandler` implement the same hybrid debounce pattern with thread-safe locks.

3. **Thread Safety**: `MemgraphIngestor` has `_conn_lock = threading.Lock()` for thread-safe database operations.

## 11. Implementation Roadmap

### Phase 1: Core Integration (MVP)
- [ ] Implement `UnifiedWatcherManager` class
- [ ] Add CLI parameters to `start` command
- [ ] Integrate with existing main application flow
- [ ] Basic error handling and logging

### Phase 2: Enhanced Features
- [ ] Add JSON file support
- [ ] Implement chat control commands
- [ ] Add status indicators and notifications
- [ ] Comprehensive testing suite

### Phase 3: Optimization and Polish
- [ ] Performance optimization for large repositories
- [ ] Advanced configuration options
- [ ] Documentation and examples
- [ ] User feedback integration

## 12. Validation Criteria

The implementation will be considered successful when:

1. **Functional**: Realtime updates work correctly for all supported file types
2. **Reliable**: No crashes or data corruption under normal usage
3. **Efficient**: Minimal performance impact on main chat session
4. **Usable**: Clear status indicators and intuitive controls
5. **Compatible**: Works with all existing features and configurations
6. **Tested**: Comprehensive test coverage with passing CI builds

## 13. References

- Existing `realtime_updater.py` implementation
- Watchdog library documentation
- Memgraph connection pooling best practices
- Threading and concurrency patterns in Python
- Existing CLI architecture in `codebase_rag/cli.py`