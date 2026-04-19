# Realtime Updater File Watcher Noise Fix

## Document Information
- **Version**: 1.1.0
- **Date**: 2026-04-19
- **Scope**: code-graph-rag realtime file watcher, document change event filtering
- **Status**: Implemented

---

## 1. Executive Summary

This specification addresses **excessive file watcher activity** observed during normal codebase operations. The realtime updater processes many spurious "opened" events for build artifacts and non-source files that should be ignored.

**Issue Summary:**
| # | Issue | Severity | Root Cause |
|---|-------|----------|------------|
| 1 | Build artifacts trigger document events | High | `.txt` pattern in `.cgrignore` not applied to `opened` events |
| 2 | No rate limiting on opened events | Medium | Every file open triggers processing |
| 3 | Document version cache lookup overhead | Low | Skipped documents still trigger log messages |

---

## 2. Issue Analysis

### Issue 1: Build Artifacts Trigger Document Events

**Severity**: High
**File**: `realtime_updater.py:470-514`
**Priority**: P1

**Problem**:
Files like `entry_points.txt`, `top_level.txt`, and other build artifacts generate continuous "opened" events that trigger the document change handler. Even though these files are in `.cgrignore` with `*.txt` pattern, the events still propagate.

**Observed Log Pattern**:
```
2026-04-19 23:27:28.983 | INFO | realtime_updater:dispatch:490 - Document change debouncing: opened on entry_points.txt
2026-04-19 23:27:34.265 | INFO | realtime_updater:_process_change:544 - Processing document change: entry_points.txt
2026-04-19 23:28:07.595 | INFO | realtime_updater:dispatch:490 - Document change debouncing: opened on incomplete_features_spec.md
...
(repeats continuously)
```

**Root Cause Analysis**:

The `DocumentChangeEventHandler` has two separate filtering mechanisms:

1. `_is_relevant()` method for document file filtering
2. `.cgrignore` pattern matching in `DocumentGraphUpdater._collect_documents()`

However, the `_is_relevant()` method may not be properly consulting `.cgrignore` patterns for the `opened` event type, or the patterns are loaded differently between the two systems.

**Current Code Flow**:
```
FileSystemEvent → dispatch() → _is_relevant() → debouncing → _process_change() → _collect_documents()
                                              ↑                           ↑
                                        May not check                Checks .cgrignore
                                        .cgrignore                   (too late - already logged)
```

**Impact**:
- Unnecessary log noise
- CPU cycles spent on debouncing irrelevant files
- Delayed processing of actual document changes due to queue congestion

---

### Issue 2: No Rate Limiting on Opened Events

**Severity**: Medium
**File**: `realtime_updater.py:490-492`

**Problem**:
The `opened` event type is logged and processed for every file access, even though it rarely indicates an actual content change. Most editors trigger `opened` events when merely viewing a file.

**Current Behavior**:
```python
if relative_path_str not in self.first_event_time:
    self.first_event_time[relative_path_str] = current_time
    logger.info(
        f"Document change debouncing: {event.event_type} on {path.name}"
    )
```

**Proposed Change**:
Only log and process `opened` events for files that have actual content changes within a short time window.

---

### Issue 3: Document Version Cache Lookup Overhead

**Severity**: Low
**File**: `codebase_rag/document/document_updater.py:838-845`

**Problem**:
Even when documents are skipped due to version caching, the system still:
1. Logs the debouncing event
2. Processes the change event
3. Looks up the version cache
4. Determines no reindex is needed
5. Logs "Skipping unchanged document"

For high-activity codebases, this creates significant log noise.

**Proposed Solution**:
Add an early-exit check in `_is_relevant()` that skips files matching `.cgrignore` patterns entirely.

---

## 3. Proposed Solution

### Solution 1: Early Filter with .cgrignore Patterns

Load and apply `.cgrignore` patterns in `DocumentChangeEventHandler._is_relevant()` to filter out excluded files before any processing.

**Implementation**:

In `realtime_updater.py`:

```python
class DocumentChangeEventHandler(FileSystemEventHandler):
    """Handle document file changes with debouncing and exclusion patterns."""

    def __init__(
        self,
        doc_updater: DocumentGraphUpdater | None,
        debounce_seconds: float = 5.0,
        max_wait_seconds: float = 30.0,
    ) -> None:
        super().__init__()
        self.doc_updater = doc_updater
        self.debounce_seconds = debounce_seconds
        self.max_wait_seconds = max_wait_seconds
        self.lock = threading.Lock()
        self.pending_events: dict[str, FileSystemEvent] = {}
        self.first_event_time: dict[str, float] = {}
        self.timers: dict[str, threading.Timer] = {}

        # Load cgrignore patterns for early filtering
        self._exclude_patterns: list[re.Pattern] = []
        self._unignore_patterns: list[re.Pattern] = []
        if doc_updater is not None:
            self._exclude_patterns = doc_updater.exclude_patterns
            self._unignore_patterns = doc_updater.unignore_patterns

    def _is_relevant(self, path: str) -> bool:
        """Check if file is relevant for document indexing.

        Applies .cgrignore patterns early to avoid processing excluded files.
        """
        from codebase_rag.shared.utils.file_classifier import is_document_file

        # Check file extension first (fast path)
        if not is_document_file(path):
            return False

        # Apply .cgrignore patterns
        relative_path = Path(path).relative_to(self.doc_updater.repo_path)
        relative_str = str(relative_path)

        # Check unignore patterns first (higher priority)
        for pattern in self._unignore_patterns:
            if pattern.search(relative_str):
                return True

        # Check exclude patterns
        for pattern in self._exclude_patterns:
            if pattern.search(relative_str):
                logger.debug(f"Ignoring document change for excluded file: {relative_str}")
                return False

        return True
```

### Solution 2: Ignore Opened Events for Documents

Add option to ignore `opened` events entirely or only process them when accompanied by a `modified` event within a time window.

```python
# In dispatch() method
if event.event_type == EventType.OPENED:
    # Only process opened events if they might indicate a real change
    # Most editors trigger opened when merely viewing files
    logger.debug(f"Ignoring opened event for {path.name}")
    return
```

### Solution 3: Add Minimal Logging for Skipped Documents

Change log level from INFO to DEBUG for:
- Debouncing messages on excluded files
- "Skipping unchanged document" messages
- Processing messages for files that will be skipped

```python
# Before
logger.info(f"Document change debouncing: {event.event_type} on {path.name}")

# After
logger.debug(f"Document change debouncing: {event.event_type} on {path.name}")
```

---

## 4. Implementation Plan

### Phase 1: Early Filtering (High Priority)

1. Add `exclude_patterns` and `unignore_patterns` properties to `DocumentGraphUpdater`
2. Modify `DocumentChangeEventHandler.__init__()` to receive patterns
3. Update `_is_relevant()` to apply patterns early
4. Add tests for pattern filtering

### Phase 2: Event Type Filtering (Medium Priority)

1. Add configuration option `IGNORE_OPENED_EVENTS` (default: True)
2. Modify `dispatch()` to skip `opened` events
3. Add tests for event type filtering

### Phase 3: Log Level Adjustment (Low Priority)

1. Change INFO to DEBUG for debouncing messages
2. Add configuration option for verbose logging
3. Update documentation

---

## 5. Configuration Options

Add new configuration options in `config.py`:

```python
# Realtime updater settings
REALTIME_IGNORE_OPENED_EVENTS: bool = True
REALTIME_LOG_SKIPPED_DOCUMENTS: bool = False  # Log at DEBUG level by default
REALTIME_DEBOUNCE_SECONDS: float = 5.0
REALTIME_MAX_WAIT_SECONDS: float = 30.0
```

---

## 6. Acceptance Criteria

- [x] Build artifacts (`.txt`, `.egg-info`, etc.) do not trigger document processing
- [x] Opened events are ignored by default
- [x] Log noise is reduced to only meaningful events
- [x] `.cgrignore` patterns are applied consistently across all file event types
- [ ] Tests verify exclusion pattern application (future enhancement)

## Implementation Notes

**Implemented 2026-04-19:**

### Changes Made

1. **Early event type filtering in `dispatch()` methods** (4 handlers):
   - `CodeChangeEventHandler.dispatch` (line 113-121)
   - `DocumentChangeEventHandler.dispatch` (line 495-503)
   - `JSONChangeEventHandler.dispatch` (line 632-640)
   - `UnifiedChangeEventHandler.dispatch` (line 794-802)

   Filters out read-only events (`opened`, `closed_no_write`) before any logging or debouncing.

2. **Removed redundant event type checks** from `_process_change()` methods:
   - `CodeChangeEventHandler._process_change` - removed redundant check
   - `DocumentChangeEventHandler._process_change` - removed redundant check
   - `JSONChangeEventHandler._process_change` - removed redundant check

3. **Added missing docstrings** for consistency:
   - `_schedule_immediate_processing()` in Document and JSON handlers
   - `_process_debounced_change()` in Document and JSON handlers

4. **Added warning logs** for missing events in `_process_debounced_change()`:
   - Code handler: `logger.warning(logs.DEBOUNCE_NO_EVENT.format(path=relative_path_str))`
   - Document handler: `logger.warning(f"No event found for {relative_path_str}")`
   - JSON handler: `logger.warning(f"No event found for {relative_path_str}")`

5. **Changed debouncing log level from INFO to DEBUG** (Solution 3):
   - Code handler: `logger.debug(logs.CHANGE_DEBOUNCING.format(...))`
   - Document handler: `logger.debug(f"Document change debouncing: ...")`
   - JSON handler: `logger.debug(f"JSON change debouncing: ...")`
   - Reduces log noise while keeping processing messages at INFO level

6. **Fixed log level** in `CodeChangeEventHandler._process_change()`:
   - Changed from `logger.warning()` to `logger.info()` for `CHANGE_DETECTED`

7. **Early .cgrignore pattern filtering** in `DocumentChangeEventHandler._is_relevant()` (Solution 1):
   - Added import for `should_skip_path` from `codebase_rag.utils.path_utils`
   - Updated `_is_relevant()` to apply `exclude_paths` and `unignore_paths` from `doc_updater`
   - This ensures `.cgrignore` patterns are applied before any logging or debouncing

### Code Pattern Used

All four handlers now use identical early filtering pattern:
```python
# Filter out read-only events early to avoid log spam
# Skip "opened", "closed_no_write" etc. that don't modify the file
relevant_events = {
    EventType.MODIFIED,
    EventType.CREATED,
    EventType.DELETED,
}
if event.event_type not in relevant_events:
    return
```

### Files Modified

- `realtime_updater.py` - All event handler classes
- `.specs/realtime_updater_noise_fix_spec.md` - This specification (status: Implemented)

---

## 7. Testing Strategy

### Unit Tests

1. Test `_is_relevant()` with various file patterns
2. Test exclusion pattern matching
3. Test event type filtering

### Integration Tests

1. Verify build artifacts don't trigger document updates
2. Verify actual document changes are still processed
3. Verify log output is clean during normal operations

---

## 8. Files to Modify

| File | Changes |
|------|---------|
| `realtime_updater.py` | Add early filtering, ignore opened events |
| `codebase_rag/config.py` | Add new configuration options |
| `codebase_rag/document/document_updater.py` | Expose exclude/unignore patterns |
| `codebase_rag/tests/test_realtime_updater.py` | Add tests for filtering |

---

## 9. Related Documents

- `.cgrignore` - Exclusion patterns for codebase
- `codebase_rag/config.py:load_cgrignore_patterns()` - Pattern loading logic
- `.specs/incomplete_features_spec.md` - Related incomplete features
