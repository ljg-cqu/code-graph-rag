# Incomplete Feature Implementations Design Specification

## Document Information
- **Version**: 1.1.0
- **Date**: 2026-04-19
- **Scope**: code-graph-rag incomplete feature implementations
- **Status**: Implementation Phase

---

## 1. Executive Summary

This specification documents **5 incomplete feature implementations** identified through code analysis. These are methods with `NotImplementedError`, `TODO` comments, or placeholder implementations that need full implementation.

**Issue Summary:**
| # | Feature | File | Severity |
|---|---------|------|----------|
| 1 | Document graph deletion | `realtime_updater.py` | High |
| 2 | JSON entity deletion tracking | `realtime_updater.py` | High |
| 3 | Validation suggestion generation | `doc_vs_code.py` | Medium |
| 4 | Task splitter node-type splitting | `task_splitter.py` | Low |
| 5 | Task splitter query/manual splitting | `task_splitter.py` | Low |

---

## 2. Issue Analysis

### Issue 1: Document Graph Deletion Not Implemented

**Severity**: High
**File**: `realtime_updater.py:552-555`
**Priority**: P1

**Problem**:
When a document file is deleted, the `DocumentChangeEventHandler` only logs a warning but does not remove the document from the graph database.

**Current Code**:
```python
if event.event_type == EventType.DELETED:
    logger.info(f"Document deleted: {path.name}")
    # TODO: Delete from document graph
```

**Impact**:
- Orphaned document nodes accumulate in the document graph
- Document search returns deleted files
- Graph size grows unnecessarily

**Proposed Solution**:
Add a `delete_file()` method to `DocumentGraphUpdater` that:
1. Validates path boundaries (security)
2. Opens a `MemgraphIngestor` connection (the established pattern)
3. Calls the existing `_delete_document_nodes()` to remove `Section` and `Chunk` children
4. Deletes the `Document` node itself with `workspace` filter
5. Removes the entry from `version_cache`
6. Returns `"deleted"`, `"skipped"`, or `"failed"`

**Implementation**:

In `codebase_rag/document/document_updater.py`:
```python
def delete_file(self, file_path: Path) -> str:
    """Delete a document and all related nodes from the graph."""
    if self._is_excluded_path(file_path):
        return "skipped"

    resolved_path = file_path.resolve()
    if not self._is_path_within_boundary(resolved_path):
        return "failed"

    doc_path = str(file_path.relative_to(self.repo_path))

    with MemgraphIngestor(
        host=self.host,
        port=self.port,
        batch_size=self.batch_size,
        connection_timeout=settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT,
    ) as ingestor:
        ingestor.ensure_constraints()
        self._delete_document_nodes(doc_path, ingestor)
        ingestor.execute_write(
            """
            MATCH (d:Document {path: $path, workspace: $workspace})
            DETACH DELETE d
            """,
            {"path": doc_path, "workspace": self.workspace},
        )
        ingestor.flush_all()
        self.version_cache.remove(doc_path)
        self.version_cache.save()
        return "deleted"
```

In `realtime_updater.py`:
```python
if event.event_type == EventType.DELETED:
    result = self.doc_updater.delete_file(path)
    logger.success(f"Document deleted: {path.name} ({result})")
```

**Key Design Decisions**:
- Reuses existing `_delete_document_nodes()` for child cleanup
- Includes `workspace` filter on all queries for multi-tenancy safety
- Removes from `version_cache` so the file is not treated as stale on re-creation

---

### Issue 2: JSON Entity Deletion Tracking Not Implemented

**Severity**: High
**File**: `realtime_updater.py:682-692`
**Priority**: P1

**Problem**:
When a JSON file is deleted, the system logs a warning but doesn't remove the JSON entities from the JSON graph (port 7689, `JsonEntity` nodes).

**Current Code**:
```python
if event.event_type == EventType.DELETED:
    operation = "delete"
    logger.warning(
        f"JSON file deleted: {path.name}. "
        f"Manual cleanup of JSON entities may be required."
    )
    # TODO: Implement proper JSON entity deletion tracking
    return
```

**Impact**:
- Orphaned `JsonEntity` nodes accumulate in the JSON graph
- Data inconsistency between files and graph
- Manual cleanup required by operators

**Root Cause Analysis**:
The `JSONChangeEventHandler` routes to `handle_json_update_event()` → `ingest_json_data()`, which operates on the JSON graph (port 7689). Entities are stored as `JsonEntity` nodes keyed by `unique_id` and scoped by `dataset_id`. The pipeline does not track which file each entity originated from, so file deletion cannot be mapped to entity deletion without additional metadata.

**Proposed Solution**:
Track the source file path in entity metadata and delete by that property on file deletion.

1. **Add `source_file` metadata**: Modify `handle_json_update_event` to accept optional `metadata` and merge it into the ingestion payload. The handler passes `{"source_file": relative_path}`.

2. **Add deletion helper**: Create `delete_entities_by_source_file()` in `json_ingestion.py` that queries `JsonEntity` nodes by `dataset_id` + `source_file` and deletes them.

**Implementation**:

In `codebase_rag/json_ingestion.py`:
```python
def delete_entities_by_source_file(
    dataset_id: str,
    source_file: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> OperationSummary:
    summary = OperationSummary()
    with _create_json_ingestor(batch_size) as graph_connection:
        graph_connection.ensure_constraints()
        rows = graph_connection.fetch_all(
            """
            MATCH (n:JsonEntity {dataset_id: $dataset_id, source_file: $source_file})
            RETURN n.unique_id AS unique_id
            """,
            {"dataset_id": dataset_id, "source_file": source_file},
        )
        for row in rows:
            unique_id = str(row["unique_id"])
            graph_connection.execute_write(
                """
                MATCH (n:JsonEntity {unique_id: $unique_id, dataset_id: $dataset_id})
                DETACH DELETE n
                """,
                {"unique_id": unique_id, "dataset_id": dataset_id},
            )
            summary.deleted += 1
    return summary
```

In `realtime_updater.py`:
```python
relative_path = str(path.relative_to(self.repo_path))

if event.event_type == EventType.DELETED:
    result = delete_entities_by_source_file(
        dataset_id=self.dataset_id,
        source_file=relative_path,
    )
    logger.success(f"JSON file deleted: {path.name} (entities deleted: {result.deleted})")
    return

# For CREATED / MODIFIED
result = handle_json_update_event(
    event=update_event,
    dataset_id=self.dataset_id,
    metadata={"source_file": relative_path},
)
```

**Key Design Decisions**:
- Targets the **JSON graph** (port 7689), not the code graph
- Uses `source_file` property for file-to-entity mapping
- Only affects entities ingested after this change (existing entities lack `source_file`)
- Gracefully handles the case where no entities match (deletes 0, logs count)

---

### Issue 3: Validation Suggestion Generation Placeholder

**Severity**: Medium
**File**: `codebase_rag/shared/validation/doc_vs_code.py:229-236`
**Priority**: P2

**Problem**:
The `_suggest_fix` method returns a generic placeholder string instead of generating context-aware suggestions.

**Current Code**:
```python
def _suggest_fix(self, claim: dict) -> str:
    # TODO: Implement suggestion generation
    return "Update the documentation to reflect current implementation."
```

**Impact**:
- Users receive generic, unhelpful suggestions
- Manual effort required to determine what to update
- Reduced value of validation feature

**Constraints**:
- `validate()` is synchronous; `_suggest_fix` must remain synchronous
- The codebase has no lightweight synchronous LLM completion utility
- `claim` is a `dict` with known keys: `description`, `code_reference`, `source_section`, `source_document`

**Proposed Solution**:
Implement context-aware heuristic suggestions based on `code_reference` type:

1. **API endpoints** (`/...`): Suggest updating route documentation
2. **Function signatures** (`...(...)`): Suggest updating function documentation
3. **Qualified names** (`...`.`...`): Suggest updating module/class docs
4. **Class names** (TitleCase): Suggest updating class documentation
5. **Default**: Suggest updating the specific reference

Also introduce a `ClaimDict` TypedDict for type safety.

**Implementation**:

In `codebase_rag/shared/validation/doc_vs_code.py`:
```python
from typing import TypedDict

class ClaimDict(TypedDict, total=False):
    description: str
    code_reference: str
    source_section: str
    source_document: str

class DocVsCodeValidator(BaseValidator):
    ...

    def _suggest_fix(self, claim: ClaimDict) -> str:
        code_reference = claim.get("code_reference", "")
        if not code_reference:
            return "Update the documentation to reflect current implementation."

        if code_reference.startswith("/"):
            return (
                f"Update API documentation for endpoint '{code_reference}' "
                f"to match the current route implementation."
            )

        if "(" in code_reference and ")" in code_reference:
            return (
                f"Update documentation for function '{code_reference}' "
                f"to match the current signature and behavior."
            )

        if "." in code_reference:
            return (
                f"Update documentation for '{code_reference}' "
                f"to reflect the current module or class structure."
            )

        if code_reference[0:1].isupper():
            return (
                f"Update documentation for class '{code_reference}' "
                f"to match the current implementation."
            )

        return (
            f"Update documentation referencing '{code_reference}' "
            f"to reflect the current code."
        )
```

**Key Design Decisions**:
- Remains synchronous to avoid breaking `validate()` callers
- Uses heuristics rather than LLM calls to avoid blocking and infrastructure dependencies
- TypedDict improves type safety over plain `dict`

---

### Issue 4: Task Splitter Node-Type Splitting Not Implemented

**Severity**: Low
**File**: `codebase_rag/orchestrator/task_splitter.py:275-280`
**Priority**: P3

**Problem**:
The `_split_by_node_type` method raises `NotImplementedError`.

**Current Code**:
```python
def _split_by_node_type(self, prompt: str) -> list[dict[str, Any]]:
    """
    Split task by graph node type (functions, classes, etc.).
    TODO: Implement in Phase 2
    """
    raise NotImplementedError("Node-type splitting will be implemented in Phase 2")
```

**Impact**:
- Cannot use node-type based task splitting
- Limited task distribution options

**Proposed Solution**:
This is a Phase 2 feature. The `NotImplementedError` is intentional and documented. Add a comprehensive docstring and replace `dict[str, Any]` with a proper `Subtask` TypedDict.

**Implementation**:
```python
class Subtask(TypedDict, total=False):
    id: str
    type: str
    query: str
    prompt: str
    file_path: str
    relative_path: str
    priority: int
    complexity: int
    target_entity: str

def _split_by_node_type(self, prompt: str) -> list[Subtask]:
    """
    Split task by graph node type (functions, classes, etc.).

    Planned Behavior:
    - Analyze prompt for mentions of specific node types
    - Create subtasks for each node type cluster
    - Enable parallel querying of different node types

    Example:
        "Find all authentication functions and classes"
        -> [
            {"type": "function", "query": "authentication"},
            {"type": "class", "query": "authentication"},
        ]

    Status: Planned for Phase 2
    """
    raise NotImplementedError("Node-type splitting will be implemented in Phase 2")
```

---

### Issue 5: Task Splitter Query/Manual Splitting Not Implemented

**Severity**: Low
**File**: `codebase_rag/orchestrator/task_splitter.py:282-296`
**Priority**: P3

**Problem**:
Both `_split_by_query` and `_split_manual` raise `NotImplementedError`.

**Current Code**:
```python
def _split_by_query(self, prompt: str) -> list[dict[str, Any]]:
    """
    Split task by independent query segments.
    TODO: Implement in Phase 2
    """
    raise NotImplementedError("Query-based splitting will be implemented in Phase 2")

def _split_manual(self, prompt: str) -> list[dict[str, Any]]:
    """
    Split task based on explicit user-defined subtasks.
    TODO: Implement in Phase 2
    """
    raise NotImplementedError("Manual splitting will be implemented in Phase 2")
```

**Impact**:
- Limited task splitting capabilities
- Cannot use advanced splitting strategies

**Proposed Solution**:
These are Phase 2 features. Add comprehensive docstrings explaining planned behavior and update return types to `list[Subtask]`.

**Implementation**:
```python
def _split_by_query(self, prompt: str) -> list[Subtask]:
    """
    Split task by independent query segments.

    Planned Behavior:
    - Parse prompt into semantically independent sub-queries
    - Create a subtask for each independent segment
    - Enable parallel execution of unrelated questions

    Example:
        "Find auth functions and list all database models"
        -> [
            {"type": "function", "query": "auth"},
            {"type": "class", "query": "database models"},
        ]

    Status: Planned for Phase 2
    """
    raise NotImplementedError("Query-based splitting will be implemented in Phase 2")

def _split_manual(self, prompt: str) -> list[Subtask]:
    """
    Split task based on explicit user-defined subtasks.

    Planned Behavior:
    - Parse prompt for numbered or bulleted subtask lists
    - Create a subtask for each explicitly listed item
    - Preserve user intent for custom decomposition

    Example:
        "1. Find auth functions 2. List database models"
        -> [
            {"type": "function", "query": "auth"},
            {"type": "class", "query": "database models"},
        ]

    Status: Planned for Phase 2
    """
    raise NotImplementedError("Manual splitting will be implemented in Phase 2")
```

---

## 3. Implementation Priority

### Phase 1: Critical (Issues 1, 2)

1. **Document Graph Deletion**
   - Add `delete_file()` to `DocumentGraphUpdater`
   - Wire into `DocumentChangeEventHandler`
   - Test with document deletion events

2. **JSON Entity Deletion**
   - Add `metadata` parameter to `handle_json_update_event`
   - Add `delete_entities_by_source_file()` to `json_ingestion.py`
   - Wire into `JSONChangeEventHandler`
   - Test with JSON file deletion

### Phase 2: Enhancement (Issue 3)

1. **Validation Suggestion Generation**
   - Add `ClaimDict` TypedDict
   - Implement `_suggest_fix` with heuristic suggestions
   - Verify suggestions are context-aware

### Phase 3: Future (Issues 4, 5)

1. Add `Subtask` TypedDict and replace `dict[str, Any]` throughout `task_splitter.py`
2. Expand docstrings for `NotImplementedError` stubs
3. Schedule Phase 2 implementation for node-type/query/manual splitting

---

## 4. Acceptance Criteria

- [ ] Document deletion removes `Document`, `Section`, and `Chunk` nodes from the document graph
- [ ] JSON file deletion removes all `JsonEntity` nodes with matching `source_file` from the JSON graph
- [ ] Validation suggestions provide specific, actionable guidance based on claim type
- [ ] `NotImplementedError` stubs have comprehensive docstrings and use `Subtask` TypedDict
- [ ] Tests pass for all deletion scenarios
- [ ] No `Any` or unparameterized `dict` types remain in touched files

---

## 5. Testing Requirements

### Unit Tests

1. `test_document_deletion_removes_nodes`: Verify `Document`, `Section`, and `Chunk` nodes are deleted
2. `test_json_deletion_by_source_file`: Verify `JsonEntity` nodes are deleted by `source_file`
3. `test_validation_suggestion_context`: Verify suggestions match claim type (API, function, class, etc.)
4. `test_delete_file_skips_excluded_paths`: Verify security boundary checks

### Integration Tests

1. Real-time file deletion scenario with document graph
2. Real-time file deletion scenario with JSON ingestion graph
3. End-to-end validation with suggestion generation

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Deletion removes wrong nodes | Low | High | Use exact path + workspace matching; `DocumentGraphUpdater` reuses tested `_delete_document_nodes` |
| JSON `source_file` missing on old entities | High | Low | Only affects entities ingested before this change; operators can run `delete_dataset` for full cleanup |
| Performance impact on large graphs | Medium | Medium | Batch deletion via individual `DETACH DELETE` per entity; consider batched delete if scale exceeds 10k entities per file |

---

## 7. Related Documents

- `.specs/data_modeling_quality_phase2_spec.md` - Data modeling quality issues
- `realtime_updater.py` - File change event handlers
- `codebase_rag/document/document_updater.py` - Document graph updater
- `codebase_rag/json_ingestion.py` - JSON graph ingestion
- `codebase_rag/shared/validation/` - Validation framework
- `codebase_rag/orchestrator/task_splitter.py` - Task splitting strategies
