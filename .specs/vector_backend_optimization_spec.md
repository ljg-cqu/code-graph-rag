# Vector Backend Initialization Optimization

## Problem Statement

When in `DOCUMENT_ONLY` mode, the system unnecessarily initializes code vector indexes (`function_embedding_index`, `method_embedding_index`, etc.) even though they will never be used.

## Evidence from Logs

```
# Document vector backend requested
Using Memgraph native vector backend for document

# But code indexes are initialized!
Vector index 'function_embedding_index' already exists
Vector index 'method_embedding_index' already exists
Vector index 'class_embedding_index' already exists
Vector index 'interface_embedding_index' already exists
...
# 17 code indexes checked!
```

This happens every time `get_shared_backend_for_documents()` is called.

## Root Cause

**Location**: `codebase_rag/vector_store_memgraph.py:144-240`

```python
def initialize(self) -> None:
    """Create vector indexes for embeddable node types."""
    for label in self.LABELS_TO_INDEX:  # Always code labels!
        index_name = f"{label.lower()}_embedding_index"
        # Check/create each code index...
```

The `is_document` flag is only used for connection parameters, not for index creation logic.

## Impact

1. **Wasted I/O**: 17 unnecessary index checks per document search
2. **Confusing Logs**: Users see code indexes being checked for document queries
3. **Resource Waste**: Memory and CPU spent checking irrelevant indexes

## Proposed Solution

### Quick Fix: Skip Index Creation for Document Backend

```python
def initialize(self) -> None:
    """Create vector indexes for embeddable node types."""
    # Document backend uses pre-created index from DocumentGraphUpdater
    if self.is_document:
        logger.debug("Document backend uses pre-created index, skipping initialization")
        return
        
    # Code backend: create indexes for each label
    for label in self.LABELS_TO_INDEX:
        ...
```

### Complete Fix: Create Document Index in Backend

The document index should be created by the vector backend, not by `DocumentGraphUpdater`:

```python
def initialize(self) -> None:
    """Create vector indexes."""
    if self.is_document:
        self._create_document_index()
    else:
        self._create_code_indexes()

def _create_document_index(self) -> None:
    """Create single index for Chunk nodes."""
    index_name = settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME
    dimension = settings.get_effective_vector_dim("document")
    
    # Check if index exists
    # Create if needed
    # ...

def _create_code_indexes(self) -> None:
    """Create indexes for code node types."""
    for label in self.LABELS_TO_INDEX:
        # Existing logic...
```

## Recommendation

**Implement the Complete Fix** to have consistent index management:
1. All vector indexes are managed by the vector backend
2. `DocumentGraphUpdater` only handles graph structure, not vector indexes
3. Clear separation of concerns

## Implementation Checklist

- [x] Add `is_document` check to `initialize()`
- [x] Move document index creation from `DocumentGraphUpdater` to `MemgraphBackend`
- [x] Update `_ensure_vector_index()` in `DocumentGraphUpdater` to use backend
- [ ] Add tests for document index creation
- [x] Remove duplicate index creation logic

## Files to Modify

1. `codebase_rag/vector_store_memgraph.py` - Add document index logic
2. `codebase_rag/document/document_updater.py` - Remove duplicate index creation
3. `codebase_rag/tests/test_vector_backend.py` - Add tests
