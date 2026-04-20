# Document Vector Search Architecture Fix

## Problem Statement

Document semantic search returns no results despite successful indexing of 204 chunks. The root cause is that the `MemgraphBackend` class uses code node labels (`Function`, `Method`, `Class`, etc.) for all vector operations, including document searches that should use `Chunk` nodes.

## Identified Bugs

### Bug 1: Wrong Vector Index Usage for Documents

**Location**: `codebase_rag/vector_store_memgraph.py:66-67`

```python
LABELS_TO_INDEX = cs.EMBEDDABLE_CODE_NODE_LABELS  # Wrong for documents!
```

When `is_document=True`, the backend still uses code labels:
- Searches `function_embedding_index`, `method_embedding_index`, etc.
- Should search `doc_embeddings` index on `Chunk` nodes

**Evidence from logs**:
```
Using Memgraph native vector backend for document
Vector index 'function_embedding_index' already exists  # Wrong!
Vector index 'method_embedding_index' already exists    # Wrong!
```

### Bug 2: Document Backend Initialization Creates Code Indexes

**Location**: `codebase_rag/vector_store_memgraph.py:144-240`

The `initialize()` method creates indexes for all `LABELS_TO_INDEX` regardless of `is_document` flag:
```python
for label in self.LABELS_TO_INDEX:  # Always code labels
    index_name = f"{label.lower()}_embedding_index"
```

**Expected behavior**: When `is_document=True`, create `doc_embeddings` index on `Chunk` nodes only.

### Bug 3: Search Method Ignores Document Index

**Location**: `codebase_rag/vector_store_memgraph.py:383-580`

The `search()` method iterates over `self.LABELS_TO_INDEX`:
```python
for label in self.LABELS_TO_INDEX:
    ...
    "index_name": f"{label.lower()}_embedding_index",
```

For documents, it should use:
- Index: `doc_embeddings` (single index for all Chunks)
- Label: `Chunk`
- Filter: `workspace` property

### Bug 4: Missing Workspace Filter in Document Search

**Location**: `codebase_rag/document/tools/document_search.py:57-66`

```python
backend_results = vector_backend.search(
    query_embedding=query_embedding,
    top_k=limit * 2,
    filters={"workspace": workspace},  # This filter is NOT handled
)
```

The `MemgraphBackend.search()` only handles `project_prefix` filter, not `workspace`:
```python
project_prefix = filters.get("project_prefix") if filters else None
```

### Bug 5: Fallback Path Uses Wrong Index Name

**Location**: `codebase_rag/document/tools/document_search.py:115-170`

The `_search_memgraph_native()` correctly uses `DOC_MEMGRAPH_VECTOR_INDEX_NAME`:
```python
index_name = settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME  # "doc_embeddings"
```

But this fallback only runs when `vector_backend is None`, which is never the case since `get_shared_backend_for_documents()` always returns a backend.

## Root Cause Analysis

```
Document Indexing:
  Document → Chunk nodes → doc_embeddings index (CORRECT)

Document Search:
  query → vector_backend.search() → Searches function_embedding_index (WRONG!)
                                   → Returns empty results
```

The `MemgraphBackend` class was designed for code vector search only. When `is_document=True` was added, only the connection parameters were changed, not the search logic.

## Proposed Solution

### Option A: Separate Document Vector Backend (Recommended)

Create a specialized `DocumentMemgraphBackend` class that handles document-specific search:

```python
class DocumentMemgraphBackend(VectorBackend):
    """Memgraph backend for document chunk embeddings."""
    
    LABEL = "Chunk"
    INDEX_NAME = "doc_embeddings"
    
    def initialize(self) -> None:
        """Create single vector index for Chunk nodes."""
        # Create doc_embeddings index only
        
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None,
        ...
    ) -> list[tuple[int, float]]:
        """Search Chunk nodes with workspace filter."""
        workspace = filters.get("workspace") if filters else None
        # Use vector_search.search('doc_embeddings', ...)
        # Filter by workspace
```

### Option B: Add Document Mode to Existing Backend

Modify `MemgraphBackend` to handle both code and document modes:

```python
class MemgraphBackend(VectorBackend):
    def __init__(self, is_document: bool = False) -> None:
        self.is_document = is_document
        if is_document:
            self._labels = ("Chunk",)
            self._index_name = "doc_embeddings"
        else:
            self._labels = cs.EMBEDDABLE_CODE_NODE_LABELS
            
    def initialize(self) -> None:
        if self.is_document:
            self._create_document_index()
        else:
            self._create_code_indexes()
            
    def search(...) -> ...:
        if self.is_document:
            return self._search_documents(...)
        else:
            return self._search_code(...)
```

## Recommendation

**Option A** is preferred because:
1. **Single Responsibility**: Each class handles one type of vector search
2. **Clear Separation**: Document and code search logic are independent
3. **Easier Testing**: Test each backend in isolation
4. **Future Flexibility**: Can optimize each independently

## Implementation Checklist

- [x] ~~Create `DocumentMemgraphBackend` class~~ (Used Option B: Extended MemgraphBackend)
- [x] Implement `initialize()` for document index creation
- [x] Implement `search()` with workspace filter support
- [x] Update `get_vector_backend()` factory to return correct backend
- [ ] Add integration tests for document vector search
- [x] Add fallback from code backend to document backend

> **Note**: Option B was chosen instead of Option A. The `MemgraphBackend` class was extended
> with `is_document` flag and `_search_documents()` method rather than creating a separate
> `DocumentMemgraphBackend` class. This provides the same functionality with less code duplication.

## Files to Modify

1. `codebase_rag/vector_store_memgraph.py` - Add `DocumentMemgraphBackend` class
2. `codebase_rag/vector_backend.py` - Update factory function
3. `codebase_rag/document/tools/document_search.py` - Ensure correct backend usage
4. `codebase_rag/tests/test_vector_backend.py` - Add document backend tests
