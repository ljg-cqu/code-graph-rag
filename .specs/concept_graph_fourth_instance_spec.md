# Fourth Memgraph Instance for Concept Extraction Graph

## Problem Statement

CGR's document concept extraction (`DOC_CONCEPT_EXTRACTION_ENABLED`) currently stores extracted `Concept` nodes, `MENTIONS` relationships (Chunk→Concept), and concept-to-concept relationships (HIERARCHICAL, COMPOSITIONAL, CAUSAL, etc.) in the **Document Graph instance** (`memgraph-doc`, port 7688) — the same instance that stores `Document`/`Chunk` nodes with their vector embeddings.

This co-location creates three problems:

1. **Resource contention** — Concept extraction performs batch writes (Concept nodes, relationship edges, index creation) while the doc instance simultaneously serves vector search queries (chunk embedding similarity). Write-heavy concept extraction can cause latency spikes in retrieval.

2. **Divergent scaling characteristics** — Document chunks scale with the volume of documentation ingested (linear); concepts and their relationships scale with the *semantic density* of that documentation (potentially quadratic, as concepts form a dense graph). A large codebase with 50K chunks might generate 200K+ concept-to-concept relationships. These workloads have fundamentally different memory, CPU, and I/O patterns.

3. **Coupled data lifecycles** — Document data is inherently cache-like: re-indexing a workspace replaces all chunks and their embeddings. The concept knowledge graph, by contrast, is a long-lived semantic asset that should persist across re-indexing events. Currently, document cleanup (`_cleanup_concepts_for_document`) cascades to concept deletion, coupling these lifecycles.

## Design Goal

Introduce a **fourth Memgraph instance** — the **Concept Graph** (`memgraph-concept`, port 7690) — dedicated to storing:

- `Concept` nodes (full properties: name, definition, confidence, entity_category, entity_subtype, entity_emoji, source_chunk_qn, aliases)
- `ChunkRef` nodes (lightweight proxy: qualified_name, workspace) — anchors for MENTIONS edges
- `MENTIONS` relationships (ChunkRef → Concept, with frequency and context properties)
- Concept-to-concept relationships (all 9 canonical categories: HIERARCHICAL, COMPOSITIONAL, CONTEXTUAL, ATTRIBUTIVE, COMPARATIVE, SEQUENTIAL, CAUSAL, ANALOGICAL, RELATED_TO — with verb, emoji, strength properties)

The **Document Graph instance** (`memgraph-doc`, port 7688) retains:
- `Document`, `Chunk` nodes (with vector embeddings)
- Document→Chunk relationships (CONTAINS_CHUNK)
- NO concept-related data (concept extraction moves entirely to the new instance)

This separation isolates the knowledge graph (persistent, dense, relationship-heavy) from the document store (ephemeral, chunk-heavy, embedding-focused).

### Comparison: Current vs. Proposed

| Aspect | Current (3 instances) | Proposed (4 instances) |
|--------|----------------------|----------------------|
| **Instance 1 (memgraph, :7687)** | Code graph | Code graph (unchanged) |
| **Instance 2 (memgraph-doc, :7688)** | Doc chunks + Concepts + MENTIONS + concept edges | Doc chunks only |
| **Instance 3 (memgraph-json, :7689)** | JSON graph | JSON graph (unchanged) |
| **Instance 4 (memgraph-concept, :7690)** | — (doesn't exist) | Concepts + ChunkRefs + MENTIONS + concept edges |

### Why Not Duplicate Concept Nodes Across Instances?

An alternative would keep lightweight Concept nodes in the doc instance (for MENTIONS traversal) and full Concept nodes in the concept instance (for concept graph queries). This was rejected because:

1. **Dual-write complexity** — Every concept extraction writes to two instances, requiring distributed transaction coordination
2. **Consistency drift** — The two copies can diverge on partial failures
3. **Cleanup amplification** — Document deletion requires coordinated cleanup across both instances

A clean split with `ChunkRef` proxy nodes in the concept instance avoids these problems: writes go to one instance, and the `ChunkRef` node (just `qualified_name` + `workspace`) is a minimal anchor — not a duplicate of chunk data.

## Data Model Changes

### Concept Instance Schema

```
Node Labels:
  :Concept {qualified_name, workspace, name, aliases, type, definition, confidence,
            source_chunk_qn, entity_category, entity_subtype, entity_emoji}
  :ChunkRef {qualified_name, workspace}

Relationship Types:
  (:ChunkRef)-[:MENTIONS {frequency, context}]->(:Concept)
  (:Concept)-[:HIERARCHICAL {verb, emoji, strength}]->(:Concept)
  (:Concept)-[:COMPOSITIONAL {verb, emoji, strength}]->(:Concept)
  (:Concept)-[:CONTEXTUAL {verb, emoji, strength}]->(:Concept)
  (:Concept)-[:ATTRIBUTIVE {verb, emoji, strength}]->(:Concept)
  (:Concept)-[:COMPARATIVE {verb, emoji, strength}]->(:Concept)
  (:Concept)-[:SEQUENTIAL {verb, emoji, strength}]->(:Concept)
  (:Concept)-[:CAUSAL {verb, emoji, strength}]->(:Concept)
  (:Concept)-[:ANALOGICAL {verb, emoji, strength}]->(:Concept)
  (:Concept)-[:RELATED_TO {verb, emoji, strength}]->(:Concept)
```

### Indexes (Concept Instance)

```cypher
-- Node indexes (same as currently in doc instance, plus ChunkRef)
CREATE INDEX ON :Concept(qualified_name);
CREATE INDEX ON :Concept(workspace);
CREATE INDEX ON :Concept(entity_category);
CREATE INDEX ON :Concept(entity_subtype);
CREATE INDEX ON :ChunkRef(qualified_name);
CREATE INDEX ON :ChunkRef(workspace);

-- Edge label indexes (moved from doc instance)
CREATE INDEX ON :HIERARCHICAL(verb);
CREATE INDEX ON :COMPOSITIONAL(verb);
CREATE INDEX ON :CONTEXTUAL(verb);
CREATE INDEX ON :ATTRIBUTIVE(verb);
CREATE INDEX ON :COMPARATIVE(verb);
CREATE INDEX ON :SEQUENTIAL(verb);
CREATE INDEX ON :CAUSAL(verb);
CREATE INDEX ON :ANALOGICAL(verb);
CREATE INDEX ON :RELATED_TO(verb);
```

### Doc Instance Schema (Post-Migration)

After migration, the doc instance drops all concept-related data:

```
Removed from doc instance:
  - :Concept nodes (moved to concept instance)
  - :Topic nodes (moved to concept instance — they are part of the concept taxonomy)
  - (:Chunk)-[:MENTIONS]->(:Concept) edges (replaced by ChunkRef pattern in concept instance)
  - All concept-to-concept edges (9 categories, moved to concept instance)

Retained in doc instance:
  - :Document, :Chunk nodes (unchanged)
  - (:Document)-[:CONTAINS_CHUNK]->(:Chunk) (unchanged)
  - Vector embeddings on Chunk nodes (unchanged)
```

### MENTIONS Edge Direction

Current: `(:Chunk)-[:MENTIONS]->(:Concept)`

Proposed: `(:ChunkRef)-[:MENTIONS]->(:Concept)`

The direction is preserved (chunk → concept). `ChunkRef` nodes are created on-demand during concept storage; they are lightweight proxies for document chunks that exist in the doc instance.

## Configuration Changes

### New Config Settings (in `config.py`)

```python
# ─────────────────────────────────────────────────────────
# CONCEPT GRAPH (NEW — 4th Memgraph instance)
# ─────────────────────────────────────────────────────────
CONCEPT_MEMGRAPH_HOST: str = "localhost"
CONCEPT_MEMGRAPH_PORT: int = 7690
CONCEPT_MEMGRAPH_USERNAME: str | None = None
CONCEPT_MEMGRAPH_PASSWORD: str | None = None
CONCEPT_MEMGRAPH_BATCH_SIZE: int = 1000
CONCEPT_MEMGRAPH_MEMORY_LIMIT: str = "4GB"
CONCEPT_MEMGRAPH_CONNECTION_TIMEOUT: int = Field(default=600, gt=0)
CONCEPT_MEMGRAPH_CONNECTION_RETRY_ATTEMPTS: int = Field(default=3, ge=0)
CONCEPT_MEMGRAPH_CONNECTION_RETRY_BASE_DELAY: float = Field(default=1.0, gt=0)
CONCEPT_LAB_PORT: int = 3003  # Memgraph Lab for concept graph

@property
def concept_memgraph(self) -> dict:
    """Concept Memgraph configuration as a dict for easy access."""
    return {
        "host": self.CONCEPT_MEMGRAPH_HOST,
        "port": self.CONCEPT_MEMGRAPH_PORT,
        "username": self.CONCEPT_MEMGRAPH_USERNAME,
        "password": self.CONCEPT_MEMGRAPH_PASSWORD,
        "batch_size": self.CONCEPT_MEMGRAPH_BATCH_SIZE,
    }
```

### Existing Settings — Changes

| Setting | Change |
|---------|--------|
| `DOC_CONCEPT_EXTRACTION_ENABLED` | No change — still controls extraction on/off |
| All `DOC_CONCEPT_*` timeout/retry settings | No change — control LLM behavior, not storage |
| `DOC_MEMGRAPH_BATCH_SIZE` | Still used for doc instance operations (chunks, documents); concept storage uses `CONCEPT_MEMGRAPH_BATCH_SIZE` |
| `DOC_MEMGRAPH_HOST:PORT` | Unchanged — doc instance still stores chunks/embeddings |
| `DOC_MEMGRAPH_VECTOR_*` | Unchanged — vector search on chunks stays in doc instance |

**Removed from doc instance scope** (no longer applicable):
- `_ensure_concept_indexes()` no longer runs against doc instance; it runs against concept instance
- Concept-related edge indexes (on HIERARCHICAL, COMPOSITIONAL, etc.) are now created in concept instance

### Environment Variables (`.env` / `.env.example`)

New block:
```bash
# ─────────────────────────────────────────────────────────
# Concept Graph (4th Memgraph instance — knowledge graph)
# ─────────────────────────────────────────────────────────
# Security: If deploying CGR as an MCP server that connects to a remote Memgraph instance,
# restrict the Memgraph port via firewall (never expose to public internet without auth)
# and bind CONCEPT_MEMGRAPH_HOST to localhost only (never 0.0.0.0) to prevent unauthorized access
CONCEPT_MEMGRAPH_HOST=localhost
CONCEPT_MEMGRAPH_PORT=7690
# Memgraph username (leave empty if no authentication configured on the server)
CONCEPT_MEMGRAPH_USERNAME=admin
# Memgraph password — replace with a strong random value in production
CONCEPT_MEMGRAPH_PASSWORD=REPLACE_WITH_SECURE_PASSWORD_IN_PRODUCTION
CONCEPT_MEMGRAPH_MEMORY_LIMIT=4GB
CONCEPT_MEMGRAPH_CONNECTION_TIMEOUT=600
CONCEPT_MEMGRAPH_CONNECTION_RETRY_ATTEMPTS=3
CONCEPT_MEMGRAPH_CONNECTION_RETRY_BASE_DELAY=1.0
CONCEPT_LAB_PORT=3003
```

## Docker Compose Changes

### New Service: `memgraph-concept`

```yaml
  # ============================================
  # Concept Graph (NEW — 4th instance)
  # ============================================
  memgraph-concept:
    image: memgraph/memgraph-mage:latest
    ports:
      - "${CONCEPT_MEMGRAPH_PORT:-7690}:7687"
    command: [
      "--schema-info-enabled",
      "--also-log-to-stderr"
    ]
    environment:
      - MEMGRAPH_MEMORY_LIMIT=${CONCEPT_MEMGRAPH_MEMORY_LIMIT:-4GB}
    volumes:
      - memgraph_concept_data:/var/lib/memgraph
    healthcheck:
      test: ["CMD-SHELL", "echo 'RETURN 1;' | mgconsole"]
      interval: 30s
      timeout: 10s
      retries: 3

  lab-concept:
    image: memgraph/lab:latest
    ports:
      - "${CONCEPT_LAB_PORT:-3003}:3000"
    environment:
      QUICK_CONNECT_MG_HOST: memgraph-concept
    depends_on:
      - memgraph-concept
```

### New Volume

```yaml
volumes:
  memgraph_code_data:
  memgraph_doc_data:
  memgraph_json_data:
  memgraph_concept_data:  # NEW Concept Graph Volume
```

### Port Allocation Summary

| Service | Bolt Port | Lab Port | Volume |
|---------|-----------|----------|--------|
| memgraph (code) | 7687 | 3000 (lab) | memgraph_code_data |
| memgraph-doc (docs) | 7688 | 3001 (lab-doc) | memgraph_doc_data |
| memgraph-json (json) | 7689 | 3002 (lab-json) | memgraph_json_data |
| **memgraph-concept** | **7690** | **3003** (lab-concept) | **memgraph_concept_data** |

## Code Changes

### 1. `codebase_rag/main.py` — New Connection Function

Add `connect_concept_memgraph()`:

```python
def connect_concept_memgraph(batch_size: int = 1000) -> MemgraphIngestor:
    """Connect to CONCEPT graph backend (CONCEPT_MEMGRAPH_HOST:CONCEPT_MEMGRAPH_PORT)."""
    return MemgraphIngestor(
        host=settings.CONCEPT_MEMGRAPH_HOST,
        port=settings.CONCEPT_MEMGRAPH_PORT,
        batch_size=batch_size,
        username=settings.CONCEPT_MEMGRAPH_USERNAME,
        password=settings.CONCEPT_MEMGRAPH_PASSWORD,
    )
```

Add concept instance to `connect_both_graphs()` → rename to `connect_all_graphs()` or add a separate `connect_concept_graph()` context manager. The existing `connect_both_graphs()` can optionally gain a concept ingestor parameter.

### 2. `codebase_rag/services/graph_service.py` — Port-Based Routing

Two methods need updates for the concept instance:

#### 2a. Add concept instance timeout routing in `_get_connection_timeout()` (line 617)

```python
def _get_connection_timeout(self) -> int:
    if self._connection_timeout is not None:
        return self._connection_timeout
    if self._port == settings.DOC_MEMGRAPH_PORT:
        return settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT
    elif self._port == settings.JSON_MEMGRAPH_PORT:
        return settings.JSON_MEMGRAPH_CONNECTION_TIMEOUT
    elif self._port == settings.CONCEPT_MEMGRAPH_PORT:     # NEW
        return settings.CONCEPT_MEMGRAPH_CONNECTION_TIMEOUT  # NEW
    return settings.MEMGRAPH_CONNECTION_TIMEOUT
```

#### 2b. Fix hardcoded graph_type in `_execute_query_with_guidance()` (line 600-601)

The method currently hardcodes:
```python
graph_type="code" if self._port == settings.MEMGRAPH_PORT else "document",
```

This produces incorrect error guidance when the concept instance encounters failures. Replace with explicit port-based dispatch:

```python
if self._port == settings.DOC_MEMGRAPH_PORT:
    graph_type = "document"
elif self._port == settings.JSON_MEMGRAPH_PORT:
    graph_type = "json"
elif self._port == settings.CONCEPT_MEMGRAPH_PORT:
    graph_type = "concept"
else:
    graph_type = "code"
```

### 3. `codebase_rag/document/document_updater.py` — Core Storage Changes

This is the primary file affected. The actual class is `DocumentGraphUpdater` (line 328), which takes `host`, `port`, `username`, `password` in its constructor — **not** a pre-built ingestor. The `MemgraphIngestor` is created inside `run()` (line 570) and `run_async()` (line 749) using `self.host`/`self.port`, then passed down to `_process_document()` → `_extract_and_store_concepts()` and `_delete_document_nodes()`.

The changes are:

#### 3a. Add Concept Instance Connection Parameters to `__init__`

Add concept instance connection parameters alongside existing doc instance parameters:

```python
class DocumentGraphUpdater:
    def __init__(
        self,
        host: str,
        port: int,
        repo_path: Path,
        batch_size: int = 1000,
        workspace: str = "default",
        exclude_paths: frozenset[str] | None = None,
        unignore_paths: frozenset[str] | None = None,
        concept_extractor: ConceptExtractor | None = None,
        embeddings_enabled: bool | None = None,
        embeddings_required: bool | None = None,
        username: str | None = None,
        password: str | None = None,
        # NEW: concept instance connection parameters
        concept_host: str | None = None,
        concept_port: int | None = None,
        concept_username: str | None = None,
        concept_password: str | None = None,
    ):
        ...
        self.concept_host = concept_host or settings.CONCEPT_MEMGRAPH_HOST
        self.concept_port = concept_port or settings.CONCEPT_MEMGRAPH_PORT
        self.concept_username = concept_username or settings.CONCEPT_MEMGRAPH_USERNAME
        self.concept_password = concept_password or settings.CONCEPT_MEMGRAPH_PASSWORD
```

#### 3b. Create Concept Ingestor in `run()` and `run_async()`

Inside `run()` (line 570), after creating the doc ingestor, create a concept ingestor when concept extraction is enabled:

```python
# Inside run() and run_async(), after doc ingestor creation:
concept_ingestor = None
if self.concept_extractor and settings.CONCEPT_MEMGRAPH_ENABLED:
    try:
        concept_ingestor = MemgraphIngestor(
            host=self.concept_host,
            port=self.concept_port,
            batch_size=settings.CONCEPT_MEMGRAPH_BATCH_SIZE,
            connection_timeout=settings.CONCEPT_MEMGRAPH_CONNECTION_TIMEOUT,
            username=self.concept_username,
            password=self.concept_password,
        ).__enter__()
    except (ConnectionError, TimeoutError, OSError) as e:
        logger.warning(
            f"Concept extraction enabled but concept graph instance unavailable: {e} — "
            "concepts will not be stored"
        )
```

Pass `concept_ingestor` alongside the doc `ingestor` to `_process_document()` / `_process_document_async()`, and from there to `_extract_and_store_concepts()`, `_delete_document_nodes()`, and `_cleanup_concepts_for_document()`.

When `DOC_CONCEPT_EXTRACTION_ENABLED` is True and `concept_ingestor` is None (connection failed or `CONCEPT_MEMGRAPH_ENABLED=False`), log a warning and skip concept storage (graceful degradation). When provided, use for all concept storage operations.

#### 3c. Call `_ensure_concept_indexes()` with Concept Ingestor

`_ensure_concept_indexes(ingestor)` (line 2516) is already parameterized. Change the call site at line 2612 from the doc ingestor to the concept ingestor:

```python
self._ensure_concept_indexes(concept_ingestor)  # was: ingestor (doc)
```

Also add a `ChunkRef` index to the method's index creation list (line 2526):

```python
indexes_to_create = [
    ("Concept", "qualified_name"),
    ("Concept", "workspace"),
    ("Concept", "entity_category"),
    ("Concept", "entity_subtype"),
    ("ChunkRef", "qualified_name"),  # NEW
    ("ChunkRef", "workspace"),       # NEW
    ("Topic", "qualified_name"),     # Topic nodes also move to concept instance
    ("Topic", "workspace"),
]
```

#### 3d. Update `_extract_and_store_concepts()` — Accept Both Ingestors

Change signature to accept both ingestors (doc ingestor no longer needed for concept operations, but concept ingestor is required):

```python
async def _extract_and_store_concepts(
    self,
    chunks: list[DocumentChunk],
    ingestor: MemgraphIngestor,              # doc ingestor (unchanged — still passed through)
    concept_ingestor: MemgraphIngestor,      # NEW: concept instance
    workspace: str,
    stats: dict[str, object] | None = None,
) -> None:
```

Internal calls to `_merge_concept_nodes_batch()`, `_create_mentions_batch()`, `_store_concept_relationships_batch()` all use `concept_ingestor` instead of the doc ingestor.

#### 3e. Update Call Sites in `_process_document()` and `_process_document_async()`

- **Sync path** (line 1466): Change from `self._extract_and_store_concepts(chunks, ingestor, self.workspace, stats)` to `self._extract_and_store_concepts(chunks, ingestor, concept_ingestor, self.workspace, stats)`
- **Async path** (line 1611): Same change — add `concept_ingestor` parameter
- **`_delete_document_nodes()`** (line 1504): Change `self._cleanup_concepts_for_document(doc_path, ingestor)` to `self._cleanup_concepts_for_document(doc_path, ingestor, concept_ingestor)`

#### 3f. Update `_create_mentions_batch()` — Use ChunkRef Nodes

Change from matching Chunk nodes (which live in doc instance) to MERGE-ing lightweight ChunkRef nodes (in concept instance):

```cypher
UNWIND $rels as rel
MERGE (c:ChunkRef {qualified_name: rel.chunk_qn, workspace: $workspace})
MERGE (concept:Concept {qualified_name: rel.concept_qn, workspace: $workspace})
MERGE (c)-[m:MENTIONS]->(concept)
SET m.frequency = rel.frequency, m.context = rel.context
```

Note: `MERGE` instead of `MATCH` for ChunkRef — creates the proxy node if it doesn't exist yet.

#### 3g. Add `_get_chunk_qns_for_document()` Helper

New helper to query chunk qualified names from the doc instance (needed for cross-instance cleanup):

```python
def _get_chunk_qns_for_document(
    self,
    document_path: str,
    doc_ingestor: MemgraphIngestor,
) -> list[str]:
    """Get chunk qualified names for a document from the doc instance."""
    cypher = """
    MATCH (d:Document {path: $doc_path, workspace: $workspace})
          -[:CONTAINS_CHUNK]->(c:Chunk)
    RETURN c.qualified_name AS chunk_qn
    """
    result = doc_ingestor.fetch_all(cypher, {
        "doc_path": document_path,
        "workspace": self.workspace,
    })
    return [r["chunk_qn"] for r in result]
```

#### 3h. Update `_cleanup_concepts_for_document()` — Cross-Instance

This method needs to coordinate cleanup across both instances. The doc instance has Document/Chunk nodes; the concept instance has ChunkRef/Concept/MENTIONS. **Both** ingestors are needed:

```python
def _cleanup_concepts_for_document(
    self,
    document_path: str,
    doc_ingestor: MemgraphIngestor,       # For querying chunk QNs
    concept_ingestor: MemgraphIngestor,   # For deleting ChunkRefs + Concepts
) -> None:
    """Remove orphaned concepts after document chunks are deleted.

    Step 1: Queries doc instance for chunk QNs belonging to the document.
    Step 2: Deletes ChunkRefs and orphaned Concepts from concept instance.
    """
    logger.info(f"Cleaning up concepts for document: {document_path}")

    # Step 1: Get chunk qualified names from doc instance
    chunk_qns = self._get_chunk_qns_for_document(document_path, doc_ingestor)
    if not chunk_qns:
        logger.debug(f"No chunks found for document: {document_path}")
        return

    # Step 2: Clean up concept instance
    cypher = """
    UNWIND $chunk_qns as chunk_qn
    MATCH (cr:ChunkRef {qualified_name: chunk_qn, workspace: $workspace})
    OPTIONAL MATCH (cr)-[m:MENTIONS]->(concept:Concept {workspace: $workspace})
    DELETE m, cr
    WITH concept
    WHERE concept IS NOT NULL
    WITH collect(DISTINCT concept) as concepts
    UNWIND concepts as concept
    OPTIONAL MATCH (:ChunkRef)-[remaining:MENTIONS]->(concept)
    WITH concept, remaining
    WHERE remaining IS NULL
    DETACH DELETE concept
    RETURN count(concept) as removed_count
    """
    result = concept_ingestor.fetch_all(cypher, {
        "chunk_qns": chunk_qns,
        "workspace": self.workspace,
    })
    removed_count = result[0].get("removed_count", 0) if result else 0
    logger.info(f"Cleaned up {removed_count} orphaned concepts for {document_path}")
```

Note: The `_get_chunk_qns_for_document()` query runs against the doc instance because that's where `Document` and `Chunk` nodes live. The cleanup Cypher runs against the concept instance because that's where `ChunkRef`, `Concept`, and `MENTIONS` live.

### 4. `codebase_rag/vector_store_memgraph.py` — Add Concept Mode

Add `is_concept: bool = False` parameter to `MemgraphBackend.__init__()`:

```python
def __init__(self, is_document: bool = False, is_concept: bool = False) -> None:
    self.is_document = is_document
    self.is_concept = is_concept
```

In `_create_connection()`, add concept instance routing:

```python
if self.is_concept:
    host = settings.CONCEPT_MEMGRAPH_HOST
    port = settings.CONCEPT_MEMGRAPH_PORT
    username = settings.CONCEPT_MEMGRAPH_USERNAME
    password = settings.CONCEPT_MEMGRAPH_PASSWORD
elif self.is_document:
    ...
```

Note: `is_concept` and `is_document` are mutually exclusive. Consider using an enum instead of two booleans, but for backward compatibility, keep both with validation.

### 5. `codebase_rag/document/graph_algorithms.py` — No Change Needed

`DocumentGraphAlgorithms` already accepts a generic `QueryProtocol` — it does not hardcode which instance it targets. The change is in the **callers** that instantiate `DocumentGraphAlgorithms`:

- **`codebase_rag/tools/document_query.py`** (line 74): Passes `router.doc_graph` — change to pass `router.concept_graph` (a new field on the router pointing to the concept instance).
- **`codebase_rag/shared/query_router.py`** (line 794): Passes `self.doc_graph` — change to pass `self.concept_graph`.

`QueryRouter` needs a new `concept_graph` field (alongside existing `code_graph` and `doc_graph`) for DocGraphTraversal intents to use the concept instance. `DocumentGraphAlgorithms` itself requires no modifications.

### 6. `codebase_rag/document/tools/document_search.py` — Update Queries

Any queries that traverse MENTIONS or concept-to-concept relationships need to target the concept instance. Currently, `document_search.py` uses `DOC_MEMGRAPH_VECTOR_INDEX_NAME` for chunk vector search (still doc instance). If it also queries concept relationships, those queries move to concept instance.

### 7. `codebase_rag/config.py` — New Settings

Add all `CONCEPT_MEMGRAPH_*` settings as listed in the Configuration Changes section above. Follow the existing pattern from `DOC_MEMGRAPH_*` and `JSON_MEMGRAPH_*`.

## Cleanup: Doc Instance Concept Data Removal

### Old Data in Doc Instance

After deploying the concept instance, the doc instance may still contain stale Concept nodes, Topic nodes, MENTIONS edges, concept-to-concept edges, and concept-related indexes from prior extractions. A one-time cleanup migration is needed:

```cypher
-- Run against doc instance (port 7688)
-- Step 1: Remove all concept/topic data
MATCH (:Chunk)-[m:MENTIONS]->(:Concept) DELETE m;
MATCH (:Chunk)-[m:MENTIONS]->(:Topic) DELETE m;
MATCH (c:Concept) DETACH DELETE c;
MATCH (t:Topic) DETACH DELETE t;

-- Step 2: Drop stale indexes (they consume memory)
DROP INDEX ON :Concept(qualified_name);
DROP INDEX ON :Concept(workspace);
DROP INDEX ON :Concept(entity_category);
DROP INDEX ON :Concept(entity_subtype);
DROP INDEX ON :Topic(qualified_name);
DROP INDEX ON :Topic(workspace);
DROP INDEX ON :HIERARCHICAL(verb);
DROP INDEX ON :COMPOSITIONAL(verb);
DROP INDEX ON :CONTEXTUAL(verb);
DROP INDEX ON :ATTRIBUTIVE(verb);
DROP INDEX ON :COMPARATIVE(verb);
DROP INDEX ON :SEQUENTIAL(verb);
DROP INDEX ON :CAUSAL(verb);
DROP INDEX ON :ANALOGICAL(verb);
DROP INDEX ON :RELATED_TO(verb);
```

Note: `DROP INDEX` commands may fail if the index was never created or is already dropped. The migration script should handle these gracefully (skip on error, log a debug message).

### Migration Script

Provide a CLI command or standalone script: `python -m codebase_rag.migrations.cleanup_doc_concepts`

This:
1. Connects to doc instance
2. Counts stale Concept/Topic nodes and related edges
3. Lists stale indexes found in doc instance
4. Asks for confirmation (unless `--yes` flag)
5. Deletes edges first, then DETACH DELETE nodes
6. Drops stale concept-related indexes (gracefully skipping any that don't exist)

## Backward Compatibility

### Graceful Degradation Without Concept Instance

When `DOC_CONCEPT_EXTRACTION_ENABLED=True` but no concept instance is configured (no `CONCEPT_MEMGRAPH_HOST` set, or connection fails):

1. Log a warning: "Concept extraction enabled but concept graph instance unavailable — concepts will not be stored"
2. Skip concept storage (concept extraction itself continues; results are discarded)
3. Do NOT fall back to storing concepts in doc instance (prevents silent data placement in wrong instance)

This is controlled by a new config flag:

```python
CONCEPT_MEMGRAPH_ENABLED: bool = True  # Set to False to skip concept storage
```

### Existing `DOC_CONCEPT_EXTRACTION_ENABLED` Behavior

If `DOC_CONCEPT_EXTRACTION_ENABLED=False`, nothing changes — no extraction, no storage, no concept instance needed.

If `DOC_CONCEPT_EXTRACTION_ENABLED=True` and `CONCEPT_MEMGRAPH_ENABLED=True` but concept instance is unreachable:
- Extraction runs (LLM calls happen)
- Storage fails gracefully with logged warnings
- The circuit breaker tracks concept instance failures
- Chunks are still indexed in doc instance (document retrieval works)

### Rollback

If the concept instance needs to be rolled back:
1. Set `CONCEPT_MEMGRAPH_ENABLED=False`
2. Concepts from the rollback period are lost (new extractions not stored)
3. Pre-existing concept data in the concept instance is untouched (can be queried, just won't get updates)
4. Old concept data in doc instance is already cleaned up (see migration)

## Implementation Phases

### Phase 1: Infrastructure (Docker + Config)
1. Add `memgraph-concept` service + `lab-concept` to `docker-compose.yaml`
2. Add `memgraph_concept_data` volume
3. Add all `CONCEPT_MEMGRAPH_*` settings to `config.py` (including `CONCEPT_MEMGRAPH_ENABLED`)
4. Add `CONCEPT_MEMGRAPH_*` entries to `.env.example` and `.env`
5. Add concept port timeout routing + fix hardcoded graph_type in `graph_service.py` (2 methods)
6. Add `connect_concept_memgraph()` to `main.py`

### Phase 2: Storage Migration (document_updater.py)
7. Add concept instance connection params to `DocumentGraphUpdater.__init__()` + create concept ingestor in `run()`/`run_async()`
8. Pass `concept_ingestor` alongside doc `ingestor` to `_process_document()` / `_process_document_async()` and downstream methods
9. Update `_extract_and_store_concepts()` signature to accept `concept_ingestor`; route all concept writes to it
10. Call `_ensure_concept_indexes(concept_ingestor)` instead of doc ingestor; add ChunkRef indexes
11. Update `_create_mentions_batch()` to use `MERGE (c:ChunkRef {...})` instead of `MATCH (c:Chunk {...})`
12. Add `_get_chunk_qns_for_document()` helper for cross-instance cleanup
13. Update `_cleanup_concepts_for_document()` to accept both ingestors and query doc instance for chunk QNs, then clean concept instance
14. Update `_delete_document_nodes()` call site: pass `concept_ingestor` to `_cleanup_concepts_for_document()`

### Phase 3: Query Routing + Graph Algorithms
15. Add `concept_graph` field to `QueryRouter` (alongside `code_graph` and `doc_graph`)
16. Update `document_query.py` and `query_router.py` to pass concept instance for DocGraphTraversal intents
17. Add `is_concept` mode to `MemgraphBackend` (only if concept vector search is needed — see Open Question 2)

### Phase 4: Cleanup + Migration
18. Create migration script to clean stale Concept/Topic nodes, edges, and indexes from doc instance
19. Add integration test: doc indexing + concept extraction → concepts land in concept instance, not doc instance
20. Add integration test: concept graph traversal queries work against concept instance

### Phase 5: Testing
21. Unit tests for `connect_concept_memgraph()`
22. Unit tests for `_create_mentions_batch()` with ChunkRef MERGE
23. Unit tests for `_get_chunk_qns_for_document()` with doc instance
24. Unit tests for `_cleanup_concepts_for_document()` with cross-instance coordination
25. Integration test: full extract → store → query → cleanup cycle against concept instance
26. Verify old doc instance no longer has concept data or indexes after migration

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `docker-compose.yaml` | Add | `memgraph-concept` service, `lab-concept` service, `memgraph_concept_data` volume |
| `.env` | Add | `CONCEPT_MEMGRAPH_*` entries |
| `.env.example` | Add | `CONCEPT_MEMGRAPH_*` entries with documentation comments |
| `codebase_rag/config.py` | Add | `CONCEPT_MEMGRAPH_*` settings + `concept_memgraph` property |
| `codebase_rag/main.py` | Add | `connect_concept_memgraph()` function |
| `codebase_rag/services/graph_service.py` | Edit | Add concept port timeout routing + fix hardcoded graph_type in error guidance (2 methods) |
| `codebase_rag/document/document_updater.py` | Edit | Add concept instance params to `__init__`, create concept ingestor in `run()`/`run_async()`, update `_create_mentions_batch()` for ChunkRef, add `_get_chunk_qns_for_document()`, update `_cleanup_concepts_for_document()` for cross-instance, update all call sites |
| `codebase_rag/vector_store_memgraph.py` | Edit | Add `is_concept` mode |
| `codebase_rag/document/graph_algorithms.py` | None | No changes — already accepts generic `QueryProtocol` |
| `codebase_rag/tools/document_query.py` | Edit | Pass `concept_graph` instead of `doc_graph` to `DocumentGraphAlgorithms` |
| `codebase_rag/shared/query_router.py` | Edit | Add `concept_graph` field; route DocGraphTraversal to concept instance |
| `codebase_rag/document/tools/document_search.py` | None | No changes — chunk vector search only, no concept queries |
| `codebase_rag/migrations/cleanup_doc_concepts.py` | **New** | One-time migration script with index cleanup |
| `codebase_rag/tests/document/test_concept_extraction.py` | Edit | Update tests for concept instance |
| `codebase_rag/tests/test_concept_storage.py` | **New** | Tests for concept instance storage, ChunkRef, cross-instance cleanup |

## Success Criteria

1. ✅ Concept extraction runs and stores all concept data in `memgraph-concept` (port 7690), not `memgraph-doc` (port 7688)
2. ✅ `memgraph-doc` contains only Document/Chunk nodes after migration — no Concept, Topic, MENTIONS, or concept-to-concept edges
3. ✅ `MENTIONS` edges in concept instance use `ChunkRef` proxy nodes (not Chunk nodes)
4. ✅ Document cleanup cascades correctly: deleting a document from doc instance removes corresponding ChunkRefs and orphaned Concepts from concept instance
5. ✅ Concept graph queries (traversal, path finding) work against concept instance without touching doc instance
6. ✅ Document retrieval (vector search on chunks) works against doc instance without touching concept instance
7. ✅ Existing `DOC_CONCEPT_EXTRACTION_ENABLED=False` behavior is unchanged (no extraction, no concept writes)
8. ✅ Graceful degradation when concept instance is unavailable: doc indexing succeeds, concept storage skips with warning
9. ✅ Backward compatible: `DOC_CONCEPT_*` timeout/retry settings unchanged, extraction logic unchanged
10. ✅ Docker Compose starts all 4 instances with correct port allocation and health checks

## Resolved Questions

1. **Topic nodes** (resolved v1.1.0) — `Topic` nodes exist alongside `Concept` nodes in `_ensure_concept_indexes()` (lines 2531-2532). They are part of the concept taxonomy and serve as broader thematic groupings. Decision: **move Topic nodes to the concept instance** alongside Concept nodes. Both node types share the same lifecycle and belong in the concept knowledge graph. `_ensure_concept_indexes()` already creates indexes for both (Concept + Topic). The migration script removes stale Topic nodes from the doc instance; new extractions create them in the concept instance.

## Open Questions

1. **Cross-instance transaction consistency** — Document deletion requires: (a) delete chunks from doc instance, (b) delete ChunkRefs from concept instance. These are two separate Memgraph transactions with no distributed coordination. If step (a) succeeds but step (b) fails, orphaned ChunkRefs remain. Mitigation: ChunkRefs are idempotent (they get re-used if the chunk is re-created), and a periodic cleanup job can remove ChunkRefs with no corresponding chunk (match ChunkRef QNs against doc instance's Chunk nodes). Is this acceptable, or do we need a compensating transaction pattern?

2. **Concept instance vector search** — Should Concept nodes get vector embeddings (for semantic concept search), or is the concept graph traversed purely via relationship edges? If embeddings are needed, the `MemgraphBackend` needs `is_concept` support. If not, skip Phase 3 vector store changes.

3. **Existing concept data migration** — Should existing Concept nodes and edges be *migrated* from doc instance to concept instance, or simply discarded (re-extracted on next doc processing)? Migration preserves historical data; discarding is simpler but loses concept graph state.

---

## Version

v1.1.0 | 2026-04-25 | Review fixes: class name correction, ingestor pattern fix, cross-instance cleanup, graph_type routing, Topic resolution, stale index cleanup
