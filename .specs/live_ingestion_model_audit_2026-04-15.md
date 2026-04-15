# Live Ingestion Model Audit And Remediation Spec

## Metadata
| Field | Value |
|---|---|
| Spec ID | SPEC-LIVE-INGEST-AUDIT-20260415 |
| Status | Implementation Ready |
| Date | 2026-04-15 |
| Scope | Code graph, document graph, JSON graph, vector retrieval |
| Repository | code-graph-rag |

## 1. Audit Scope And Evidence Snapshot

This specification is based on a live audit of the currently ingested graphs plus direct code-path inspection.

### Live state observed
| Graph | Nodes | Relationships | Key observations |
|---|---:|---:|---|
| Code | 8057 | 14423 | 7287 node embeddings present, 1024 dimensions, 1 Project node, 0 File nodes |
| Document | 996 | 1135 | 35 Documents, 787 Sections, 174 Chunks, all stored Chunk embeddings are present |
| JSON | 0 | 0 | No datasets or entities present |

### Validation evidence collected
1. Built-in health checks passed for disconnected production nodes, required properties, and embedding model correlation on the code graph.
2. Direct graph queries showed the code graph has no isolated nodes and no internal nodes outside the project component.
3. Direct graph queries showed the document graph has no orphan Sections or Chunks.
4. Direct graph queries showed the JSON graph is empty.
5. Dry-run validation of `sample_json_ingest.json` failed under the current official schema.
6. Targeted test run of `codebase_rag/tests/test_json_ingestion.py` failed with 3 failing tests.
7. Manual Memgraph `vector_search.search()` worked against an existing code vector index, while the application backend returned an empty result set for the same embedding.

## 2. Non-Issues Confirmed During Audit

These cases looked suspicious initially but are not defects in the current model:

1. The 80 Function nodes not directly attached to a Module are nested functions. They are linked by `(:Function)-[:DEFINES]->(:Function)` and are not true orphans.
2. The 146 Module nodes without `CONTAINS_MODULE` parents are external/import-derived modules such as `os`, `sys`, and `typing`; they are not internal repository modules.
3. Two of the three documents without Chunks are tiny `.txt` files with 0 or 2 words. Those are expected to produce no semantic chunks under the current minimum chunk token filter.

## 3. Confirmed Issues

### Issue A: Code graph is missing the File layer entirely

#### Severity
Critical

#### Evidence
1. Live query result: `MATCH (f:File) RETURN count(f)` returned `0`.
2. The current code graph contains Modules, Functions, Classes, Folders, Packages, and Project nodes, but no File nodes.
3. `GraphUpdater._process_single_file()` calls `process_generic_file()`, but the production parallel path in `GraphUpdater._process_files()` uses `_process_worker_chunk()`, and `_process_worker_chunk()` never calls `process_generic_file()`.

#### Root cause
The serial and parallel ingestion paths have diverged. File node creation exists only in the serial helper path, while the actual worker-based path only emits Module, Function, Class, Method, and relationship data.

#### Impact
1. File-level queries and traversals are structurally incomplete.
2. The intended container tree is missing one full entity layer.
3. Any downstream logic expecting `:File` nodes, `CONTAINS_FILE` relationships, or file-level metadata in the graph cannot work correctly.
4. The built-in health checks miss this because they do not verify expected label presence.

#### Design decision
Restore File nodes in the worker path and remove the serial/parallel ingestion drift.

#### Implementation approach
1. Extract the shared per-file graph emission logic into one helper used by both `_process_single_file()` and `_process_worker_chunk()`.
2. In that shared helper, always emit the File node and `CONTAINS_FILE` edge for every eligible file, regardless of whether parser-based definition extraction succeeds.
3. Preserve the current Module ingestion flow so the change is additive and low risk.
4. Add an optional explicit File-to-Module edge for internal source files after File restoration.

#### Required code changes
1. Refactor `codebase_rag/graph_updater.py` so the worker path calls the same file-node creation logic as the serial path.
2. Ensure worker-collected node and relationship results include File nodes and `CONTAINS_FILE` relationships.
3. Add regression tests that run the parallel ingestion path and assert File nodes exist.

#### Acceptance criteria
1. A full ingest of this repository produces a non-zero File node count.
2. The number of internal Module nodes and internal source File nodes is no longer separated by a missing entity layer.
3. Queries against `:File` return current repository files.
4. Health/audit checks fail if internal Modules exist but File count is zero.

### Issue B: Stored embeddings exist, but application-level vector search is disabled by false capability detection

#### Severity
Critical

#### Evidence
1. Live code graph stores 7287 embeddings with dimension 1024.
2. Manual query `CALL vector_search.search('function_embedding_index', 3, $embedding)` succeeds and returns the source Function as the top hit.
3. The application path `search_embeddings()` returns `[]` for the same embedding.
4. Logs show `MemgraphBackend.search()` exits early because capability detection incorrectly concludes vector search is unsupported.

#### Root cause
`MemgraphQueryGenerator._detect_capabilities()` probes procedure support by calling `vector_search.search()` against a non-existent test index. That error is treated as lack of procedure support, so `supports_vector_search` becomes false even on a Memgraph instance where the procedure works.

#### Impact
1. Semantic retrieval can silently fail even though embeddings and indexes are present.
2. Code semantic search is effectively disabled.
3. The same shared capability logic also threatens document and future JSON semantic retrieval.

#### Design decision
Detect procedure support by classifying procedure-not-found errors separately from missing-index errors.

#### Implementation approach
1. Update `codebase_rag/graph/query_generator.py` to treat errors such as missing vector index as evidence that the procedure exists.
2. Only mark procedure support false when the error clearly indicates that the procedure itself does not exist.
3. Add a post-initialize smoke test using a real existing vector index when available.
4. Keep the early-return guard in `MemgraphBackend.search()`, but ensure capability detection is correct first.

#### Required code changes
1. Adjust `MemgraphQueryGenerator._detect_capabilities()` in `codebase_rag/graph/query_generator.py`.
2. Add a focused test that mocks Memgraph responses for:
   - procedure exists but index is missing
   - procedure missing entirely
   - procedure succeeds
3. Add an integration test that stores one embedding and confirms `search_embeddings()` returns the node itself as the top hit.

#### Acceptance criteria
1. `search_embeddings()` returns non-empty results for a stored embedding on Memgraph 3.9.0.
2. Capability detection no longer logs false vector-search unsupported warnings on working indexes.
3. Retrieval works for both code and document backends that use the shared detection logic.

### Issue C: PDF extraction produces Sections but almost no Chunks, so PDF content is not semantically retrievable

#### Severity
High

#### Evidence
1. The document graph contains one PDF document with 622 Sections and 175987 words but 0 stored Chunks.
2. In-memory reproduction of extraction plus chunking on `optimize/EXPERT_PYTHON_PROGRAMMING_FOURTH_EDITION.pdf` produced only a single 3-token preamble chunk.
3. The PDF extractor builds one `ExtractedSection` per page, but each page Section is created with `start_line == end_line == page_index`.
4. The chunker assumes `start_line` is the header line and `section.content` begins at `start_line + 1`, so every PDF page looks like a header-only section with no body.

#### Root cause
The PDF extractor and semantic chunker use incompatible line semantics. PDF sections are emitted as page buckets, but their `start_line` and `end_line` metadata do not span the actual page content.

#### Impact
1. Large PDF content is indexed structurally but not semantically.
2. No Chunk embeddings are created for PDF content, so semantic document retrieval misses that content entirely.
3. The document graph gives a false impression of successful ingestion because Document and Section nodes exist.

#### Design decision
Fix the PDF extractor as the primary source-of-truth repair and add a defensive chunker fallback for flat-content sections.

#### Implementation approach
1. In `codebase_rag/document/extractors/pdf_extractor.py`, compute cumulative line offsets per page and set page Section line spans to cover the actual extracted text.
2. In `codebase_rag/document/chunking.py`, add a fallback path for sections where `section.content` is non-empty but the computed own-content span is empty.
3. Re-index the PDF after the fix and verify Chunk and embedding creation.

#### Required code changes
1. Update both `_extract_with_pdfplumber()` and `_extract_with_pypdf2()` to maintain cumulative page line offsets.
2. Add a chunker regression test using a synthetic flat section with non-empty content and collapsed line metadata.
3. Add an integration test that confirms a PDF with multiple pages yields more than zero Chunks.

#### Acceptance criteria
1. The audited PDF produces non-zero Chunks after re-index.
2. Those Chunks receive embeddings and `BELONGS_TO_SECTION` relationships.
3. Tiny text files may still produce zero Chunks if below threshold, but large PDFs no longer do.

### Issue D: JSON graph is empty because the sample and tests are still using a legacy shape that the current schema rejects

#### Severity
Critical

#### Evidence
1. Live JSON graph node count is `0`.
2. `sample_json_ingest.json` fails dry-run validation under the current official schema.
3. `codebase_rag/tests/test_json_ingestion.py` currently fails in 3 places.
4. The current schema requires top-level `entity.name`, `relationship.source`, `relationship.target`, and `relationship.relationship`, but the sample/tests still reflect an older shape in parts of the JSON contract.

#### Root cause
The canonical JSON ingestion contract moved to the official `ingestion_schema.json`, but the sample data and tests were not fully migrated.

#### Impact
1. JSON ingestion appears to be supported, but the shipped sample path does not ingest.
2. The JSON graph remains empty even when users try the provided sample.
3. Tests now validate outdated assumptions instead of the live contract.

#### Design decision
Keep the official schema as the canonical contract and update shipped artifacts to match it. Backward compatibility, if desired, should be explicit rather than implicit.

#### Implementation approach
1. Update `sample_json_ingest.json` so every entity includes top-level `name`.
2. Update `codebase_rag/tests/test_json_ingestion.py` to assert the canonical shape instead of legacy keys.
3. Update JSON ingestion docs and examples to stop using legacy relationship field names and nested-only entity names.
4. If backwards compatibility is still required, add an explicit pre-validation normalizer behind a clearly named option rather than silently transforming input.

#### Required code changes
1. Update `sample_json_ingest.json`.
2. Update `codebase_rag/tests/test_json_ingestion.py` and any doc/spec examples still asserting the legacy shape.
3. Add one canonical happy-path integration test that ingests the sample in dry-run mode and expects non-zero processed counts.

#### Acceptance criteria
1. Dry-run ingestion of `sample_json_ingest.json` succeeds.
2. `codebase_rag/tests/test_json_ingestion.py` passes.
3. A real JSON ingest produces non-zero nodes in the JSON graph.

### Issue E: JSON embedding persistence is internally inconsistent and cannot correctly correlate embeddings to JSON graph entities

#### Severity
Critical

#### Evidence
1. `json_ingestion.py` imports the shared vector backend on module import, and a JSON dry-run initializes the code vector backend.
2. `ingest_entities()` passes a string `unique_id` into `vector_store.add_item()`, but `MemgraphBackend.add_item()` matches `id(n) = $node_id`, which requires an internal Memgraph integer node ID.
3. `ingest_relationships()` accepts `rel_embeddings` but never persists them.
4. `delete_dataset()` calls `vector_store.delete_by_metadata(...)`, but the Memgraph backend does not implement that API.

#### Root cause
The JSON path is reusing a vector abstraction that was designed for code/document node IDs and fixed label indexes. That abstraction does not match the JSON entity model, JSON database scope, or Memgraph's inability to vector-index relationships directly.

#### Impact
1. Even after JSON entities start ingesting, entity embeddings cannot be reliably attached to the correct JSON graph nodes through the current API.
2. Relationship embeddings are generated but discarded.
3. Dry-run and import-time behavior can touch the wrong backend.
4. Dataset deletion is not coherent with Memgraph-native embedding storage.

#### Design decision
Do not use the shared `vector_store.py` path for JSON ingestion. Persist JSON entity embeddings directly in the JSON graph and make relationship embeddings an explicit later-phase feature.

#### Implementation approach
1. Remove the shared `vector_store` dependency from `codebase_rag/json_ingestion.py`.
2. During entity merge, always add a stable base label such as `JsonEntity` in addition to user-provided labels.
3. Return `id(n)` from JSON entity MERGE queries and set embedding properties directly on the JSON graph nodes.
4. Add one JSON vector index on `:JsonEntity(embedding)` and route JSON semantic search through the JSON Memgraph instance.
5. Stop generating relationship embeddings in phase 1.
6. If relationship semantic retrieval is a hard requirement, introduce reified relationship nodes in a later phase instead of pretending edge embeddings are searchable.
7. Fix `delete_dataset()` to delete only JSON graph data and rely on node deletion to remove Memgraph-stored embeddings with the nodes.

#### Required code changes
1. Refactor `codebase_rag/json_ingestion.py` to set embeddings on merged JSON nodes instead of calling `vector_store.add_item()`.
2. Introduce JSON vector index initialization for the stable `JsonEntity` label.
3. Remove unused relationship embedding persistence parameters or gate them behind a future reified-relationship design.
4. Fix `delete_dataset()` result handling and remove the unsupported `delete_by_metadata()` call.

#### Acceptance criteria
1. A successful JSON ingest stores entity embeddings on JSON graph nodes.
2. JSON entity semantic search operates against the JSON Memgraph instance, not the code graph.
3. Dataset deletion works without backend-specific metadata delete calls.
4. Phase 1 no longer claims to support persisted relationship embeddings.

### Issue F: The current health checks are too weak to detect real ingestion model defects

#### Severity
Medium

#### Evidence
1. Built-in health checks passed while the code graph still had no File nodes.
2. Built-in health checks passed while the document graph contained a large PDF with no semantic chunks.
3. Built-in health checks do not validate JSON ingestion success at all.

#### Root cause
The health checks focus on connectivity, required properties, and embedding model metadata, but they do not assert expected entity-layer presence, chunk coverage, or vector retrieval functionality.

#### Impact
Operators can receive a fully green health report from an incompletely modeled ingest.

#### Design decision
Add a post-ingestion audit layer with graph-shape, coverage, and retrieval smoke tests.

#### Implementation approach
1. Extend the existing audit tooling to add graph-specific assertions:
   - Code graph: internal File count, internal Module container coverage, semantic retrieval smoke test.
   - Document graph: large-document chunk coverage, Chunk embedding coverage, no orphan Chunks.
   - JSON graph: processed dataset count, sample dry-run validity, JSON entity embedding coverage.
2. Produce machine-readable and human-readable audit summaries.
3. Fail CI or CLI validation when critical checks fail.

#### Acceptance criteria
1. A green audit implies the File layer exists, large documents chunk correctly, and vector retrieval is operational.
2. Critical regressions are surfaced immediately after ingest.

## 4. Ordered Remediation Plan

### Phase 1: Restore correctness of currently broken production behavior
1. Fix Memgraph vector capability detection.
2. Restore File nodes in the parallel code ingestion path.
3. Repair PDF extractor/chunker compatibility.
4. Update shipped JSON sample and failing JSON tests to the canonical schema.

### Phase 2: Make JSON ingestion structurally sound end-to-end
1. Remove JSON dependence on the shared code vector backend.
2. Persist JSON entity embeddings directly in the JSON graph on a stable `JsonEntity` label.
3. Fix dataset deletion and remove unsupported vector backend calls.
4. Disable or defer relationship embeddings until a reified relationship model exists.

### Phase 3: Prevent recurrence
1. Add the stronger post-ingestion audit suite.
2. Add regression tests for File nodes, vector retrieval, PDF chunk coverage, and JSON sample ingestion.
3. Document the canonical JSON contract and deprecation path for any legacy compatibility helpers.

## 5. Reindex And Reingest Requirements

After the fixes above are implemented, the current live data must be refreshed:

1. Re-index the code graph to backfill File nodes.
2. Re-index the document graph to regenerate PDF chunks and embeddings.
3. Re-ingest the JSON sample or target dataset after the canonical sample/tests are fixed and the JSON embedding path is corrected.

## 6. Final Expected State

The target end state after remediation is:

1. Code graph contains Project, Package, Folder, File, Module, and symbol nodes with no missing File layer.
2. Code semantic search returns non-empty results on live Memgraph indexes.
3. Large PDFs produce Chunks and Chunk embeddings, not only Section nodes.
4. JSON ingestion succeeds for the shipped sample and writes data into the JSON graph.
5. JSON entity embeddings are stored in the JSON graph itself and searched there.
6. Health checks fail loudly when any of the above invariants regress.