# GraphRAG Data Model & Ingestion Fix Design Specification
## Executive Summary
This repository already uses three separate Memgraph backends: the code graph on `MEMGRAPH_HOST`/`MEMGRAPH_PORT`, the document graph on `DOC_MEMGRAPH_HOST`/`DOC_MEMGRAPH_PORT`, and the JSON graph on `JSON_MEMGRAPH_HOST`/`JSON_MEMGRAPH_PORT`. Embeddings are stored directly on graph nodes. The optimal design therefore is not to add embedding nodes, UUID join tables, or direct cross-database edges. The correct fix is to preserve the split-graph architecture, repair legacy graph inconsistencies, and add orchestration metadata that lets the query layer correlate results safely.

## Ground Truth Alignment
- Code ingestion already creates `Project`, `Package`, `Folder`, `File`, `Module`, `Function`, `Class`, and `Method` nodes through `GraphUpdater` and the parser processors.
- Document ingestion already creates `Document`, `Section`, and `Chunk` nodes through `DocumentGraphUpdater`.
- JSON ingestion already ingests canonical `metadata/entities/relationships` payloads into a dedicated JSON graph as `JsonEntity` nodes and typed relationships.
- Code and document vector retrieval already operate on node-owned `embedding` properties; there is no external embedding identity layer to reconcile.

## Root-Cause Gaps
| ID | Gap | Current State | Root Cause | Required Fix |
|----|-----|---------------|------------|--------------|
| 1 | Legacy code graphs can still contain `Function` nodes without incoming `DEFINES` relationships. | Current ingestion creates parent relationships, but old data is not repaired automatically. | Historical ingestion drift and partial re-indexes. | Add an idempotent post-ingestion repair query scoped to the current project. |
| 2 | Document extraction finds code references, but indexing stores only raw reference text on `Document` nodes. | `Chunk` nodes do not carry resolved code metadata, so semantic document hits cannot bridge back into code. | No resolution step against the code graph during document indexing. | Build a code reference index from the code graph and persist resolved references on documents and chunks. |
| 3 | `BOTH_MERGED` mode merges code and document answers, but does not expand document hits into referenced code entities. | The query layer ignores document-side reference metadata. | The merged response has no code-context bridge from document chunks. | Expose resolved references in document search results and append code context by `qualified_name` lookup. |
| 4 | `validate_ingestion_quality` accepts multi-label input but emits invalid Memgraph syntax. | `GraphUpdater.run()` passes `Function|Method|Class`, which produces `MATCH (n:Function|Method|Class)` in the health checker. | Validation drift from Memgraph query constraints already enforced elsewhere in the repo. | Convert label expressions to `ANY(label IN labels(n) ...)` filters. |
| 5 | The original spec assumed arbitrary JSON object decomposition into `JSONObject` / `JSONField` / `JSONValue` nodes. | The shipped JSON pipeline intentionally models canonical entity/relationship datasets in a dedicated JSON graph. | Spec mismatch, not an implementation bug. | Keep the current JSON data model unless product requirements explicitly change. |

## Implemented Design
### 1. Code Graph Repair
- Add an idempotent repair pass after code ingestion flushes to Memgraph.
- For each `Function` without an incoming `DEFINES`, derive the parent from `qualified_name`.
- Reattach to a parent `Function` when the function is nested, otherwise to the containing `Module`.
- Scope the repair by `project_name` so unrelated projects are untouched.

### 2. Document-to-Code Bridge Metadata
- Load a code reference index from the code graph using existing `qualified_name` and `name` properties.
- Resolve only unambiguous references.
- Persist on `Document` nodes:
  - `code_references`
  - `resolved_code_references`
  - `resolved_code_reference_count`
- Persist the same metadata on `Chunk` nodes so semantic document hits can bridge back to code at retrieval time.

### 3. Query Orchestration
- Extend document search results to return resolved code references and chunk end lines.
- In `DOCUMENT_ONLY`, surface resolved code references as document metadata without querying the code graph.
- In `BOTH_MERGED`, look up the resolved `qualified_name` values in the code graph and append the referenced code entities as explicit cross-graph context.

### 4. Validation
- Fix ingestion quality checks to support multi-label validation without invalid Cypher syntax.
- Continue using node-owned embeddings. No `HAS_EMBEDDING` nodes, UUID joins, or separate embedding records are introduced.

## Non-Goals
- No direct `REFERENCES_CODE` edges between the document graph and the code graph, because those graphs live in separate Memgraph instances.
- No separate embedding nodes or shared UUID mapping layer.
- No arbitrary JSON object decomposition for the canonical JSON ingestion payload format.

## Validation & Success Criteria
1. A code re-index repairs legacy missing `Module` / `Function` -> `DEFINES` -> `Function` relationships for the indexed project.
2. Document and chunk nodes retain both raw references and resolved code qualified names when resolution is unambiguous.
3. Document semantic search returns resolved code references alongside chunk content and line metadata.
4. `BOTH_MERGED` mode surfaces code context referenced by matching document chunks.
5. Ingestion quality checks run without invalid multi-label Cypher.

## Rollout
1. Re-index the code graph once to apply the legacy relationship repair pass.
2. Re-index documents to populate resolved code reference metadata on `Document` and `Chunk` nodes.
3. Use merged query mode for cross-modal investigation. JSON ingestion remains unchanged.