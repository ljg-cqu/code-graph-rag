# Code Graph Data Modeling Quality Fix - Reviewed Alignment Revision

## Status

- Reviewed on 2026-04-16 against the current implementation in `definition_processor.py`, `call_processor.py`, `graph_updater.py`, `json_content_processor.py`, `document_updater.py`, `json_ingestion.py`, `vector_store_memgraph.py`, `types_defs.py`, `constants.py`, and `cypher_queries.py`.
- This document supersedes stale parts of the earlier v2.0 draft that described already-shipped behavior as still pending.
- Goal: define the correct data-model contract for the code graph, document graph, canonical JSON graph, and their corresponding vector indexes; distinguish already-implemented fixes from the remaining work needed for a fully sound model.

## Executive Summary

This repository does not maintain a single unified graph. It maintains three graph domains:

1. A **code graph** in the default Memgraph instance for source code structure, call/import relationships, and arbitrary JSON file structure discovered during code indexing.
2. A **document graph** in the document Memgraph instance for `Document`, `Section`, and `Chunk` nodes.
3. A **canonical JSON graph** in the JSON Memgraph instance for user-provided `JsonEntity` datasets and dataset-scoped relationships.

All vector storage is Memgraph-native. There is no supported external vector database backend.

After reviewing the current codebase, the original problem set breaks down into two categories:

| # | Area | Current State | Required Action |
|---|------|---------------|-----------------|
| 1 | File <-> Module relationships | Already implemented in code | Backfill legacy graphs only |
| 2 | Builtin `CALLS` noise | Already implemented in code | Clean historical builtin `CALLS` edges only |
| 3 | Arbitrary JSON file structure in the code graph | Partially implemented | Correct the structural model before considering it complete |
| 4 | Project deletion for code-graph JSON structure | Current query is too broad and misses `HAS_VALUE` traversal | Fix query |
| 5 | Code embedding validation coverage | Current post-ingestion validation covers only a subset of embeddable labels | Expand validation scope |

The most important remaining design work is **not** file-module linking or builtin filtering. Those are already present. The real remaining gap is making arbitrary JSON-file structure semantically correct, collision-safe, deletion-safe, and clearly separated from the canonical JSON entity graph.

---

## 1. Architecture Baseline

### 1.1 Graph Domains

| Domain | Memgraph Instance | Primary Nodes | Primary Relationships | Embeddings | Notes |
|--------|-------------------|---------------|-----------------------|------------|-------|
| Code graph | `MEMGRAPH_HOST:MEMGRAPH_PORT` | `Project`, `Package`, `Folder`, `File`, `Module`, code definition nodes, arbitrary JSON structure nodes | `CONTAINS_*`, `DEFINES`, `DEFINES_METHOD`, `IMPORTS`, `CALLS`, JSON structure edges | Embeddings on embeddable code nodes only; stored in the same code graph | Arbitrary JSON file structure belongs here, not in the canonical JSON graph |
| Document graph | `DOC_MEMGRAPH_HOST:DOC_MEMGRAPH_PORT` | `Document`, `Section`, `Chunk` | `CONTAINS_SECTION`, `HAS_SUBSECTION`, `CONTAINS_CHUNK`, `BELONGS_TO_SECTION` | Embeddings on `Chunk` only | Cross-graph code linkage is metadata-only via resolved qualified names |
| Canonical JSON graph | `JSON_MEMGRAPH_HOST:JSON_MEMGRAPH_PORT` | `JsonEntity` | Dataset-scoped dynamic relationship types from ingestion payloads | Embeddings on `JsonEntity` only | This is the authoritative graph for user-supplied JSON entity/relationship datasets |

### 1.2 Vector Index Contract

The vector model is graph-local and must be described explicitly:

- The **code graph** uses Memgraph native vectors stored on embeddable code nodes in the same graph. The current backend creates **per-label vector indexes** such as `function_embedding_index`, `method_embedding_index`, and similar label-specific indexes. The config field `MEMGRAPH_VECTOR_INDEX_NAME` is not the canonical runtime index identifier for code embeddings and should not be treated as the schema contract.
- The **document graph** uses a single graph-local vector index named `settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME` on `Chunk(embedding)`.
- The **canonical JSON graph** uses a single graph-local vector index named `settings.JSON_MEMGRAPH_VECTOR_INDEX_NAME` on `JsonEntity(embedding)`.

### 1.3 Non-Negotiable Invariants

1. There are no direct runtime relationships between the code graph and the document graph, or between the code graph and the canonical JSON graph.
2. Document-to-code linkage is represented by metadata fields such as `resolved_code_references`, not by cross-database edges like `REFERENCES_CODE`.
3. Arbitrary JSON file structure discovered during code indexing is **derived data** and is disposable. If its identity scheme changes, the correct remediation is full re-indexing, not manual migration.
4. Canonical JSON entity ingestion is a separate product surface from arbitrary JSON-file structure extraction. The two must never be conflated.
5. Vector indexes must follow graph-local lifecycle rules. `clean_database()` removes nodes and relationships but does not remove vector indexes, so dimension mismatches must continue to trigger index recreation where already implemented.

---

## 2. Code Graph Corrections

### 2.1 File <-> Module Relationships

#### Current State

This fix is already implemented.

- `DefinitionProcessor.process_file()` creates:
  - `File -[:CONTAINS_MODULE]-> Module`
  - `Module -[:BELONGS_TO_FILE]-> File`
- `RelationshipType.BELONGS_TO_FILE` exists.
- `RELATIONSHIP_SCHEMAS` already includes both `CONTAINS_MODULE` from `File` and `BELONGS_TO_FILE` from `Module`.
- Unit tests already cover the emitted relationships.

#### Correct Invariant

The invariant is:

- Every **internal module created from an indexed repository file** must have exactly one `BELONGS_TO_FILE` edge and one corresponding incoming `CONTAINS_MODULE` edge from the matching `File` node.
- **External or import-only `Module` nodes are exempt**. They may exist without a matching `File` node.

The original draft's success criterion of "all modules must belong to a file" is too strong and is incorrect for the current codebase.

#### Legacy Backfill

Existing graphs created before this fix may still be missing the relationships. The safe join key is `Module.path == File.path` for internal modules.

```cypher
MATCH (m:Module)
WHERE (m.qualified_name = $project_name OR m.qualified_name STARTS WITH ($project_name + '.'))
  AND (m.is_external IS NULL OR m.is_external = false)
  AND m.path IS NOT NULL
MATCH (f:File {path: m.path})
MERGE (f)-[:CONTAINS_MODULE]->(m)
MERGE (m)-[:BELONGS_TO_FILE]->(f);
```

#### Verification

```cypher
MATCH (m:Module)
WHERE (m.qualified_name = $project_name OR m.qualified_name STARTS WITH ($project_name + '.'))
  AND (m.is_external IS NULL OR m.is_external = false)
  AND m.path IS NOT NULL
  AND NOT (m)-[:BELONGS_TO_FILE]->(:File)
RETURN count(m) AS internal_orphan_modules;
-- Expected: 0

MATCH (f:File {path: 'codebase_rag/config.py'})
-[:CONTAINS_MODULE]->(m:Module)
-[:DEFINES]->(c:Class)
RETURN c.name, c.qualified_name;
```

No further schema change is required here.

---

### 2.2 Builtin Function Noise in `CALLS`

#### Current State

This fix is already implemented.

- `settings.INCLUDE_BUILTIN_CALLS` exists and defaults to `False`.
- `CallProcessor` filters `CALLS` edges whose target qualified name starts with `builtin.` before calling `ensure_relationship_batch()`.
- Builtin nodes may still be created and marked with `is_builtin = true`; the graph simply avoids connecting normal `CALLS` edges to them by default.
- Unit tests already cover both default filtering and opt-in inclusion behavior.

#### Correct Invariant

The invariant is:

- When `INCLUDE_BUILTIN_CALLS = false`, normal ingestion must not create `CALLS` edges whose target qualified name starts with `builtin.`.
- Builtin nodes may remain in the graph as explicit nodes for resolution/debugging, but they must not dominate call-chain traversal.
- When `INCLUDE_BUILTIN_CALLS = true`, builtin `CALLS` edges are allowed.

#### Historical Cleanup

Graphs created before the filtering change may still contain old builtin `CALLS` edges. Those should be cleaned once.

```cypher
MATCH ()-[r:CALLS]->(t)
WHERE t.qualified_name STARTS WITH 'builtin.'
DELETE r;
```

#### Verification

```cypher
MATCH ()-[r:CALLS]->(t)
WHERE t.qualified_name STARTS WITH 'builtin.'
RETURN count(r) AS builtin_calls_remaining;
-- Expected: 0 when INCLUDE_BUILTIN_CALLS = false and historical cleanup has been run

MATCH (caller)-[:CALLS]->(callee)
WHERE NOT caller.qualified_name STARTS WITH 'builtin.'
  AND NOT callee.qualified_name STARTS WITH 'builtin.'
RETURN count(*) AS meaningful_calls;
```

The original draft's root-cause analysis about `ensure_relationship_batch()` is now only historical context. The authoritative behavior today is the guard in `CallProcessor`.

---

### 2.3 Arbitrary JSON File Structure in the Code Graph

#### Current State

The code graph already has an initial implementation for arbitrary JSON-file structure:

- `JsonContentProcessor` exists.
- `GraphUpdater._process_worker_chunk()` already invokes it for `.json` files during parallel indexing.
- `NodeLabel.JSON_OBJECT`, `JSON_ARRAY`, `JSON_FIELD`, and `JSON_VALUE` exist.
- `RelationshipType.CONTAINS_JSON`, `HAS_FIELD`, `HAS_VALUE`, and `HAS_ELEMENT` exist.

This means the feature is no longer hypothetical. However, the **current structural model is not yet fully sound** and should not be treated as complete.

#### Remaining Modeling Defects

The current implementation still has several correctness issues:

1. **Qualified-name collision risk**
   The current root qualified name uses the file stem only. Files such as `config/server.json` and `deploy/server.json` collide.

2. **Unsafe key-to-qualified-name mapping**
   Raw JSON keys are inserted into qualified names. Keys containing dots, slashes, spaces, brackets, or other punctuation can produce ambiguous or colliding identifiers.

3. **Incorrect root typing**
   The current implementation always starts with a `JsonObject` root. JSON files whose root is an array or scalar are not modeled correctly.

4. **Incorrect parent-child semantics for complex field values**
   A field whose value is an object or array must point to that value via `HAS_VALUE`. The current logic instead infers parent labels from the qualified-name pattern and can emit `HAS_ELEMENT` from non-array parents.

5. **Schema mismatch for `HAS_ELEMENT`**
   Semantically, only arrays have elements. `HAS_ELEMENT` must never originate from `JsonObject` or `JsonField`.

6. **Project deletion is not JSON-safe today**
   The current delete query matches JSON content via an unscoped `File` pattern and traverses only `HAS_FIELD|HAS_ELEMENT`, missing `HAS_VALUE`.

7. **Canonical-payload detection is too weak**
   Skipping a file only because it contains top-level keys named `entities` and `relationships` is too broad. A normal configuration file can legitimately have those keys.

8. **Test coverage is missing for the actual hard cases**
   There is no focused regression coverage for root arrays, root scalars, duplicate basenames, escaped keys, or delete-query scoping.

#### Target Model

The target model for arbitrary JSON-file structure in the code graph is:

```text
(File)-[:CONTAINS_JSON]->(JsonObject|JsonArray|JsonValue)
(JsonObject)-[:HAS_FIELD]->(JsonField)
(JsonField)-[:HAS_VALUE]->(JsonObject|JsonArray|JsonValue)
(JsonArray)-[:HAS_ELEMENT {index: int}]->(JsonObject|JsonArray|JsonValue)
```

This model gives each concept a single responsibility:

- `JsonObject` represents object values.
- `JsonField` represents named object members.
- `JsonArray` represents array values.
- `JsonValue` represents scalar values.
- `HAS_VALUE` means "this field's value is ..." regardless of whether the value is scalar, object, or array.
- `HAS_ELEMENT` means "this array contains element at position `index`" and must be emitted only from `JsonArray`.

#### Node Properties

The node-property expectations should be:

| Node | Required Properties | Notes |
|------|---------------------|-------|
| `JsonObject` | `qualified_name`, `path`, `depth` | Represents a JSON object value |
| `JsonArray` | `qualified_name`, `path`, `depth`, `length` | Represents a JSON array value |
| `JsonField` | `qualified_name`, `path`, `key`, `depth` | `value` and `value_type` may be retained as denormalized convenience fields for scalar values only, but the canonical structure is the `HAS_VALUE` edge |
| `JsonValue` | `qualified_name`, `path`, `value`, `value_type`, `depth` | Represents string, number, boolean, or null |

#### Qualified Name Rules

The qualified-name scheme must satisfy these requirements:

1. It must include the **full relative path without extension**, not only the basename.
2. It must use a **reversible encoding** for JSON keys before placing them into qualified-name segments.
3. It must create **distinct identifiers** for field nodes versus value nodes so that a field and its value cannot collide.
4. It must support object roots, array roots, and scalar roots without synthetic mislabeling.

Recommended convention:

```text
base_qn = {project_name}.json.{encoded_relative_path_without_ext}

root object: {base_qn}._object
root array:  {base_qn}._array
root scalar: {base_qn}._value

field node:  {parent_object_qn}._field.{encoded_key}
field value object: {field_qn}._object
field value array:  {field_qn}._array
field value scalar: {field_qn}._value

array element object: {array_qn}._index.{n}._object
array element array:  {array_qn}._index.{n}._array
array element scalar: {array_qn}._index.{n}._value
```

This convention preserves uniqueness, preserves parentage, and avoids collisions between field metadata and value nodes.

#### Required Code Changes

1. **Refactor `JsonContentProcessor`** so that recursive ingestion carries the explicit parent label instead of inferring it from the qualified-name string.
2. **Allow `CONTAINS_JSON` to target any root value type**: `JsonObject`, `JsonArray`, or `JsonValue`.
3. **Allow `HAS_VALUE` to target any value type**: `JsonValue`, `JsonObject`, or `JsonArray`.
4. **Keep `HAS_ELEMENT` source restricted to `JsonArray` only**.
5. **Add element ordering explicitly** on `HAS_ELEMENT`, preferably via an `index` property.
6. **Improve canonical-payload detection** so that the worker skips only files that actually match the canonical ingestion-payload shape, not any file that merely contains top-level keys named `entities` or `relationships`.
7. **Require a full re-index** after the qualified-name model changes. Arbitrary JSON structure is derived data and should be regenerated, not migrated in place.

#### Required Schema Alignment

At the shared-schema level, the relationship endpoints must be:

```python
RelationshipSchema(
    (NodeLabel.FILE,),
    RelationshipType.CONTAINS_JSON,
    (NodeLabel.JSON_OBJECT, NodeLabel.JSON_ARRAY, NodeLabel.JSON_VALUE),
)
RelationshipSchema(
    (NodeLabel.JSON_OBJECT,),
    RelationshipType.HAS_FIELD,
    (NodeLabel.JSON_FIELD,),
)
RelationshipSchema(
    (NodeLabel.JSON_FIELD,),
    RelationshipType.HAS_VALUE,
    (NodeLabel.JSON_VALUE, NodeLabel.JSON_OBJECT, NodeLabel.JSON_ARRAY),
)
RelationshipSchema(
    (NodeLabel.JSON_ARRAY,),
    RelationshipType.HAS_ELEMENT,
    (NodeLabel.JSON_VALUE, NodeLabel.JSON_OBJECT, NodeLabel.JSON_ARRAY),
)
```

#### Project Deletion Fix

The current delete query is not safe enough because it matches JSON subtrees from an unscoped `File` pattern and does not traverse `HAS_VALUE`.

The query must be scoped to the project's own `File` descendants and must traverse all JSON-structure relationships:

```cypher
MATCH (p:Project {name: $project_name})
OPTIONAL MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER|CONTAINS_FILE|CONTAINS_MODULE*]->(container)
OPTIONAL MATCH (container)-[:DEFINES|DEFINES_METHOD*]->(defined)
OPTIONAL MATCH (container:File)-[:CONTAINS_JSON]->(json_root)
OPTIONAL MATCH (json_root)-[:HAS_FIELD|HAS_VALUE|HAS_ELEMENT*]->(json_content)
DETACH DELETE p, container, defined, json_root, json_content
```

#### Verification

```cypher
// Every indexed arbitrary JSON file should have one structural root
MATCH (f:File)
WHERE f.extension = '.json'
  AND NOT f.name IN ['package.json', 'package-lock.json']
OPTIONAL MATCH (f)-[:CONTAINS_JSON]->(root)
RETURN f.path, count(root) AS root_count;

// Only arrays may own HAS_ELEMENT
MATCH (n)-[r:HAS_ELEMENT]->()
WHERE NOT n:JsonArray
RETURN count(r) AS invalid_has_element_edges;
-- Expected: 0

// Every JsonField should have exactly one value edge
MATCH (field:JsonField)
OPTIONAL MATCH (field)-[r:HAS_VALUE]->()
WITH field, count(r) AS value_edge_count
WHERE value_edge_count <> 1
RETURN count(field) AS invalid_field_value_counts;
-- Expected: 0

// No root qualified-name collisions across distinct files
MATCH (f:File)-[:CONTAINS_JSON]->(root)
WITH root.qualified_name AS qn, collect(DISTINCT f.path) AS paths
WHERE size(paths) > 1
RETURN qn, paths;
-- Expected: no rows
```

#### Embedding Policy for Arbitrary JSON Structure

Arbitrary JSON structure in the code graph should remain **structural-only** in this spec.

- Do not create embeddings for `JsonObject`, `JsonArray`, `JsonField`, or `JsonValue` as part of this fix.
- Do not add a dedicated vector index for arbitrary JSON structure nodes in the code graph in this revision.
- If semantic retrieval over configuration JSON is needed later, that should be handled by a separate retrieval design after query evidence justifies the storage and indexing cost.

---

### 2.4 Code Embedding Validation Coverage

#### Current State

The code graph's embedding generation and vector backend support more labels than the current post-ingestion quality check validates.

- `CYPHER_QUERY_EMBEDDINGS` includes `Function`, `Method`, `Class`, `Interface`, `Contract`, `Library`, `Enum`, `Type`, `Union`, `Event`, `Modifier`, `StateVariable`, `CustomError`, `Hotkey`, `Hotstring`, `Label`, and `AhkClass`.
- `MemgraphBackend.LABELS_TO_INDEX` likewise covers the full set of embeddable code-node labels.
- `GraphUpdater` currently runs `HealthChecker.validate_ingestion_quality()` with `embedded_node_label="Function|Method|Class"` only.

#### Required Change

The validation scope should be expanded so that the post-ingestion check covers the full embeddable code-node contract, not only three labels.

Recommended approach:

1. Derive the validation label set from the same source of truth used for embedding generation or index creation.
2. Validate all embeddable code-node labels that can appear in the code graph.
3. Keep document-graph and canonical-JSON-graph embedding validation separate, because they are separate databases with separate index lifecycles.

This is a data-quality alignment task, not a schema expansion.

---

## 3. Document Graph Contract

No schema expansion is required here, but the spec must accurately reflect the current architecture.

### Invariants

1. `Document` nodes are identified by `path`, not by `qualified_name`.
2. `Chunk -[:BELONGS_TO_SECTION]-> Section` must remain child-to-parent.
3. There must be no direct `REFERENCES_CODE` edges in the live document graph, because the code graph is stored in a separate Memgraph instance.
4. Code linkage is represented by metadata arrays such as `code_references` and `resolved_code_references` on `Document` and `Chunk` nodes.
5. Chunk embeddings are stored on `Chunk(embedding)` and indexed via `settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME`.

### Verification

```cypher
MATCH (c:Chunk)
WHERE NOT (c)-[:BELONGS_TO_SECTION]->(:Section)
RETURN count(c) AS orphan_chunks;
-- Expected: 0

MATCH ()-[r:REFERENCES_CODE]->()
RETURN count(r) AS live_cross_graph_edges;
-- Expected: 0

SHOW VECTOR INDEX INFO;
-- Expected to include settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME with the effective document embedding dimension
```

---

## 4. Canonical JSON Graph Contract

No schema redesign is required here, but the spec must represent the current model accurately.

### Invariants

1. The canonical JSON graph lives in the separate JSON Memgraph instance.
2. `JsonEntity` is the canonical node label for JSON dataset ingestion.
3. Each entity is scoped by `dataset_id` and a dataset-unique identifier such as `unique_id = {dataset_id}::{entity_id}`.
4. Relationship types are dynamic and come from ingestion payloads. They are not restricted to `RelationshipType` enum values used by the code graph.
5. Embeddings belong on `JsonEntity` only and are indexed by `settings.JSON_MEMGRAPH_VECTOR_INDEX_NAME`.
6. Dataset-wide relationship-reference validation must remain in place so that missing or ambiguous entity references fail before writes.

### Verification

```cypher
MATCH (n:JsonEntity)
RETURN count(n) AS entity_count;

SHOW VECTOR INDEX INFO;
-- Expected to include settings.JSON_MEMGRAPH_VECTOR_INDEX_NAME with the effective JSON embedding dimension
```

The code graph's arbitrary JSON structure must not be described as a replacement for this graph. The two serve different retrieval and modeling purposes.

---

## 5. Required Tests

The reviewed spec is not implementation-ready unless the following regression coverage exists:

1. **File <-> Module**
   - Internal source-file module creates both edges.
   - External/import-only module remains allowed without `BELONGS_TO_FILE`.

2. **Builtin `CALLS` filtering**
   - Default path omits builtin `CALLS` edges.
   - Opt-in path retains builtin `CALLS` edges.

3. **Arbitrary JSON structure**
   - Root object file.
   - Root array file.
   - Root scalar file.
   - Two JSON files with the same basename in different directories do not collide.
   - Keys containing punctuation or dots are encoded without collisions.
   - `JsonField -> HAS_VALUE -> JsonObject` for object-valued fields.
   - `JsonField -> HAS_VALUE -> JsonArray` for array-valued fields.
   - `JsonArray -> HAS_ELEMENT` only from arrays, with ordering preserved.
   - Config files that merely contain keys named `entities` or `relationships` are still structurally indexed unless they actually match canonical ingestion-payload shape.

4. **Deletion safety**
   - Deleting one project removes only that project's JSON structural subtree.
   - JSON-value nodes reachable only through `HAS_VALUE` are not leaked.

5. **Embedding validation alignment**
   - Post-ingestion quality validation covers the full embeddable code-label set, or a shared single source of truth proves the label scope is synchronized.

---

## 6. Rollout Plan

1. Keep the already-implemented File <-> Module and builtin-filtering behavior unchanged.
2. Run one-time cleanup/backfill for legacy graphs:
   - Backfill `CONTAINS_MODULE` and `BELONGS_TO_FILE` where missing.
   - Delete historical builtin-target `CALLS` edges.
3. Land the remaining code-graph fixes for arbitrary JSON structure:
   - Correct root typing.
   - Correct `HAS_VALUE` and `HAS_ELEMENT` semantics.
   - Fix qualified-name identity.
   - Improve canonical-payload detection.
   - Fix project-delete traversal and scoping.
4. Expand code-graph embedding validation coverage to the full embeddable label set.
5. Run a **full code-graph re-index** after the arbitrary JSON qualified-name changes.
6. Leave the document-graph and canonical-JSON-graph schemas unchanged, while preserving their existing vector-index dimension-reconciliation behavior.

---

## 7. Final Acceptance Criteria

The design is considered complete only when all of the following are true:

1. Internal modules derived from indexed source files always link to exactly one `File` via `BELONGS_TO_FILE`, while external/import-only modules remain allowed.
2. Default ingestion produces zero builtin-target `CALLS` edges in new graphs.
3. Arbitrary JSON files in the code graph are modeled with correct root typing, collision-safe identities, correct `HAS_VALUE` semantics, and array-only `HAS_ELEMENT` ownership.
4. Deleting a project removes only that project's code nodes and code-graph JSON structure.
5. Document/code linkage continues to use `resolved_code_references` metadata rather than cross-database edges.
6. Canonical JSON entity ingestion remains isolated to the JSON Memgraph instance with dataset-scoped validation and `JsonEntity` embeddings.
7. Vector indexes remain graph-appropriate:
   - Per-label code indexes for embeddable code nodes.
   - `DOC_MEMGRAPH_VECTOR_INDEX_NAME` for document chunks.
   - `JSON_MEMGRAPH_VECTOR_INDEX_NAME` for canonical JSON entities.
8. Post-ingestion validation checks the same embedding contract that the runtime actually stores and indexes.

At that point, the spec is logically sound, aligned with the existing codebase architecture, and implementation-ready for the remaining work.