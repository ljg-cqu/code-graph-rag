# Data Model Quality Fixes — Design Specifications

> **Status**: Implementation-Ready  
> **Scope**: Code-graph + JSON-entity + Document-graph data modeling  
> **Generated from**: Full codebase audit of `types_defs.py`, `constants.py`, `schemas.py`, `models.py`, `json_ingestion.py`, `schema_builder.py`, `migrations/data_model_migrations.py`, `cypher_queries.py`, `graph_service.py`, and ingestion mixins  

---

## 1. Issue: `Method` Node Missing `is_exported` Property

**Severity**: HIGH — Query inconsistency & embedding coverage gap

### Problem
The `NodeSchema` for `Method` in `NODE_SCHEMAS` (`types_defs.py`) does **not** include `is_exported`:
```python
NodeSchema(
    NodeLabel.METHOD,
    "{qualified_name: string, name: string, decorators: list[string], start_line: int, end_line: int, docstring: string, path: string, absolute_path: string}",
)
```
But the sibling schemas for `Function`, `Class`, `Interface`, `Enum`, `Type`, `Union`, `Contract`, `Library` all include `is_exported`.

The `ingest_method()` function in `parsers/utils.py` also omits `is_exported` from the method's `PropertyDict`. Methods bypass `_build_function_props` (which sets `is_exported` for standalone functions) and go through `ingest_method()` exclusively.

### Impact
- `CYPHER_QUERY_EMBEDDINGS` and `CYPHER_QUERY_PROJECT_NODE_IDS` query methods via `Module->DEFINES->(c)->DEFINES_METHOD->(n:Method)`, but filters on `is_exported` are impossible for methods.
- LLM-generated Cypher queries that reference `n.is_exported` on Method nodes will always return `null`, producing incorrect results.
- The schema text fed to the LLM (via `build_graph_schema_text`) incorrectly implies methods cannot be exported.

### Fix
**File**: `codebase_rag/types_defs.py` — `NODE_SCHEMAS`

Update the `Method` `NodeSchema` to include `is_exported`:
```python
NodeSchema(
    NodeLabel.METHOD,
    "{qualified_name: string, name: string, decorators: list[string], start_line: int, end_line: int, docstring: string, is_exported: bool, path: string, absolute_path: string}",
)
```

**File**: `codebase_rag/parsers/utils.py` — `ingest_method` function

Add `is_exported` parameter and propagate it into the method's `PropertyDict`:
```python
def ingest_method(
    method_node: ASTNode,
    container_qn: str,
    container_type: cs.NodeLabel,
    ingestor: IngestorProtocol,
    function_registry: FunctionRegistryTrieProtocol,
    simple_name_lookup: SimpleNameLookup,
    get_docstring_func: Callable[[ASTNode], str | None],
    language: cs.SupportedLanguage | None = None,
    extract_decorators_func: Callable[[ASTNode], list[str]] | None = None,
    method_qualified_name: str | None = None,
    file_path: Path | None = None,
    repo_path: Path | None = None,
    is_exported: bool = False,
) -> None:
    ...
    method_props: PropertyDict = {
        cs.KEY_QUALIFIED_NAME: method_qn,
        cs.KEY_NAME: method_name,
        cs.KEY_DECORATORS: decorators,
        cs.KEY_START_LINE: method_node.start_point[0] + 1,
        cs.KEY_END_LINE: method_node.end_point[0] + 1,
        cs.KEY_DOCSTRING: get_docstring_func(method_node),
        cs.KEY_IS_EXPORTED: is_exported,
    }
    ...
```

**Cross-language semantics**:
- `SupportedLanguage.CPP`: `is_exported` is inherited from the parent class/module's export status (use `cpp_utils.is_exported` on the parent container if applicable, otherwise `False`).
- All other languages: default to `False` unless the language handler provides explicit export detection.

**File**: `codebase_rag/parsers/function_ingest.py` — `_handle_cpp_out_of_class_method`

Pass `is_exported=False` (or derive from parent class status) to the `ingest_method` call.

**Migration**: `codebase_rag/migrations/data_model_migrations.py`

Add a new migration `_migrate_method_is_exported`:
```cypher
MATCH (m:Method)
WHERE m.is_exported IS NULL
SET m.is_exported = false
```

---

## 2. Issue: `CodeChunk` Nodes Queried but Never Ingested

**Severity**: HIGH — Dead query branch & schema inconsistency

### Problem
`CYPHER_QUERY_EMBEDDINGS` and `CYPHER_QUERY_PROJECT_NODE_IDS` both include a `UNION` branch for `:CodeChunk` nodes:
```cypher
MATCH (m:Module)
WHERE m.qualified_name STARTS WITH ($project_name + '.')
MATCH (m)-[:DEFINES]->(n:CodeChunk)
RETURN id(n) AS node_id, ...
```

However, `CodeChunk` nodes are **never ingested into the graph**. The `SemanticCodeChunker` in `utils/code_chunker.py` produces `CodeChunk` dataclass instances for embedding oversized code, but these chunks are only used to generate embedding vectors. They are not written to Memgraph via `ensure_node_batch`.

Additionally, `NodeType` enum in `types_defs.py` lacks `CODE_CHUNK = "CodeChunk"`, even though `NodeLabel.CODE_CHUNK` exists.

### Impact
- The `:CodeChunk` query branch always returns zero rows, wasting query execution time.
- The LLM sees `CodeChunk` in `EMBEDDABLE_CODE_NODE_LABELS` and may generate queries for nodes that do not exist.
- `NodeType` and `NodeLabel` are inconsistent.

### Fix
**Option A** (Recommended — minimal change): Remove the `:CodeChunk` branch from `CYPHER_QUERY_EMBEDDINGS` and `CYPHER_QUERY_PROJECT_NODE_IDS`.

**File**: `codebase_rag/constants.py`

Delete the final `UNION` branch in both queries:
```cypher
UNION
MATCH (m:Module)
WHERE m.qualified_name STARTS WITH ($project_name + '.')
MATCH (m)-[:DEFINES]->(n:CodeChunk)
RETURN id(n) AS node_id, n.qualified_name AS qualified_name,
       n.start_line AS start_line, n.end_line AS end_line, n.path AS path
```

Also remove `"CodeChunk"` from `EMBEDDABLE_CODE_NODE_LABELS`.

**Option B** (Future feature): Implement actual `CodeChunk` node ingestion in the embedding pipeline (`embedder.py` or `graph_updater.py`), create `HAS_CHUNK` relationships, and add `NodeType.CODE_CHUNK`.

**File**: `codebase_rag/types_defs.py` — `NodeType` enum

If Option A is chosen, no change needed to `NodeType`. If Option B is chosen, add:
```python
CODE_CHUNK = "CodeChunk"
```

---

## 3. Issue: `RELATIONSHIP_SCHEMAS` Missing Several Relationship Types (Used) and Contains Unused Enum Values

**Severity**: MEDIUM — Schema documentation gap affecting LLM query generation

### Problem
The `RelationshipType` enum defines types that are **not** present in `RELATIONSHIP_SCHEMAS`. However, many of these are also **not created during ingestion**, meaning the LLM should not know about them.

| RelationshipType | Ingestion Code Exists? | In RELATIONSHIP_SCHEMAS? | Action |
|---|---|---|---|
| `HAS_CHUNK` | No | No | Remove from enum (no chunks ingested) |
| `REFERENCES_CODE` | No | No | Remove from enum (unused) |
| `DEFINES_HOTKEY` | No | No | Remove from enum (unused) |
| `DEFINES_HOTSTRING` | No | No | Remove from enum (unused) |
| `DEFINES_LABEL` | No | No | Remove from enum (unused) |
| `TRIGGERS_HOTKEY` | No | No | Remove from enum (unused) |
| `CALLS_COMMAND` | No | No | Remove from enum (unused) |
| `CREATES_COM_OBJECT` | No | No | Remove from enum (unused) |
| `SENDS_KEYS` | No | No | Remove from enum (unused) |
| `CLICKS_ELEMENT` | No | No | Remove from enum (unused) |
| `CREATES_GUI` | No | No | Remove from enum (unused) |
| `CONTROLS_GUI` | No | No | Remove from enum (unused) |
| `INCLUDES_FILE` | No | No | Remove from enum (unused) |
| `RELATES_TO` | No | **Yes** | Remove from enum (unused; schema exists but no data) |

The AutoHotkey parser (`parsers/handlers/autohotkey.py`) handles hotkeys/hotstrings/labels as standard `Function`/`Method` nodes via `DEFINES`/`DEFINES_METHOD`. It does **not** create any AutoHotkey-specific relationships.

### Impact
- Unused enum values bloat the codebase and confuse future maintainers.
- `RELATES_TO` has a schema entry but no ingestion code, which could mislead the LLM if the full schema is ever fed to it.

### Fix
**File**: `codebase_rag/constants.py` — `RelationshipType` enum

Remove all unused relationship types:
```python
# REMOVE these entirely:
HAS_CHUNK = "HAS_CHUNK"
REFERENCES_CODE = "REFERENCES_CODE"
DEFINES_HOTKEY = "DEFINES_HOTKEY"
DEFINES_HOTSTRING = "DEFINES_HOTSTRING"
DEFINES_LABEL = "DEFINES_LABEL"
TRIGGERS_HOTKEY = "TRIGGERS_HOTKEY"
CALLS_COMMAND = "CALLS_COMMAND"
CREATES_COM_OBJECT = "CREATES_COM_OBJECT"
SENDS_KEYS = "SENDS_KEYS"
CLICKS_ELEMENT = "CLICKS_ELEMENT"
CREATES_GUI = "CREATES_GUI"
CONTROLS_GUI = "CONTROLS_GUI"
INCLUDES_FILE = "INCLUDES_FILE"
RELATES_TO = "RELATES_TO"
```

**File**: `codebase_rag/types_defs.py` — `RELATIONSHIP_SCHEMAS`

Remove the `RELATES_TO` schema entry (the only unused one that had a schema):
```python
# REMOVE:
RelationshipSchema(
    (NodeLabel.JSON_ENTITY,),
    RelationshipType.RELATES_TO,
    (NodeLabel.JSON_ENTITY,),
),
```

**File**: `codebase_rag/tests/test_schemas.py`

Update or remove `test_references_code_is_not_declared_in_split_graph_schema` since `REFERENCES_CODE` will no longer exist.

---

## 4. Issue: `JsonEntity` `labels` Property Conflicts with Memgraph `labels()` Function

**Severity**: MEDIUM — Potential query confusion

### Problem
In `_entity_properties` (`json_ingestion.py`), the `labels` property is set on `JsonEntity` nodes:
```python
properties["labels"] = labels
```
In Memgraph/Cypher, `labels(n)` is a built-in function returning node labels. Having a property also named `labels` creates semantic ambiguity. While Memgraph stores labels and properties separately, the LLM orchestrator receiving schema text that says `labels: list[string]` as a *property* may generate incorrect Cypher (e.g., `n.labels` instead of `labels(n)`).

### Impact
- LLM-generated queries may use `n.labels` (property access) instead of `labels(n)` (function call), returning the entity's type labels list rather than the Memgraph label set, or vice versa.

### Fix
**File**: `codebase_rag/json_ingestion.py` — `_entity_properties`

Rename the property from `labels` to `entity_labels`:
```python
properties["entity_labels"] = labels
```

**File**: `codebase_rag/types_defs.py` — `NODE_SCHEMAS` for `JSON_ENTITY`

```python
NodeSchema(
    NodeLabel.JSON_ENTITY,
    "{unique_id: string, entity_id: string, name: string, type: string, dataset_id: string, entity_labels: list[string], embedding: list[float], embedding_model: string, embedding_version: int}",
),
```

**Migration**: `_migrate_json_entity_labels`
```cypher
MATCH (n:JsonEntity) WHERE n.labels IS NOT NULL
SET n.entity_labels = n.labels, n.labels = null
```

---

## 5. Issue: `Test` Node Label Declared in Health Check Queries but Not in `NodeLabel` Enum

**Severity**: MEDIUM — Schema/health-check inconsistency

### Problem
In `constants.py`, the `QUERY_GEN_REQUIRED_PROPS` Cypher query references `n:Test`:
```cypher
... OR n:Test)
  AND (n.name IS NULL OR n.qualified_name IS NULL OR n.path IS NULL ...)
```
And the migration `_cleanup_incomplete_test_nodes` also operates on `(t:Test)` nodes.
But `NodeLabel` enum has **no** `TEST = "Test"` entry. There is no `NodeSchema` for `Test`, no `_NODE_LABEL_UNIQUE_KEYS` mapping, and no `NODE_UNIQUE_CONSTRAINTS` entry.

### Impact
- Test nodes can exist in the graph (from ingestion or legacy data) but have no defined schema, unique key, or constraint.
- The health check `QUERY_GEN_REQUIRED_PROPS` references `Test` but the schema definition fed to the LLM does not mention it, causing confusion.
- The `_cleanup_incomplete_test_nodes` migration deletes Test nodes with null name/qualified_name, but there's no mechanism to prevent their recreation.

### Fix
**Option A** (Recommended): Remove `Test` from health check queries since it's not a defined node type.

**File**: `codebase_rag/constants.py` — `QUERY_GEN_REQUIRED_PROPS`

Remove `OR n:Test` from the WHERE clause.

**File**: `codebase_rag/migrations/data_model_migrations.py`

Remove `_cleanup_incomplete_test_nodes` from `run_migrations` and delete the function.

**Option B**: If Test nodes are intentionally supported, add `TEST = "Test"` to `NodeLabel`, `NODE_SCHEMAS`, `_NODE_LABEL_UNIQUE_KEYS`, and create a proper schema definition.

---

## 6. Issue: Inconsistent `path` vs `absolute_path` Handling Across Node Types

**Severity**: LOW — Property naming clarity (documentation only)

### Problem
Multiple node types have both `path` and `absolute_path` properties. Their semantics are consistent for code-graph nodes but could be documented more explicitly:
- **Module**: `path: string | null, import_path: string | null, absolute_path: string` — `path` can be null for external modules.
- **Class/Function/Method/Interface/etc.**: `path: string, absolute_path: string` — both required, `path` is relative to repo root.
- **File**: `path: string, name: string, extension: string, absolute_path: string` — `path` is relative to repo root.
- **Folder/Package**: Same pattern.
- **JsonObject/JsonArray/JsonField/JsonValue**: `path: string` — the relative filesystem path of the JSON file (same semantics as code nodes), **not** a JSON document path.

The JSON document path is encoded in `qualified_name` (e.g., `project.json.config._object._field.name`).

### Impact
- Low. Current naming is actually consistent (all `path` values are relative filesystem paths). The potential confusion is documentation-level.

### Fix
No code changes required. Add a clarifying comment in `types_defs.py` above the JSON node schemas:
```python
# JSON content node schemas
# NOTE: `path` on JSON nodes is the relative filesystem path of the source file,
# identical to code-graph nodes. The JSON document path is encoded in `qualified_name`.
```

---

## 7. Issue: `JsonEntity` Node Schema Missing `embedding`, `embedding_model`, `embedding_version` Properties

**Severity**: LOW — Schema documentation gap

### Problem
The `NodeSchema` for `JSON_ENTITY` in `NODE_SCHEMAS` already includes embedding properties. However, the **code-graph node schemas** (Function, Class, Method, etc.) do **not** declare these embedding properties, even though the embedding pipeline (`CYPHER_QUERY_EMBEDDINGS`) and vector search queries rely on them being present.

### Impact
- LLM-generated Cypher that filters on `n.embedding_model` or `n.embedding_version` for code nodes appears schema-invalid according to the schema text.
- The health check `QUERY_GEN_EMBEDDING_MODEL_MISMATCH` queries `n.embedding_model` on code nodes, but the schema doesn't document these properties.

### Fix
**File**: `codebase_rag/types_defs.py` — `NODE_SCHEMAS`

Add `embedding`, `embedding_model`, `embedding_version` as optional properties to all embeddable code node schemas:

```python
NodeSchema(
    NodeLabel.FUNCTION,
    "{qualified_name: string, name: string, decorators: list[string], start_line: int, end_line: int, docstring: string, is_exported: bool, path: string, absolute_path: string, embedding: list[float] | null, embedding_model: string | null, embedding_version: int | null}",
),
```

Repeat for: `CLASS`, `METHOD`, `INTERFACE`, `ENUM`, `TYPE`, `UNION`, `CONTRACT`, `LIBRARY`, `EVENT`, `MODIFIER`, `STATE_VARIABLE`, `CUSTOM_ERROR`, `HOTKEY`, `HOTSTRING`, `LABEL`, `CLASS_AHK`.

---

## 8. Issue: `_NODE_LABEL_UNIQUE_KEYS` Runtime Check Can Miss Future Additions

**Severity**: LOW — Defensive coding improvement

### Problem
`constants.py` includes a runtime check:
```python
_missing_keys = set(NodeLabel) - set(_NODE_LABEL_UNIQUE_KEYS.keys())
if _missing_keys:
    raise RuntimeError(...)
```
This is excellent defensive coding, but it runs at **module import time**. If a developer adds a new `NodeLabel` but forgets the unique key, the error surfaces only at import — which may be late in the pipeline.

### Impact
- No functional impact currently (all labels are covered), but the check should also be mirrored in test suites for early detection.

### Fix
**File**: `codebase_rag/tests/test_schemas.py` — Add tests

```python
def test_all_node_labels_have_unique_keys():
    """Every NodeLabel MUST have a corresponding entry in _NODE_LABEL_UNIQUE_KEYS."""
    from codebase_rag.constants import NodeLabel, _NODE_LABEL_UNIQUE_KEYS
    missing = set(NodeLabel) - set(_NODE_LABEL_UNIQUE_KEYS.keys())
    assert not missing, f"NodeLabel(s) missing unique keys: {missing}"


def test_all_relationship_types_have_schemas():
    """Every RelationshipType used in ingestion MUST have a schema entry."""
    from codebase_rag.constants import RelationshipType
    from codebase_rag.types_defs import RELATIONSHIP_SCHEMAS
    schema_rel_types = {schema.rel_type for schema in RELATIONSHIP_SCHEMAS}
    missing = set(RelationshipType) - schema_rel_types
    assert not missing, f"RelationshipType(s) missing schemas: {missing}"
```

---

## 9. Issue: `IngestionResult` and `UpdateResult` Models Have No Dataset Validation

**Severity**: LOW — Data integrity

### Problem
In `schemas.py`, `IngestionResult` accepts `dataset_id: str` with no validation. The `json_ingestion.py` code auto-populates `dataset_id` from the first prepared file, but if multiple files have different `dataset_id` values, only the first is recorded in the result.

### Impact
- The result object may report an incorrect `dataset_id` when multiple datasets are ingested in a single call.

### Fix
**File**: `codebase_rag/schemas.py` — `IngestionResult`

Change `dataset_id: str` to `dataset_ids: list[str]` to support multi-dataset ingestion results:

```python
class IngestionResult(BaseModel):
    dataset_ids: list[str] = Field(default_factory=list)
    ...
```

**File**: `codebase_rag/json_ingestion.py` — `ingest_json_data`

Collect all unique dataset IDs from `prepared_files`:
```python
result.dataset_ids = sorted({pf.dataset_id for pf in prepared_files})
```

Remove the single `dataset_id` fallback assignment.

---

## 10. Issue: `JSON_ENTITY` Unique Key is `unique_id` but Constraint Scope Is Overly Broad

**Severity**: LOW — Constraint cleanup

### Problem
`ensure_constraints()` in `graph_service.py` iterates over `NODE_UNIQUE_CONSTRAINTS`, which includes mappings for **all** node labels (code, document, JSON). The JSON ingestor calls `ensure_constraints()` on its own `MemgraphIngestor` instance (potentially pointing to a separate Memgraph instance via `JSON_MEMGRAPH_PORT`). This creates constraints for code-graph labels on the JSON instance and vice versa, which is wasteful but not harmful.

### Impact
- Wasteful constraint creation on multi-instance setups.
- No data integrity impact.

### Fix
**File**: `codebase_rag/services/graph_service.py` — `ensure_constraints`

Add an optional `labels_filter` parameter:

```python
def ensure_constraints(self, labels: tuple[str, ...] | None = None) -> None:
    logger.info(ls.MG_ENSURING_CONSTRAINTS)
    targets = (
        {k: v for k, v in NODE_UNIQUE_CONSTRAINTS.items() if k in labels}
        if labels else NODE_UNIQUE_CONSTRAINTS
    )
    for label, prop in targets.items():
        try:
            self._execute_query(build_constraint_query(label, prop))
        except Exception:
            pass
    logger.info(ls.MG_CONSTRAINTS_DONE)
    self._ensure_indexes(targets)
```

Update callers (`graph_updater.py`, `json_ingestion.py`) to pass only the labels they actually create.

**Note**: Do **not** attempt to add a composite unique constraint on `(dataset_id, entity_id)` — Memgraph's `ASSERT n.prop IS UNIQUE` syntax does not support composite constraints. The existing `unique_id` constraint (`unique_id = dataset_id::entity_id`) already prevents duplicates.

---

## Implementation Priority Order

| Priority | Issue | Effort | Risk if Unfixed |
|----------|-------|--------|-----------------|
| P0 | #1 Method `is_exported` missing | Small | High — query correctness |
| P0 | #2 CodeChunk dead query branch | Small | High — wasted queries & LLM confusion |
| P1 | #3 Remove unused RelationshipType values | Medium | Medium — schema bloat & LLM confusion |
| P1 | #4 `labels` → `entity_labels` rename | Small | Medium — LLM confusion |
| P1 | #5 `Test` node inconsistency | Small | Low — health check noise |
| P2 | #6 `path` documentation (no code change) | Tiny | None — documentation only |
| P2 | #7 Embedding properties in schema | Small | Low — documentation gap |
| P2 | #8 Runtime check test coverage | Small | None — defensive |
| P2 | #9 Multi-dataset result tracking | Small | None — accuracy |
| P3 | #10 Constraint scope cleanup | Small | None — optimization |

---

## Validation Checklist

Before closing each issue, verify:

- [ ] `NODE_SCHEMAS` entries match actual properties set during ingestion
- [ ] `RELATIONSHIP_SCHEMAS` entries cover all `RelationshipType` enum values that have corresponding ingestion code
- [ ] `_NODE_LABEL_UNIQUE_KEYS` covers all `NodeLabel` enum values
- [ ] `NODE_UNIQUE_CONSTRAINTS` covers all unique keys
- [ ] `build_graph_schema_text()` output includes only relationships that exist in actual graph data
- [ ] Cypher query templates do not reference node labels that are never ingested
- [ ] Migration scripts exist for property additions/renames
- [ ] Health check queries reference only defined node labels
- [ ] Cypher query templates reference only schema-defined properties
- [ ] `ingestion_schema.json` remains the single source of truth for JSON ingestion
- [ ] Test suite includes `test_all_node_labels_have_unique_keys` and `test_all_relationship_types_have_schemas`
- [ ] `NodeType` enum is consistent with `NodeLabel` for all ingested node types
