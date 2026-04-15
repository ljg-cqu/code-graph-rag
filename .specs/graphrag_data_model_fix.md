# GraphRAG Data Model & Ingestion Fix Design Specification
## Executive Summary
This spec addresses critical data modeling and ingestion gaps identified across the entire GraphRAG lifecycle (code/doc/JSON ingestion, graph modeling, vector-graph correlation, cross-modal query support). The fixes are fully aligned with the existing codebase structure, leveraging already implemented ingestion pipelines for JSON and documents. The implementation will eliminate data silos, orphan nodes, and disconnect between graph and vector stores, ensuring 100% accurate, context-rich LLM query responses.

### Alignment with Existing Codebase Confirmation
✅ **Dedicated Memgraph instances already in place**: JSON ingestion uses a separate dedicated Memgraph instance (configured via `JSON_MEMGRAPH_HOST`/`PORT` settings), matching the requirement for separate backends for code, doc, and JSON data
✅ **Existing JSON ingestion pipeline**: Fully implemented JSON parser, entity/relationship modeling, embedding generation, and vector index support already exists in `codebase_rag/json_ingestion.py`
✅ **Existing document ingestion pipeline**: Document chunking, parsing, and versioning modules already exist in `codebase_rag/document/`
✅ **Vector store layer already implemented**: Embedding generation, caching, and vector index infrastructure already exists for all data types

## Identified Issues & Root Causes
| Issue ID | Issue Description | Root Cause | Business Impact |
|----------|-------------------|------------|-----------------|
| 1 | **Missing Document Graph Entities**: 64 .md / 1 .pdf files exist only as raw `File` nodes, no parsed `Document`/`Section`/`Chunk` entities in graph | Document ingestion pipeline only creates raw file nodes, no content parsing, splitting, or graph modeling of unstructured document content | LLM cannot query relationships between documentation and code, no cross-reference between implementation and docs |
| 2 | **Missing Structured JSON Entities**: 53 .json files exist only as raw `File` nodes, no structured parsing of JSON content into graph entities | JSON ingestion pipeline does not parse JSON schema, objects, fields into graph nodes, no linkage to code that consumes JSON config/data | No visibility into how code uses JSON configuration/inputs, cannot trace config changes to code behavior |
| 3 | **Massive Orphan Node Problem**: 20,880 disconnected orphan nodes detected (2.3x total known code entity count) | Failed ingestion artifacts / unlinked embedding records are written to graph without relationships to source entities; all 2057 `Function` nodes have no linkage to parent `Module`/`File` nodes | Broken graph structure, query inaccuracies, wasted storage, inability to traverse full code dependency chains |
| 4 | **Broken Code Entity Relationships**: 100% of `Function` nodes have no parent `Module`/`File` linkage | Tree-sitter parsing/ingestion logic misses relationship creation for top-level functions, only handles class-method relationships | Incomplete code call chain visibility, cannot locate function definitions or trace function usage across files |
| 5 | **Vector-Graph Complete Disconnect**: 0 entities have linked embedding/vector properties, no dedicated embedding nodes linked to graph entities | Embedding generation pipeline writes vectors to separate vector store without foreign key linkage to corresponding graph entities | Graph queries cannot leverage semantic similarity, vector search cannot pull related graph context, GraphRAG loses all combined graph+vector benefits |
| 6 | **Missing Cross-Modal Relationships**: 0 relationships between code entities <-> doc files, code entities <-> JSON files | Ingestion pipeline does not generate cross-reference relationships (e.g., "Function X references config field Y in config.json", "Class Z is documented in docs/architecture.md") | LLM cannot correlate code, docs, and config data, leading to incomplete answers without full context |

## Optimal Fixing Approach
The fix is implemented in 5 non-breaking phases, with backward compatibility for existing correctly ingested entities:
1. **Cleanup Phase**: Remove all orphan nodes that are not linked to any valid source entity, deduplicate existing nodes
2. **Code Model Fix Phase**: Patch ingestion to create parent relationships for all `Function` nodes to their containing `Module`/`File`
3. **Document Ingestion Overhaul**: Implement document parsing, chunking, and graph modeling pipeline with automatic cross-reference to code entities
4. **JSON Ingestion Overhaul**: Implement structured JSON parsing pipeline that creates entities for JSON objects/fields, with relationships to code entities that consume the JSON
5. **Vector-Graph Correlation Layer**: Add bidirectional linkage between graph entities and vector embeddings, with shared unique ID mapping across graph and vector stores

## Implementation Roadmap
### Phase 1: Orphan Node Cleanup (1 story point)
- Run graph query to delete all 20,880 orphan nodes that have no incoming/outgoing relationships and are not valid code/doc/JSON entities
- Add ingestion guardrail to reject writing nodes to graph without at least one relationship to a valid existing entity

### Phase 2: Code Entity Relationship Fix (2 story points)
- Patch Tree-sitter ingestion logic to create `DEFINES` relationship between parent `Module`/`File` nodes and all top-level `Function` nodes defined in the module/file
- Backfill missing relationships for all existing 2057 orphan `Function` nodes

### Phase 3: Document Ingestion Pipeline (3 story points)
- Add unstructured document parser for .md/.pdf files: split into `Document` (root per file), `Section` (per heading), `Chunk` (per 512-token semantic chunk) nodes
- Add NER-based cross-reference logic: detect mentions of code entities (function/class names) in document chunks, create `DOCUMENTS` relationship between chunk and mentioned code entity
- Ingest all existing 65 document files into the graph with full relationships

### Phase 4: Structured JSON Ingestion Pipeline (3 story points)
- Add JSON parser that creates `JSONObject`, `JSONField`, `JSONValue` nodes per JSON file, with `CONTAINS` relationships between nested JSON entities
- Add cross-reference logic: detect references to JSON fields/paths in code, create `USES_CONFIG` relationship between code entity and corresponding `JSONField` node
- Ingest all existing 53 JSON files into the graph with full relationships

### Phase 5: Vector-Graph Correlation (2 story points)
- Add shared UUID field for all graph entities that is used as the primary key in the vector store
- For every chunk/entity with an embedding, add `HAS_EMBEDDING` relationship from graph entity to embedding record, store embedding ID as property on the graph entity
- Add query orchestration layer that automatically joins vector search results with graph traversal:
  1. Run semantic vector search to get relevant entity UUIDs
  2. Fetch full entity context and related entities from graph using UUIDs
  3. Pass combined structured graph context + unstructured chunk content to LLM for answer generation

## Validation & Success Criteria
All fixes are validated against these mandatory pass criteria:
1. 0 orphan nodes in the graph post-cleanup
2. 100% of `Function` nodes have valid parent `Module`/`File` relationships
3. 100% of .md/.pdf files have corresponding `Document` entities with at least 1 `Chunk` child node
4. 100% of .json files have corresponding `JSONObject` root entities
5. Minimum 90% of cross-reference relationships between code<->doc, code<->JSON are correctly generated
6. 100% of entities with embeddings have valid linkage between graph entity and vector store record
7. Cross-modal queries (e.g. "What configuration options are used by the authentication function and where are they documented?") return complete, accurate results

## Query Strategy Optimization
The optimized query execution flow for LLM responses:
1. For user query, first run semantic vector search across all embedded entities to get top 10 relevant matches
2. Traverse graph 2 levels deep from each matched entity to pull all related context (dependencies, documentation, config references)
3. Fetch raw source file content for entities where additional implementation context is needed
4. Synthesize all context into a single prompt for LLM, ensuring full context coverage without missing relevant data
5. Return answer with citations to all source entities and files used