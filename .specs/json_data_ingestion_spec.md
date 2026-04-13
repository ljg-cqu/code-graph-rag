# JSON Data Ingestion Design Specification
## Version: 1.0 | Last Updated: 2024
## Target Path: `/home/zealy/github/ljg-cqu/code-graph-rag/.specs/json_data_ingestion_spec.md`

---

## 1. Overview & Purpose
This feature enables users to ingest pre-processed JSON data containing entities, relationships, and metadata directly into both the Code-Graph RAG knowledge graph (Neo4j/Memgraph) and vector database (Qdrant/Memgraph vector store). The outcome supports both semantic similarity search and graph traversal queries over user-provided custom domain data.

The feature includes **optional opt-in real-time incremental update and enterprise monitoring support**, fully reusing existing system infrastructure for unified workflow with unstructured document ingestion. All core functionality remains backwards compatible with existing batch ingestion use cases.

---

## 2. Input JSON Schema Definition
All ingested JSON files MUST conform to the following schema, aligned with existing internal models (`GraphRelationship`, `RelationshipData` from `/codebase_rag/models.py` and `/codebase_rag/types_defs.py`):

```json
{
  "metadata": {
    "dataset_id": "string (required, unique identifier for the dataset)",
    "source": "string (optional, data source attribution)",
    "created_at": "ISO 8601 timestamp (optional)",
    "default_entity_labels": ["string (optional, default labels to apply to all entities)"]
  },
  "entities": [
    {
      "id": "string (required, unique entity ID within dataset)",
      "labels": ["string (required, node labels e.g. ['Job', 'Company'])"],
      "properties": {
        "name": "string (required, human-readable entity name)",
        "description": "string (required, content used for embedding generation)",
        "any_custom_property": "any (optional custom metadata fields)"
      }
    }
  ],
  "relationships": [
    {
      "id": "string (optional, unique relationship ID)",
      "source_entity_id": "string (required, references entity.id in same dataset)",
      "target_entity_id": "string (required, references entity.id in same dataset)",
      "type": "string (required, relationship type e.g. 'REQUIRES_SKILL', 'WORKS_AT')",
      "properties": {
        "description": "string (optional, used for relationship embedding if provided)",
        "any_custom_property": "any (optional custom metadata fields)"
      }
    }
  ]
}
```

### Example Valid JSON Structure
```json
{
  "metadata": {
    "dataset_id": "job_hunting_jd_1",
    "source": "public job board"
  },
  "entities": [
    {
      "id": "job_001",
      "labels": ["JobPosting", "SoftwareEngineer"],
      "properties": {
        "name": "Senior Backend Engineer",
        "description": "Senior backend engineer position requiring Python, Neo4j, and microservices experience. 5+ years of experience required.",
        "salary_range": "$150k-$200k",
        "location": "Remote"
      }
    },
    {
      "id": "skill_001",
      "labels": ["Skill", "Technology"],
      "properties": {
        "name": "Python",
        "description": "Python programming language, version 3.8+"
      }
    }
  ],
  "relationships": [
    {
      "source_entity_id": "job_001",
      "target_entity_id": "skill_001",
      "type": "REQUIRES_SKILL",
      "properties": {
        "description": "Job requires Python proficiency",
        "minimum_experience_years": 3
      }
    }
  ]
}
```

---

## 3. Core Ingestion Workflow
### Step 1: Input Validation
1. Validate JSON structure against schema using Pydantic (reuse existing type definitions from `types_defs.py`)
2. Check for duplicate entity IDs within the dataset
3. Verify all relationship source/target entity IDs exist in the dataset
4. Validate that all required fields are present

### Step 2: Embedding Generation
1. Use existing embedding provider infrastructure (`get_embedding_provider_instance()` from `/codebase_rag/embedder.py`)
2. Generate embeddings for:
   - Entity `properties.description` field (required for all entities)
   - Relationship `properties.description` field (optional, only if present)
3. Reuse existing embedding cache (`EmbeddingCache` class) to avoid duplicate embedding generation for identical content
4. Store embedding results with associated metadata linking to graph node/relationship IDs

### Step 3: Knowledge Graph Ingestion
1. Use existing Cypher query builders (`build_merge_relationship_query`, `build_create_relationship_query` from `/codebase_rag/cypher_queries.py`)
2. Merge entities (upsert) to avoid duplicate nodes: use `dataset_id + entity.id` as the unique key for matching existing nodes
3. Create relationships between merged entities
4. Attach all custom properties to nodes and relationships
5. Add `source_dataset` property to all nodes/relationships from the same dataset for easy filtering/cleanup later

### Step 4: Vector Database Ingestion
1. Use existing vector store backend (`store_embedding_batch()` from `/codebase_rag/vector_store.py`)
2. Store embeddings with metadata including:
   - Graph node/relationship ID reference
   - Dataset ID
   - Entity/relationship labels/type
   - All custom properties required for search filtering
3. Maintain 1:1 mapping between graph nodes/relationships and vector store entries for cross-reference between semantic search results and graph traversal

---

## 4. Integration with Existing System Components
| Existing Component | Usage Purpose |
|---------------------|---------------|
| `codebase_rag.embedder.EmbeddingProvider` | Reuse for embedding generation, supports all configured embedding models |
| `codebase_rag.vector_store.store_embedding_batch` | Bulk insert embeddings into configured vector backend (Qdrant/Memgraph) |
| `codebase_rag.cypher_queries` | Reuse existing query builders for graph CRUD operations |
| `codebase_rag.models.GraphRelationship` | Align relationship data model with existing internal type system |
| `codebase_rag.types_defs.RelationshipData` | Validate input data against existing type definitions |

---

## 5. User Interface
### CLI Interface
Add new CLI command to `main.py`:
```bash
cgr ingest-json --input-path /path/to/json/files/ [--dataset-id custom_id] [--skip-existing] [--incremental] [--dry-run] [--conflict-resolution <strategy>]
```
- `--input-path`: Path to single JSON file or directory of JSON files
- `--dataset-id`: Optional override for dataset ID (defaults to value in JSON metadata)
- `--skip-existing`: Skip entities/relationships that already exist in the graph
- `--incremental`: Run incremental update, only process changed entities/relationships (uses `last_updated` timestamps to detect changes)
- `--dry-run`: Validate input and calculate changes without writing to databases
- `--conflict-resolution`: Conflict resolution strategy for overlapping updates (options: `last-write-wins` [default], `highest-confidence-wins`, `manual-review`)

### Python API
Expose function for programmatic usage:
```python
def ingest_json_data(
    input_path: str,
    dataset_id: Optional[str] = None,
    skip_existing: bool = False,
    batch_size: int = 100,
    incremental: bool = False,
    dry_run: bool = False,
    conflict_resolution: str = "last-write-wins"
) -> IngestionResult:
    """
    Ingest JSON data into graph and vector database
    Returns: IngestionResult with counts of entities/relationships ingested, updated, deleted, skipped, failed
    """
```

### Event-Driven Streaming API (Optional)
Extends existing `realtime_updater.py` to support JSON update events from streaming sources (Kafka, RabbitMQ, webhooks) for low-latency graph updates:
```python
def handle_json_update_event(
    event: dict,
    dataset_id: str,
    conflict_resolution: str = "last-write-wins"
) -> UpdateResult:
    """Process single incremental JSON update event from streaming source"""
```

---

## 6. Error Handling & Resilience
1. Atomic batch operations: Ingest entities and relationships in batches, rollback entire batch if any error occurs
2. Partial failure support: Option to continue ingestion on error, with detailed error log of failed entries
3. Idempotency: Re-running ingestion with the same dataset ID will update existing entities/relationships instead of creating duplicates
4. Dry run mode: Add `--dry-run` flag to validate input and estimate ingestion size without writing to databases
5. Cleanup support: Add CLI command `cgr delete-dataset --dataset-id <id>` to remove all nodes/relationships and vector entries for a specific dataset
6. Incremental update consistency: Graph and vector database writes are atomic per update operation, no partial state changes if an update fails
7. Monitoring alerts: Integration with existing alerting pipeline to notify on failed ingestion batches, high error rates, or performance degradation

---

## 7. Real-Time Incremental Updates & Monitoring Extension
All features in this section are **optional opt-in**, no changes required for existing batch ingestion workflows.

### 7.1 Supported Operation Types
JSON payloads can include optional operation directives to control incremental updates:
| Operation | Description |
|-----------|-------------|
| `add` (default) | Create new entity/relationship, or update if it already exists (upsert) |
| `update` | Explicitly update an existing entity/relationship, fail if it does not exist |
| `delete` | Remove an existing entity/relationship and its corresponding vector entry |

Operations can be specified at 3 levels (in order of priority):
1. Per-entity / per-relationship level (overrides all higher level settings)
2. Batch level (applies to all entities/relationships in the payload)
3. CLI/API level (default if no operation specified in payload)

### 7.2 Incremental Update Workflow
1. Change detection: Compare incoming entity/relationship `last_updated` timestamps with existing values in the graph
2. Only process entries with newer timestamps (or explicit `delete` operations)
3. Atomic update: Apply graph changes first, then sync corresponding vector database changes
4. Audit logging: Record all changes to an immutable audit trail for point-in-time rollback support

### 7.3 Monitoring Integration
Reuses existing system metrics pipeline to track JSON ingestion performance:
| Metric | Description |
|--------|-------------|
| `json_ingestion_entities_total` | Total entities processed (ingested/updated/deleted/skipped) |
| `json_ingestion_relationships_total` | Total relationships processed |
| `json_ingestion_error_count` | Number of failed ingestion operations |
| `json_ingestion_embedding_latency_seconds` | Time taken to generate embeddings for entities/relationships |
| `json_ingestion_graph_write_latency_seconds` | Time taken to write changes to the knowledge graph |
| `json_ingestion_vector_write_latency_seconds` | Time taken to sync changes to the vector database |
| `json_ingestion_batch_duration_seconds` | Total time to process an ingestion batch |

All metrics are tagged with `dataset_id`, `operation_type`, and `status` for granular filtering and alerting.

### 7.4 Conflict Resolution Strategies
For overlapping updates to the same entity/relationship:
1. **Last Write Wins (default)**: Accept the update with the most recent `last_updated` timestamp
2. **Highest Confidence Wins**: Accept the update with the highest `confidence` score (for relationships with confidence values)
3. **Manual Review**: Flag conflicting updates for user review instead of auto-applying them

### 7.5 Consistency Guarantees
- **Atomicity**: Each update operation is fully applied or fully rolled back, no partial state changes
- **Durability**: All changes are persisted to both graph and vector databases before an operation is marked successful
- **Eventual Consistency**: Graph and vector database are typically synced within <100ms for incremental updates
- **Idempotency**: Reapplying the same update multiple times produces the same result as applying it once

---

## 7. Performance Considerations
1. Batch processing: Process entities/relationships in configurable batch sizes (default 100) to optimize database throughput
2. Parallel embedding generation: Use async processing for embedding generation to reduce ingestion time for large datasets
3. Bulk insert support: Use database bulk import APIs for large datasets (>10k entities) instead of individual queries
4. Progress tracking: Add progress bar for CLI ingestion showing number of entities/relationships processed

---

## 8. Testing Requirements
1. Unit tests: Validate schema validation, batch processing, and idempotency
2. Integration tests: Test end-to-end ingestion with sample JSON data, verify entities/relationships exist in graph and embeddings exist in vector store
3. Edge case tests: Test invalid JSON, duplicate IDs, missing required fields, and large datasets
4. Performance tests: Benchmark ingestion time for 1k, 10k, and 100k entity datasets

---

## 9. Out of Scope for v1
1. Automatic entity extraction from unstructured text (this feature only handles pre-extracted entities/relationships)
2. Cross-dataset relationship support (relationships only reference entities within the same dataset)
3. Custom embedding field configuration (defaults to using `description` field for embeddings)
