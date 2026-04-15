# JSON Data Ingestion Design Specification
## Version: 1.0 | Last Updated: 2024
## Target Path: `/home/zealy/github/ljg-cqu/code-graph-rag/.specs/json_data_ingestion_spec.md`

---

## 1. Overview & Purpose
This feature enables users to ingest pre-processed JSON data containing entities, relationships, and metadata directly into the dedicated JSON Memgraph graph. Entity embeddings are stored on JSON graph nodes themselves, so semantic similarity search and graph traversal operate over the same dataset boundary.

The feature includes **optional opt-in real-time incremental update and enterprise monitoring support**, fully reusing existing system infrastructure for unified workflow with unstructured document ingestion. All core functionality remains backwards compatible with existing batch ingestion use cases.

---

## 2. Input JSON Schema Definition
All ingested JSON files MUST conform to the official ingestion schema. The canonical contract is top-level `entity.name`, optional `entity.type`, optional `entity.labels`, and relationship fields `source`, `target`, and `relationship`.

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
      "id": "string (optional, unique entity ID within dataset; auto-generated from name if omitted)",
      "name": "string (required, human-readable entity name)",
      "type": "string (optional, primary entity type)",
      "labels": ["string (optional, node labels e.g. ['Job', 'Company'])"],
      "properties": {
        "description": "string (recommended, content used for embedding generation)",
        "any_custom_property": "any (optional custom metadata fields)"
      }
    }
  ],
  "relationships": [
    {
      "id": "string (optional, unique relationship ID)",
      "source": "string (required, references entity.id or entity.name in same dataset)",
      "target": "string (required, references entity.id or entity.name in same dataset)",
      "relationship": "string (required, relationship type e.g. 'REQUIRES_SKILL', 'WORKS_AT')",
      "properties": {
        "description": "string (optional relationship metadata)",
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
      "name": "Senior Backend Engineer",
      "type": "JobPosting",
      "labels": ["JobPosting", "SoftwareEngineer"],
      "properties": {
        "description": "Senior backend engineer position requiring Python, Neo4j, and microservices experience. 5+ years of experience required.",
        "salary_range": "$150k-$200k",
        "location": "Remote"
      }
    },
    {
      "id": "skill_001",
      "name": "Python",
      "type": "Skill",
      "labels": ["Skill", "Technology"],
      "properties": {
        "description": "Python programming language, version 3.8+"
      }
    }
  ],
  "relationships": [
    {
      "source": "job_001",
      "target": "skill_001",
      "relationship": "REQUIRES_SKILL",
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
1. Validate JSON structure against the official schema and Pydantic models
2. Check for duplicate entity IDs within the dataset
3. Verify relationship references resolve by entity ID or name
4. Validate that all required fields are present

### Step 2: Embedding Generation
1. Use existing embedding provider infrastructure (`get_embedding_provider_instance()` from `/codebase_rag/embedder.py`)
2. Generate embeddings for:
  - Entity `properties.description` field, falling back to `name`
3. Reuse existing embedding cache (`EmbeddingCache` class) to avoid duplicate embedding generation for identical content
4. Store embeddings directly on `JsonEntity` nodes in the JSON graph

### Step 3: Knowledge Graph Ingestion
1. Merge entities (upsert) using `dataset_id + entity.id` as the unique key
2. Always add the stable `JsonEntity` label to JSON nodes
3. Create relationships only after entity ingestion completes, so cross-file references resolve correctly
4. Attach all custom properties to nodes and relationships

### Step 4: Vector Indexing
1. Create a JSON-specific vector index on `:JsonEntity(embedding)` in the JSON Memgraph instance
2. Do not persist relationship embeddings in phase 1
3. Rely on node deletion to remove JSON embeddings alongside their owning nodes

---

## 4. Integration with Existing System Components
| Existing Component | Usage Purpose |
|---------------------|---------------|
| `codebase_rag.embedder.EmbeddingProvider` | Reuse for embedding generation, supports all configured embedding models |
| `codebase_rag.services.graph_service.MemgraphIngestor` | Write JSON entities and relationships directly to the JSON Memgraph instance |
| `codebase_rag.embedder.EmbeddingCache` | Reuse cached entity embeddings across ingest runs |

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
