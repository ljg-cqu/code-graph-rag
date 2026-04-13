---
description: "Ingest custom entities, relationships, and domain knowledge from JSON files into Code-Graph-RAG."
---

# JSON Data Ingestion Guide

The JSON data ingestion feature allows you to load pre-processed custom domain data, entities, relationships, and metadata directly into both the Code-Graph-RAG knowledge graph (Memgraph) and vector database. This enables you to combine your custom domain knowledge with code analysis for comprehensive RAG queries.

## Overview

This feature supports:
- Bulk ingestion of custom entities and relationships from JSON files
- Automatic embedding generation for all entities and relationships
- Incremental updates for changing datasets
- Atomic batch operations with rollback support
- Idempotent ingestion (re-running the same command does not create duplicates)
- Complete integration with existing Code-Graph-RAG querying and validation capabilities
- Optional real-time streaming ingestion for low-latency updates

## Input JSON Schema

All ingested JSON files must conform to the following schema, which aligns with existing internal Code-Graph-RAG models:

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
      "labels": ["string (required, node labels e.g. ['Job', 'Company'])", "inherited from metadata.default_entity_labels if not specified"],
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

### Example Valid JSON

```json
{
  "metadata": {
    "dataset_id": "job_hunting_jd_2024",
    "source": "public job board (2024 Q4)"
  },
  "entities": [
    {
      "id": "job_001",
      "labels": ["JobPosting", "SoftwareEngineer"],
      "properties": {
        "name": "Senior Backend Engineer",
        "description": "Senior backend engineer position requiring Python, Neo4j, and microservices experience. 5+ years of experience required.",
        "salary_range": "$150k-$200k",
        "location": "Remote",
        "posted_date": "2024-10-01"
      }
    },
    {
      "id": "skill_001",
      "labels": ["Skill", "Technology"],
      "properties": {
        "name": "Python",
        "description": "Python programming language, version 3.8+",
        "category": "Programming Language"
      }
    },
    {
      "id": "skill_002",
      "labels": ["Skill", "Technology"],
      "properties": {
        "name": "Neo4j",
        "description": "Neo4j graph database, including Cypher query language",
        "category": "Database"
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
        "minimum_experience_years": 3,
        "required_level": "Expert"
      }
    },
    {
      "source_entity_id": "job_001",
      "target_entity_id": "skill_002",
      "type": "REQUIRES_SKILL",
      "properties": {
        "description": "Job requires Neo4j experience",
        "minimum_experience_years": 2,
        "required_level": "Intermediate"
      }
    }
  ]
}
```

## Core Ingestion Workflow

### 1. Input Validation
- JSON structure is validated against the schema using Pydantic
- Duplicate entity IDs within the dataset are checked
- All relationship source and target entity IDs are verified to exist in the dataset
- Required fields are validated for presence and correct type

### 2. Embedding Generation
- Uses your configured embedding provider (same as code GraphRAG)
- Generates embeddings for entity `properties.description` (required for all entities)
- Generates embeddings for relationship `properties.description` (optional, only if present)
- Reuses existing embedding cache to avoid duplicate work for identical content
- Embeddings are stored with metadata linking to the corresponding graph node/relationship IDs

### 3. Knowledge Graph Ingestion
- Uses existing Cypher query builders from Code-Graph-RAG
- Entities are merged (upserted) to avoid duplicates: `dataset_id + entity.id` is used as the unique key
- Relationships are created between the merged entities
- All custom properties are attached to nodes and relationships
- All nodes and relationships from the same dataset get a `source_dataset` property for easy filtering and cleanup

### 4. Vector Database Ingestion
- Embeddings are stored in the configured vector backend (Qdrant/Memgraph)
- Metadata includes:
  - Graph node/relationship ID reference
  - Dataset ID
  - Entity/relationship labels/type
  - All custom properties needed for search filtering
- Maintains 1:1 mapping between graph nodes/relationships and vector store entries for cross-reference between semantic search results and graph traversal

## CLI Usage

### Basic Ingestion
Ingest a single JSON file:
```bash
cgr ingest-json --input-path /path/to/your/data.json
```

Ingest an entire directory of JSON files:
```bash
cgr ingest-json --input-path /path/to/json/directory
```

### Options
| Option | Description |
|--------|-------------|
| `--dataset-id <id>` | Override the dataset ID defined in JSON metadata |
| `--skip-existing` | Skip entities/relationships that already exist in the graph |
| `--incremental` | Run incremental update, only process changed entities/relationships using last modified timestamps |
| `--dry-run` | Validate input and calculate changes without writing to databases |
| `--conflict-resolution <strategy>` | Conflict resolution strategy for overlapping updates: `last-write-wins` (default), `highest-confidence-wins`, `manual-review` |
| `--batch-size <number>` | Override default batch size for database operations (default: 100) |

### Examples
Dry run to validate your JSON before ingestion:
```bash
cgr ingest-json --input-path ./dataset.json --dry-run
```

Ingest with custom dataset ID and skip existing entries:
```bash
cgr ingest-json --input-path ./datasets/2024 --dataset-id internal-knowledge-base --skip-existing
```

Incremental update for a changing dataset:
```bash
cgr ingest-json --input-path ./live-dataset --incremental --conflict-resolution highest-confidence-wins
```

## Python API

You can also use the JSON ingestion functionality programmatically via the Python API:

```python
from codebase_rag.ingestion.json_ingestor import ingest_json_data, IngestionResult

result: IngestionResult = ingest_json_data(
    input_path="/path/to/your/data.json",
    dataset_id="custom-dataset-id",
    skip_existing=False,
    batch_size=100,
    incremental=True,
    dry_run=False,
    conflict_resolution="last-write-wins"
)

print(f"Ingestion completed:")
print(f"  Entities ingested: {result.entities_ingested}")
print(f"  Entities updated: {result.entities_updated}")
print(f"  Entities skipped: {result.entities_skipped}")
print(f"  Relationships ingested: {result.relationships_ingested}")
print(f"  Relationships updated: {result.relationships_updated}")
print(f"  Relationships skipped: {result.relationships_skipped}")
print(f"  Errors: {len(result.errors)}")
```

## Real-Time Streaming Ingestion (Optional)

For low-latency updates, you can use the streaming API to ingest individual JSON update events:

```python
from codebase_rag.ingestion.json_stream_handler import handle_json_update_event, UpdateResult

event = {
    "operation": "add",
    "entities": [
        {
            "id": "new_job_002",
            "labels": ["JobPosting"],
            "properties": {
                "name": "Data Engineer",
                "description": "Data engineer position requiring Python, Spark, and SQL experience."
            }
        }
    ],
    "relationships": []
}

result: UpdateResult = handle_json_update_event(
    event=event,
    dataset_id="live-job-postings",
    conflict_resolution="last-write-wins"
)
```

### Supported Operation Types
| Operation | Description |
|-----------|-------------|
| `add` (default) | Create new entity/relationship, or update if it already exists (upsert) |
| `update` | Explicitly update an existing entity/relationship, fail if it does not exist |
| `delete` | Remove an existing entity/relationship and its corresponding vector entry |

Operations can be specified at three levels (priority order):
1. Per entity/relationship level
2. Batch level (applies to all entries in the payload)
3. CLI/API level (default if no operation is specified)

## Dataset Management

### Delete an Entire Dataset
Remove all nodes, relationships, and vector entries for a specific dataset:
```bash
cgr delete-dataset --dataset-id your-dataset-id
```

### List All Datasets
Show all datasets currently in the knowledge graph:
```bash
cgr list-datasets
```

## Querying Ingested Data

Once ingested, your custom JSON data is fully integrated into the Code-Graph-RAG system and can be queried just like code entities:

### Natural Language Queries
```
> Find all job postings that require Python skills
> What skills are required for senior backend engineer positions?
> Show me all remote jobs with salary above $150k
```

### Combined Code + Custom Data Queries
You can query across both your ingested custom data and codebase at the same time:
```
> What Python functions in the codebase are related to the job requirements for backend engineers?
> Compare the skill requirements in our job postings with the skills used in our actual codebase
```

### Validation
Use the existing validation capabilities to verify your data:
```bash
# Validate that your ingested data matches a specification document
cgr validate-spec --spec-path ./job-posting-requirements.md --dataset-id job_hunting_jd_2024
```

## Error Handling & Resilience

- **Atomic Batch Operations**: Entire batches are rolled back if any error occurs during ingestion
- **Partial Failure Support**: Option to continue ingestion on error with detailed error logging
- **Idempotent Ingestion**: Re-running the same ingestion command produces the same result without creating duplicates
- **Audit Trail**: All changes are logged with timestamps and operation details for compliance
- **Conflict Resolution**: Multiple strategies available for handling overlapping updates

## Performance Considerations

- **Batch Processing**: Configurable batch sizes to optimize database throughput (default: 100)
- **Parallel Embedding Generation**: Async processing reduces ingestion time for large datasets
- **Bulk Import**: Uses database bulk import APIs for datasets with >10k entities
- **Progress Tracking**: CLI shows progress bar with estimated time remaining for large ingestion jobs

## Out of Scope for v1
- Automatic entity extraction from unstructured text (this feature only handles pre-processed entities)
- Cross-dataset relationships (relationships can only reference entities within the same dataset)
- Custom embedding field configuration (defaults to using the `description` field for embeddings)
