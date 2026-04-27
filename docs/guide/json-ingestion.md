---
description: "Ingest custom entities, relationships, and domain knowledge from JSON files into Code-Graph-RAG."
---

# JSON Data Ingestion Guide

The JSON data ingestion feature allows you to load pre-processed custom domain data, entities, relationships, and metadata directly into the dedicated JSON Memgraph graph. Entity embeddings are stored on the JSON nodes themselves, which keeps semantic retrieval and graph traversal in the same data store.

## Overview

This feature supports:
- Bulk ingestion of custom entities and relationships from JSON files
- Automatic embedding generation for entities
- Incremental updates for changing datasets
- Idempotent ingestion (re-running the same command does not create duplicates)
- Complete integration with existing Code-Graph-RAG querying and validation capabilities
- Optional real-time streaming ingestion for low-latency updates

## Input JSON Schema

All ingested JSON files must conform to the official ingestion schema. Entities use top-level `name`, `type`, and `labels` fields. Relationships use `source`, `target`, and `relationship` fields.

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
      "type": "string (optional, primary entity type used as a label)",
      "labels": ["string (optional, node labels e.g. ['Job', 'Company'])", "inherited from metadata.default_entity_labels if not specified"],
      "properties": {
        "description": "string (recommended, content used for embedding generation)",
        "any_custom_property": "any (optional custom metadata fields)"
      }
    }
  ],
  "relationships": [
    {
      "id": "string (optional, unique relationship ID)",
      "source": "string (required, references entity.id or entity.name in the same dataset)",
      "target": "string (required, references entity.id or entity.name in the same dataset)",
      "relationship": "string (required, relationship type e.g. 'REQUIRES_SKILL', 'WORKS_AT')",
      "properties": {
        "description": "string (optional relationship metadata)",
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
      "name": "Senior Backend Engineer",
      "type": "JobPosting",
      "labels": ["JobPosting", "SoftwareEngineer"],
      "properties": {
        "description": "Senior backend engineer position requiring Python, Neo4j, and microservices experience. 5+ years of experience required.",
        "salary_range": "$150k-$200k",
        "location": "Remote",
        "posted_date": "2024-10-01"
      }
    },
    {
      "id": "skill_001",
      "name": "Python",
      "type": "Skill",
      "labels": ["Skill", "Technology"],
      "properties": {
        "description": "Python programming language, version 3.8+",
        "category": "Programming Language"
      }
    },
    {
      "id": "skill_002",
      "name": "Neo4j",
      "type": "Skill",
      "labels": ["Skill", "Technology"],
      "properties": {
        "description": "Neo4j graph database, including Cypher query language",
        "category": "Database"
      }
    }
  ],
  "relationships": [
    {
      "source": "job_001",
      "target": "skill_001",
      "relationship": "REQUIRES_SKILL",
      "category": "CAUSAL",
      "properties": {
        "description": "Job requires Python proficiency",
        "minimum_experience_years": 3,
        "required_level": "Expert"
      }
    },
    {
      "source": "job_001",
      "target": "skill_002",
      "relationship": "REQUIRES_SKILL",
      "category": "CAUSAL",
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
- JSON structure is validated against the official schema and Pydantic models
- Duplicate entity IDs within the dataset are checked
- Relationship references are validated against entity IDs or names
- Required fields are validated for presence and correct type

### 2. Embedding Generation
- Uses your configured embedding provider (same as code GraphRAG)
- Generates embeddings for entity `properties.description`, falling back to `name` if needed
- Reuses existing embedding cache to avoid duplicate work for identical content
- Embeddings are stored directly on `JsonEntity` nodes in the JSON Memgraph instance

**Optional Embeddings**: JSON embeddings are optional. If your embedding provider is unavailable (e.g., PyTorch is not installed for local embeddings), ingestion continues without embeddings and entities are stored without vector properties. You can control this behavior with:
- `CGR_JSON_EMBEDDINGS_ENABLED=false` — disable JSON embeddings entirely
- `CGR_JSON_EMBEDDINGS_REQUIRED=true` — fail ingestion when embeddings are unavailable

### 3. Knowledge Graph Ingestion
- Entities are merged (upserted) using `dataset_id + entity.id` as the unique key
- Relationships are created after entity ingestion, so cross-file references resolve correctly
- All custom properties are attached to nodes and relationships
- All JSON entities receive the stable `JsonEntity` label in addition to user-provided labels

### 4. Vector Indexing
- JSON entity embeddings are indexed directly in the JSON Memgraph instance
- The JSON ingestion path does not persist relationship embeddings
- Dataset deletion removes JSON nodes and relationships without a separate vector-store cleanup step

### 5. Entity Category Taxonomy
All JSON entities are automatically classified into one of 7 MECE (Mutually Exclusive, Collectively Exhaustive) categories:

| Category | Emoji | Description | Example Types |
|----------|-------|-------------|---------------|
| `CONCRETE_ENTITY` | 🧱 | Physical objects or instances | Tool, Product, Facility |
| `EVENT_PROCESS` | ⏱️ | Things that unfold over time | ProgressionStage, Deployment, Migration |
| `INFORMATION_EXPRESSION` | 📨 | Representations of data/knowledge | GovernanceRule, Reference, API |
| `PROPERTY_ATTRIBUTE` | 📏 | Characteristics or measurements | SafetyBoundary, SLI, Metric |
| `SYSTEM_STRUCTURE` | 🏗️ | Organized collections or frameworks | Framework, Layer, GovernanceConstruct |
| `AGENT_ROLE` | 🎭 | Entities that exercise intention | Role, Stakeholder, CI Bot |
| `ABSTRACT_CONCEPT` | 💡 | Pure ideas or mental constructs | Mindset, Competency, AntiPattern |

**Automatic Classification**: Entities are classified based on their `type` field using a 122+ entry subtype registry. If no match is found, entities default to `ABSTRACT_CONCEPT`.

**Stored Properties**:
- `entity_category` — The MECE category name (e.g., `"AGENT_ROLE"`)
- `entity_subtype` — The specific type (e.g., `"Role"`)
- `entity_emoji` — The category emoji (e.g., `"🎭"`)

### 6. Relationship Verb Inference
Relationships automatically receive inferred properties for precise semantic queries:

| Category | Emoji | Default Verb | Type-Specific Examples |
|----------|-------|--------------|------------------------|
| `CAUSAL` | ⚡ | `influences` | Mindset → Competency: `enables` |
| `COMPOSITIONAL` | 🧩 | `part-of` | Framework → Framework: `contains` |
| `HIERARCHICAL` | 🌳 | `is-a` | Competency → Competency: `subtype-of` |
| `CONTEXTUAL` | 🎯 | `contextualizes` | Layer → Construct: `defines-context` |
| `SEQUENTIAL` | ⏩ | `precedes` | ProgressionStage → ProgressionStage: `precedes` |
| `COMPARATIVE` | ⚖️ | `compares-to` | Competency → Competency: `contrasts-with` |
| `ATTRIBUTIVE` | 💭 | `has-property` | Framework → Property: `characterizes` |
| `ANALOGICAL` | 🌉 | `analogous-to` | Concept → Concept: `similar-to` |
| `RELATED_TO` | 🔗 | `related-to` | Generic fallback |

**Stored Properties**:
- `relationship_category` — The normalized category (e.g., `"CAUSAL"`)
- `relationship_emoji` — The category emoji (e.g., `"⚡"`)
- `verb` — The inferred or provided verb (e.g., `"enables"`)

**Category Normalization**: JSON categories like `ATTRIBUTE` are automatically mapped to internal taxonomy (e.g., `ATTRIBUTIVE`).

### 7. Emoji Extraction
Entity names with emoji prefixes are automatically parsed:

```json
{"name": "👤 CTO Role"}
```

Results in stored properties:
- `name`: `"CTO Role"` (clean name for search)
- `emoji`: `"👤"` (extracted prefix)
- `display_name`: `"👤 CTO Role"` (original for display)

### 8. Post-Ingestion Processing
After successful ingestion:

1. **Property Indexes Created**:
   - `entity_category` — Fast filtering by MECE category
   - `type` — Fast filtering by entity type
   - `pagerank_score` — Efficient importance ranking

2. **PageRank Computation** (optional, enabled by default):
   - Calculates importance scores for all entities
   - Improves ranking quality for semantic search
   - Requires MAGE procedures (gracefully skipped if unavailable)

Control with CLI flag:
```bash
cgr start --ingest-json --json-compute-pagerank  # enable (default)
cgr start --ingest-json --no-json-compute-pagerank  # disable
```

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
from codebase_rag.json_ingestion import ingest_json_data, IngestionResult

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

Once ingested, your custom JSON data can be queried using the `query_json_graph` tool available in the interactive chat:

### JSON Graph Query Tool

The `query_json_graph` tool provides semantic search, category filtering, and relationship traversal for JSON-ingested entities:

**Natural Language Queries**:
```
> What competencies does a CTO need?
> Show me all stakeholders in the organization
> What does Strategic Thinking influence?
> Find all frameworks related to governance
```

**Category Filtering**:
Query by MECE entity category:
```
> Show me all agents and roles in the system
> List all abstract concepts in the CTO framework
> What concrete entities (tools) are available?
```

**Relationship Traversal**:
Explore connections between entities:
```
> What competencies are enabled by Strategic Thinking?
> Show the hierarchy under the CTO Framework
> What relationships involve the Technical Architecture competency?
```

### Query Methods Available

| Method | Description | Requirements |
|--------|-------------|--------------|
| `semantic_search` | Vector similarity + PageRank ranking | Embeddings, vector index |
| `keyword_search` | Name/description keyword matching | None |
| `find_related_entities` | Graph traversal from entity | Relationships |
| `find_relationships` | Filter by category/verb | Relationships |
| `get_entities_by_category` | Filter by MECE category | Entity categories |
| `get_important_entities` | Top entities by PageRank | PageRank scores |
| `get_entity_taxonomy` | Hierarchical taxonomy view | Relationships |

### Graceful Degradation

All query methods degrade gracefully when optional features are unavailable:

| Scenario | Fallback Behavior |
|----------|-------------------|
| No embedding provider | Falls back to keyword search |
| No vector index | Falls back to keyword search |
| No PageRank scores | Uses default score 0.1 |
| No entity category | Treats as `ABSTRACT_CONCEPT` |
| No relationship category | Uses `"RELATED_TO"` |
| No relationship emoji | Uses `"🔗"` |
| No relationship verb | Uses `"related-to"` |

**Backward Compatibility**: The `find_relationships` method uses `COALESCE(r.category, r.relationship_category, type(r))` for filtering, so it works correctly with:
- New data that has the `category` property
- Legacy data that has the `relationship_category` property
- Very old data that only has the relationship type (e.g., `CAUSAL`)

### Python API

You can also query JSON data programmatically:

```python
from codebase_rag.json_queries import JsonGraphQueryEngine
from codebase_rag.json_ingestion import _create_json_ingestor
from codebase_rag.config import settings

with _create_json_ingestor(settings.JSON_MEMGRAPH_BATCH_SIZE) as executor:
    engine = JsonGraphQueryEngine(executor=executor)
    
    # Semantic search
    results = engine.semantic_search("strategic thinking", top_k=10)
    
    # Filter by category
    roles = engine.get_entities_by_category("AGENT_ROLE", top_k=50)
    
    # Find relationships
    rels = engine.find_relationships("Strategic Thinking", relationship_category="CAUSAL")
    
    # Get important entities
    important = engine.get_important_entities(top_k=20)
```

### Combined Code + Custom Data Queries

You can query across both your ingested custom data and codebase:

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
