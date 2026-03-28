# Design Specification: Dual Vector Backend Support for Code-Graph-RAG

**Status**: Draft | **Version**: 1.0.0 | **Date**: 2026-03-28

---

## Executive Summary

This specification defines the architecture for supporting both **Qdrant** and **Memgraph native vector storage** as backends for semantic search in Code-Graph-RAG, with **Memgraph native as the default**.

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Default to Memgraph native | Simplifies deployment, single database, unified queries |
| Keep Qdrant as option | Remote deployment, proven scale (billions of vectors), specialized optimization |
| Backend abstraction layer | Clean separation, runtime switching, independent testing |

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Memgraph Vector Capabilities Research](#2-memgraph-vector-capabilities-research)
3. [Proposed Architecture](#3-proposed-architecture)
4. [Data Model Changes](#4-data-model-changes)
5. [Configuration Changes](#5-configuration-changes)
6. [API Design](#6-api-design)
7. [Cypher Query Templates](#7-cypher-query-templates)
8. [GraphRAG Lifecycle Integration](#8-graphrag-lifecycle-integration)
9. [MCP Tool Updates](#9-mcp-tool-updates)
10. [Docker Deployment](#10-docker-deployment)
11. [Migration Path](#11-migration-path)
12. [Testing Strategy](#12-testing-strategy)

---

## 1. Current Architecture Analysis

### 1.1 Current Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CURRENT ARCHITECTURE (Qdrant)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Source Files ──► Tree-sitter ──► AST ──► Memgraph (Nodes/Relationships)    │
│                                                  │                          │
│                                                  ▼                          │
│                              Pass 4: Embedding Generation                    │
│                                                  │                          │
│                    ┌─────────────────────────────┴───────────────────┐       │
│                    ▼                                                 ▼       │
│              Memgraph (metadata)                            Qdrant (vectors) │
│              - node_id                                      - point_id      │
│              - qualified_name                               - vector[768]   │
│              - name                                         - payload       │
│              - labels                                                      │
│                    │                                                 │       │
│                    └─────────────────────┬───────────────────────────┘       │
│                                          ▼                                   │
│                              Two-Phase Query:                                │
│                              1. Qdrant: vector similarity                    │
│                              2. Memgraph: fetch node metadata                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Current Components

| Component | File | Purpose |
|-----------|------|---------|
| **Embedder** | `embedder.py` | UniXcoder model (768-dim) |
| **Vector Store** | `vector_store.py` | Qdrant client operations |
| **Graph Updater** | `graph_updater.py` | Pass 4 embedding generation |
| **Semantic Search** | `tools/semantic_search.py` | Two-phase query execution |

### 1.3 Current Node-ID Mapping

```python
# Qdrant Point Structure (vector_store.py lines 73-83)
PointStruct(
    id=node_id,              # Memgraph internal ID → Qdrant point ID (1:1 mapping)
    vector=embedding,        # 768-dimensional UniXcoder embedding
    payload={
        "node_id": node_id,
        "qualified_name": qualified_name,
    },
)
```

### 1.4 Issues with Current Architecture

| Issue | Description |
|-------|-------------|
| **Two systems** | Memgraph + Qdrant increases deployment complexity |
| **Two-phase query** | Latency from cross-database coordination |
| **Separate storage** | Vectors stored outside graph, no native hybrid queries |
| **No graph context** | Qdrant can't leverage graph relationships for ranking |

---

## 2. Memgraph Vector Capabilities Research

### 2.1 Native Vector Index Support (Memgraph v3.0.0+)

**Stable since January 2024** - Memgraph has native vector index support via Cypher DDL:

```cypher
-- Create vector index (stable API since v3.0.0)
-- Note: CONFIG uses JSON-style quoted keys
CREATE VECTOR INDEX index_name
ON :Label(property)
WITH CONFIG {"dimension": 768, "capacity": 10000, "metric": "cos"};

-- Drop vector index
DROP VECTOR INDEX index_name;

-- Show index info
SHOW VECTOR INDEX INFO;
CALL vector_search.show_index_info() YIELD * RETURN *;
```

**Version History:**
| Version | Date | Milestone |
|---------|------|-----------|
| v3.0.0 | Jan 2024 | Vector search became stable API (removed from experimental) |
| v3.9.0 | Mar 2026 | Vector indexes support dynamic config, memory tracking |

**Required Configuration Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `dimension` | int | **Required** - Vector dimension (must match embedding model) |
| `capacity` | int | **Required** - Minimum index capacity (prefers powers of 2) |
| `metric` | string | Optional - Distance metric (default: `l2sq`) |
| `resize_coefficient` | int | Optional - Multiplier when index fills (default: 2) |
| `scalar_kind` | string | Optional - Data type (default: `f32`) |

**Supported Metrics:**
| Metric | Description |
|--------|-------------|
| `l2sq` | Squared Euclidean distance (default) |
| `cos` | Cosine similarity (recommended for embeddings) |
| `ip` | Inner product (dot product) |
| `pearson` | Pearson correlation coefficient |
| `haversine` | Haversine distance (geographic) |
| `hamming`, `tanimoto`, `sorensen`, `jaccard` | Specialized metrics |

**Scalar Kinds (Memory Optimization):**
| Kind | Description |
|------|-------------|
| `f32` | 32-bit float (default) |
| `f64` | 64-bit float (double precision) |
| `f16` | 16-bit float (reduced precision, lower memory) |
| `bf16` | 16-bit bfloat16 |
| `f8` | 8-bit float (minimal memory) |

#### 2.1.1 Vector Search Query Syntax

```cypher
-- Basic vector search (correct parameter order)
CALL vector_search.search("index_name", $top_k, $query_vector)
YIELD node, distance, similarity
RETURN node.qualified_name, distance, similarity
ORDER BY distance ASC;

-- Search with post-filtering
CALL vector_search.search("code_embedding_index", 10, $embedding)
YIELD node, distance, similarity
WHERE node.qualified_name STARTS WITH $project_prefix
RETURN node, distance, similarity;

-- Search on edges (if indexed)
CALL vector_search.search_edges("edge_index", $top_k, $query_vector)
YIELD edges, distance, similarity
RETURN edges, distance;
```

**Output Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `node` | Vertex | Matching node |
| `distance` | double | Distance from query vector |
| `similarity` | double | Similarity score |

**Important:** Query planner does NOT automatically use vector indices - must call procedures explicitly.

#### 2.1.2 Utility Functions

```cypher
-- Cosine similarity between two vectors (no index needed)
RETURN vector_search.cosine_similarity([1.0, 2.0], [1.0, 3.0]) AS similarity;
-- Returns: 0.9899...
```

#### 2.1.3 Memory Tracking

```cypher
-- Check memory usage
SHOW STORAGE INFO;
-- Returns:
-- graph_memory_tracked: memory for graph structures
-- vector_index_memory_tracked: memory for vector indices
-- memory_tracked: total (sum)
```

#### 2.1.2 Hybrid Search Pattern (Native)

```cypher
-- Vector search + graph traversal in single query
CALL vector_search.search('embedding_index', $embedding, 10)
YIELD node, score
MATCH (node)-[:CALLS*1..3]->(callee:Function)
RETURN
    node.qualified_name as similar_function,
    score,
    collect(DISTINCT callee.qualified_name) as call_chain
ORDER BY score DESC;
```

### 2.2 MAGE (Memgraph Advanced Graph Extensions)

**Current Docker Image**: `memgraph/memgraph-mage` (already in use)

MAGE provides additional algorithms for enhanced search:

| Module | Use Case |
|--------|----------|
| `pagerank_module` | Rank search results by importance |
| `betweenness_centrality_module` | Find critical functions |
| `weakly_connected_components` | Group related code |
| `node_similarity_module` | Graph-based similarity (Jaccard, Cosine) |
| `embeddings.py` | SentenceTransformer embedding utilities |

#### 2.2.1 MAGE Embedding Worker

```python
# MAGE's embed_worker uses SentenceTransformer
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_id, device=device)
embs = model.encode(texts, normalize_embeddings=True)
```

### 2.3 Key Insight: Hybrid Query Advantage

**Memgraph enables hybrid queries impossible with Qdrant alone:**

```cypher
-- Vector search + PageRank ranking
CALL vector_search.search('embedding_index', $embedding, 50)
YIELD node, score
WITH node, score,
     size((node)-[:CALLS]->()) + size((:Function)-[:CALLS]->(node)) AS degree
RETURN
    node.qualified_name,
    score as vector_similarity,
    degree,
    (score * 0.7 + degree * 0.01) AS hybrid_score
ORDER BY hybrid_score DESC
LIMIT 10;
```

---

## 3. Proposed Architecture

### 3.1 Dual Backend Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROPOSED DUAL BACKEND ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌──────────────────┐                           │
│                              │  VectorBackend   │ (Protocol/ABC)            │
│                              │  - store_batch() │                           │
│                              │  - search()      │                           │
│                              │  - delete()      │                           │
│                              └────────┬─────────┘                           │
│                                       │                                     │
│                    ┌──────────────────┴──────────────────┐                  │
│                    ▼                                     ▼                  │
│         ┌─────────────────────┐             ┌─────────────────────┐         │
│         │  MemgraphBackend    │             │   QdrantBackend     │         │
│         │  (DEFAULT)          │             │   (OPTIONAL)        │         │
│         ├─────────────────────┤             ├─────────────────────┤         │
│         │ - node.embedding    │             │ - Qdrant collection │         │
│         │ - knn_module        │             │ - HNSW index        │         │
│         │ - native Cypher     │             │ - Remote URI        │         │
│         │ - hybrid queries    │             │ - quantization      │         │
│         └─────────────────────┘             └─────────────────────┘         │
│                                                                             │
│  Backend Selection: VECTOR_STORE_BACKEND env var (default: "memgraph")     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Backend Comparison Matrix

| Feature | Memgraph Native | Qdrant |
|---------|-----------------|--------|
| **Deployment** | Single container | Extra container or cloud |
| **Vector Storage** | Node property | Dedicated collection |
| **Index Algorithm** | KNN module | HNSW (optimized) |
| **Distance Metrics** | COSINE, EUCLIDEAN | COSINE, L2, DOT_PRODUCT |
| **Hybrid Queries** | ✅ Native | ❌ Requires two queries |
| **Graph Traversal** | ✅ Same query | ❌ Separate query |
| **Scale Limit** | ~Millions | ~Billions |
| **Remote Support** | ❌ No | ✅ QDRANT_URI |
| **Quantization** | ❌ No | ✅ Scalar/Product/Binary |

### 3.3 When to Use Each Backend

| Use Case | Recommended Backend |
|----------|---------------------|
| **Single-machine deployment** | Memgraph (simpler) |
| **Codebases < 1M functions** | Memgraph |
| **Need hybrid search** | Memgraph |
| **Cloud/remote vectors** | Qdrant |
| **Large scale (>1M vectors)** | Qdrant |
| **Memory optimization needed** | Qdrant (quantization) |

---

## 4. Data Model Changes

### 4.1 Node Schema Extension

Add `embedding` property to embeddable node types:

```python
# types_defs.py - Extended Node Schema

EMBEDDABLE_NODE_LABELS = frozenset({
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
    NodeLabel.CLASS,
    NodeLabel.INTERFACE,
    NodeLabel.CONTRACT,  # Solidity
})

# Node properties for Memgraph native storage
NODE_EMBEDDING_PROPERTIES = {
    "embedding": list[float],        # 768-dimensional vector
    "embedding_model": str,          # "microsoft/unixcoder-base"
    "embedding_version": int,        # For migration tracking
    "embedding_hash": str,           # Content hash for deduplication
}
```

### 4.2 Memgraph Node Creation with Embedding

```cypher
-- Create Function node with embedding
MERGE (f:Function {qualified_name: $qualified_name})
SET f.name = $name,
    f.start_line = $start_line,
    f.end_line = $end_line,
    f.embedding = $embedding,
    f.embedding_model = $model_name,
    f.embedding_version = 1
RETURN id(f) as node_id;
```

### 4.3 Vector Index Creation

```cypher
-- Create KNN index for Function embeddings
CALL knn_module.create_index(
  "embedding",     -- property name
  768,             -- dimension
  "cosine"         -- distance metric
);

-- Create index for Method embeddings (same property)
CALL knn_module.create_index("embedding", 768, "cosine");
```

---

## 5. Configuration Changes

### 5.1 New Environment Variables

```python
# config.py - New settings

class AppConfig(BaseSettings):
    # Backend Selection
    VECTOR_STORE_BACKEND: Literal["memgraph", "qdrant"] = "memgraph"

    # Memgraph Vector Settings (REQUIRED for native vector index)
    MEMGRAPH_VECTOR_INDEX_NAME: str = "code_embeddings"
    MEMGRAPH_VECTOR_DIM: int = 768              # Must match embedding model
    MEMGRAPH_VECTOR_CAPACITY: int = 100000      # REQUIRED - index capacity (use power of 2)
    MEMGRAPH_VECTOR_METRIC: str = "cos"         # Options: l2sq, cos, ip, pearson
    MEMGRAPH_VECTOR_SCALAR_KIND: str = "f32"    # Options: f32, f64, f16, bf16, f8

    # Unified Vector Settings
    VECTOR_SEARCH_TOP_K: int = 5
    VECTOR_EMBEDDING_BATCH_SIZE: int = 50
    VECTOR_MIN_SIMILARITY: float = 0.0

    # Keep Qdrant settings for backward compatibility
    QDRANT_DB_PATH: str = "./.qdrant_code_embeddings"
    QDRANT_COLLECTION_NAME: str = "code_embeddings"
    QDRANT_VECTOR_DIM: int = 768
    QDRANT_URI: str | None = None
    QDRANT_TOP_K: int = 5
```

### 5.2 .env.example Additions

```bash
# ============================================
# Vector Store Backend Configuration
# ============================================

# Backend: "memgraph" (default) or "qdrant"
VECTOR_STORE_BACKEND=memgraph

# Memgraph Vector Settings (when backend=memgraph)
MEMGRAPH_VECTOR_INDEX_NAME=code_embeddings
MEMGRAPH_VECTOR_DIM=768
MEMGRAPH_VECTOR_CAPACITY=100000     # REQUIRED - estimate ~2x your function count
MEMGRAPH_VECTOR_METRIC=cos          # l2sq, cos, ip, pearson
MEMGRAPH_VECTOR_SCALAR_KIND=f32     # f32 (default), f16 (save memory)

# Qdrant Settings (when backend=qdrant)
# QDRANT_URI=http://localhost:6333  # Remote Qdrant server
# QDRANT_API_KEY=your-api-key       # For Qdrant Cloud

# Unified Settings
VECTOR_SEARCH_TOP_K=5
VECTOR_EMBEDDING_BATCH_SIZE=50
```

---

## 6. API Design

### 6.1 VectorBackend Protocol

```python
# vector_backend.py - Abstract backend interface

from abc import ABC, abstractmethod
from typing import Protocol, Sequence

class VectorBackend(Protocol):
    """Protocol for vector storage backends."""

    def initialize(self) -> None:
        """Initialize the backend (create indexes, collections)."""
        ...

    def store_batch(
        self,
        points: Sequence[tuple[int, list[float], str]]
    ) -> int:
        """
        Store embeddings in batch.

        Args:
            points: Sequence of (node_id, embedding, qualified_name)

        Returns:
            Number of successfully stored points
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None
    ) -> list[tuple[int, float]]:
        """
        Search for similar embeddings.

        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Optional filters (e.g., {"labels": ["Function"]})

        Returns:
            List of (node_id, score) tuples
        """
        ...

    def delete_batch(self, node_ids: Sequence[int]) -> int:
        """Delete embeddings by node IDs."""
        ...

    def verify_ids(self, expected_ids: set[int]) -> set[int]:
        """Verify which IDs are stored."""
        ...

    def close(self) -> None:
        """Cleanup resources."""
        ...

    def get_stats(self) -> dict:
        """Return backend statistics."""
        ...


def get_vector_backend() -> VectorBackend:
    """Factory function to get configured backend."""
    backend = settings.VECTOR_STORE_BACKEND.lower()

    if backend == "qdrant":
        from .vector_store_qdrant import QdrantBackend
        return QdrantBackend()
    elif backend == "memgraph":
        from .vector_store_memgraph import MemgraphBackend
        return MemgraphBackend()
    else:
        raise ValueError(f"Unknown vector backend: {backend}")
```

### 6.2 Memgraph Backend Implementation

```python
# vector_store_memgraph.py

from typing import Sequence
from .vector_backend import VectorBackend
from .services.graph_service import MemgraphIngestor
from .config import settings

class MemgraphBackend(VectorBackend):
    """Memgraph native vector storage using vector_search.

    Creates per-label indexes for embeddable node types:
    - Function, Method, Class, Interface, Contract, Library

    This allows searching across all code constructs uniformly.
    """

    LABELS_TO_INDEX = ("Function", "Method", "Class", "Interface", "Contract", "Library")
    INDEX_NAME = "code_embedding_index"

    def __init__(self):
        self._conn: mgclient.Connection | None = None

    def initialize(self) -> None:
        """Create vector indexes for each embeddable node type.

        Memgraph requires explicit capacity estimation (~2x function count).
        Per-label indexes allow efficient filtering by node type.

        Note: CONFIG uses JSON-style quoted keys: "dimension" not dimension
        """
        for label in self.LABELS_TO_INDEX:
            index_name = f"{label.lower()}_embedding_index"
            cypher = f"""
            CREATE VECTOR INDEX {index_name}
            ON :{label}(embedding)
            WITH CONFIG {{
                "dimension": {settings.MEMGRAPH_VECTOR_DIM},
                "capacity": {settings.MEMGRAPH_VECTOR_CAPACITY},
                "metric": "{settings.MEMGRAPH_VECTOR_METRIC}"
            }};
            """

            try:
                self._execute_query(cypher)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    raise

    def store_batch(
        self,
        points: Sequence[tuple[int, list[float], str]]
    ) -> int:
        """Store embeddings as node properties."""
        if not points:
            return 0

        # Build batch update query
        cypher = """
        UNWIND $points AS p
        MATCH (n) WHERE id(n) = p.node_id
        SET n.embedding = p.embedding,
            n.embedding_model = $model_name,
            n.embedding_version = 1
        RETURN count(n) as stored;
        """

        params = {
            "points": [
                {"node_id": nid, "embedding": emb}
                for nid, emb, _ in points
            ],
            "model_name": "microsoft/unixcoder-base",
        }

        result = self._execute_query(cypher, params)
        return result[0]["stored"] if result else 0

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict | None = None
    ) -> list[tuple[int, float]]:
        """Search across all per-label indexes and combine results.

        Queries each label-specific index (function_embedding_index,
        method_embedding_index, etc.) and merges results by similarity.

        Note: Parameter order for vector_search.search is (index_name, limit, query_vector)
        Returns: (node_id, similarity) tuples sorted by similarity descending.
        """
        results: list[tuple[int, float]] = []
        project_prefix = filters.get("project_prefix") if filters else None

        for label in self.LABELS_TO_INDEX:
            index_name = f"{label.lower()}_embedding_index"

            # Fetch more results if filtering by project prefix
            fetch_count = top_k * 3 if project_prefix else top_k

            if project_prefix:
                cypher = """
                CALL vector_search.search($index_name, $fetch_count, $embedding)
                YIELD node, distance, similarity
                WHERE node.qualified_name STARTS WITH $project_prefix
                RETURN id(node) AS node_id, similarity
                ORDER BY distance ASC;
                """
            else:
                cypher = """
                CALL vector_search.search($index_name, $top_k, $embedding)
                YIELD node, distance, similarity
                RETURN id(node) AS node_id, similarity
                ORDER BY distance ASC;
                """

            params = {
                "index_name": index_name,
                "embedding": query_embedding,
                "top_k": top_k,
                "fetch_count": fetch_count,
                "project_prefix": project_prefix,
            }

            try:
                label_results = self._execute_query(cypher, params)
                for r in label_results:
                    results.append((int(r["node_id"]), float(r.get("similarity", 0.0))))
            except Exception:
                continue  # Skip if index doesn't exist

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def delete_batch(self, node_ids: Sequence[int]) -> int:
        """Remove embeddings from nodes."""
        cypher = """
        MATCH (n)
        WHERE id(n) IN $node_ids
        SET n.embedding = NULL,
            n.embedding_model = NULL,
            n.embedding_version = NULL
        RETURN count(n) as deleted;
        """
        result = self._execute_query(
            cypher, {"node_ids": list(node_ids)}
        )
        return result[0]["deleted"] if result else 0

    def verify_ids(self, expected_ids: set[int]) -> set[int]:
        """Check which IDs have embeddings."""
        cypher = """
        MATCH (n)
        WHERE id(n) IN $node_ids AND n.embedding IS NOT NULL
        RETURN collect(id(n)) as found_ids;
        """
        result = self._execute_query(
            cypher, {"node_ids": list(expected_ids)}
        )
        return set(result[0]["found_ids"]) if result else set()

    def get_stats(self) -> dict:
        """Return embedding statistics."""
        cypher = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN
          count(n) as total_embeddings,
          count(DISTINCT labels(n)) as node_types,
          max(size(n.embedding)) as max_dimension;
        """
        result = self._execute_query(cypher)
        return result[0] if result else {}

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
```

---

## 7. Cypher Query Templates

### 7.1 Index Creation (Per-Label Indexes)

```cypher
-- Create vector indexes for each embeddable node type
-- Note: CONFIG uses JSON-style quoted keys
CREATE VECTOR INDEX function_embedding_index
ON :Function(embedding)
WITH CONFIG {"dimension": 768, "capacity": 10000, "metric": "cos"};

CREATE VECTOR INDEX method_embedding_index
ON :Method(embedding)
WITH CONFIG {"dimension": 768, "capacity": 10000, "metric": "cos"};

CREATE VECTOR INDEX class_embedding_index
ON :Class(embedding)
WITH CONFIG {"dimension": 768, "capacity": 10000, "metric": "cos"};

CREATE VECTOR INDEX interface_embedding_index
ON :Interface(embedding)
WITH CONFIG {"dimension": 768, "capacity": 10000, "metric": "cos"};

CREATE VECTOR INDEX contract_embedding_index
ON :Contract(embedding)
WITH CONFIG {"dimension": 768, "capacity": 10000, "metric": "cos"};

CREATE VECTOR INDEX library_embedding_index
ON :Library(embedding)
WITH CONFIG {"dimension": 768, "capacity": 10000, "metric": "cos"};

-- Show all vector indexes
SHOW VECTOR INDEX INFO;
```

### 7.2 Basic Vector Search

```cypher
-- Search for similar functions (parameter order: index_name, limit, query_vector)
CALL vector_search.search("function_embedding_index", $top_k, $query_vector)
YIELD node, distance, similarity
RETURN
    id(node) as node_id,
    node.qualified_name as qualified_name,
    node.name as name,
    labels(node) as type,
    distance,
    similarity
ORDER BY distance ASC;
```

### 7.3 Hybrid: Vector + Graph Traversal

```cypher
-- Find similar functions and their call targets
CALL vector_search.search("function_embedding_index", $top_k, $query_vector)
YIELD node, distance, similarity
MATCH (node)-[:CALLS*1..3]->(callee:Function|Method)
RETURN
    id(node) as node_id,
    node.qualified_name as qualified_name,
    distance,
    collect(DISTINCT callee.qualified_name) as calls_to,
    length((node)-[:CALLS]->()) as call_count
ORDER BY distance ASC;
```

### 7.4 Hybrid: Vector + Graph Context

```cypher
-- Similar functions with class and module context
CALL vector_search.search("code_embedding_index", $top_k, $query_vector)
YIELD node, distance, similarity
MATCH (m:Module)-[:DEFINES]->(node)
OPTIONAL MATCH (c:Class)-[:DEFINES_METHOD]->(node)
RETURN
    id(node) as node_id,
    node.qualified_name as qualified_name,
    distance,
    similarity,
    m.path as file_path,
    c.name as parent_class,
    node.start_line as start_line,
    node.end_line as end_line
ORDER BY distance ASC;
```

### 7.5 Filtered Vector Search

```cypher
-- Search with project filter (post-filter)
CALL vector_search.search("code_embedding_index", $top_k * 2, $query_vector)
YIELD node, distance, similarity
WHERE node.qualified_name STARTS WITH $project_prefix
RETURN
    id(node) as node_id,
    node.qualified_name as qualified_name,
    distance,
    similarity
ORDER BY distance ASC
LIMIT $top_k;
```

### 7.6 Hybrid Ranking: Vector + Centrality

```cypher
-- Rank by vector similarity + call degree
CALL vector_search.search("code_embedding_index", $top_k * 2, $query_vector)
YIELD node, distance, similarity
WITH node, distance, similarity,
     size((node)-[:CALLS]->()) + size((:Function|Method)-[:CALLS]->(node)) AS degree
-- Hybrid ranking: lower distance + higher degree is better
WITH node, distance, similarity, degree,
     (1.0 - similarity) * 0.7 - degree * 0.01 AS hybrid_score
ORDER BY hybrid_score ASC
LIMIT $top_k
RETURN
    id(node) as node_id,
    node.qualified_name as qualified_name,
    similarity,
    degree as call_degree,
    hybrid_score
ORDER BY hybrid_score ASC;
```

### 7.7 Call Chain Discovery

```cypher
-- Find entry points leading to similar functions
CALL vector_search.search("code_embedding_index", $top_k, $query_vector)
YIELD node, distance, similarity
MATCH path = (entry:Function|Method)-[:CALLS*]->(node)
WHERE NOT EXISTS((:Function|Method)-[:CALLS]->(entry))
  AND entry.qualified_name STARTS WITH $project_prefix
RETURN
    entry.qualified_name as entry_point,
    node.qualified_name as similar_target,
    distance,
    length(path) as call_depth,
    [n IN nodes(path) | n.qualified_name] as call_chain
ORDER BY distance ASC, call_depth ASC;
```

---

## 8. GraphRAG Lifecycle Integration

### 8.1 Current Pipeline (4 Passes)

| Pass | Purpose | Current Embedding Handling |
|------|---------|----------------------------|
| 1 | Structure identification | None |
| 2 | File/AST processing | None |
| 3 | Call resolution | None |
| 4 | Embedding generation | Query Memgraph → Generate → Store in Qdrant |

### 8.2 Proposed Pipeline (Memgraph Native)

| Pass | Purpose | Embedding Handling |
|------|---------|-------------------|
| 1 | Structure identification | None |
| 2 | File/AST processing | **Embed functions inline** |
| 3 | Call resolution | None |
| 4 | **Embedding verification** | Verify counts, re-embed failures |

### 8.3 Inline Embedding During Node Creation

```python
# function_ingest.py - Modified to include embedding

def _register_function(self, func_node, resolution, module_qn):
    # Build function properties
    func_props = self._build_function_props(func_node, resolution)

    # NEW: Compute embedding inline
    if self._should_embed_functions:
        source_code = self._extract_function_source(func_node)
        if source_code:
            embedding = embed_code(source_code)
            func_props["embedding"] = embedding
            func_props["embedding_model"] = UNIXCODER_MODEL
            func_props["embedding_version"] = 1

    # Create node with embedding
    self.ingestor.ensure_node_batch(NodeLabel.FUNCTION, func_props)
```

### 8.4 Benefits of Inline Embedding

| Benefit | Description |
|---------|-------------|
| **AST in memory** | Source extraction is precise (no file re-read) |
| **Single pass** | Eliminates separate Pass 4 for Memgraph backend |
| **Consistent state** | Node and embedding created atomically |
| **Memory efficient** | No separate embedding cache needed |

---

## 9. MCP Tool Updates

### 9.1 Tool Changes Summary

| Tool | Change | Description |
|------|--------|-------------|
| `semantic_search` | Updated | Works with both backends via abstraction |
| `get_function_source` | Unchanged | Uses node_id regardless of backend |
| `index_repository` | Updated | Uses backend abstraction |
| `update_repository` | Updated | Uses backend abstraction |
| `delete_project` | Updated | Deletes from appropriate backend |

### 9.2 New Vector Management Tools

#### 9.2.1 `list_vector_status`

```python
MCP_LIST_VECTOR_STATUS = """
Show the status of vector embeddings for all projects.
Returns count of nodes with embeddings vs total nodes per type.
"""

async def list_vector_status(self) -> dict:
    cypher = """
    MATCH (n)
    WHERE n:Function OR n:Method OR n:Class
    WITH labels(n)[0] as type,
         CASE WHEN n.embedding IS NOT NULL THEN 1 ELSE 0 END as has_embedding
    RETURN type,
           sum(has_embedding) as embedded,
           count(*) as total
    ORDER BY type;
    """
    results = self.ingestor.fetch_all(cypher)
    return {r["type"]: {"embedded": r["embedded"], "total": r["total"]} for r in results}
```

#### 9.2.2 `reindex_embeddings`

```python
MCP_REINDEX_EMBEDDINGS = """
Regenerate embeddings for a specific project without re-parsing files.
Useful when embedding model changes or embeddings are corrupted.
"""

async def reindex_embeddings(self, project_name: str | None = None) -> str:
    # Get all functions/methods for project
    cypher = """
    MATCH (m:Module)-[:DEFINES]->(n)
    WHERE (n:Function OR n:Method)
      AND m.qualified_name STARTS WITH $project_prefix
    RETURN id(n) as node_id, n.qualified_name as qname,
           m.path as path, n.start_line as start, n.end_line as end;
    """
    # Re-embed and update
    ...
```

#### 9.2.3 `delete_embeddings`

```python
MCP_DELETE_EMBEDDINGS = """
Delete all embeddings for a project while preserving the graph structure.
Useful for freeing memory or preparing for re-indexing.
"""

async def delete_embeddings(self, project_name: str) -> str:
    cypher = """
    MATCH (m:Module)-[:DEFINES]->(n)
    WHERE m.qualified_name STARTS WITH $project_prefix
    SET n.embedding = NULL, n.embedding_model = NULL
    RETURN count(n) as cleared;
    """
    ...
```

---

## 10. Docker Deployment

### 10.1 Updated docker-compose.yaml

```yaml
version: "3.8"

services:
  memgraph:
    image: memgraph/memgraph-mage:latest
    container_name: memgraph
    ports:
      - "${MEMGRAPH_PORT:-7687}:7687"
      - "${MEMGRAPH_HTTP_PORT:-7444}:7444"
    volumes:
      - memgraph_data:/var/lib/memgraph
      - memgraph_log:/var/log/memgraph
    environment:
      - MEMGRAPH_LOG_LEVEL=INFO
    command: ["--also-log-to-stderr"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7444/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  lab:
    image: memgraph/lab:latest
    container_name: memgraph-lab
    ports:
      - "${LAB_PORT:-3000}:3000"
    environment:
      - QUICK_CONNECT_MG_HOST=memgraph
    depends_on:
      - memgraph

  # Optional: Qdrant for remote/cloud deployments
  qdrant:
    image: qdrant/qdrant:latest
    profiles:
      - qdrant
    container_name: qdrant
    ports:
      - "${QDRANT_PORT:-6333}:6333"
      - "${QDRANT_GRPC_PORT:-6334}:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT_API_KEY=${QDRANT_API_KEY:-}

volumes:
  memgraph_data:
  memgraph_log:
  qdrant_data:
```

### 10.2 Deployment Modes

| Mode | Command | Backend |
|------|---------|---------|
| **Default** | `docker compose up` | Memgraph |
| **With Qdrant** | `docker compose --profile qdrant up` | Qdrant (if configured) |

---

## 11. Migration Path

### 11.1 Migration Strategy

```
Phase 1: Add Backend Abstraction
├── Create vector_backend.py protocol
├── Refactor vector_store.py → QdrantBackend
├── Create vector_store_memgraph.py
└── Add VECTOR_STORE_BACKEND config

Phase 2: Update Components
├── Update graph_updater.py to use backend abstraction
├── Update semantic_search.py to use backend abstraction
├── Update MCP tools
└── Add new vector management tools

Phase 3: Testing & Validation
├── Unit tests for both backends
├── Integration tests
├── Performance benchmarks
└── Documentation updates

Phase 4: Deprecation (Future)
├── Qdrant becomes optional dependency
├── Memgraph native is default
└── Migration guide for existing users
```

### 11.2 Backward Compatibility

| Aspect | Compatibility |
|--------|---------------|
| **Existing Qdrant users** | Set `VECTOR_STORE_BACKEND=qdrant` |
| **Existing .env** | Qdrant settings still work |
| **MCP tools** | No breaking changes |
| **API** | All endpoints unchanged |

### 11.3 Migration Script

```python
# scripts/migrate_to_memgraph_vectors.py

def migrate_qdrant_to_memgraph():
    """Migrate embeddings from Qdrant to Memgraph native storage."""

    from codebase_rag.vector_store import get_qdrant_client, settings
    from codebase_rag.services.graph_service import MemgraphIngestor

    qdrant = get_qdrant_client()
    ingestor = MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
    )

    # Scroll all points from Qdrant
    offset = None
    batch_size = 100

    while True:
        results, offset = qdrant.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
        )

        if not results:
            break

        # Batch update Memgraph
        cypher = """
        UNWIND $points AS p
        MATCH (n) WHERE id(n) = p.node_id
        SET n.embedding = p.embedding
        """

        points = [
            {"node_id": p.id, "embedding": p.vector}
            for p in results
        ]
        ingestor.execute_write(cypher, {"points": points})

        if offset is None:
            break

    print("Migration complete!")
```

---

## 12. Testing Strategy

### 12.1 Unit Tests

```python
# tests/test_vector_backend.py

import pytest
from codebase_rag.vector_backend import get_vector_backend

class TestMemgraphBackend:

    def test_store_batch(self, mock_ingestor):
        backend = MemgraphBackend()
        backend._ingestor = mock_ingestor

        points = [
            (1, [0.1] * 768, "func1"),
            (2, [0.2] * 768, "func2"),
        ]

        count = backend.store_batch(points)
        assert count == 2

    def test_search_returns_node_ids(self, mock_ingestor):
        backend = MemgraphBackend()
        backend._ingestor = mock_ingestor

        results = backend.search([0.1] * 768, top_k=5)
        assert len(results) <= 5
        assert all(isinstance(r[0], int) for r in results)


class TestQdrantBackend:

    def test_store_batch(self, mock_qdrant_client):
        backend = QdrantBackend()
        # ... similar tests
```

### 12.2 Integration Tests

```python
# tests/integration/test_vector_backend_integration.py

@pytest.mark.integration
class TestVectorBackendIntegration:

    def test_memgraph_backend_full_cycle(self, memgraph_container):
        backend = MemgraphBackend()
        backend.initialize()

        # Store
        points = [(1, [0.1] * 768, "test.func")]
        assert backend.store_batch(points) == 1

        # Search
        results = backend.search([0.1] * 768, top_k=1)
        assert len(results) == 1
        assert results[0][0] == 1

        # Delete
        assert backend.delete_batch([1]) == 1

        # Verify
        assert backend.verify_ids({1}) == set()

    def test_backend_switching(self):
        # Test that switching backends works at runtime
        os.environ["VECTOR_STORE_BACKEND"] = "memgraph"
        backend1 = get_vector_backend()
        assert isinstance(backend1, MemgraphBackend)

        os.environ["VECTOR_STORE_BACKEND"] = "qdrant"
        backend2 = get_vector_backend()
        assert isinstance(backend2, QdrantBackend)
```

### 12.3 Performance Benchmarks

| Metric | Memgraph | Qdrant |
|--------|----------|--------|
| **Embedding Storage (1000)** | TBD | TBD |
| **Search Latency (p50)** | TBD | TBD |
| **Search Latency (p99)** | TBD | TBD |
| **Hybrid Query Latency** | TBD | N/A |

---

## Appendix A: Hybrid Search Patterns

### A.1 Ranking Strategies

```python
# Ranking formula combining vector similarity + graph centrality
def hybrid_ranking_score(
    vector_similarity: float,
    graph_centrality: float,
    path_distance: int,
    alpha: float = 0.6,  # Vector weight
    beta: float = 0.3,   # Centrality weight
    gamma: float = 0.1,  # Distance penalty
) -> float:
    distance_score = 1.0 / (1.0 + path_distance)
    return alpha * vector_similarity + beta * graph_centrality + gamma * distance_score
```

### A.2 Unique Memgraph Capabilities

| Capability | Query Pattern |
|------------|---------------|
| Call chain traversal | `(f)-[:CALLS*]->(target)` |
| Inheritance analysis | `(c)-[:INHERITS]->(parent)` |
| Interface matching | `(c)-[:IMPLEMENTS]->(i)` |
| Dead code detection | `NOT EXISTS((:Function)-[:CALLS]->(f))` |
| Community detection | `CALL wcc.get()` |
| Centrality ranking | `CALL pagerank.get()` |

---

## Appendix B: Dependencies

### B.1 pyproject.toml Updates

```toml
[project.optional-dependencies]
# Full semantic support with Qdrant
semantic-qdrant = [
    "qdrant-client>=1.9.0",
    "torch>=2.6.0",
    "transformers>=4.0.0",
]

# Lite semantic support (Memgraph only)
semantic-lite = [
    "torch>=2.6.0",
    "transformers>=4.0.0",
]

# Full support (backward compatible)
semantic = [
    "qdrant-client>=1.9.0",
    "torch>=2.6.0",
    "transformers>=4.0.0",
]
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-03-28 | Claude (10 parallel workers) | Initial design specification |

---

## References

1. Memgraph MAGE Documentation: https://memgraph.com/docs/mage
2. Memgraph MAGE Repository: https://github.com/memgraph/mage
3. Qdrant Documentation: https://qdrant.tech/documentation/
4. UniXcoder Model: https://huggingface.co/microsoft/unixcoder-base