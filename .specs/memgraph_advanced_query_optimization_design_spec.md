# Memgraph Advanced Query & Algorithm Optimization Design Spec

## Executive Summary

This specification outlines comprehensive optimizations to fully leverage Memgraph's advanced capabilities for the Code-Graph-RAG system. The current implementation uses basic Memgraph features but misses significant opportunities for performance, relevance, and UX improvements available through Memgraph's MAGE library, advanced algorithms, and hybrid query capabilities.

## Current State Analysis

### What's Currently Implemented

1. **Basic Vector Search**: Native Memgraph vector indexes with `vector_search.search()`
2. **PageRank**: Using `pagerank.get()` for node importance scoring
3. **Community Detection**: Leiden/Louvain algorithms for clustering
4. **Basic BFS**: Simple breadth-first traversal for context expansion
5. **Hybrid Scoring**: Combining vector similarity + PageRank scores

### Key Gaps & Missed Opportunities

1. **No Text Search**: Missing Memgraph's text index for keyword-based retrieval
2. **Limited Traversal Algorithms**: Not using weighted shortest path, all shortest paths, or k-shortest paths
3. **No Node Similarity**: Missing `node_similarity` module for structural similarity
4. **No Dynamic Algorithms**: Not using dynamic PageRank/community detection for real-time updates
5. **Limited Hybrid Queries**: Not combining vector + text + graph in single atomic queries
6. **No Query-Focused Summarization**: Missing QFS with community summaries
7. **No Deep Path Analysis**: Missing multi-hop reasoning with path algorithms

## Proposed Optimizations

### 1. Multi-Modal Hybrid Retrieval (Atomic Pipelines)

#### 1.1 Unified Hybrid Search Query

Create atomic queries that combine multiple retrieval methods in single Cypher execution:

```cypher
// Hybrid search: Vector + Text + Graph in one query
WITH $embedding AS query_vec, $keywords AS keywords, $top_k AS top_k

// Step 1: Parallel vector searches across multiple indexes
CALL vector_search.search("function_embedding_index", top_k * 2, query_vec)
YIELD node AS vf, similarity AS sim_f

UNION ALL

CALL vector_search.search("class_embedding_index", top_k * 2, query_vec)
YIELD node AS vc, similarity AS sim_c

WITH collect({node: coalesce(vf, vc), sim: coalesce(sim_f, sim_c)}) AS vector_results

// Step 2: Text search for keyword matches
UNWIND keywords AS keyword
MATCH (t:Function|Class|Method)
WHERE t.name CONTAINS keyword OR t.docstring CONTAINS keyword
WITH vector_results, collect(DISTINCT t) AS text_results

// Step 3: Merge and rank with graph centrality
UNWIND vector_results AS vr
WITH vr.node AS n, vr.sim AS vector_score, text_results
WHERE n:Function OR n:Class OR n:Method

// Check if also in text results (boost score)
WITH n, vector_score,
     CASE WHEN n IN text_results THEN 0.2 ELSE 0 END AS text_boost,
     COALESCE(n.pagerank_score, 0.1) AS pagerank,
     COALESCE(n.community_importance, 0.0) AS community_score

// Final ranking formula
WITH n,
     (vector_score * 0.6) + (text_boost * 0.2) + (pagerank * 0.15) + (community_score * 0.05) AS final_score
ORDER BY final_score DESC
LIMIT top_k

RETURN id(n) AS node_id, n.name AS name, n.qualified_name AS qualified_name,
       labels(n)[0] AS node_type, final_score AS score
```

#### 1.2 Implementation

Create new module: `codebase_rag/memgraph_advanced/hybrid_retrieval.py`

```python
"""Hybrid retrieval combining vector, text, and graph signals."""

from dataclasses import dataclass
from typing import Any

from ..graph_algorithms import GraphAlgorithms


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    node_id: int
    name: str
    qualified_name: str
    node_type: str
    vector_score: float
    text_score: float
    pagerank_score: float
    community_score: float
    final_score: float
    metadata: dict[str, Any]


class HybridRetriever:
    """Multi-modal retrieval combining vector, text, and graph signals."""

    def __init__(self, algorithms: GraphAlgorithms | None = None):
        self.algo = algorithms or GraphAlgorithms()

    def search(
        self,
        query_embedding: list[float],
        keywords: list[str],
        top_k: int = 10,
        weights: dict[str, float] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        """
        Execute hybrid search combining multiple signals.

        Args:
            query_embedding: Vector representation of query
            keywords: Text keywords for text search
            top_k: Number of results to return
            weights: Optional custom weights for scoring components
            filters: Optional filters (e.g., project_prefix, node_types)

        Returns:
            List of hybrid search results ranked by combined score
        """
        # Use atomic hybrid query
        ...
```

### 2. Advanced Graph Algorithms for Relevance

#### 2.1 Node Similarity (Jaccard/Overlap)

Use MAGE's `node_similarity` module to find structurally similar functions/classes:

```cypher
// Find functions similar to a given function based on call patterns
MATCH (f:Function {qualified_name: $target_qn})-[:CALLS]->(target_calls)
WITH f, collect(DISTINCT id(target_calls)) AS f_calls

MATCH (candidate:Function)-[:CALLS]->(callee)
WHERE candidate <> f
WITH f, f_calls, candidate, collect(DISTINCT id(callee)) AS c_calls

// Calculate Jaccard similarity
WITH f, candidate,
     size([x IN f_calls WHERE x IN c_calls]) AS intersection,
     size(f_calls) + size(c_calls) - size([x IN f_calls WHERE x IN c_calls]) AS union

WITH f, candidate, toFloat(intersection) / union AS jaccard_score
WHERE jaccard_score > 0.3

RETURN candidate.qualified_name AS similar_function,
       candidate.name AS name,
       jaccard_score
ORDER BY jaccard_score DESC
LIMIT 10;
```

#### 2.2 Path-Based Reasoning with K-Shortest Paths

Implement multi-path reasoning for finding relationships between components:

```cypher
// Find multiple paths between two functions (K-shortest paths)
MATCH (start:Function {qualified_name: $start_qn}),
      (end:Function {qualified_name: $end_qn})

// Use K-shortest paths algorithm
MATCH path = (start)-[:CALLS|:DEFINES *KSHORTEST 3]->(end)

WITH path, length(path) AS path_length, 
     [n IN nodes(path) | n.qualified_name] AS node_names,
     [r IN relationships(path) | type(r)] AS rel_types

RETURN node_names, rel_types, path_length
ORDER BY path_length
```

#### 2.3 Weighted Shortest Path for Call Chains

Find optimal paths weighted by complexity, frequency, or other metrics:

```cypher
// Find the "simplest" call path (fewest intermediate calls)
MATCH (start:Function {qualified_name: $start_qn}),
      (end:Function {qualified_name: $end_qn})

MATCH path = (start)-[:CALLS *WSHORTEST (r, n | r.call_count + 1)]->(end)

RETURN [n IN nodes(path) | n.qualified_name] AS call_chain,
       [n IN nodes(path) | n.complexity_score] AS complexities,
       reduce(s = 0, r IN relationships(path) | s + r.call_count) AS total_weight
```

### 3. Query-Focused Summarization (QFS) with Communities

#### 3.1 Community-Aware Summarization

Build community summaries for global question answering:

```python
# codebase_rag/memgraph_advanced/qfs.py

"""Query-Focused Summarization using community detection."""

from dataclasses import dataclass


@dataclass
class CommunitySummary:
    """Summary of a detected community."""
    community_id: int
    node_count: int
    representative_nodes: list[str]
    summary_text: str
    key_functions: list[str]
    key_classes: list[str]


class CommunityQFS:
    """Query-Focused Summarization using community detection."""

    def build_community_summaries(self) -> list[CommunitySummary]:
        """
        Build summaries for each detected community.

        Returns:
            List of community summaries
        """
        cypher = """
        // First, ensure communities are detected
        CALL community_detection.leiden(
            "CALLS",
            "OUTGOING",
            { community_property: "community_id", weight_property: "weight" }
        ) YIELD node, community_id

        // Collect nodes by community
        WITH community_id, collect(node) AS nodes

        // Get representative nodes (highest PageRank in each community)
        UNWIND nodes AS n
        WITH community_id, nodes, n
        ORDER BY COALESCE(n.pagerank_score, 0) DESC
        WITH community_id, nodes, collect(n)[0..5] AS representatives

        // Calculate community statistics
        WITH community_id,
             size(nodes) AS node_count,
             [r IN representatives | r.qualified_name] AS rep_names,
             [n IN nodes WHERE n:Function] AS functions,
             [n IN nodes WHERE n:Class] AS classes

        RETURN community_id, node_count, rep_names,
               size(functions) AS function_count,
               size(classes) AS class_count,
               [f IN functions | f.qualified_name][0..10] AS key_functions,
               [c IN classes | c.qualified_name][0..5] AS key_classes
        ORDER BY node_count DESC
        """

        # Execute query and build summaries
        # ... implementation

    def query_focused_summary(
        self,
        question: str,
        top_communities: int = 3
    ) -> str:
        """
        Generate query-focused summary from community summaries.

        Args:
            question: User question
            top_communities: Number of top communities to include

        Returns:
            Summarized answer
        """
        # Find most relevant communities based on question
        # Use LLM to generate summary from community info
        # ... implementation
```

### 4. Real-Time Dynamic Algorithms

#### 4.1 Dynamic PageRank for Incremental Updates

Use dynamic algorithms for real-time centrality updates:

```python
# codebase_rag/memgraph_advanced/dynamic_algorithms.py

"""Dynamic graph algorithms for real-time updates."""

class DynamicGraphAlgorithms:
    """Wrapper for dynamic MAGE algorithms."""

    def update_pagerank_dynamic(
        self,
        new_relationships: list[tuple[int, int]],
        deleted_relationships: list[tuple[int, int]]
    ) -> dict:
        """
        Incrementally update PageRank scores when graph changes.

        Args:
            new_relationships: List of (source_id, target_id) for new edges
            deleted_relationships: List of (source_id, target_id) for deleted edges

        Returns:
            Updated PageRank statistics
        """
        # Check if dynamic algorithm is available (Enterprise feature)
        cypher = """
        // Check if dynamic pagerank is available
        CALL dynamic.pagerank_online.update($new_edges, $deleted_edges)
        YIELD node, rank
        SET node.pagerank_score = rank
        RETURN count(node) AS updated
        """

        params = {
            "new_edges": [{"from": s, "to": t} for s, t in new_relationships],
            "deleted_edges": [{"from": s, "to": t} for s, t in deleted_relationships],
        }

        # Execute and return results
        # ... implementation

    def update_communities_dynamic(
        self,
        changed_nodes: list[int]
    ) -> dict:
        """
        Incrementally update community assignments.

        Args:
            changed_nodes: List of node IDs that changed

        Returns:
            Updated community statistics
        """
        # Use dynamic community detection
        cypher = """
        CALL dynamic.community_detection_online.update($changed_nodes)
        YIELD node, community_id
        SET node.community_id = community_id
        RETURN community_id, count(node) AS size
        ORDER BY size DESC
        """

        params = {"changed_nodes": changed_nodes}
        # ... implementation
```

### 5. Advanced Traversal & Path Analysis

#### 5.1 Multi-Hop Path Analysis

Implement sophisticated path-based reasoning:

```python
# codebase_rag/memgraph_advanced/path_analysis.py

"""Advanced path analysis for code relationships."""

from dataclasses import dataclass
from enum import Enum


class PathType(Enum):
    CALL_CHAIN = "call_chain"
    DEPENDENCY = "dependency"
    INHERITANCE = "inheritance"
    IMPORT = "import"


@dataclass
class PathAnalysis:
    """Analysis of paths between code entities."""
    path_type: PathType
    paths: list[dict]
    shortest_path_length: int
    alternative_paths: int
    critical_nodes: list[str]
    bottlenecks: list[str]


class PathAnalyzer:
    """Analyze paths between code entities."""

    def analyze_call_chain(
        self,
        start_qn: str,
        end_qn: str,
        max_paths: int = 3
    ) -> PathAnalysis:
        """
        Analyze call chains between two functions.

        Uses k-shortest paths to find multiple call chains.
        """
        cypher = """
        MATCH (start:Function {qualified_name: $start_qn}),
              (end:Function {qualified_name: $end_qn})

        // Find k shortest paths using CALLS relationship
        MATCH path = (start)-[:CALLS *KSHORTEST $max_paths]->(end)

        WITH path,
             length(path) AS path_length,
             [n IN nodes(path) | {
                 qualified_name: n.qualified_name,
                 name: n.name,
                 type: labels(n)[0],
                 pagerank: COALESCE(n.pagerank_score, 0)
             }] AS node_details,
             [r IN relationships(path) | type(r)] AS relationship_types

        // Find critical nodes (high betweenness - appear in multiple paths)
        WITH collect({
            path: path,
            length: path_length,
            nodes: node_details,
            rels: relationship_types
        }) AS all_paths

        UNWIND all_paths AS p
        UNWIND p.nodes AS node
        WITH all_paths, node, count(*) AS node_frequency
        WHERE node_frequency > 1

        RETURN all_paths,
               collect(DISTINCT node.qualified_name) AS critical_nodes,
               size(all_paths) AS total_paths,
               reduce(min_len = 1000, p IN all_paths | CASE WHEN p.length < min_len THEN p.length ELSE min_len END) AS shortest_length
        """

        params = {"start_qn": start_qn, "end_qn": end_qn, "max_paths": max_paths}

        # Execute and build PathAnalysis
        # ... implementation

    def find_bottlenecks(self, function_qn: str) -> list[dict]:
        """
        Find functions that are bottlenecks (high betweenness centrality).

        Uses betweenness centrality to identify critical functions that
        control flow between different parts of the codebase.
        """
        cypher = """
        // Use betweenness centrality to find bottlenecks
        CALL betweenness_centrality.get("CALLS", "BOTH")
        YIELD node, betweenness

        WHERE node:Function AND betweenness > 0.01

        // Get additional context
        OPTIONAL MATCH (node)-[:CALLS]->(callee)
        WITH node, betweenness, count(callee) AS out_degree

        OPTIONAL MATCH (caller)-[:CALLS]->(node)
        WITH node, betweenness, out_degree, count(caller) AS in_degree

        RETURN node.qualified_name AS function_name,
               node.name AS name,
               betweenness AS centrality_score,
               in_degree,
               out_degree,
               (in_degree + out_degree) AS total_connections
        ORDER BY betweenness DESC
        LIMIT 20
        """

        # Execute and return results
        # ... implementation
```

### 6. Query-Focused Summarization (QFS) Implementation

#### 6.1 Community-Based Summarization

```python
# codebase_rag/memgraph_advanced/qfs.py

"""Query-Focused Summarization using community detection."""

from dataclasses import dataclass


@dataclass
class CommunitySummary:
    """Summary of a detected community."""
    community_id: int
    node_count: int
    representative_nodes: list[str]
    summary_text: str
    key_functions: list[str]
    key_classes: list[str]


class CommunityQFS:
    """Query-Focused Summarization using community detection."""

    def build_community_summaries(self) -> list[CommunitySummary]:
        """Build summaries for each detected community."""
        cypher = """
        // First, ensure communities are detected
        CALL community_detection.leiden(
            "CALLS",
            "OUTGOING",
            { community_property: "community_id", weight_property: "weight" }
        ) YIELD node, community_id

        // Collect nodes by community
        WITH community_id, collect(node) AS nodes

        // Get representative nodes (highest PageRank in each community)
        UNWIND nodes AS n
        WITH community_id, nodes, n
        ORDER BY COALESCE(n.pagerank_score, 0) DESC
        WITH community_id, nodes, collect(n)[0..5] AS representatives

        // Calculate community statistics
        WITH community_id,
             size(nodes) AS node_count,
             [r IN representatives | r.qualified_name] AS rep_names,
             [n IN nodes WHERE n:Function] AS functions,
             [n IN nodes WHERE n:Class] AS classes

        RETURN community_id, node_count, rep_names,
               size(functions) AS function_count,
               size(classes) AS class_count,
               [f IN functions | f.qualified_name][0..10] AS key_functions,
               [c IN classes | c.qualified_name][0..5] AS key_classes
        ORDER BY node_count DESC
        """

        # Execute query and build summaries
        ...

    def query_focused_summary(
        self,
        question: str,
        top_communities: int = 3
    ) -> str:
        """Generate query-focused summary from community summaries."""
        # Find most relevant communities based on question
        # Use LLM to generate summary from community info
        ...
```

### 7. Real-Time Dynamic Algorithms

#### 7.1 Dynamic PageRank for Incremental Updates

```python
# codebase_rag/memgraph_advanced/dynamic_algorithms.py

"""Dynamic graph algorithms for real-time updates."""

class DynamicGraphAlgorithms:
    """Wrapper for dynamic MAGE algorithms."""

    def update_pagerank_dynamic(
        self,
        new_relationships: list[tuple[int, int]],
        deleted_relationships: list[tuple[int, int]]
    ) -> dict:
        """
        Incrementally update PageRank scores when graph changes.

        Args:
            new_relationships: List of (source_id, target_id) for new edges
            deleted_relationships: List of (source_id, target_id) for deleted edges

        Returns:
            Updated PageRank statistics
        """
        # Check if dynamic algorithm is available (Enterprise feature)
        cypher = """
        // Check if dynamic pagerank is available
        CALL dynamic.pagerank_online.update($new_edges, $deleted_edges)
        YIELD node, rank
        SET node.pagerank_score = rank
        RETURN count(node) AS updated
        """

        params = {
            "new_edges": [{"from": s, "to": t} for s, t in new_relationships],
            "deleted_edges": [{"from": s, "to": t} for s, t in deleted_relationships],
        }

        # Execute and return results
        ...

    def update_communities_dynamic(
        self,
        changed_nodes: list[int]
    ) -> dict:
        """
        Incrementally update community assignments.

        Args:
            changed_nodes: List of node IDs that changed

        Returns:
            Updated community statistics
        """
        # Use dynamic community detection
        cypher = """
        CALL dynamic.community_detection_online.update($changed_nodes)
        YIELD node, community_id
        SET node.community_id = community_id
        RETURN community_id, count(node) AS size
        ORDER BY size DESC
        """

        params = {"changed_nodes": changed_nodes}
        ...
```

### 8. Advanced Traversal & Path Analysis

#### 8.1 Multi-Hop Path Analysis

```python
# codebase_rag/memgraph_advanced/path_analysis.py

"""Advanced path analysis for code relationships."""

from dataclasses import dataclass
from enum import Enum


class PathType(Enum):
    CALL_CHAIN = "call_chain"
    DEPENDENCY = "dependency"
    INHERITANCE = "inheritance"
    IMPORT = "import"


@dataclass
class PathAnalysis:
    """Analysis of paths between code entities."""
    path_type: PathType
    paths: list[dict]
    shortest_path_length: int
    alternative_paths: int
    critical_nodes: list[str]
    bottlenecks: list[str]


class PathAnalyzer:
    """Analyze paths between code entities."""

    def analyze_call_chain(
        self,
        start_qn: str,
        end_qn: str,
        max_paths: int = 3
    ) -> PathAnalysis:
        """
        Analyze call chains between two functions.

        Uses k-shortest paths to find multiple call chains.
        """
        cypher = """
        MATCH (start:Function {qualified_name: $start_qn}),
              (end:Function {qualified_name: $end_qn})

        // Find k shortest paths using CALLS relationship
        MATCH path = (start)-[:CALLS *KSHORTEST $max_paths]->(end)

        WITH path,
             length(path) AS path_length,
             [n IN nodes(path) | {
                 qualified_name: n.qualified_name,
                 name: n.name,
                 type: labels(n)[0],
                 pagerank: COALESCE(n.pagerank_score, 0)
             }] AS node_details,
             [r IN relationships(path) | type(r)] AS relationship_types

        // Find critical nodes (high betweenness - appear in multiple paths)
        WITH collect({
            path: path,
            length: path_length,
            nodes: node_details,
            rels: relationship_types
        }) AS all_paths

        UNWIND all_paths AS p
        UNWIND p.nodes AS node
        WITH all_paths, node, count(*) AS node_frequency
        WHERE node_frequency > 1

        RETURN all_paths,
               collect(DISTINCT node.qualified_name) AS critical_nodes,
               size(all_paths) AS total_paths,
               reduce(min_len = 1000, p IN all_paths | CASE WHEN p.length < min_len THEN p.length ELSE min_len END) AS shortest_length
        """

        params = {"start_qn": start_qn, "end_qn": end_qn, "max_paths": max_paths}

        # Execute and build PathAnalysis
        ...

    def find_bottlenecks(self, function_qn: str) -> list[dict]:
        """
        Find functions that are bottlenecks (high betweenness centrality).

        Uses betweenness centrality to identify critical functions that
        control flow between different parts of the codebase.
        """
        cypher = """
        // Use betweenness centrality to find bottlenecks
        CALL betweenness_centrality.get("CALLS", "BOTH")
        YIELD node, betweenness

        WHERE node:Function AND betweenness > 0.01

        // Get additional context
        OPTIONAL MATCH (node)-[:CALLS]->(callee)
        WITH node, betweenness, count(callee) AS out_degree

        OPTIONAL MATCH (caller)-[:CALLS]->(node)
        WITH node, betweenness, out_degree, count(caller) AS in_degree

        RETURN node.qualified_name AS function_name,
               node.name AS name,
               betweenness AS centrality_score,
               in_degree,
               out_degree,
               (in_degree + out_degree) AS total_connections
        ORDER BY betweenness DESC
        LIMIT 20
        """

        # Execute and return results
        ...
```

## Implementation Roadmap

### Phase 1: Core Hybrid Retrieval (Week 1-2)

1. **Hybrid Search Module**
   - Implement `hybrid_retrieval.py` with unified atomic queries
   - Add weights configuration for scoring components
   - Integrate with existing vector store

2. **Text Search Integration**
   - Add text index creation for node names and docstrings
   - Implement text search fallback
   - Combine with vector results

### Phase 2: Advanced Algorithms (Week 3-4)

1. **Path Analysis Module**
   - Implement `path_analysis.py` with k-shortest paths
   - Add bottleneck detection using betweenness centrality
   - Create call chain analysis

2. **Node Similarity**
   - Implement Jaccard similarity for structural matching
   - Add overlap similarity option
   - Create similar function/class recommendations

### Phase 3: QFS & Communities (Week 5-6)

1. **Community QFS Module**
   - Implement `qfs.py` with community detection
   - Add community summary generation
   - Create query-focused summarization

2. **Global Question Answering**
   - Add cross-community reasoning
   - Implement community ranking by relevance
   - Create final summary generation

### Phase 4: Dynamic Algorithms (Week 7-8)

1. **Dynamic Algorithm Module**
   - Implement `dynamic_algorithms.py`
   - Add dynamic PageRank updates
   - Implement dynamic community detection

2. **Real-Time Updates**
   - Add incremental graph updates
   - Create score recalculation triggers
   - Implement batch update optimization

## Configuration

Add to `config.py`:

```python
# Hybrid Retrieval Settings
class HybridRetrievalConfig:
    """Configuration for hybrid retrieval."""
    vector_weight: float = 0.6
    text_weight: float = 0.2
    pagerank_weight: float = 0.15
    community_weight: float = 0.05
    top_k: int = 10
    max_context_depth: int = 2

# Path Analysis Settings
class PathAnalysisConfig:
    """Configuration for path analysis."""
    max_paths: int = 3
    max_path_length: int = 10
    bottleneck_threshold: float = 0.01

# QFS Settings
class QFSConfig:
    """Configuration for query-focused summarization."""
    top_communities: int = 3
    min_community_size: int = 5
    summary_max_tokens: int = 500
```

## Benefits

1. **Improved Relevance**: Hybrid scoring combines multiple signals for better results
2. **Faster Queries**: Atomic queries reduce round-trips
3. **Better UX**: Path analysis shows code relationships visually
4. **Global Insights**: QFS answers high-level questions about the codebase
5. **Real-Time Updates**: Dynamic algorithms keep scores current
6. **Deeper Analysis**: Advanced algorithms find hidden patterns

## Dependencies

- Memgraph MAGE library (already used)
- Enterprise features for dynamic algorithms (optional)
- Additional Python packages: none (uses existing stack)

## Testing Strategy

1. **Unit Tests**: Each new module with mocked queries
2. **Integration Tests**: Against real Memgraph instance
3. **Performance Tests**: Query latency benchmarks
4. **Relevance Tests**: Human evaluation of search results
5. **Stress Tests**: Large graph performance

## Migration Path

1. New features are additive - no breaking changes
2. Existing code continues to work unchanged
3. New modules are opt-in via configuration
4. Gradual migration from old to new implementations
5. Feature flags for A/B testing new algorithms

## Conclusion

This specification provides a roadmap for significantly enhancing the Code-Graph-RAG system's Memgraph utilization. By implementing these advanced features, the system will provide:

- **Better search relevance** through hybrid retrieval
- **Deeper code insights** through path analysis
- **Global understanding** through QFS
- **Real-time performance** through dynamic algorithms

The phased implementation allows for incremental delivery of value while maintaining system stability.
