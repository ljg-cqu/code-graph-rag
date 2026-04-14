# Memgraph Advanced Query & Algorithm Optimization Design Spec

## Executive Summary

This specification outlines comprehensive optimizations to fully leverage Memgraph's advanced capabilities for the Code-Graph-RAG system. The current implementation uses basic Memgraph features but misses significant opportunities for performance, relevance, and UX improvements available through Memgraph's MAGE library, advanced algorithms, and hybrid query capabilities.

## Current State Analysis

### What's Currently Implemented

1. **Vector Search**: Native Memgraph vector indexes with `vector_search.search()` - ✅ Implemented in `vector_store_memgraph.py`
2. **PageRank**: Using `pagerank.get()` for node importance scoring - ✅ Implemented in `graph_algorithms.py:69-93`
3. **Community Detection**: Leiden/Louvain algorithms - ✅ Implemented in `graph_algorithms.py:95-144`
4. **BFS Context Expansion**: Breadth-first traversal with depth control - ✅ Implemented in `graph_algorithms.py:146-182`
5. **Node Similarity (Jaccard)**: Structural similarity using call patterns - ✅ Implemented in `graph_algorithms.py:184-229`
6. **Basic Hybrid Scoring**: Combining vector similarity + PageRank - ✅ Implemented in `vector_store_memgraph.py:189-301`

### Key Gaps & Missed Opportunities

1. **No Text Search**: Missing Memgraph's text index for keyword-based retrieval
2. **Limited Traversal Algorithms**: Not using weighted shortest path, all shortest paths, or k-shortest paths
3. **No MAGE Node Similarity**: Using manual Jaccard instead of MAGE's `node_similarity` module
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

// Step 1: Vector searches across multiple indexes (fixed parameter order)
CALL vector_search.search("function_embedding_index", query_vec, top_k * 2)
YIELD node, similarity
WITH query_vec, keywords, top_k, 
     collect({node: node, similarity: similarity, source: 'function'}) AS function_results

CALL vector_search.search("class_embedding_index", query_vec, top_k * 2)
YIELD node, similarity
WITH query_vec, keywords, top_k, function_results,
     collect({node: node, similarity: similarity, source: 'class'}) AS class_results

// Combine results from both indexes
WITH query_vec, keywords, top_k,
     function_results + class_results AS vector_results

// Step 2: Text search for keyword matches
UNWIND keywords AS keyword
MATCH (t:Function|Class|Method)
WHERE t.name CONTAINS keyword OR t.docstring CONTAINS keyword
WITH vector_results, collect(DISTINCT t) AS text_results

// Step 3: Merge and rank with graph centrality
UNWIND vector_results AS vr
WITH vr.node AS n, vr.similarity AS vector_score, text_results

// Check if also in text results (boost score) - compare by ID
WITH n, vector_score,
     CASE WHEN ANY(t IN text_results WHERE id(t) = id(n)) THEN 0.2 ELSE 0 END AS text_boost,
     COALESCE(n.pagerank_score, 0.1) AS pagerank,
     COALESCE(n.community_importance, 0.0) AS community_score

// Final ranking formula
WITH n,
     (vector_score * 0.6) + (text_boost * 0.2) + (pagerank * 0.15) + (community_score * 0.05) AS final_score
ORDER BY final_score DESC
LIMIT top_k

RETURN id(n) AS node_id, n.name AS name, n.qualified_name AS qualified_name,
       labels(n)[0] AS node_type, final_score AS score,
       vector_score, text_boost AS text_score, pagerank AS pagerank_score, community_score
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
    final_score: float
    vector_score: float
    text_score: float
    pagerank_score: float
    community_score: float
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
            weights: Optional custom weights for scoring components.
                Default: {"vector": 0.6, "text": 0.2, "pagerank": 0.15, "community": 0.05}
            filters: Optional filters (e.g., {"project_prefix": "src/", "node_types": ["Function"]})

        Returns:
            List of hybrid search results ranked by combined score
        """
        # Use default weights if not provided
        weights = weights or {
            "vector": 0.6,
            "text": 0.2,
            "pagerank": 0.15,
            "community": 0.05
        }

        # Build parameterized Cypher query with filters
        cypher = self._build_hybrid_query(filters, weights)

        params = {
            "embedding": query_embedding,
            "keywords": keywords,
            "top_k": top_k
        }

        # Execute query
        results = self.algo._execute_query(cypher, params)

        # Build result objects
        return [
            HybridSearchResult(
                node_id=r["node_id"],
                name=r["name"],
                qualified_name=r["qualified_name"],
                node_type=r["node_type"],
                final_score=r["score"],
                vector_score=r.get("vector_score", 0.0),
                text_score=r.get("text_score", 0.0),
                pagerank_score=r.get("pagerank_score", 0.0),
                community_score=r.get("community_score", 0.0),
                metadata={}
            )
            for r in results
        ]

    def _build_hybrid_query(self, filters: dict[str, Any] | None, weights: dict[str, float]) -> str:
        """Build hybrid search query with optional filters."""
        # Build the corrected Cypher query
        query = """
        WITH $embedding AS query_vec, $keywords AS keywords, $top_k AS top_k

        // Step 1: Vector searches across multiple indexes
        CALL vector_search.search("function_embedding_index", query_vec, top_k * 2)
        YIELD node, similarity
        WITH query_vec, keywords, top_k,
             collect({node: node, similarity: similarity}) AS function_results

        CALL vector_search.search("class_embedding_index", query_vec, top_k * 2)
        YIELD node, similarity
        WITH query_vec, keywords, top_k, function_results,
             collect({node: node, similarity: similarity}) AS class_results

        WITH query_vec, keywords, top_k,
             function_results + class_results AS vector_results

        // Step 2: Text search for keyword matches
        UNWIND keywords AS keyword
        MATCH (t:Function|Class|Method)
        WHERE t.name CONTAINS keyword OR t.docstring CONTAINS keyword
        WITH vector_results, collect(DISTINCT t) AS text_results

        // Step 3: Merge and rank with graph centrality
        UNWIND vector_results AS vr
        WITH vr.node AS n, vr.similarity AS vector_score, text_results

        WITH n, vector_score,
             CASE WHEN ANY(t IN text_results WHERE id(t) = id(n)) THEN $text_weight ELSE 0 END AS text_score,
             COALESCE(n.pagerank_score, 0.1) AS pagerank_score,
             COALESCE(n.community_importance, 0.0) AS community_score

        WITH n,
             (vector_score * $vector_weight) + (text_score * $text_weight) +
             (pagerank_score * $pagerank_weight) + (community_score * $community_weight) AS final_score
        ORDER BY final_score DESC
        LIMIT top_k

        RETURN id(n) AS node_id, n.name AS name, n.qualified_name AS qualified_name,
               labels(n)[0] AS node_type, final_score AS score,
               vector_score, text_score, pagerank_score, community_score
        """
        return query
