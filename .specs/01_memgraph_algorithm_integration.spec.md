# Memgraph Algorithm Integration Design Specification

## Purpose
Leverage Memgraph's built-in graph algorithms and MAGE library to improve GraphRAG retrieval quality, performance, and functionality.

## Current State
The codebase currently only uses Memgraph as a graph store and for basic Cypher queries. No use of advanced built-in algorithms:
- No community detection (Leiden/Louvain)
- No centrality measures (PageRank, Betweenness)
- No use of built-in traversal algorithms (BFS/DFS/shortest path)
- No use of similarity algorithms from MAGE
- All traversal and ranking logic is implemented in Python, leading to higher latency and lower performance

## Proposed Implementation

### 1. Community Detection Integration
Implement automatic community detection for code and document graphs using Memgraph's optimized Leiden/Louvain algorithms:
- **Run on ingestion**: After each repository/index update, run community detection to group related code entities (functions, classes, modules) and document chunks
- **Community properties**: Store community ID on each node, plus pre-computed community summaries (using LLM) for faster retrieval
- **Supported algorithms**: 
  - Use `community_detection.leiden()` for better quality communities
  - Fallback to `community_detection.louvain()` for larger graphs
- **Cypher implementation**:
  ```cypher
  CALL community_detection.leiden(
    'CALLS', 
    'OUTGOING', 
    { community_property: 'community_id', weight_property: 'weight' }
  ) YIELD node, community_id
  RETURN count(*);
  ```

### 2. Centrality Measures Integration
Compute centrality scores for all nodes to improve result ranking:
- **PageRank**: Measure influence/importance of code entities (core classes, frequently called functions get higher scores)
- **Betweenness Centrality**: Identify bridge nodes that connect different parts of the codebase
- **Implementation**:
  - Run during ingestion/post-processing step
  - Store scores as node properties (`pagerank_score`, `betweenness_score`)
  - Use in semantic search queries to boost results from important entities
- **Cypher implementation**:
  ```cypher
  CALL pagerank.get() YIELD node, rank
  SET node.pagerank_score = rank;
  ```

### 3. Built-in Traversal Algorithm Usage
Replace Python-side traversal logic with Memgraph's optimized built-in traversal algorithms:
- **BFS for context expansion**: When retrieving related context for a code entity, use BFS to get the relevant subgraph instead of multiple Cypher queries
- **Weighted shortest path for dependency analysis**: Use built-in weighted shortest path for dependency chain queries
- **Example BFS query for context expansion**:
  ```cypher
  MATCH (start:Function {qualified_name: $start_qn})
  MATCH path = (start)-[:CALLS*BFS..3]->(related:Function)
  RETURN path, nodes(path) as context_nodes;
  ```

### 4. Similarity Algorithm Integration
Leverage MAGE similarity algorithms for better retrieval:
- **Node similarity**: Find similar code entities based on graph structure (in addition to vector similarity)
- **Jaccard/Cosine similarity**: For comparing function call patterns, dependency graphs
- **Use case**: Improve "find similar functions" queries, duplicate code detection

## Expected Benefits
1. **Improved RAG Relevance**: Community detection + centrality ranking ensures more relevant context is retrieved
2. **Performance Improvements**: Offloading traversal and algorithm work to Memgraph (implemented in optimized C++) reduces query latency by 50-80%
3. **New Functionality**: Enables use cases like dependency chain analysis, duplicate code detection, architecture visualization
4. **Better Global Queries**: Community summaries enable answering global questions about codebase structure, architecture, and patterns

## Implementation Steps
1. **Step 1 (Low Effort)**: Enable and test PageRank calculation during ingestion, integrate with search result ranking
2. **Step 2**: Implement community detection on ingestion, add community summary generation using LLM
3. **Step 3**: Refactor existing traversal queries to use built-in BFS/DFS algorithms instead of multiple separate queries
4. **Step 4**: Add similarity algorithm integration for advanced use cases

### Reliability & Validation Controls
- **Idempotency**: Add unique run ID and processing timestamp properties to all nodes modified during algorithm runs. Skip processing nodes with timestamps newer than the current run start time to avoid duplicate work.
- **Rollback Plan**: All algorithm-generated properties (community_id, pagerank_score, betweenness_score) will be written in a temporary transaction. If any step fails, the entire transaction is rolled back automatically to restore the previous state. For partial failures in incremental runs, a cleanup query will delete all properties set during the failed run before retrying.
- **Performance Benchmarking**: Run validation tests against 100k+ node production-sized graphs before production release to confirm the 50-80% latency reduction claim, with public benchmark results published for reference.

## Dependencies
- Memgraph MAGE library enabled in the Docker container (already included in standard Memgraph images)
- No additional external dependencies required

## Priority
- **High**: PageRank and community detection provide immediate relevance improvements
- **Medium**: Traversal algorithm refactoring provides performance gains
- **Low**: Similarity algorithms for advanced use cases
