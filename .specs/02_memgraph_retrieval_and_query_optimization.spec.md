# Memgraph Retrieval and Query Optimization Design Specification

## Purpose
Optimize the GraphRAG retrieval pipeline, Cypher query generation, and vector search capabilities by leveraging Memgraph's native features to improve result relevance, query performance, and reduce architectural complexity.

## Current State
The codebase has several critical gaps in Memgraph feature utilization that impact performance and result quality:
1. **Separate Vector Store**: Uses Qdrant for vector search instead of Memgraph's native vector storage/indexing, adding unnecessary cross-service latency and architectural complexity
2. **Suboptimal Cypher Generation**: LLM prompts for Cypher generation do not include Memgraph-specific best practices, unsupported construct warnings, or guidance to use MAGE procedures
3. **Basic Retrieval Logic**: Only implements simple semantic + call graph retrieval, no support for advanced retrieval patterns (Local Graph Search, Global QFS, analytical queries)
4. **No Query Performance Optimization**: No use of profiling, index hinting, `ANALYZE GRAPH`, or parallel execution features to improve query speed
5. **No Hybrid Search**: Does not combine vector, text, and graph traversal in a single query, leading to less relevant results

## Proposed Implementation

### 1. Native Vector Search Integration
Replace Qdrant with Memgraph's native vector storage and search to eliminate cross-service calls and enable hybrid search capabilities:
- **Migration Steps**:
  1. Update vector index initialization to use Memgraph's native vector indexes (already partially implemented in `/codebase_rag/vector_store_memgraph.py`, but not used as the primary vector store)
  2. Modify the semantic search function to use Memgraph's `vector_search.search()` procedure directly instead of calling Qdrant
  3. Remove Qdrant dependency and all related service initialization code
- **Query Structure for Hybrid Search**:
  ```cypher
  CALL vector_search.search("function_embedding_index", $top_k, $query_vector) YIELD node, similarity
  MATCH (node)-[:CALLS*1..2]->(related:Function)
  WHERE node.name CONTAINS $text_filter
  RETURN node, similarity, collect(related) as context
  ORDER BY node.pagerank_score DESC, similarity DESC
  ```
- **Benefits**: 30-50% reduction in semantic search latency, single source of truth for all graph and vector data, enable true hybrid retrieval in one query.

### 2. Text2Cypher Prompt Optimization
Update the Cypher generation prompts to follow Memgraph best practices and avoid common pitfalls:
- **Add Memgraph-Specific Rules to Prompt**:
  1. Use Memgraph MAGE procedures instead of Neo4j APOC
  2. Avoid unsupported constructs like atom expressions `size((n)-->())`; replace with `OPTIONAL MATCH (n)-[r]->() RETURN count(r)`
  3. Use Memgraph index syntax `CREATE INDEX ON :Label(property)` not Neo4j's `CREATE INDEX ... FOR (n:Label) ON (n.property)`
  4. Use built-in traversal algorithms (*BFS, *DFS, *WSHORTEST) instead of manual pattern matching for path queries
  5. Avoid `size()` with pattern parameters, use explicit `OPTIONAL MATCH` + `count()`
  6. Use `valueType()` function instead of `IS :: TYPE` type predicate expressions
- **Add Error Correction Logic**: Feed Memgraph query errors back to the LLM with specific instructions to fix Memgraph-specific syntax issues
- **Add Query Validation**: Automatically check generated queries for unsupported constructs before execution

### 3. Retrieval Pipeline Enhancement
Implement the full set of Memgraph-powered GraphRAG retrieval patterns:
- **Analytical Queries (Text2Cypher)**:
  - For deterministic, fact-based queries, generate optimized Cypher to get exact answers directly from the graph
  - Example queries: "How many functions call UserService.create()?", "What is the average number of parameters per function in the auth module?"
- **Local Graph Search**:
  - Step 1: Use vector search to find the most relevant starting node
  - Step 2: Use Memgraph's built-in BFS/DFS to expand context up to 2-3 hops away
  - Step 3: Rank results by PageRank score + vector similarity
  - Use case: Retrieving all relevant context for a specific function or module to answer implementation questions
- **Global Query-focused Summarization (QFS)**:
  - For high-level questions that require analysis across the entire codebase:
    1. Use precomputed community summaries (from community detection)
    2. Generate partial answers for each relevant community using LLM
    3. Synthesize final answer from partial community responses
  - Use case: "What are the security vulnerabilities in the payment processing module?", "How can we improve the performance of the API rate limiting system?"

### 4. Query Performance Optimization
Implement Memgraph-specific query optimizations to improve execution speed:
- **Automatic Index Optimization**:
  - Run `ANALYZE GRAPH` after each repository indexing to help the query planner choose optimal execution plans
  - Add automatic index hinting to complex queries to ensure the best indexes are used
  - Example:
    ```cypher
    MATCH (f:Function {qualified_name: $qfn})-[:CALLS]->(related:Function)
    USING INDEX :Function(qualified_name)
    RETURN related
    ```
- **Query Profiling**:
  - Add an optional debug mode that runs `EXPLAIN`/`PROFILE` on all generated queries and logs the execution plan for performance tuning
  - Automatically detect full graph scans (`ScanAll` operators) and suggest new indexes
- **Parallel Execution**:
  - Add `USING PARALLEL EXECUTION` to large analytical queries to use multiple CPU cores for faster execution
  - Use for community detection, PageRank calculation, and large cross-codebase analytical queries
- **Round-Robin Parallel Worker Pool Implementation**
  - Implement a dedicated pool of 10 persistent parallel workers for ingestion, batch query processing, and community summary generation workloads
  - **Scheduling**: Use strict round-robin task distribution to ensure even workload allocation across all 10 workers, eliminating worker starvation
  - **Workload Partitioning Rules**:
    1. Ingestion tasks: Split repository files into 10 equal size chunks, assign one chunk per worker for parallel parsing, embedding generation, and graph insertion
    2. Batch analytical queries: Split full-graph processing jobs (PageRank calculation, bulk community summary generation, index rebuilds) into 10 parallel sub-queries, assigned round-robin to workers
    3. Retrieval batch jobs: Split large bulk query workloads into individual query tasks distributed round-robin across workers
  - **Resource & Throughput Controls**:
    1. Per-worker resource limits: Configure default CPU limit of 0.5 cores and memory limit of 512MB per worker, configurable via environment variables to prevent resource contention on host systems
    2. Memgraph rate limiting: Enforce a maximum of 5 concurrent queries per worker against Memgraph, with a global pool limit of 30 concurrent connections to avoid overwhelming the database
    3. Retry logic: Implement exponential backoff retry for all failed worker tasks (initial delay 1s, max delay 30s, 3 retries total) for transient failures (network issues, lock conflicts) before marking tasks as failed for reassignment
  - **Reliability Features**: Add task idempotency checks, automatic worker health monitoring, and failed task reassignment to avoid data consistency issues
- **Best Practice Enforcement**:
  - Automatically rewrite queries to follow Memgraph best practices: avoid Cartesian products for large datasets, limit path traversal depth, use explicit relationship types in matches

## Expected Benefits
1. **Performance**: 2x improvement in overall query latency from removing Qdrant and using optimized Memgraph queries
2. **Relevance**: 20-30% improvement in retrieval relevance from hybrid search and centrality-based ranking
3. **Simplicity**: Reduced architectural complexity by removing an external service dependency
4. **Capability**: Enable advanced global and analytical queries that were not possible with the previous architecture

## Implementation Steps
1. **Phase 1 (High Priority)**: Complete Memgraph vector store integration, remove Qdrant dependency, implement hybrid search
2. **Phase 2 (Medium Priority)**: Update Cypher generation prompts with Memgraph best practices, add error correction logic
3. **Phase 3 (Medium Priority)**: Implement Local Graph Search retrieval pipeline, add PageRank ranking to results
4. **Phase 4 (Medium Priority)**: Implement 10-worker round-robin parallel worker pool for ingestion and batch processing workloads, including task idempotency and health check features
5. **Phase 5 (Low Priority)**: Implement Global QFS pipeline, add automatic query profiling and optimization

## Dependencies
- Memgraph >= 2.15 with MAGE library (for vector search and graph algorithms)
- No additional external dependencies required
