# Memgraph Native Vector Retrieval & Atomic GraphRAG Pipeline Design Specification

## Purpose
Refactor the entire vector retrieval layer to use Memgraph's native vector storage and indexing capabilities entirely, remove external Qdrant dependency, and implement full Atomic GraphRAG pipelines running directly within Memgraph using MAGE library features for improved performance, reduced architectural complexity, and advanced retrieval capabilities.

## Current State
The existing implementation has critical inefficiencies and missed opportunities to leverage Memgraph's native GraphRAG capabilities:
1. **Separate Qdrant vector store**: Stores vectors in an external database leading to cross-service latency, data duplication, and consistency issues when the code graph changes.
2. **Decoupled retrieval**: Vector search runs separately from graph traversal, requiring multiple database queries and application-side data merging, eliminating the possibility of atomic hybrid vector+graph queries.
3. **No use of MAGE embedding/LLM procedures**: All embedding generation and LLM inference runs in the application layer, preventing implementation of fully-atomic GraphRAG pipelines within Memgraph.
4. **Limited retrieval patterns**: Only implements basic semantic lookup, no support for advanced GraphRAG patterns like community-augmented retrieval or query-focused summarization using Memgraph's graph analytics capabilities.

## Proposed Solution
Fully migrate vector storage and retrieval to Memgraph's native vector capabilities, eliminate Qdrant dependency entirely, and implement Atomic GraphRAG pipelines running as single Memgraph queries using MAGE's `embeddings` and `llm` modules.

### Implementation Details

#### 1. Vector Layer Refactoring
- **Deprecate Qdrant entirely**: Remove all Qdrant-related code, configuration, and dependencies.
- **Upgrade Memgraph vector store implementation**:
  - Store embeddings directly on code nodes (Function, Class, Module, Interface, etc.) as properties, eliminating duplicate data storage.
  - Create optimized per-node-type vector indexes during ingestion:
    ```cypher
    CREATE VECTOR INDEX function_embedding_index ON :Function(embedding)
    WITH CONFIG {
        dimension: 768,
        capacity: 100000,
        metric: "cosine"
    };
    CREATE VECTOR INDEX class_embedding_index ON :Class(embedding)
    WITH CONFIG {
        dimension: 768,
        capacity: 50000,
        metric: "cosine"
    };
    -- Similar indexes for Method, Module, Interface, Contract nodes
    ```
  - Automatically update embeddings incrementally during code ingestion/updates, no full reindex required for small changes.
- **Consistency guarantee**: Vector updates run in the same transaction as graph updates, ensuring no mismatches between graph structure and embeddings.

#### 2. Atomic GraphRAG Retrieval Pipelines
Implement all retrieval patterns as single atomic Memgraph queries to eliminate application-side processing overhead:

##### A. Local Graph Search Pipeline (Default Retrieval)
For standard code questions requiring context around relevant code entities:
```cypher
WITH $query_embedding AS query_vec, $top_k AS top_k
-- Step 1: Vector search to find top semantically matching code nodes
CALL vector_search.search('function_embedding_index', top_k * 2, query_vec)
YIELD node AS start_node, similarity AS sim
-- Step 2: BFS traversal to get related context (callers, dependencies, parent classes)
MATCH path = (start_node)-[:CALLS|:DEFINES|:IMPORTS*1..2]-(related_node)
WHERE related_node:Function OR related_node:Class OR related_node:Module
-- Step 3: Rank by similarity + PageRank to prioritize important entities
WITH DISTINCT related_node, sim, COALESCE(related_node.pagerank_score, 0.1) AS pr_score
ORDER BY (sim * 0.7) + (pr_score * 0.3) DESC
LIMIT top_k
-- Step 4: Return node data + related paths for context
RETURN
    related_node.name AS name,
    related_node.qualified_name AS qualified_name,
    related_node.path AS file_path,
    related_node.docstring AS docstring,
    related_node.start_line AS start_line,
    collect(DISTINCT [n IN nodes(path) | n.qualified_name]) AS context_path,
    sim AS similarity
```
- Entire retrieval runs in a single Memgraph query with no application-side processing, cutting latency by 50%+ compared to separate Qdrant + Memgraph calls.

##### B. Query-Focused Summarization Pipeline (Global/Architecture Questions)
For high-level questions requiring analysis across the entire codebase (e.g. "How does the authentication system work?", "What are the performance bottlenecks in the API?"):
```cypher
WITH $user_query AS user_query, $query_embedding AS query_vec
-- Step 1: Find relevant communities using semantic similarity to community summaries
CALL vector_search.search('community_embedding_index', 5, query_vec)
YIELD node AS community_node, similarity
WHERE community_node.nodes_count > 3
-- Step 2: Generate partial answer per relevant community using Memgraph LLM procedure
WITH user_query, community_node, similarity,
    llm.complete(
        'You are a code expert. Answer this question using only the provided community summary: '
        + user_query + '\nCommunity Summary: ' + community_node.summary
    ) AS partial_answer
WHERE partial_answer IS NOT NULL
-- Step 3: Aggregate partial answers into final consolidated response
WITH user_query, collect(partial_answer) AS partial_answers
WITH
    'Synthesize these partial answers from different parts of the codebase into a single coherent answer to the question: '
    + user_query + '\nPartial Answers:\n' + reduce(s = '', ans IN partial_answers | s + '- ' + ans + '\n') AS prompt
RETURN llm.complete(prompt) AS final_answer
```
- Entire summarization pipeline runs within Memgraph, no data leaves the database.

#### 3. Integration with MAGE Embedding & LLM Modules
- **In-database embedding generation**: Use MAGE's `embeddings.text` procedure to generate embeddings directly in Memgraph during ingestion, eliminating data transfer overhead:
  ```cypher
  CALL embeddings.text($code_snippet, {provider: $provider, model: $model})
  YIELD embedding
  MATCH (f:Function {qualified_name: $qn})
  SET f.embedding = embedding
  ```
- **Supported embedding providers**: OpenAI, Anthropic, Ollama, all configurable via existing environment variables, no application-side changes needed.
- **LLM procedure support**: Use MAGE's `llm.complete` with any supported LLM provider for in-database summarization, eliminating application-side LLM calls for retrieval processing.

#### 4. Community-Augmented Retrieval Improvements
- Precompute community summaries during ingestion using Leiden community detection:
  1. Run `community_detection.leiden()` to group code entities into logical communities.
  2. Generate text summaries of each community using LLM.
  3. Embed community summaries and store in `Community` nodes linked to member code entities.
- Improve retrieval relevance by adding community context to all local search results, enabling the LLM to understand code structure better.

## Expected Benefits
1. **Performance**: 40-60% reduction in retrieval latency by eliminating cross-service calls to Qdrant and running entire pipelines within Memgraph.
2. **Simplified Architecture**: Single database dependency for all graph, vector, and LLM operations, eliminating Qdrant deployment and management overhead.
3. **Advanced Retrieval Capabilities**: Enable hybrid vector+graph queries, community-augmented retrieval, and query-focused summarization that were not possible with separate Qdrant deployment.
4. **Consistency**: Atomic embedding + graph updates guarantee no mismatches between code structure and vector data.
5. **Cost**: Reduce operational costs by removing an external service and reducing data transfer between services.

## Implementation Roadmap
### Phase 1 (High Priority, ~2 weeks)
- [ ] Remove all Qdrant-related code, configuration, and dependencies.
- [ ] Upgrade `MemgraphVectorStore` to store embeddings directly on code nodes.
- [ ] Update ingestion pipeline to generate and store embeddings during code parsing.
- [ ] Migrate semantic search functionality to use Memgraph native vector search procedures.
- [ ] Run full test suite to ensure retrieval relevance matches or exceeds previous Qdrant implementation.

### Phase 2 (Medium Priority, ~3 weeks)
- [ ] Implement local graph search atomic retrieval pipeline as single Cypher queries.
- [ ] Integrate PageRank scoring into retrieval ranking to prioritize important code entities.
- [ ] Update chat interface to use new atomic retrieval pipeline by default.
- [ ] Add configurable retrieval parameters (hop depth, ranking weight, result count) per query type.

### Phase 3 (Medium Priority, ~3 weeks)
- [ ] Implement community detection during ingestion using MAGE Leiden algorithm.
- [ ] Add community summary generation flow using LLM during ingestion.
- [ ] Implement query-focused summarization pipeline for global/architecture questions.
- [ ] Add query type detection to automatically select between local search and QFS pipelines based on user question.

### Phase 4 (Low Priority, ~2 weeks)
- [ ] Implement incremental embedding updates for real-time code changes, avoiding full reindex on small edits.
- [ ] Add in-database embedding generation using MAGE `embeddings` module, eliminate application-side embedding generation.
- [ ] Add vector search query caching using Memgraph built-in query caching for repeat queries.

### Migration & Compatibility Controls
- **Existing Qdrant Migration Path**: Add a one-way migration script that:
  1. Reads all existing embeddings from Qdrant in batches of 1000
  2. Writes embeddings directly to matching Memgraph code nodes using qualified_name as the join key
  3. Validates 10% of migrated embeddings automatically to ensure no data loss
  4. Includes a rollback option to revert to Qdrant if migration fails
- **Large Graph Fallback**: Add a feature flag `MEMGRAPH_VECTOR_SEARCH_ENABLED` that defaults to `true`, but can be disabled to fall back to Qdrant for deployments with >1M nodes where Memgraph vector search performance is insufficient
- **Embedding Dimension Validation**: Add automatic validation during index creation that confirms the configured embedding model dimension matches the vector index dimension, with clear error messages for mismatches to avoid failed index creation

## Dependencies
- Memgraph >= 2.15 with MAGE library installed (included in standard `memgraph/memgraph-mage` Docker image).
- No additional external service dependencies required.
