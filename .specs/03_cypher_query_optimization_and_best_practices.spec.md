# Cypher Query Optimization and Memgraph Best Practices Integration Design Specification

## Purpose
Optimize all Cypher query generation, execution patterns, and query planning to fully leverage Memgraph's performance features, eliminate invalid/non-optimal query patterns, and ensure maximum query speed and correctness.

## Current State
The codebase has several critical gaps in Memgraph Cypher best practices:
1. **Suboptimal Text2Cypher generation prompts**: No Memgraph-specific syntax guidance, so generated queries may use Neo4j-specific patterns that don't work in Memgraph, unsupported constructs (like atom expressions `size((n)-->())`), or inefficient patterns.
2. **No use of query planner optimization features**: No `ANALYZE GRAPH` run after ingestion to help the query planner choose optimal execution plans, no index hinting for complex queries.
3. **No use of parallel execution**: Large analytical queries don't use `USING PARALLEL EXECUTION` to leverage multiple CPU cores for faster results.
4. **Missing query performance observability**: No built-in profiling/debug mode to identify slow queries or full graph scans.
5. **Suboptimal existing query patterns**: Some queries don't follow Memgraph best practices for traversal, filtering, and result projection, leading to slower execution and larger result roundtrips.

## Proposed Solution
Implement full Memgraph Cypher best practices across the entire query lifecycle, from generation to execution to observability.

### Implementation Details

#### 1. Text2Cypher Prompt Optimization
Update all Cypher generation prompts to include Memgraph-specific rules and constraints to ensure valid, optimal queries:
- **Add Memgraph-specific syntax rules** to the prompt:
  - Use Memgraph MAGE procedures instead of Neo4j APOC procedures
  - Avoid unsupported constructs: atom expressions like `size((n)-->())`; replace with `OPTIONAL MATCH (n)-[r]->() RETURN count(r)`
  - Use Memgraph index syntax `CREATE INDEX ON :Label(property)` not Neo4j's `CREATE INDEX ... FOR (n:Label) ON (n.property)`
  - Use built-in traversal syntax `*BFS`, `*DFS`, `*KSHORTEST` instead of Neo4j's `shortestPath`/`kShortestPaths` functions
  - Use `valueType()` function instead of `IS :: TYPE` type predicate expressions
  - Avoid Neo4j-specific `count()`/`collect()` subqueries; use standard aggregation functions
- **Add performance best practices** to the prompt:
  - Prefer single pattern matches for traversals to leverage Cyphermorphism's EdgeUniquenessFilter and avoid duplicate results
  - Use explicit relationship types in matches to reduce scan scope
  - Limit traversal depth with range patterns `*1..3` to avoid full graph scans
  - Project only required properties in results to reduce roundtrip time
- **Add feedback loop**: When a generated query fails with a Memgraph error, automatically feed the error message back to the LLM to regenerate a corrected query following Memgraph syntax rules.

#### 2. Query Planner Optimization
Add automatic query planner optimization steps to improve execution speed:
- **Run `ANALYZE GRAPH` automatically after every ingestion/update**: This computes statistics about node counts, relationship distributions, and property cardinalities to help the query planner choose the optimal execution plan.
- **Add index hinting to complex queries**: For known slow queries (like multi-hop traversals, analytical aggregations), add explicit index hints to ensure the planner uses the optimal index:
  ```cypher
  MATCH (f:Function {qualified_name: $qn})-[:CALLS *1..3]->(related:Function)
  USING INDEX :Function(qualified_name)
  RETURN related
  ```
- **Enable parallel execution for large queries**: Add `USING PARALLEL EXECUTION` to all analytical queries that process large portions of the graph (like codebase-wide statistics, global dependency analysis) to use multiple CPU cores and reduce execution time by up to 80% for large graphs.

#### 3. Existing Query Pattern Optimization
Rewrite all existing Cypher queries to follow Memgraph best practices:
- **Replace atom expressions**: Convert any queries using `size((n)-->())` to use proper `OPTIONAL MATCH` + aggregation:
  *Before*:
  ```cypher
  MATCH (f:Function) RETURN f.name, size((f)-[:CALLS]->()) as call_count
  ```
  *After*:
  ```cypher
  MATCH (f:Function)
  OPTIONAL MATCH (f)-[c:CALLS]->()
  RETURN f.name, count(c) as call_count
  ```
- **Optimize traversal queries**: Use Memgraph's built-in traversal algorithms (`BFS`, `DFS`) for path queries instead of generic variable-length patterns to get optimized execution:
  *Before*:
  ```cypher
  MATCH path = (start:Function {name: $name})-[:CALLS *]->(end:Function)
  RETURN path
  ```
  *After*:
  ```cypher
  MATCH path = (start:Function {name: $name})-[:CALLS *BFS]->(end:Function)
  RETURN path
  ```
- **Optimize result projection**: Modify queries to return only required properties instead of full nodes, use `project(path)` for path queries to reduce result size and roundtrip time:
  *Before*:
  ```cypher
  MATCH path = (f:Function)-[:CALLS *1..2]->(related)
  RETURN path
  ```
  *After*:
  ```cypher
  MATCH path = (f:Function)-[:CALLS *1..2]->(related)
  WITH project(path) as result
  RETURN result.nodes {.name, .qualified_name, .path}, result.relationships {.type}
  ```
- **Optimize filter patterns**: For `OR` filters on the same property, use `IN []` instead of multiple `OR` clauses to leverage label-property indexes:
  *Before*:
  ```cypher
  MATCH (f:Function) WHERE f.language = "Python" OR f.language = "TypeScript"
  RETURN f
  ```
  *After*:
  ```cypher
  MATCH (f:Function) WHERE f.language IN ["Python", "TypeScript"]
  RETURN f
  ```

#### 4. Query Performance Observability
Add built-in query profiling and slow query detection to identify optimization opportunities:
- **Add debug query profiling mode**: When enabled, prepend `PROFILE` to all queries, log the execution plan, and highlight queries with full graph scans (`ScanAll` operators) or high `actual hits` values that indicate potential performance issues.
- **Add slow query logging**: Automatically log queries that take longer than a configurable threshold (default 500ms) along with their execution plan for later optimization.
- **Add automatic index recommendation**: Analyze slow query plans to identify missing indexes, generate recommendations for new label/label-property indexes that would speed up common queries.

## Expected Benefits
1. **Query speed improvements**: 30-70% faster query execution from optimized patterns, better query planning, and parallel execution.
2. **Eliminate invalid queries**: Memgraph-specific prompt guidance ensures all generated queries are valid and use optimal Memgraph patterns.
3. **Better observability**: Slow query logging and profiling help identify and fix performance bottlenecks proactively.
4. **Reduced roundtrip time**: Optimized result projection reduces the size of data returned from Memgraph by 30-60% for most queries, improving end-to-end response time.

## Implementation Roadmap
### Phase 1 (High Priority, ~1 week)
- [ ] Update all Text2Cypher prompts with Memgraph-specific syntax rules and best practices
- [ ] Add automatic `ANALYZE GRAPH` run after all ingestion/update operations
- [ ] Rewrite all existing Cypher queries in `cypher_queries.py` to follow Memgraph best practices
- [ ] Add `USING PARALLEL EXECUTION` to all large analytical queries

### Phase 2 (Medium Priority, ~2 weeks)
- [ ] Implement query feedback loop for failed generated Cypher queries
- [ ] Add debug profiling mode with slow query logging
- [ ] Add index hinting to all complex traversal/filter queries

### Phase 3 (Low Priority, ~1 week)
- [ ] Implement automatic index recommendation feature based on slow query analysis
- [ ] Add performance regression tests for common query patterns to prevent performance regressions in future releases

### Implementation Guardrails (Pre-Development Requirements)
- **Existing Query Inventory**: Complete a full inventory of all hardcoded Cypher queries in the codebase (including file paths, usage frequency, and purpose) before any rewrite work begins, to ensure no queries are missed during optimization
- **Benchmarking Plan**: Establish pre-optimization performance baselines for all common query patterns (semantic search, multi-hop traversal, analytical aggregation queries) against 1k, 10k, and 100k node test graphs, to validate performance improvement claims post-optimization
- **Query Versioning & Rollback System**: Implement a versioned query repository for all hardcoded and optimized queries, with automatic fallback to the previous working query version if optimized queries fail validation, return incorrect results, or degrade performance
- **Pre-Execution Validation**: Add a static validation step for all generated and hardcoded queries before execution that checks for Memgraph syntax compatibility, performance anti-patterns (unbounded traversals, full graph scans, missing index hints), and invalid constructs to prevent runtime errors
