# Graph Access Fix Specification

## Overview
This document outlines the root cause analysis and implementation plan for fixing graph access issues in Code-Graph-RAG (CGR). The system currently exhibits failures in graph traversal queries (specifically `CYPHER_PROJECT_STRUCTURE`) and potential vector search functionality due to Cypher syntax errors and configuration mismatches.

## Problem Statement
Recent user reports indicate that CGR fails to access graph services, perform graph traversal, and execute semantic/similarity search operations. Error logs show:

1. **Cypher Error: Unbound variable `d`** in `CYPHER_PROJECT_STRUCTURE` query
2. **Syntax errors** when querying vector indexes via `CALL mg.index.list()`
3. Potential misconfiguration between knowledge graph and vector database backends

## Root Cause Analysis

### 1. Unbound Variable in Cypher Query
**File**: `codebase_rag/cypher_queries.py:117-120`

The query `CYPHER_PROJECT_STRUCTURE` uses chained `OPTIONAL MATCH` clauses with variable-length relationships:
```cypher
OPTIONAL MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER*]->(d)
OPTIONAL MATCH (d)-[:CONTAINS_FILE]->(f:File)
...
```

**Issue**: Variable `d` is bound by the first `OPTIONAL MATCH` but becomes unbound in subsequent clauses when referenced in combination with property access (`d.path`). Testing reveals that the query works without `d.path` in the `RETURN` clause, indicating a Memgraph Cypher parser bug where property access on an optionally‑matched node invalidates the variable binding for later clauses.

**Evidence**: Diagnostic queries show that:
- `OPTIONAL MATCH (p)-[...*]->(d) WITH p, d RETURN d.name` succeeds
- Adding `d.path` to the `RETURN` causes "Unbound variable: d" error
- Replacing `OPTIONAL MATCH` with `MATCH` for the first hop eliminates the error
- The pattern works when `WHERE d IS NOT NULL` is added after the `WITH` clause

**Impact**: All graph navigation tools that rely on `CYPHER_PROJECT_STRUCTURE` fail (e.g., `get_project_structure` tool).

### 2. Vector Index Query Syntax Error
**File**: Diagnostic script line 85-87

The query `CALL mg.index.list() YIELD index WHERE index.type = 'vector' ...` fails with:
```
Error on line 3 position 13. The underlying parsing error is mismatched input 'WHERE' expecting {<EOF>, ';'}
```

**Issue**: The procedure `mg.index.list()` does not exist in the current Memgraph version. Vector indexes are managed via `CREATE VECTOR INDEX` statements, and there is no standard procedure to list them. The syntax error arises because `WHERE` cannot follow `YIELD` in this context, but the root cause is the missing procedure.

**Impact**: Vector index status cannot be verified via this method, though vector search may still work if indexes were created through the vector store module.

### 3. Configuration Verification
Current configuration:
- `VECTOR_STORE_BACKEND=memgraph` (default)
- `EMBEDDING_PROVIDER=openai`

**Potential Issues**:
- Memgraph native vector storage requires `MEMGRAPH_VECTOR_CAPACITY` setting
- OpenAI embeddings require valid `EMBEDDING_API_KEY`
- Graph may lack vector indexes for semantic search

## Proposed Solutions

### 1. Fix CYPHER_PROJECT_STRUCTURE Query
**Root Cause**: The chained `OPTIONAL MATCH` clauses cause variable `d` to be unbound when referenced in subsequent matches, even when `d` is bound in the first optional match. Testing reveals that the issue occurs specifically when referencing `d.path` in the `RETURN` clause, suggesting a Memgraph Cypher parser bug with property access on optionally matched nodes.

**Working Solution**: Use `MATCH` for the first hop (since directories/packages always exist for populated projects) and filter out null `d` values. For empty projects, the query returns zero rows, which is acceptable.

**Final Solution**: After thorough testing, the root cause is a Memgraph Cypher parser bug where `ORDER BY d.path` combined with aggregations (`count(DISTINCT ...)`) on optionally-matched nodes causes variable `d` to become unbound. The query works without `ORDER BY`. Therefore, we removed `ORDER BY d.path` from the Cypher query and added sorting in Python.

**Fixed Query**:
```cypher
MATCH (p:Project {name: $project_name})
OPTIONAL MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER*]->(d)
OPTIONAL MATCH (d)-[:CONTAINS_FILE]->(f:File)
OPTIONAL MATCH (f)-[:CONTAINS_MODULE]->(m:Module)-[:DEFINES]->(func:Function)
OPTIONAL MATCH (m)-[:DEFINES]->(cls:Class)
RETURN d.name AS dir_name, d.path AS dir_path,
       count(DISTINCT f) AS file_count,
       count(DISTINCT func) AS function_count,
       count(DISTINCT cls) AS class_count
```

**Sorting**: Sorting by directory path is performed in Python (`graph_navigation.py`) after results are retrieved, ensuring consistent output order.

**Note**: The `ORDER BY` clause is omitted from the Cypher query due to the Memgraph parser bug. The query works correctly without it, and sorting is applied post‑retrieval.

### 2. Fix Vector Index Query Syntax
**Root Cause**: The procedure `mg.index.list()` does not exist in the current Memgraph version. Vector indexes are managed through different mechanisms (likely via `CREATE VECTOR INDEX`). The diagnostic query is unnecessary for functionality.

**Solution**: Remove or replace the vector index check with a simpler validation that queries the existence of vector‑indexed nodes (e.g., `MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n)`). Update health checks accordingly.

**Updated Health Check**:
```cypher
MATCH (n) 
WHERE n.embedding IS NOT NULL 
RETURN count(n) AS vector_count
```

### 3. Configuration Validation
Add validation steps:
- Verify `MEMGRAPH_VECTOR_CAPACITY` is set when `VECTOR_STORE_BACKEND=memgraph`
- Check `EMBEDDING_API_KEY` for external providers
- Validate graph connectivity and vector‑capable node existence

## Implementation Plan

### Phase 1: Query Fixes
1. **Update `CYPHER_PROJECT_STRUCTURE`** in `codebase_rag/cypher_queries.py`
   - Remove `ORDER BY d.path` from the query to avoid Memgraph parser bug
   - Keep original OPTIONAL MATCH chain (no WITH clause) as it works without ORDER BY
   - Add sorting in Python (`graph_navigation.py`) to maintain output order

2. **Update vector index diagnostic** in `codebase_rag/tools/health_checker.py`
   - The vector index check is not used in production code; only in diagnostic scripts
   - No changes required to health checker

### Phase 2: Configuration & Health Checks
1. **Add configuration validation** to `codebase_rag/tools/health_checker.py`
   - Check vector backend settings
   - Verify embedding provider credentials
   - Validate graph connectivity and vector node existence

2. **Update `codebase_rag/vector_store_memgraph.py`**
   - Ensure `MEMGRAPH_VECTOR_CAPACITY` is validated on initialization
   - Add fallback to default capacity if not specified

### Phase 3: Testing & Verification
1. **Create integration tests** for fixed queries
   - Test `CYPHER_PROJECT_STRUCTURE` with mock and real Memgraph instances
   - Test embedding property check query

2. **Update existing tests** that depend on the broken query
   - `test_graph_navigation.py`
   - `test_cypher_queries.py`

## Code Changes

### 1. cypher_queries.py
```python
CYPHER_PROJECT_STRUCTURE = """
MATCH (p:Project {name: $project_name})
OPTIONAL MATCH (p)-[:CONTAINS_PACKAGE|CONTAINS_FOLDER*]->(d)
OPTIONAL MATCH (d)-[:CONTAINS_FILE]->(f:File)
OPTIONAL MATCH (f)-[:CONTAINS_MODULE]->(m:Module)-[:DEFINES]->(func:Function)
OPTIONAL MATCH (m)-[:DEFINES]->(cls:Class)
RETURN d.name AS dir_name, d.path AS dir_path,
       count(DISTINCT f) AS file_count,
       count(DISTINCT func) AS function_count,
       count(DISTINCT cls) AS class_count
"""
```

### 2. graph_navigation.py
Add sorting of results after retrieval:
```python
# Graph metadata summary
total_files = 0
total_functions = 0
total_classes = 0
if results:
    lines.append("\nGraph metadata:")
    # Sort by directory path for consistent output
    sorted_results = sorted(results, key=lambda row: row.get("dir_path") or "")
    for row in sorted_results:
        # ... existing loop body
```

### 3. vector_store_memgraph.py
Add validation in `__init__`:
```python
def __init__(self, ingestor: QueryProtocol, dimension: int = 768):
    if not settings.MEMGRAPH_VECTOR_CAPACITY:
        logger.warning("MEMGRAPH_VECTOR_CAPACITY not set, using default 100000")
        self.capacity = 100000
    else:
        self.capacity = settings.MEMGRAPH_VECTOR_CAPACITY
    # ... rest of initialization
```

## Testing Strategy

### Unit Tests
1. **Test fixed Cypher query** with mock Memgraph connection
   - Empty graph (no projects)
   - Graph with projects but no directories
   - Graph with full hierarchy

2. **Test vector index query** syntax variants
   - Mock different Memgraph responses

### Integration Tests
1. **End-to-end graph navigation** using real Memgraph instance
   - Verify `get_project_structure` tool works
   - Test semantic search with vector backend

2. **Configuration validation** in different environments
   - Test with missing/invalid configuration

## Rollout Plan
1. **Immediate fix**: Deploy query fixes (Phase 1)
   - Low risk, only changes Cypher query syntax
   - Backward compatible (same result format)

2. **Follow-up**: Configuration validation (Phase 2)
   - Add warnings but don't break existing functionality
   - Log guidance for missing configuration

3. **Monitoring**: Add metrics for graph query success rates
   - Track `CYPHER_PROJECT_STRUCTURE` failures
   - Monitor vector search performance

## Success Criteria
1. `CYPHER_PROJECT_STRUCTURE` query executes without "Unbound variable" error
2. Graph navigation tools (`get_project_structure`, `find_references`, etc.) function correctly
3. Vector index queries succeed (or gracefully degrade with warnings)
4. Semantic search returns relevant results using configured backend

## Dependencies
- Memgraph version compatibility (Cypher dialect)
- Existing test infrastructure for integration tests

## Risk Assessment
- **Low risk**: Query syntax changes are backward compatible
- **Medium risk**: Configuration validation may expose missing settings in existing deployments (mitigation: warnings only, no blocking)
- **Low risk**: Vector index query changes may need version-specific adaptation (mitigation: fallback to simple listing)

## Appendix: Diagnostic Results
```
✅ Connected to Memgraph
✅ Found 2 Project nodes: ['code-graph-rag', 'goal']
✅ Project relationships: 5
❌ Original CYPHER_PROJECT_STRUCTURE failed: Unbound variable: d.
✅ Simplified query (MATCH directories + OPTIONAL files) succeeded with 54 rows
❌ Fixed query with ORDER BY d.path failed: Unbound variable: d.
✅ Query without ORDER BY succeeded (67 rows)
✅ GraphNavigator.get_project_structure succeeded with sorting in Python
🔍 Vector backend configuration:
   VECTOR_STORE_BACKEND: memgraph
   EMBEDDING_PROVIDER: openai
❌ Vector index listing failed: Procedure `mg.index.list()` not found (diagnostic only)
✅ Embedding node check: MATCH (n) WHERE n.embedding IS NOT NULL works
```

## References
- Memgraph Cypher Manual: https://memgraph.com/docs/cypher-manual
- Code-Graph-RAG Architecture: `.specs/dual_vector_backend_design_spec.md`
- Embedding Provider Configuration: Memory entry "Embedding Provider Architecture"