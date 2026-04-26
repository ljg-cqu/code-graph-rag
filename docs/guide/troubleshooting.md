# Troubleshooting Guide

## Document Graph Issues

### Document indexing fails with "Connection refused"

**Symptom**: `ERROR | codebase_rag.services.graph_service:_execute_query:501 - !!! Cypher Error: couldn't connect to host: Connection refused`

**Cause**: The document graph Memgraph instance (port 7688) is not running or crashed.

**Resolution**:
1. Check container status: `docker ps | grep memgraph-doc`
2. Restart: `docker-compose up -d memgraph-doc`
3. Re-run indexing: `cgr index /path/to/repo --with-docs`
4. Partial progress is automatically preserved. Only unflushed documents need re-processing.

### Document graph unavailable during indexing

**Symptom**: CLI reports "Document graph unavailable (localhost:7688). Continuing with code graph only."

**Cause**: The document Memgraph instance was unreachable at startup or during indexing.

**Resolution**:
- Start the document graph: `docker-compose up -d memgraph-doc`
- Re-run indexing with `--with-docs` if needed
- The code graph indexing completes normally; only document indexing is skipped

### Partial document indexing progress lost

**Symptom**: Fewer documents indexed than expected after a restart.

**Cause**: With `DOC_INCREMENTAL_FLUSH_INTERVAL > 0`, documents processed before a flush are persisted. However, if a document fails after its version was cached incrementally, it may be skipped on restart.

**Resolution**:
- Check the dead letter queue (DLQ) in `.cgr/doc_errors/` for failed documents
- Re-index with `--force` or `--clean` to re-process all documents
- Adjust `DOC_INCREMENTAL_FLUSH_INTERVAL` to balance throughput vs. recovery

## JSON Graph Issues

### JSON ingestion fails with "Connection refused"

**Symptom**: `ERROR | codebase_rag.services.graph_service - couldn't connect to host: Connection refused` during JSON ingestion.

**Cause**: The JSON graph Memgraph instance (port 7689) is not running.

**Resolution**:
1. Check container status: `docker ps | grep memgraph-json`
2. Start the JSON graph: `docker-compose up -d memgraph-json`
3. Re-run ingestion: `cgr ingest-json /path/to/data.json`

### JSON entities missing embeddings

**Symptom**: Semantic search returns no results for JSON entities, or entities have no `embedding` property.

**Cause**: Embedding provider is unavailable or disabled.

**Resolution**:
1. Check embedding status: `CGR_JSON_EMBEDDINGS_ENABLED=true` (default)
2. For local embeddings, ensure `torch` and `transformers` are installed: `uv sync --extra semantic`
3. If embeddings are optional, ingestion continues without them (keyword search still works)
4. To fail on missing embeddings: `CGR_JSON_EMBEDDINGS_REQUIRED=true`

### PageRank scores not computed for JSON entities

**Symptom**: JSON entities have `pagerank_score=0.1` (default) after ingestion.

**Cause**: PageRank computation failed or was disabled.

**Resolution**:
1. Ensure MAGE procedures are installed in Memgraph
2. Check logs for "pagerank.get() not available" message
3. PageRank is skipped gracefully on Memgraph Community (no MAGE)
4. Re-run ingestion with `--json-compute-pagerank` (default: enabled)

### JSON query returns empty results

**Symptom**: `query_json_graph` tool returns "No results found" for queries.

**Cause**: No JSON data ingested, or filters don't match any entities.

**Resolution**:
1. Verify JSON data is ingested: `cgr list-datasets`
2. Check entity count in JSON graph: query `MATCH (n:JsonEntity) RETURN count(n)`
3. Try broader queries without category filters
4. Use keyword search fallback: queries work even without embeddings
