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
