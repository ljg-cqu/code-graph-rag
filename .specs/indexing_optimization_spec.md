# Code Graph RAG Indexing Optimization Specification
## Issues Identified & Fixes
---
### 1. Missing Code Node Embeddings
**Issue**: 81 code nodes (functions/classes/modules) have no embeddings after ingestion
**Root Cause**: Embedding generation skipped for certain node types, error handling missing for failed embedding requests
**Proposed Fix**:
- Add retry mechanism (3 attempts) for failed embedding requests
- Add validation step to identify missing embeddings before finalization
- Add fallback embedding generation for edge case node types
- Log all missing embedding node IDs for debugging
---
### 2. Memgraph Connection Stability Issues
**Issue**: "Broken pipe" / "bad session" errors when deleting existing document nodes during indexing
**Root Cause**: Connection pool exhaustion, idle connections not recycled properly
**Proposed Fix**:
- Implement connection health check before executing write queries
- Add connection retry logic (2 attempts) for failed database operations
- Increase connection pool size to 10 from default 5
- Add explicit connection reset on error to avoid stale sessions
- Increase Document Graph Memgraph memory limit from 2GB to 4GB (prevents OOM/segfault crashes during large document indexing)
- Fix embedding dimension mismatch between OpenAI text-embedding-v4 (1024d) and local fallback model microsoft/unixcoder-base (768d) to prevent indexing failures on network/API errors
---
### 3. Large Document Chunking Limit
**Issue**: Large documents hit MAX_CHUNKS_PER_DOCUMENT (1000) limit, partial content indexing
**Root Cause**: Static limit not scaled for large documentation files
**Proposed Fix**:
- Increase MAX_CHUNKS_PER_DOCUMENT to 5000 for documentation graph
- Add dynamic chunk size adjustment for large documents (increase chunk size by 50% for documents over 100k words)
- Add warning to CLI when content is truncated
- Allow configuration of chunk limit via environment variable `MAX_CHUNKS_PER_DOCUMENT`
---
### 4. Community Detection Algorithm Unavailability
**Issue**: Leiden/Louvain community detection skipped in Memgraph Community edition
**Root Cause**: Algorithms only available in Memgraph Enterprise edition
**Proposed Fix**:
- Implement lightweight native community detection fallback for Community edition (graph-based connected components with naming)
- Add configuration toggle to disable community detection entirely for users who don't need it
- Update logging to clearly indicate which algorithm is being used
---
### 5. Embedding Batch Size Limitation
**Issue**: API endpoint caps batch size to 10, but system is configured for 50 causing repeated requests
**Root Cause**: Static batch size not adjusted for different embedding providers
**Proposed Fix**:
- Add provider-specific batch size configuration
- Auto-detect batch size limits from API responses
- Add exponential backoff for rate limited embedding requests
---
## Implementation Priority
1. **High**: Memgraph connection stability fixes
2. **High**: Missing embeddings validation & retry
3. **Medium**: Large document chunking limit adjustment
4. **Low**: Community detection fallback
5. **Low**: Batch size auto-configuration
---
## Success Metrics
- 100% of code nodes have embeddings after indexing
- 0 database connection errors during indexing
- 0 content truncation for documents under 1 million words
- Indexing speed improved by 20% via optimized batch handling
