# Code Graph RAG Indexing Optimization Specification

## Purpose

This revision narrows the scope to indexing changes that are both needed in the current codebase and safe to implement without changing the data model or deployment topology.

The original draft mixed together three categories of work:

- defects that are still open in the indexing path
- behaviors that are already implemented elsewhere in the repository
- speculative changes that do not map cleanly to current components

This spec keeps only the first category as implementation work, documents the second as already satisfied, and rejects the third for now.

## Codebase Findings

### Already implemented and not part of this change

1. Embedding reconciliation already exists in `GraphUpdater._reconcile_embeddings()` and logs missing stored IDs after batch persistence.
2. OpenAI-compatible embedding batch caps are already handled in `codebase_rag/embeddings/openai.py` via endpoint-specific limits, including DashScope's batch size of 10.
3. Dimension-mismatched local fallback embeddings already fail fast with a clear error instead of silently storing incompatible vectors.
4. Community detection already falls back from Leiden to Louvain and then exits cleanly when the procedures are unavailable.

### Problems that remain open

1. Memgraph indexing operations use long-lived connections without automatic reconnect on transient session failures such as `broken pipe` or `bad session`.
2. Post-ingestion graph algorithm execution in `GraphUpdater.run()` does not fully honor existing configuration flags:
	- `ALGORITHM_RUN_POST_INGESTION`
	- `ALGORITHM_ENABLE_PAGERANK`
	- `ALGORITHM_ENABLE_COMMUNITY_DETECTION`
	- `ALGORITHM_COMMUNITY_ALGORITHM`
3. Community detection gating is incorrectly tied to the PageRank update count instead of actual graph size, so disabling PageRank implicitly disables community detection.

### Rejected from this revision

1. Raising a document `MAX_CHUNKS_PER_DOCUMENT` limit is not applicable because the current document updater does not enforce that static cap.
2. Adding a new native community-detection algorithm for Memgraph Community edition is not justified in this patch set because the repository already treats missing procedures as a supported degraded mode.
3. Increasing Docker memory defaults is an operational change, not an indexing implementation change, and should be handled separately if needed.

## Accepted Design

### 1. Transient Memgraph retry and reconnect

Add bounded retry support to `MemgraphIngestor` for the persistent connection path used by indexing writes and reads.

Requirements:

1. Retry only transient session/transport failures.
2. On retry, close the stale connection, create a fresh connection, and rerun the query.
3. Keep retries bounded and configurable.
4. Preserve existing behavior for non-transient errors.
5. Apply the retry behavior to:
	- `_execute_query()`
	- `_execute_batch_on()` when using the shared connection
	- `_execute_batch_with_return_on()` when using the shared connection

Configuration:

1. `MEMGRAPH_QUERY_MAX_RETRIES`: default `2`
2. `MEMGRAPH_RETRY_BASE_DELAY`: default `0.25`

Non-goals:

1. No connection pooling changes
2. No retry loop for short-lived worker-owned flush connections
3. No retry for deterministic Cypher/model errors

### 2. Config-driven post-ingestion algorithms

Refactor post-ingestion graph algorithm execution into a dedicated helper so behavior is explicit and testable.

Requirements:

1. Always run `ANALYZE GRAPH` after ingestion.
2. Treat `ALGORITHM_RUN_POST_INGESTION` as the master switch for optional enrichment steps after `ANALYZE GRAPH`.
3. Run PageRank only when both `ALGORITHM_RUN_POST_INGESTION` and `ALGORITHM_ENABLE_PAGERANK` are true.
4. Run community detection only when both `ALGORITHM_RUN_POST_INGESTION` and `ALGORITHM_ENABLE_COMMUNITY_DETECTION` are true.
5. Use `ALGORITHM_COMMUNITY_ALGORITHM` to choose Leiden vs Louvain.
6. Gate community detection on total node count, not on PageRank output.
7. If the configured algorithm value is invalid, fall back to Leiden and log a warning.

### 3. Validation strategy

Tests added in this revision must prove:

1. A transient Memgraph query failure reconnects and succeeds on retry.
2. Non-transient failures are not retried.
3. PageRank can be disabled without suppressing `ANALYZE GRAPH`.
4. Community detection respects both the enable flag and the configured algorithm choice.

## Implementation Plan

1. Extend config with bounded Memgraph retry settings.
2. Add retry/reconnect helpers to `MemgraphIngestor` and wire them into query execution.
3. Extract post-ingestion algorithm handling from `GraphUpdater.run()` into a helper method that preserves the existing master config gate.
4. Add focused unit tests for retry behavior and algorithm configuration.

## Success Criteria

1. Indexing survives transient Memgraph session failures without manual rerun.
2. Post-ingestion algorithm execution matches the configuration surface already exposed by `settings`, including the existing master post-ingestion toggle.
3. The revised spec maps directly to concrete code paths and tests in the repository.
