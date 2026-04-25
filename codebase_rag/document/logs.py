"""Log message constants for document graph operations."""

# Concept extraction
DOC_CONCEPT_EXTRACT_START = "Extracting concepts from {chunk_count} chunks"
DOC_CONCEPT_EXTRACT_DONE = "Extracted {concept_count} concepts from document"
DOC_CONCEPT_STORE_BATCH = "Storing {count} concept nodes via batch MERGE"
DOC_MENTIONS_STORE_BATCH = "Creating {count} MENTIONS relationships"
DOC_REL_STORE_BATCH = "Creating {count} concept-to-concept relationships"
DOC_CONCEPT_CIRCUIT_BREAKER_OPEN = "Circuit breaker OPEN for concept extraction, skipping {chunk_qn}"
DOC_CONCEPT_PROVIDER_UNHEALTHY = "Provider unhealthy after timeout, skipping retry for {chunk_qn}"
DOC_CONCEPT_BREAKER_SKIP_DOC = "Concept extraction circuit breaker is OPEN ({remaining:.0f}s remaining), skipping concept extraction for document {doc}"
DOC_CONCEPT_RETRY_ATTEMPT = "Concept extraction retry {attempt}/{max_retries} for {chunk_qn} after {delay:.1f}s ({error_type}, timeout={timeout:.1f}s)"
DOC_CONCEPT_RETRY_EXHAUSTED = "Concept extraction exhausted retries for {chunk_qn}, queued to DLQ"

# Graph algorithms
DOC_SHORTEST_PATH_QUERY = "Finding shortest path: {source} -> {target}"
DOC_SHORTEST_PATH_FOUND = "Found path of length {length} between {source} and {target}"
DOC_SHORTEST_PATH_NONE = "No path found between {source} and {target}"
DOC_RELATED_CONCEPTS_QUERY = "Finding concepts related to: {concept}"
DOC_NEIGHBORS_QUERY = "Finding neighbors of: {concept} at depth {depth}"
DOC_GRAPH_ALGO_ERROR = "Document graph algorithm failed: {error}"

# Real-time updates
DOC_CONCEPT_CLEANUP_START = "Cleaning up orphaned concepts for document: {doc_path}"
DOC_CONCEPT_CLEANUP_DONE = "Removed {count} orphaned concepts"

# Document updater resilience
DOC_GRAPH_CONNECT_FAILED = "Document graph connection failed: {error}"
DOC_GRAPH_UNAVAILABLE = "Document graph unavailable: {error}"
DOC_SETUP_OP_FAILED = "Setup operation '{op}' failed: {error}"
DOC_STALE_CLEANUP_FAILED = "Stale document cleanup failed: {error}"
DOC_EXCLUDED_CLEANUP_FAILED = "Excluded document cleanup failed: {error}"
EMBEDDING_PROVIDER_CLOSE_FAILED = "Could not close embedding provider cleanly: {error}"
DOC_INCREMENTAL_FLUSH_PARTIAL = (
    "Incremental flush partial failure at doc {index}: {stats}"
)
DOC_INCREMENTAL_FLUSH_OK = "Incremental flush success at doc {index}: {stats}"
DOC_INCREMENTAL_FLUSH_FAILED = "Incremental flush failed at doc {index}: {error}"
DOC_FINAL_FLUSH_PARTIAL = "Final flush had partial failures: {stats}"
DOC_FINAL_FLUSH_OK = "Final flush complete: {stats}"
DOC_FINAL_FLUSH_FAILED = "Final flush failed: {error}"

# Missing document handling
DOC_FILE_MISSING_SKIP = "Skipping missing document {path}: {error}"
DOC_EXTRACTION_FAILED = "Failed to process {path}: {error}"
DOC_DLQ_ENQUEUE_FAILED = "Could not enqueue error for {path}: {error}"
DOC_PREFLIGHT_MISSING = "Document disappeared between scan and extraction: {path}"
DOC_PRE_VERIFICATION_FILTERED = "Filtered {count} documents that no longer exist (likely temporary/generated files)"
DOC_DLQ_CLEANUP = "Cleaned up {count} stale error files from {path}"
DOC_DLQ_SIZE_WARNING = "DLQ has accumulated {count} errors — consider investigating root cause"
DOC_CACHE_PRUNE = "Pruned stale version cache entry: {path}"
