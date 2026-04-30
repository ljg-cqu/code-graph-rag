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
DOC_CONCEPT_PROACTIVE_SPLIT = "Proactive split for {chunk_qn}: {tokens} tokens > {limit} limit"
DOC_CONCEPT_RECURSIVE_SPLIT = "Recursive split depth {depth} for {chunk_qn}"
DOC_CONCEPT_SPLIT_ATTEMPT = "Attempting split extraction for {chunk_qn}"
DOC_CONCEPT_SPLIT_SUCCESS = "Split extraction succeeded for {chunk_qn}"
DOC_CONCEPT_SPLIT_FAILED = "Split extraction failed for {chunk_qn}: {error}"
DOC_CONCEPT_ADAPTIVE_RETRY = "Retrying {chunk_qn} with increased max_tokens: {max_tokens}"
DOC_CONCEPT_OUTPUT_ADAPTIVE_RETRY = (
    "Retrying {chunk_qn} with increased max_tokens after output limit: {max_tokens}"
)
DOC_CONCEPT_RETRY_SUCCESS = "Concept extraction succeeded on attempt {attempt} for {chunk_qn}"
DOC_CONCEPT_QUOTA_EXCEEDED = "LLM quota exceeded. Reset at: {retry_after}. Stopping all retries to preserve remaining quota."
DOC_CONCEPT_NON_RECOVERABLE = "Non-recoverable error: {error_type}"

# Chunk merging
DOC_CHUNK_MERGED = "Merged {count} tiny chunks (< {min_tokens} tokens) during chunking for {doc_path}"

# Graph algorithms
DOC_SHORTEST_PATH_QUERY = "Finding shortest path: {source} -> {target}"
DOC_SHORTEST_PATH_FOUND = "Found path of length {length} between {source} and {target}"
DOC_SHORTEST_PATH_NONE = "No path found between {source} and {target}"
DOC_RELATED_CONCEPTS_QUERY = "Finding concepts related to: {concept}"
DOC_NEIGHBORS_QUERY = "Finding neighbors of: {concept} at depth {depth}"
DOC_GRAPH_ALGO_ERROR = "Document graph algorithm failed: {error}"

# Real-time updates
DOC_ORPHANED_CONCEPTS_CLEANUP_START = "Cleaning up orphaned concepts for document: {doc_path}"
DOC_ORPHANED_CONCEPTS_CLEANUP_DONE = "Removed {count} orphaned concepts"

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
DOC_DLQ_NO_OVERFLOW_ERRORS = "No context-overflow or output-limit errors to retry"
DOC_DLQ_RETRY_START = "Retrying {count} overflow chunks from DLQ"
DOC_DLQ_NO_CONTENT = "No content available for DLQ retry: {chunk_qn}"
DOC_DLQ_UNAVAILABLE = "No dead letter queue available for retry"
DOC_DLQ_ADAPTIVE_FAILED = "Adaptive retry failed for {chunk_qn}: {reason}. Falling back to split retry."
DOC_DLQ_ADAPTIVE_SUCCESS = "DLQ adaptive retry succeeded for {chunk_qn} with {tokens_used} tokens"
DOC_DLQ_ADAPTIVE_MISSING_CONTENT = "No chunk content in DLQ entry"
DOC_DLQ_RETRY_MISSING = "DLQ retry skipped — file no longer exists: {path}"
DOC_CACHE_PRUNE = "Pruned stale version cache entry: {path}"

# Concept runner — dual-graph connection
CONCEPT_GRAPH_DISABLED = "Concept graph disabled, concept extraction unavailable"
CONCEPT_GRAPH_CONNECT_FAILED = "Concept graph connection failed: {error}"
CONCEPT_GRAPH_CLOSE_FAILED = "Error closing concept ingestor: {error}"
DOC_GRAPH_CLOSE_FAILED = "Error closing document ingestor: {error}"
DUAL_GRAPH_CONNECT_FAILED = "Failed to connect to required graphs"

# Concept runner — initialization and control flow
DOC_CONCEPT_EXTRACTION_DISABLED = "Concept extraction disabled via DOC_CONCEPT_EXTRACTION_ENABLED"
DOC_CONCEPT_INIT_FAILED = "Failed to initialize concept extractor: {error}"
DOC_CONCEPT_NO_CHUNKS = "No chunks found needing concept extraction"
DOC_CONCEPT_ALL_EXISTING = "{count} chunks already have concepts. Use --force to re-extract."
DOC_CONCEPT_LIMIT_APPLIED = "Processing limited to {limit} chunks"
DOC_CONCEPT_DRY_RUN = "DRY RUN: Would process {count} chunks"
DOC_CONCEPT_INDEX_STATUS = "Index status: {status}"
DOC_CONCEPT_CLEANUP_START = "Cleaning up existing concepts for {count} chunks before re-extraction"

# Concept runner — per-chunk extraction
DOC_CONCEPT_EXTRACT_FAILED = "Concept extraction failed for {chunk_qn}: {error}"

# Concept runner — index management
DOC_CONCEPT_INDEX_EXISTS = "Index on :{label}({prop}) already exists"
DOC_CONCEPT_INDEX_FAILED = "Failed to create index on :{label}({prop}): {error}"

# Concept extraction — verb override suppression
DOC_CONCEPT_VERB_OVERRIDE_SUPPRESSED = (
    "Verb '{verb}' registry override (suppressed): LLM declared '{declared_category}', registry says '{registry_category}'"
)
