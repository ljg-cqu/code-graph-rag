"""Log message constants for document graph operations."""

# Concept extraction
DOC_CONCEPT_EXTRACT_START = "Extracting concepts from {chunk_count} chunks"
DOC_CONCEPT_EXTRACT_DONE = "Extracted {concept_count} concepts from document"
DOC_CONCEPT_STORE_BATCH = "Storing {count} concept nodes via batch MERGE"
DOC_MENTIONS_STORE_BATCH = "Creating {count} MENTIONS relationships"
DOC_REL_STORE_BATCH = "Creating {count} concept-to-concept relationships"
DOC_CONCEPT_CIRCUIT_BREAKER_OPEN = "Circuit breaker OPEN for concept extraction, skipping {chunk_qn}"
DOC_CONCEPT_RETRY_ATTEMPT = "Concept extraction retry {attempt}/{max_retries} for {chunk_qn} after {delay:.1f}s ({error_type})"
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
