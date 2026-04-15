# Unified Ingestion Data Quality Fix Design Specification
Version: 1.0 | Status: Implementation Ready | Last Updated: 2024-05-20

## Executive Summary
This unified specification addresses all identified ingestion and data modeling issues in the Code Graph RAG system, with prioritized, implementation-ready fixes. The specification covers:
- Critical functionality restoration
- Data quality validation framework
- Ingestion component improvements
- Full validation test suite

---

## 1. Identified Issues & Root Causes (Prioritized)
| Severity | Issue Description | Root Cause | Business Impact |
|----------|-------------------|------------|-----------------|
| **CRITICAL** | Document semantic search fails with SSL EOF error when calling OpenAI embedding API | SSL certificate/proxy configuration not supported in OpenAI client initialization, no fallback for API failures | 100% of document search functionality broken, hybrid code+document queries fail |
| **CRITICAL** | Structural graph queries fail with syntax/permission errors | Query generator assumes enterprise Memgraph license with parallel execution support, uses unsupported window function syntax | All data quality validation queries fail, cannot audit graph integrity |
| **MEDIUM** | No post-ingestion data quality validation | Built-in HealthChecker only validates external dependencies, not ingested graph data | Ingestion failures (disconnected nodes, missing properties, invalid relationships) go undetected |
| **MEDIUM** | No pre-validation of JSON ingestion files | Ingestion pipeline accepts non-conformant JSON files that violate the official ingestion schema | Invalid/corrupted data is written to the graph |
| **MEDIUM** | Document parsing generates chunks with no source attribution | Splitter logic does not preserve source location metadata | Cannot trace semantic search results back to original document location |
| **LOW** | Tree-sitter parsing may generate disconnected production code nodes | Reference resolution logic fails for some cross-file imports/calls | Incomplete call graphs, missing relationships |
| **LOW** | No ingestion audit reporting | No summary of ingestion results or quality scores | Cannot track ingestion success rate or debug failures |

---

## 2. Unified Design Requirements
### 2.1 Core Functional Requirements
1. **Embedding Generation**: Support custom SSL certificates, proxies, and local embedding fallback for OpenAI API failures
2. **Query Compatibility**: Auto-detect Memgraph license/version and generate compatible query syntax
3. **Data Validation**: Automatically validate 100% of ingested data against schema and quality rules
4. **Ingestion Integrity**: Guarantee no invalid data is written to the graph
5. **Auditability**: Provide complete ingestion reports with quality scores and error details

### 2.2 Non-Functional Requirements
1. Backwards compatible with existing ingestion workflows
2. Less than 5% performance overhead added to ingestion jobs
3. All fixes are configurable via environment variables
4. All validation errors include clear, actionable fix recommendations

---

## 3. Implementation Details (Exact Code Changes)
### 3.1 Critical Fix: Embedding Generation Improvements
**File path**: `codebase_rag/embeddings/openai.py` (matches actual codebase structure)
**Root Cause Confirmation**: The existing `_get_client()` method creates `httpx.Client` with no SSL verification or proxy configuration, causing the SSL EOF error. Local embedding provider already exists in the codebase (at `codebase_rag/embeddings/local.py`) so we reuse existing components for fallback instead of adding new dependencies.
**Changes**:
```python
from ..config import settings  # Add this import at the top of the file

def _get_client(self) -> httpx.Client:
    """Get or create HTTP client with SSL/proxy support (FIX)."""
    if self._client is None:
        import httpx
        # New configuration parameters added to settings (all backwards compatible, default to existing behavior):
        # EMBEDDING_OPENAI_SSL_VERIFY: bool | str = True (path to custom CA cert if string)
        # EMBEDDING_OPENAI_PROXY: str | None = None
        # EMBEDDING_USE_LOCAL_FALLBACK: bool = True
        self._client = httpx.Client(
            timeout=60.0,
            verify=settings.EMBEDDING_OPENAI_SSL_VERIFY,
            proxy=settings.EMBEDDING_OPENAI_PROXY,
        )
    return self._client

# Update the _make_request method to add fallback logic (replace existing exception handling):
def _make_request(self, texts: list[str], batch_size: int) -> list[list[float]]:
    """Make embedding request to OpenAI API with local fallback (FIX)."""
    client = self._get_client()

    headers = {
        "Authorization": f"Bearer {self._api_key}",
        "Content-Type": "application/json",
    }

    # Determine effective batch size based on endpoint
    endpoint_limit = _get_batch_limit_for_endpoint(self._endpoint)
    effective_batch_size = min(batch_size, endpoint_limit)
    if batch_size > endpoint_limit:
        logger.debug(
            f"Batch size capped from {batch_size} to {effective_batch_size} for endpoint {self._endpoint}"
        )

    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), effective_batch_size):
        batch = texts[start : start + effective_batch_size]

        payload = {
            "model": self.model_id,
            "input": batch,
        }

        # encoding_format is only supported by OpenAI's native API
        # Many OpenAI-compatible APIs (DashScope, Azure, etc.) reject this parameter
        if self._endpoint.startswith("https://api.openai.com"):
            payload["encoding_format"] = "float"

        try:
            response = client.post(
                self._endpoint,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings in order
            embeddings_data = data.get("data", [])
            # Sort by index to maintain order
            embeddings_data.sort(key=lambda x: x.get("index", 0))

            for item in embeddings_data:
                embedding = item.get("embedding", [])
                all_embeddings.append(embedding)

        except Exception as e:
            # Fallback to local embedding provider if enabled (FIX)
            if settings.EMBEDDING_USE_LOCAL_FALLBACK:
                logger.warning(f"OpenAI embedding failed, falling back to local provider: {e}")
                from .local import LocalEmbeddingProvider
                local_provider = LocalEmbeddingProvider()
                return local_provider.embed_batch(texts, batch_size)
            raise EmbeddingGenerationError(
                f"OpenAI embedding request failed: {e}",
                provider="openai",
                model=self.model_id,
            ) from e

    return all_embeddings
```

### 3.2 Critical Fix: Memgraph Query Compatibility
**File path**: `codebase_rag/graph/query_generator.py`
**Changes**:
```python
import mgclient
from codebase_rag.config import settings

class QueryGenerator:
    def __init__(self):
        self.memgraph_version = None
        self.has_enterprise_license = False
        self._detect_memgraph_capabilities()

    def _detect_memgraph_capabilities(self):
        """Auto-detect Memgraph version and license to generate compatible queries"""
        try:
            conn = mgclient.connect(host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT)
            cursor = conn.cursor()
            
            # Get version
            cursor.execute("SHOW VERSION;")
            version = cursor.fetchone()[0]
            self.memgraph_version = tuple(map(int, version.split('.'))) if version else (0,0,0)
            
            # Check for enterprise license
            cursor.execute("SHOW LICENSE;")
            license = cursor.fetchone()[0]
            self.has_enterprise_license = "enterprise" in license.lower()
            
            cursor.close()
            conn.close()
        except Exception as e:
            # Fallback to minimum supported capabilities
            self.memgraph_version = (2,0,0)
            self.has_enterprise_license = False

    def get_disconnected_nodes_query(self) -> str:
        """Return compatible query for disconnected nodes based on detected capabilities"""
        if self.has_enterprise_license and self.memgraph_version >= (2,5,0):
            # Use optimized parallel query for enterprise license
            return """
                USING PARALLEL EXECUTION
                MATCH (n)
                OPTIONAL MATCH (n)-[out]->()
                WITH n, count(out) as outgoing
                OPTIONAL MATCH ()-[in]->(n)
                WITH n, outgoing, count(in) as incoming
                WHERE outgoing = 0 AND incoming = 0
                RETURN count(*) as total_disconnected, collect(n) as nodes
            """
        else:
            # Compatible query for standard license/older versions
            return """
                MATCH (n)
                OPTIONAL MATCH (n)-[out]->()
                WITH n, count(out) as outgoing
                OPTIONAL MATCH ()-[in]->(n)
                WITH n, outgoing, count(in) as incoming
                WHERE outgoing = 0 AND incoming = 0
                RETURN count(*) as total_disconnected, collect(n) as nodes
            """
```

### 3.3 Medium Fix: Data Quality Validation Framework
**File path**: `codebase_rag/tools/health_checker.py`
**Add new methods to HealthChecker class**:
```python
def check_disconnected_nodes(self) -> HealthCheckResult:
    """Check for abnormal disconnected production nodes (excludes test code)"""
    query = query_generator.get_disconnected_nodes_query()
    # Execute query, filter out test-related nodes
    # Return HealthCheckResult with list of abnormal nodes

def check_required_properties(self) -> HealthCheckResult:
    """Validate all entities have required properties"""
    required_properties = {
        "Code": ["qualified_name", "path", "start_line", "end_line"],
        "Document": ["source_file", "title"],
        "Chunk": ["content", "embedding_id", "start_offset", "end_offset"]
    }
    # Run validation queries for each entity type
    # Return HealthCheckResult with list of entities missing properties

def check_embedding_correlation(self) -> HealthCheckResult:
    """Validate embeddings correspond correctly to chunk content"""
    # Run test semantic searches and confirm expected results are returned
    # Return HealthCheckResult with pass/fail status and correlation score

def check_json_ingestion_schema(self, json_path: str) -> HealthCheckResult:
    """Validate JSON ingestion file against official schema"""
    import jsonschema
    with open("ingestion_schema.json", "r") as f:
        schema = json.load(f)
    with open(json_path, "r") as f:
        data = json.load(f)
    try:
        jsonschema.validate(instance=data, schema=schema)
        return HealthCheckResult(passed=True, message="JSON schema validation passed")
    except jsonschema.ValidationError as e:
        return HealthCheckResult(passed=False, message=f"JSON schema validation failed: {e.message}")
```

### 3.4 Medium Fix: Document Parsing Improvements
**File path**: `codebase_rag/ingestion/parsers/document_parser.py`
**Changes**:
Add mandatory source metadata to all generated chunks:
```python
def split_document(self, file_path: str, content: str) -> list[Chunk]:
    # Existing splitting logic
    chunks = self.splitter.split_text(content)
    
    # Add source metadata to each chunk
    for idx, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = file_path
        chunk.metadata["start_offset"] = chunk.start_index
        chunk.metadata["end_offset"] = chunk.end_index
        if file_path.endswith(".pdf"):
            chunk.metadata["page_number"] = chunk.page_number
    
    return chunks
```

---

## 4. Validation Plan
### 4.1 Pre-Deployment Validation
1. Run all existing unit tests to confirm no regression
2. Test embedding generation with SSL certificate, proxy, and fallback scenarios
3. Test graph queries against both enterprise and standard Memgraph instances
4. Test JSON ingestion validation with valid/invalid schema files
5. Test document parsing to confirm source metadata is correctly added to chunks

### 4.2 Post-Deployment Validation
Run the following validation steps after every ingestion job:
1. Execute `HealthChecker.run_all_checks()` to confirm all health checks pass
2. Run semantic search for known content to confirm embeddings are working correctly
3. Review ingestion report to confirm quality score is >95%
4. Spot check 5 random graph nodes to confirm required properties are present

---

## 5. Rollout Strategy
### Phase 1 (Immediate, 4 hours effort)
1. Deploy embedding generation and query compatibility critical fixes
2. Validate core functionality is restored: document search works, structural queries run without errors

### Phase 2 (Next Sprint, 8 hours effort)
1. Deploy data quality validation framework
2. Deploy JSON schema validation and document parsing improvements
3. Run full ingestion of sample codebase and confirm all quality checks pass

### Phase 3 (Future Sprint, 3 hours effort)
1. Deploy tree-sitter parsing improvements for edge case import resolution
2. Deploy ingestion audit reporting functionality

---

## 6. Success Metrics
| Metric | Target |
|--------|--------|
| Embedding generation success rate | 100% (with fallback) |
| Graph query success rate | 100% |
| Ingestion quality score | >=95% |
| Invalid JSON ingestion rejection rate | 100% |
| Document chunk source attribution completeness | 100% |
