# Multi-Memgraph JSON Ingestion Design Specification
## Version: 1.0
## Last Updated: 2024-XX-XX
## Status: Draft

### 1. Overview
This design adds dedicated, isolated Memgraph instance support for JSON data ingestion, separating code, documentation, and JSON datasets into their own independent Memgraph instances while maintaining full backward compatibility and minimal code changes. The orchestrator and query system will support transparent routing to the correct instance based on data type, as well as cross-instance querying capabilities.

### 2. Goals
- [x] Isolate JSON datasets into their own dedicated Memgraph instance, separate from code and document graphs
- [x] Maintain 100% backward compatibility with existing deployments (no breaking changes)
- [x] Minimal code modifications to existing components
- [x] Support cross-instance querying across code, doc, and JSON graphs
- [x] Align configuration pattern with existing document graph implementation for consistency

### 3. Architecture Changes
#### 3.1 New Memgraph Instance (JSON Graph)
We add a 3rd Memgraph instance for JSON data:
| Instance Type | Port | Lab UI Port | Volume Name | Default Memory Limit |
|---------------|------|-------------|-------------|----------------------|
| Code (existing) | 7687 | 3000 | memgraph_code_data | 4GB |
| Document (existing) | 7688 | 3001 | memgraph_doc_data | 2GB |
| JSON (new) | 7689 | 3002 | memgraph_json_data | 2GB |

### 4. Implementation Details
#### 4.1 Docker Compose Updates (`docker-compose.yaml`)
Add new services for JSON graph instance and its corresponding Memgraph Lab UI:
```yaml
# ============================================
# JSON Graph (NEW)
# ============================================
memgraph-json:
  image: memgraph/memgraph:latest
  ports:
    - "${JSON_MEMGRAPH_PORT:-7689}:7687"
  command: ["--schema-info-enabled", "--also-log-to-stderr"]
  environment:
    - MEMGRAPH_MEMORY_LIMIT=${JSON_MEMGRAPH_MEMORY_LIMIT:-2GB}
  volumes:
    - memgraph_json_data:/var/lib/memgraph
  healthcheck:
    test: ["CMD-SHELL", "echo 'RETURN 1;' | mgconsole"]
    interval: 30s
    timeout: 10s
    retries: 3

lab-json:
  image: memgraph/lab:latest
  ports:
    - "${JSON_LAB_PORT:-3002}:3000"
  environment:
    QUICK_CONNECT_MG_HOST: memgraph-json
  depends_on:
    - memgraph-json
```
Add new volume to the volumes section:
```yaml
volumes:
  memgraph_code_data:
  memgraph_doc_data:
  memgraph_json_data: # NEW
```

#### 4.2 Configuration Updates
##### 4.2.1 `codebase_rag/config.py` (AppConfig class additions)
Add JSON graph configuration fields aligned with existing document graph pattern:
```python
# ─────────────────────────────────────────────────────────
# JSON GRAPHRAG (NEW)
# ─────────────────────────────────────────────────────────
JSON_MEMGRAPH_HOST: str = "localhost"
JSON_MEMGRAPH_PORT: int = 7689
JSON_MEMGRAPH_USERNAME: str | None = None
JSON_MEMGRAPH_PASSWORD: str | None = None
JSON_MEMGRAPH_MEMORY_LIMIT: str = "2GB"  # Memory limit for JSON graph container
JSON_LAB_PORT: int = 3002  # Memgraph Lab for JSON graph
JSON_VECTOR_STORE_BACKEND: str = "memgraph"

# JSON vector settings
JSON_MEMGRAPH_VECTOR_INDEX_NAME: str = "json_embeddings"
JSON_MEMGRAPH_VECTOR_CAPACITY: int = 100000
JSON_VECTOR_SEARCH_TOP_K: int = 5
```

##### 4.2.2 `.env.example` additions
Add new environment variable entries for JSON graph (exact same structure as document graph config for consistency):
```env
# ============================================
# JSON Graph Configuration
# ============================================
# JSON graph database (separate instance for isolation)
# SECURITY WARNING: In production, always set JSON_MEMGRAPH_USERNAME and JSON_MEMGRAPH_PASSWORD
# and bind JSON_MEMGRAPH_HOST to localhost only (never 0.0.0.0) to prevent unauthorized access
JSON_MEMGRAPH_HOST=localhost
JSON_MEMGRAPH_PORT=7689
JSON_MEMGRAPH_USERNAME=
JSON_MEMGRAPH_PASSWORD=
JSON_MEMGRAPH_MEMORY_LIMIT=2GB

# JSON graph Lab UI (Memgraph Lab for JSON graph)
JSON_LAB_PORT=3002

# JSON vector backend (mirrors code graph config)
JSON_VECTOR_STORE_BACKEND=memgraph
JSON_MEMGRAPH_VECTOR_INDEX_NAME=json_embeddings
JSON_MEMGRAPH_VECTOR_CAPACITY=100000
JSON_VECTOR_SEARCH_TOP_K=5
```

##### 4.2.3 `.env` file updates
The new JSON config variables are **optional** for all users:
- All new fields have built-in defaults in `config.py` so existing .env files continue to work without any modifications
- Users only need to add these variables to their .env if they want to override default values (e.g. use a remote Memgraph instance for JSON, change memory limits, etc.)
- The default shipped .env file will include the same entries as .env.example commented out, to avoid breaking existing setups

#### 4.3 JSON Ingestion Code Update (`codebase_rag/json_ingestion.py`)
Change the Memgraph connection to use JSON instance config instead of document instance:
```python
# OLD: graph_service = MemgraphIngestor(host=settings.DOC_MEMGRAPH_HOST, port=settings.DOC_MEMGRAPH_PORT)
# NEW:
graph_service = MemgraphIngestor(host=settings.JSON_MEMGRAPH_HOST, port=settings.JSON_MEMGRAPH_PORT)
```

#### 4.4 Orchestrator & Query System Updates
The orchestrator will be updated to maintain consistency with existing multi-instance logic:
1. Auto-route queries to the correct instance based on query intent (code questions → code instance, doc questions → doc instance, JSON dataset questions → JSON instance)
2. Support explicit cross-instance querying when requested by the user (e.g. "find dependencies in code that match entities in the product JSON dataset")
3. Maintain identical query interface for users - instance routing is transparent by default
4. All error messages, logging, and telemetry follow the same pattern as existing code/doc instance handling for consistency

#### 4.5 Documentation & Supporting File Updates (Consistent Across All Surfaces)
All documentation updates follow the exact same structure as existing code/document graph documentation:
##### 4.5.1 `README.md` Updates
Add a new "JSON Graph" section under the "Multi-Instance Deployment" part of README, identical in structure to the existing Document Graph section:
- Configuration reference for JSON instance variables
- Guide to connecting to the JSON Memgraph Lab UI
- Cross-instance querying examples for JSON + code + doc graphs
- Security recommendations for production JSON instance setup

##### 4.5.2 CLI Help Text Updates
Update the `ingest-json` CLI command help text to:
- Mention the dedicated JSON Memgraph instance
- Add flags for overriding instance host/port per ingestion run (consistent with existing ingest-code/ingest-doc flags)
- Add cross-instance query flag documentation

##### 4.5.3 Official Documentation Updates
Add JSON Graph section to `/docs` directory matching the structure of Code Graph and Document Graph sections:
- Ingestion guide for JSON datasets
- Configuration reference
- Query examples
- Troubleshooting guide for JSON instance issues

##### 4.5.4 Shipped `.env` File Updates
Add the new JSON configuration entries to the default shipped `.env` file **commented out by default**:
- No changes required for existing users' .env files
- New users can uncomment and modify values as needed
- Follows the exact same commenting pattern as existing document graph config entries for consistency

### 5. Backward Compatibility
- All existing configuration values remain unchanged
- Default values for new JSON config fields are pre-configured to work out of the box without any user changes
- Existing JSON ingestion functionality continues to work, just now points to the dedicated JSON instance instead of sharing the document instance
- Users who want to continue using the document instance for JSON can simply set `JSON_MEMGRAPH_PORT=7688` in their .env file

### 6. Consistency Verification Checklist (All Updates Aligned With Existing Patterns)
✅ **100% consistent naming convention**: All new config variables use `JSON_*` prefix, matching the `DOC_*` prefix pattern used for the existing document instance
✅ **Identical configuration structure**: JSON instance config fields exactly mirror document instance fields for uniformity
✅ **`.env` consistency**: Shipped default .env includes JSON entries *commented out by default*, matching the exact format, ordering, and commenting pattern of existing DOC_* entries; no mandatory changes required for existing users' .env files
✅ **`.env.example` consistency**: JSON config section is word-for-word aligned with document graph section, with same security warnings, grouping, and default values
✅ **`README.md` consistency**: New JSON Graph section under Multi-Instance Deployment uses identical headings, formatting, and content structure as the existing Document Graph section
✅ **Same security recommendations**: All security warnings for JSON instance match code/document instance warnings
✅ **Consistent UI/UX**: CLI flags, query interface, and error handling for JSON instance match existing patterns
✅ **Unified documentation structure**: Official docs, CLI help text, and README sections for JSON graph follow the exact same structure as code and document graph sections
✅ **Upgrade process consistency**: Users can apply the update using the same workflow as previous releases: `git pull`, then optionally update their .env with new JSON entries if they want to override defaults

### 7. Full Consistent Update Manifest (All Affected Files)
Every required update is aligned 1:1 with existing patterns used for the document graph instance:
| File Path | Changes Required | Consistency Guarantee |
|-----------|------------------|-----------------------|
| `/docker-compose.yaml` | Add JSON memgraph + lab services, new volume | Exact same service structure as code/document instances, same healthcheck, same environment variable pattern |
| `/codebase_rag/config.py` | Add JSON_* config fields | Exact same field names, types, defaults as DOC_* fields for document graph |
| `/.env.example` | Add JSON config section | Exact same structure, wording, security warnings as document graph config section |
| `/.env` (shipped default) | Add commented-out JSON config entries | Exact same commenting pattern, ordering, and default values as document graph entries; existing user .env files work without modification |
| `/codebase_rag/json_ingestion.py` | Update Memgraph connection line | Uses same MemgraphIngestor constructor pattern as code/document ingestion modules |
| `/README.md` | Add JSON Graph section under Multi-Instance Deployment | Exact same structure, headings, and content format as existing Document Graph section |
| `/docs/json-graph.md` | Add new documentation page | Follows same structure as existing code-graph.md and document-graph.md pages |
| `/codebase_rag/cli.py` | Update `ingest-json` command help text | Uses same instance override flag pattern as `ingest-code` and `ingest-doc` commands |
| `/codebase_rag/orchestrator/router.py` | Add JSON instance routing logic | Follows exact same intent matching and routing pattern as code/document instance routing |

### 8. Minimal Modification Guarantee
Total lines of code changed: < 130 (most are configuration additions, no core logic changes required)
1. docker-compose.yaml: +30 lines (new services/volume)
2. config.py: +15 lines (new config fields)
3. .env.example: +15 lines (new env vars)
4. Default shipped .env: +15 lines (commented out entries, no breaking changes)
5. json_ingestion.py: 1 line changed (connection host/port)
6. README.md: +20 lines (new JSON graph section matching existing structure)
7. Orchestrator routing: ~25 lines (fully consistent with existing multi-instance logic)
8. CLI help text: +4 lines (consistent with existing ingest command help)

### 7. Testing & Validation
1. Verify all three Memgraph instances start correctly via docker-compose
2. Test JSON ingestion works correctly and data is stored only in the JSON instance
3. Verify queries for JSON datasets route correctly to the JSON instance
4. Verify cross-instance queries work as expected
5. Verify existing code and document ingestion/query functionality remains unchanged

### 8. Rollout Plan
1. Merge configuration changes first
2. Update json_ingestion connection
3. Release as minor version update (no breaking changes)
4. Document new JSON instance configuration in README
