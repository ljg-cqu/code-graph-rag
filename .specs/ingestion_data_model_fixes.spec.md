# Ingestion Data Model & Quality Fix Design Specification

## Identified Issues & Root Causes

### 1. Critical: Document Embedding Generation Failure
- **Status**: Confirmed
- **Symptom**: Document graph queries fail with error: `OpenAI embedding request failed: [SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1000)`
- **Root Cause**: 
  1. Network/SSL proxy/firewall configuration is blocking secure connection to OpenAI embedding API endpoints
  2. Embeddings are not being generated/stored for document nodes during ingestion
- **Impact**:
  - Semantic search over all document content is completely non-functional
  - Hybrid code+document queries fall back to broken basic search
  - All document nodes lack required embedding property, breaking correlation between vector data and graph entities
- **Fix Approach**:
  1. Add SSL certificate configuration option for OpenAI client to handle custom corporate/self-signed certificates
  2. Implement embedding generation retry logic with fallback to local embedding models (e.g. sentence-transformers) for cases where OpenAI API is unreachable
  3. Add ingestion validation step that confirms embeddings are successfully generated for all document chunks before storing to graph
  4. Add offline embedding generation support to avoid runtime API dependency

---

### 2. Graph Query Compatibility Issue
- **Status**: Confirmed
- **Symptom**: Structural graph queries fail with two errors:
  1. `Your license has an invalid type. To use PARALLEL EXECUTION you need to have an enterprise license.`
  2. `mismatched input 'OVER' expecting {<EOF>, ';'}` (window functions not supported)
- **Root Cause**:
  1. Automatic query generator assumes enterprise Memgraph license with parallel execution support, but current deployment uses standard license
  2. Query generator uses modern Memgraph window function syntax that is not supported in the deployed Memgraph version
- **Impact**:
  - Default structural graph queries fail, requiring manual query adjustment
  - Data quality validation workflows cannot run out of the box
- **Fix Approach**:
  1. Update query generation logic to detect Memgraph license/version automatically and generate compatible syntax
  2. Disable parallel execution by default for standard license deployments
  3. Replace window function usage with compatible count/collect logic for older Memgraph versions
  4. Add query compatibility test suite for all supported Memgraph versions/licenses

---

### 3. Missing Ingestion Quality Validation Capability
- **Status**: Confirmed
- **Symptom**: The built-in `HealthChecker` class only validates external dependencies (Docker, Memgraph connection, API keys, external tools) but does not validate ingested graph data quality
- **Root Cause**: No implemented data quality checks for post-ingestion validation
- **Impact**:
  - Disconnected nodes, missing properties, invalid relationships, and embedding mismatches cannot be detected automatically
  - Ingestion failures can go undetected
- **Fix Approach**:
  1. Extend `HealthChecker` class with data quality validation methods:
     - `check_disconnected_nodes()`: Identifies abnormal disconnected nodes (excludes expected test-related nodes)
     - `check_required_properties()`: Validates all entities have required properties (qualified name for code nodes, embedding for chunk nodes, path for file nodes, etc.)
     - `check_relationship_validity()`: Validates relationship cardinality and required properties (e.g. CALLS relationship has correct source/target types)
     - `check_embedding_correlation()`: Validates embeddings match entity content by running test semantic searches and confirming expected results
  2. Add post-ingestion quality check hook that runs automatically after every ingestion job
  3. Add quality score reporting that gives pass/fail status for ingestion jobs

---

### 4. Disconnected Nodes (Expected + Pending Validation)
- **Status**: Partially Confirmed
- **Symptom**: Initial query results return only test-related classes and methods as disconnected nodes
- **Root Cause**: Test code is intentionally not referenced by production code, so disconnected test nodes are expected
- **Pending Validation**: Confirm there are no abnormal disconnected nodes (production code nodes with no relationships)
- **Impact (if abnormal nodes exist)**: Broken call graphs, incorrect dependency tracking, incomplete semantic search results
- **Fix Approach (if abnormal nodes found)**:
  1. Fix tree-sitter parser logic to correctly capture references/calls between production code nodes
  2. Fix import resolution logic to correctly map cross-file references
  3. Add validation for disconnected production nodes during ingestion

---

### 5. Tree-Sitter & Document Parsing Validation (Pending)
- **Status**: Pending Check
- **Pending Validation Items**:
  1. Tree-sitter parsing correctly captures all function/class/method nodes with correct qualified names
  2. Call graph edges are correctly generated between caller/callee functions
  3. Document parsing correctly splits documents into sections/chunks with correct source attribution
  4. JSON file parsing correctly ingests JSON structure as graph nodes
- **Fix Approach (if issues found)**:
  1. Update tree-sitter grammar rules for relevant languages to fix parsing gaps
  2. Update document splitter logic to handle edge cases (large documents, non-standard formatting)
  3. Update JSON ingestion logic to correctly map nested JSON structures to graph nodes

---

## Implementation Priority
1. **High**: Document embedding failure fix (restores core document search functionality)
2. **High**: Graph query compatibility fix (restores structural query functionality)
3. **Medium**: Ingestion quality validation extension (prevents future undetected ingestion issues)
4. **Medium**: Disconnected node validation + tree-sitter/document parsing validation
