# Document GraphRAG Extension - Implementation-Ready Design Specification

**Version:** 2.5
**Status:** Implementation-Ready (Reviewed + Critical Fixes Applied + 10-Worker Parallel Review + Reference Project Alignment)
**Last Updated:** 2026-03-30
**Author:** Code-Graph-RAG Architecture Team

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Principles](#2-architecture-principles)
3. [System Architecture](#3-system-architecture)
4. [Module Structure](#4-module-structure)
5. [Database Design](#5-database-design)
6. [Query Modes](#6-query-modes)
7. [Document Ingestion Pipeline](#7-document-ingestion-pipeline)
8. [Validation System](#8-validation-system)
9. [Real-Time Updater](#9-real-time-updater)
10. [Configuration](#10-configuration)
11. [CLI & MCP Interface](#11-cli--mcp-interface)
12. [Implementation Phases](#12-implementation-phases)
13. [Testing Strategy](#13-testing-strategy)
14. [Migration Path](#14-migration-path)
15. [Appendix](#15-appendix)

---

## 1. Executive Summary

### 1.1 Purpose

This specification defines the architecture for extending Code-Graph-RAG to support **Document GraphRAG** while maintaining strict separation between code and document graphs to prevent data dilution.

### 1.2 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Same codebase, separate modules** | 40-50% code reuse (embedding, vector, MCP, config) without graph pollution; novel concepts (explicit modes, on-demand validation) require new implementation |
| **Parallel database instances** | Independent scaling, backup, schemas; no cross-contamination; optional namespace-prefix fallback for simpler deployments |
| **Uniform indexing** | All documents indexed the same way; no pre-classification |
| **On-demand validation** | Validate only when user requests; saves 60-80% LLM costs; cost estimation before execution |
| **Explicit query modes** | User specifies intent; no automatic guessing; not present in LightRAG or graphrag upstream |
| **Bidirectional validation** | Code→Doc AND Doc→Code; authority is contextual |

### 1.3 Requirements Summary

| ID | Requirement | Priority |
|----|-------------|----------|
| R1 | Support 5 explicit query modes | MUST |
| R2 | Maintain separate code/document graphs | MUST |
| R3 | No pre-classification of document relevance | MUST |
| R4 | On-demand validation (not pre-computed) | MUST |
| R5 | Unified real-time updater for both graphs | MUST |
| R6 | Bidirectional validation (code↔doc) | MUST |
| R7 | Clear source attribution in all responses | MUST |
| R8 | Support multiple document formats (.md, .pdf, .docx, .rst) | SHOULD |
| R9 | Incremental document updates | SHOULD |
| R10 | Document accuracy reporting | COULD |

---

## 2. Architecture Principles

### 2.1 Core Principles

1. **Isolation by Default, Integration on Demand**
   - Code and document graphs NEVER mix automatically
   - User explicitly chooses query mode
   - Every response shows which graph(s) were queried

2. **Uniform Indexing**
   - All documents indexed the same way
   - No pre-classification of relevance or authority
   - Minimal metadata only (path, type, counts, extracted references)

3. **On-Demand Validation**
   - No pre-computed validation at index time
   - Validation happens at query time when requested
   - Saves 60-80% on LLM costs

4. **Explicit Mode Selection**
   - User MUST specify query mode
   - No automatic guessing of intent
   - Clear error messages if mode is ambiguous

5. **Bidirectional Validation**
   - Code is NOT always source of truth
   - API specs, regulations, contracts may be authoritative
   - Support both Code→Doc and Doc→Code validation

### 2.2 Anti-Patterns (What We Avoid)

| Anti-Pattern | Why Avoid | Alternative |
|--------------|-----------|-------------|
| Pre-classifying document relevance | Error-prone, creates false confidence | Uniform indexing, query-time ranking |
| Pre-validating all documents | Wastes 60-80% of LLM costs | On-demand validation |
| Automatic query mode guessing | Often wrong, hides user intent | Explicit mode selection |
| Single graph for code+docs | Schema conflicts, query contamination | Parallel graph instances |
| Assuming code is always truth | API specs, regulations are often authoritative | Contextual authority |

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Code-Graph-RAG System                            │
│                    (Extended with Document GraphRAG)                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                        PRESENTATION LAYER                         │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │   │
│  │  │   CLI (cgr)  │  │  MCP Server  │  │  Python API  │           │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │   │
│  └─────────┼─────────────────┼─────────────────┼──────────────────┘   │
│            │                 │                 │                       │
│            └─────────────────┴────────┬────────┘                       │
│                                       │                                │
│  ┌────────────────────────────────────▼────────────────────────────┐   │
│  │                        QUERY ROUTER                              │   │
│  │  Routes queries based on EXPLICIT mode (user-specified):        │   │
│  │  - CODE_ONLY | DOCUMENT_ONLY | BOTH_MERGED                      │   │
│  │  - CODE_VS_DOC | DOC_VS_CODE                                    │   │
│  └────────────────────────────────────┬────────────────────────────┘   │
│                                       │                                │
│            ┌──────────────────────────┼──────────────────────────┐     │
│            │                          │                          │     │
│  ┌─────────▼──────────┐    ┌─────────▼──────────┐    ┌─────────▼──────────┐
│  │   CODE MODULE      │    │   DOCUMENT MODULE  │    │   SHARED MODULE    │
│  │   (Existing)       │    │   (NEW)            │    │   (Existing)       │   │
│  ├────────────────────┤    ├────────────────────┤    ├────────────────────┤
│  │ • parsers/         │    │ • extractors/      │    │ • embeddings/      │
│  │ • graph_updater    │    │ • document_updater │    │ • vector_backend   │
│  │ • tools/           │    │ • tools/           │    │ • mcp/             │
│  └─────────┬──────────┘    └─────────┬──────────┘    └─────────┬──────────┘
│            │                          │                          │
│  ┌─────────▼──────────────────────────▼──────────────────────────▼────┐
│  │                         DATA LAYER                                  │
│  │  ┌────────────────────┐         ┌────────────────────┐             │
│  │  │   CODE GRAPH       │         │  DOCUMENT GRAPH    │             │
│  │  │   Memgraph:7687    │         │  Memgraph:7688     │             │
│  │  │   + Vectors        │         │  + Vectors         │             │
│  │  └────────────────────┘         └────────────────────┘             │
│  └─────────────────────────────────────────────────────────────────────┘
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    REAL-TIME UPDATER (Unified)                    │   │
│  │  Watches entire codebase → Routes to appropriate updater         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Existing Patterns to Reuse

**From current codebase:**

1. **Embedding Provider System** (`codebase_rag/embeddings/`)
   - Registry pattern with `_EMBEDDING_PROVIDER_REGISTRY`
   - Factory function `get_embedding_provider()`
   - Abstract base class `EmbeddingProvider`
   - Providers: local, openai, google, ollama

2. **Vector Backend Protocol** (`codebase_rag/vector_backend.py`)
   - Protocol-based abstraction for Memgraph/Qdrant
   - Factory function `get_vector_backend()`
   - Shared backend via `get_shared_backend()`

3. **MCP Tools Registry** (`codebase_rag/mcp/tools.py`)
   - `MCPToolsRegistry` class with `ToolMetadata`
   - Schema-driven tool definitions
   - Handler functions with async support

4. **Configuration** (`codebase_rag/config.py`)
   - Pydantic Settings with environment variables
   - ModelConfig, EmbeddingConfig dataclasses
   - Field validators

---

## 4. Module Structure

### 4.1 Directory Layout (Incremental Extension)

```
codebase_rag/
├── __init__.py
├── cli.py                          # Extended with document commands
├── config.py                       # Extended with DOC_* config
├── main.py                         # Main application entry
├── realtime_updater.py             # Extended to route to both updaters
│
├── parsers/                        # EXISTING: Code parsers
│   ├── __init__.py
│   ├── definition_processor.py
│   ├── call_processor.py
│   ├── import_processor.py
│   └── handlers/                   # Language-specific handlers
│
├── embeddings/                     # EXISTING: Reused for documents
│   ├── __init__.py                 # Registry
│   ├── base.py                     # Abstract base class
│   ├── local.py                    # UniXcoder provider
│   ├── openai.py                   # OpenAI provider
│   ├── google.py                   # Google provider
│   ├── ollama.py                   # Ollama provider
│   └── switching.py                # Provider switching utilities
│
├── document/                       # NEW: Document GraphRAG
│   ├── __init__.py
│   ├── extractors/                 # Document format extractors
│   │   ├── __init__.py             # Registry pattern
│   │   ├── base.py                 # Abstract extractor (like EmbeddingProvider)
│   │   ├── markdown_extractor.py   # .md, .rst extraction
│   │   ├── pdf_extractor.py        # .pdf (metadata + text)
│   │   └── docx_extractor.py       # .docx
│   ├── document_updater.py         # Document graph ingestion
│   ├── section_indexer.py          # Extract sections, headers
│   ├── tools/                      # Document-specific tools
│   │   ├── __init__.py
│   │   ├── document_query.py       # Query document graph
│   │   ├── document_search.py      # Semantic search in documents
│   │   └── document_reader.py      # Read document content
│   └── utils/
│       ├── text_extraction.py
│       └── reference_extractor.py  # Extract code references
│
├── shared/                         # NEW: Shared infrastructure
│   ├── __init__.py
│   ├── query_router.py             # Query mode routing
│   ├── validation/                 # Bidirectional validation
│   │   ├── __init__.py
│   │   ├── validator.py            # Base validation engine
│   │   ├── code_vs_doc.py          # Code→Doc validation
│   │   └── doc_vs_code.py          # Doc→Code validation
│   └── utils/
│       ├── file_classifier.py      # Code vs document classification
│       └── source_attribution.py   # Source labeling utilities
│
├── mcp/                            # EXISTING: Extended with doc tools
│   ├── __init__.py
│   ├── server.py
│   ├── client.py
│   └── tools.py                    # Extended MCPToolsRegistry
│
├── services/                       # EXISTING: Shared services
│   ├── graph_service.py            # MemgraphIngestor (reusable)
│   └── llm.py                      # LLM services
│
└── tests/
    ├── document/                   # Document GraphRAG tests
    └── shared/                     # Shared infrastructure tests
```

### 4.2 Module Dependencies

```
document/ ──┬──► embeddings/         (reuse for document embeddings)
            ├──► vector_backend.py   (reuse for vector storage)
            ├──► services/           (reuse MemgraphIngestor)
            └──► shared/validation/  (cross-validation)

shared/ ──► (no internal dependencies, only external)

mcp/ ──┬──► document/tools/
       └──► shared/query_router.py
```

---

## 5. Database Design

### 5.1 Code Graph Schema (Existing - Unchanged)

```cypher
// Node Types
:Project {name: string}
:Package {qualified_name: string, name: string, path: string}
:Folder {path: string, name: string}
:File {path: string, name: string, extension: string}
:Module {qualified_name: string, name: string, path: string}
:Function {qualified_name: string, name: string, path: string, start_line: int, end_line: int}
:Class {qualified_name: string, name: string, path: string}
:Method {qualified_name: string, name: string, path: string}
:ExternalPackage {name: string, version_spec: string}

// Relationships
(:Project)-[:CONTAINS_PACKAGE]->(:Package)
(:Project)-[:CONTAINS_FOLDER]->(:Folder)
(:Package)-[:CONTAINS_FILE]->(:File)
(:Module)-[:DEFINES]->(:Function)
(:Module)-[:DEFINES]->(:Class)
(:Class)-[:DEFINES_METHOD]->(:Method)
(:Function)-[:CALLS]->(:Function)
(:Project)-[:DEPENDS_ON_EXTERNAL]->(:ExternalPackage)

// Indexes
CREATE CONSTRAINT ON (f:Function) ASSERT f.qualified_name IS UNIQUE;
CREATE CONSTRAINT ON (c:Class) ASSERT c.qualified_name IS UNIQUE;
CREATE INDEX ON :Function(name);
CREATE INDEX ON :File(path);
```

### 5.2 Document Graph Schema (NEW)

```cypher
// Node Types
:Document {
    path: string,                    // Unique identifier
    workspace: string,                // Workspace isolation (LightRAG pattern, default: "default")
    file_type: string,               // .md, .pdf, .docx, .rst
    section_count: int,              // Number of sections
    code_block_count: int,           // Number of code blocks
    code_references: list[string],   // Extracted function/class names
    word_count: int,                 // Total word count
    modified_date: string,           // ISO 8601 timestamp
    indexed_at: string,              // ISO 8601 timestamp
    content_hash: string,            // SHA-256 for incremental updates (§7.9)
    section_hashes: list[string],    // Per-section hashes for diff
    extractor_version: string        // Algorithm version for re-index trigger
}

// Composite index for workspace isolation
CREATE INDEX ON :Document(workspace, path);

:Section {
    qualified_name: string,          // Unique: document_path#section_id
    title: string,                   // Section title
    level: int,                      // Header level (1-6)
    start_line: int,                 // Line number in document
    end_line: int,                   // Line number in document
    content_snippet: string          // First 500 characters
}

:Chunk {
    qualified_name: string,          // Unique: document_path#chunk_index
    content: string,                 // Chunk text content
    token_count: int,                // Number of tokens
    section_title: string,           // Parent section title
    start_line: int,                 // Line number in document
    end_line: int                    // Line number in document
}

// Relationships
(:Document)-[:CONTAINS_SECTION]->(:Section)
(:Section)-[:HAS_SUBSECTION]->(:Section)
(:Document)-[:CONTAINS_CHUNK]->(:Chunk)     // For chunk-level embeddings
(:Section)-[:CONTAINS_CHUNK]->(:Chunk)      // Optional: section-to-chunk mapping
(:Document)-[:REFERENCES_CODE]->(:Function)  // Extracted references (not validated)
(:Section)-[:REFERENCES_CODE]->(:Function)

// Indexes
CREATE CONSTRAINT ON (d:Document) ASSERT d.path IS UNIQUE;
CREATE CONSTRAINT ON (s:Section) ASSERT s.qualified_name IS UNIQUE;
CREATE INDEX ON :Document(path);
CREATE INDEX ON :Document(file_type);
CREATE INDEX ON :Document(modified_date);  // For incremental update performance
CREATE INDEX ON :Section(title);
CREATE INDEX ON :Section(level);           // For hierarchical queries
// Note: code_references is a list, use array containment in queries instead of index
// For fast code reference lookups, denormalize into separate CodeReference nodes if needed

// Vector Index (Memgraph native syntax - matches vector_store_memgraph.py pattern)
// Note: capacity is REQUIRED in Memgraph; adjust based on expected document count
CREATE VECTOR INDEX document_embedding_index
ON :Document(embedding)
WITH CONFIG {
    "dimension": 768,
    "capacity": 100000,
    "metric": "cosine"
};

CREATE VECTOR INDEX section_embedding_index
ON :Section(embedding)
WITH CONFIG {
    "dimension": 768,
    "capacity": 50000,
    "metric": "cosine"
};

CREATE VECTOR INDEX chunk_embedding_index
ON :Chunk(embedding)
WITH CONFIG {
    "dimension": 768,
    "capacity": 500000,
    "metric": "cosine"
};

CREATE CONSTRAINT ON (c:Chunk) ASSERT c.qualified_name IS UNIQUE;
CREATE INDEX ON :Chunk(section_title);
```

### 5.3 Physical Deployment

```yaml
# docker-compose.yaml (extended)

services:
  # Code Graph (existing)
  memgraph-code:
    image: memgraph/memgraph:latest
    ports:
      - "7687:7687"
    environment:
      - MEMGRAPH_MEMORY_LIMIT=4GB
    volumes:
      - memgraph_code_data:/var/lib/memgraph
      - ./memgraph/code/init.cypherl:/docker-entrypoint-initdb.d/init.cypherl

  # Document Graph (new)
  memgraph-doc:
    image: memgraph/memgraph:latest
    ports:
      - "7688:7687"    # Host 7688 → Container 7687
    environment:
      - MEMGRAPH_MEMORY_LIMIT=2GB
    volumes:
      - memgraph_doc_data:/var/lib/memgraph
      - ./memgraph/doc/init.cypherl:/docker-entrypoint-initdb.d/init.cypherl

  # Vector Stores (optional, if using Qdrant instead of Memgraph native)
  qdrant-code:
    ports:
      - "6333:6333"

  qdrant-doc:
    ports:
      - "6335:6333"    # Host 6335 → Container 6333

volumes:
  memgraph_code_data:
  memgraph_doc_data:
```

### 5.4 Namespace-Prefix Fallback (Optional)

For simpler deployments, a single Memgraph instance with namespace prefixes can be used instead of parallel instances.

**Trade-off Analysis:**

| Aspect | Parallel Instances | Namespace Prefix |
|--------|-------------------|------------------|
| **Isolation strength** | Strong (hard boundary) | Moderate (logical separation) |
| **Operational overhead** | 2x containers | 1x container |
| **Cross-graph queries** | Application layer | Native Cypher possible |
| **Backup/restore** | Independent per graph | Combined backup |
| **Schema conflicts** | Impossible | Possible if namespaces collide |

```python
# codebase_rag/document/namespace_config.py

from dataclasses import dataclass
from enum import Enum

class IsolationMode(Enum):
    """Graph isolation strategy."""
    PARALLEL_INSTANCES = "parallel"  # Two Memgraph containers
    NAMESPACE_PREFIX = "namespace"   # Single container with prefixes


@dataclass
class NamespaceConfig:
    """Configuration for namespace-prefix isolation."""

    CODE_NAMESPACE = "code"
    DOC_NAMESPACE = "doc"

    # Node labels become: code:Function, doc:Document
    # Relationships become: code:CALLS, doc:CONTAINS_SECTION

    # Reserved namespace prefix to avoid collision with default labels
    NAMESPACE_PREFIX = "ns_"

    def _sanitize_namespace(self, namespace: str) -> str:
        """
        Sanitize namespace for safe use in Cypher queries.

        - Escapes backticks by doubling them (Cypher injection prevention)
        - Restricts to [a-zA-Z0-9_] characters for label safety
        Pattern from LightRAG memgraph_impl.py:61-73
        """
        if not namespace or not namespace.strip():
            return "base"
        # Escape backticks to prevent Cypher injection
        return namespace.replace("`", "``")

    def _validate_namespace(self, namespace: str) -> str:
        """Validate namespace contains only safe characters."""
        import re
        if not re.match(r'^[a-zA-Z0-9_]+$', namespace):
            raise ValueError(
                f"Invalid namespace '{namespace}': must contain only [a-zA-Z0-9_]"
            )
        return namespace

    def prefix_node(self, namespace: str, label: str) -> str:
        """Generate prefixed node label with backtick escaping."""
        safe_ns = self._sanitize_namespace(namespace)
        safe_label = self._sanitize_namespace(label)
        return f"`{safe_ns}`:`{safe_label}`"

    def prefix_relationship(self, namespace: str, rel_type: str) -> str:
        """Generate prefixed relationship type with backtick escaping."""
        safe_ns = self._sanitize_namespace(namespace)
        safe_rel = self._sanitize_namespace(rel_type)
        return f"`{safe_ns}`:`{safe_rel}`"

    def build_code_node_query(self, label: str, properties: dict) -> str:
        """Build Cypher query with code namespace (backtick-escaped)."""
        prefixed_label = self.prefix_node(self.CODE_NAMESPACE, label)
        return f"CREATE (n:{prefixed_label} $props)"

    def build_doc_node_query(self, label: str, properties: dict) -> str:
        """Build Cypher query with doc namespace (backtick-escaped)."""
        prefixed_label = self.prefix_node(self.DOC_NAMESPACE, label)
        return f"CREATE (n:{prefixed_label} $props)"


# Schema for namespace mode (single database)
"""
// Code namespace nodes
CREATE CONSTRAINT ON (n:code:Function) ASSERT n.qualified_name IS UNIQUE;
CREATE CONSTRAINT ON (n:code:Class) ASSERT n.qualified_name IS UNIQUE;

// Document namespace nodes
CREATE CONSTRAINT ON (n:doc:Document) ASSERT n.path IS UNIQUE;
CREATE CONSTRAINT ON (n:doc:Section) ASSERT n.qualified_name IS UNIQUE;

// Cross-namespace relationships (explicitly prefixed)
// (:doc:Document)-[:doc:REFERENCES_CODE]->(:code:Function)
"""
```

**Environment Configuration:**

```bash
# .env (namespace mode)
ISOLATION_MODE=namespace  # or parallel
MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687       # Single port for both graphs
CODE_NAMESPACE=code
DOC_NAMESPACE=doc
```

**When to use Namespace Mode:**

- Development/testing environments (simpler setup)
- Small codebases (< 10K documents)
- When cross-graph Cypher queries are needed
- Resource-constrained deployments

**When to use Parallel Instances:**

- Production deployments (strong isolation)
- Large codebases (> 10K documents)
- Independent backup/restore requirements
- Different scaling needs per graph

---

## 6. Query Modes

### 6.1 Mode Enumeration

```python
# codebase_rag/shared/query_router.py

from enum import Enum

class QueryMode(Enum):
    """
    Explicit query routing modes.
    User MUST specify mode — no automatic guessing.
    """

    CODE_ONLY = "code_only"
    """
    Query CODE graph/vector ONLY.
    Document graph is NOT touched.

    Use for: Function lookups, call graphs, class hierarchies.
    Example: "What functions call authenticate_user?"
    """

    DOCUMENT_ONLY = "document_only"
    """
    Query DOCUMENT graph/vector ONLY.
    Code graph is NOT touched.

    Use for: Tutorials, guides, API documentation.
    Example: "How do I use the authentication API?"
    """

    BOTH_MERGED = "both_merged"
    """
    Query BOTH graphs, merge results with clear attribution.

    Use for: Comprehensive research.
    Example: "Tell me everything about authentication"
    """

    CODE_VS_DOC = "code_vs_doc"
    """
    Validate CODE against DOCUMENT specifications.

    Document is SOURCE OF TRUTH.

    Use for: API spec compliance, regulatory requirements.
    Example: "Does code implement all endpoints in OpenAPI spec?"
    """

    DOC_VS_CODE = "doc_vs_code"
    """
    Validate DOCUMENT against actual CODE.

    Code is SOURCE OF TRUTH.

    Use for: Documentation audits, finding outdated docs.
    Example: "Is docs/api.md still accurate?"
    """
```

### 6.2 Query Request/Response Schema

```python
# codebase_rag/shared/query_router.py

from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class QueryRequest:
    """
    Explicit query request with mode specification.
    """
    question: str
    mode: QueryMode
    validate: bool = False              # Enable cross-validation
    include_metadata: bool = True       # Include source info
    top_k: int = 5                      # Results per graph
    scope: str = "all"                  # Validation scope: "all", "sections", "claims"


@dataclass
class Source:
    """
    Source attribution for query results.
    """
    type: Literal["code", "document"]
    path: str
    node_type: Optional[str] = None     # Function, Class, Section, etc.
    qualified_name: Optional[str] = None


@dataclass
class ValidationResult:
    """
    Single validation result.
    """
    element: str                        # Function name, section title, etc.
    status: Literal["VALID", "OUTDATED", "MISSING", "ACCURATE"]
    direction: Literal["CODE_VS_DOC", "DOC_VS_CODE"]
    suggestion: Optional[str] = None    # Suggested fix


@dataclass
class ValidationReport:
    """
    Validation report for bidirectional validation.
    """
    total: int
    passed: int
    failed: int
    direction: Literal["CODE_VS_DOC", "DOC_VS_CODE"]
    results: list[ValidationResult]
    accuracy_score: float               # passed / total


@dataclass
class QueryResponse:
    """
    Query response with clear source attribution.
    """
    answer: str
    sources: list[Source]
    mode: QueryMode
    validation_report: Optional[ValidationReport] = None
    warnings: list[str] = None          # e.g., "Document graph was NOT queried"
```

### 6.3 Query Router Implementation

```python
# codebase_rag/shared/query_router.py

from ..services.graph_service import MemgraphIngestor
from ..vector_backend import VectorBackend, get_vector_backend

class QueryRouter:
    """
    Routes queries to appropriate graph(s) based on EXPLICIT mode.

    Pattern follows existing MCPToolsRegistry from codebase_rag/mcp/tools.py
    """

    def __init__(
        self,
        code_graph: MemgraphIngestor,
        doc_graph: MemgraphIngestor,
        code_vector: VectorBackend | None = None,
        doc_vector: VectorBackend | None = None,
    ):
        self.code_graph = code_graph
        self.doc_graph = doc_graph
        self.code_vector = code_vector or get_vector_backend()
        # Document vector uses separate backend instance
        self.doc_vector = doc_vector

    def query(self, request: QueryRequest) -> QueryResponse:
        """Route query based on EXPLICIT mode."""

        if request.mode == QueryMode.CODE_ONLY:
            return self._query_code_only(request)

        elif request.mode == QueryMode.DOCUMENT_ONLY:
            return self._query_document_only(request)

        elif request.mode == QueryMode.BOTH_MERGED:
            return self._query_both_merged(request)

        elif request.mode == QueryMode.CODE_VS_DOC:
            return self._validate_code_against_doc(request)

        elif request.mode == QueryMode.DOC_VS_CODE:
            return self._validate_doc_against_code(request)

        else:
            raise ValueError(f"Unknown query mode: {request.mode}")

    def _query_code_only(self, request: QueryRequest) -> QueryResponse:
        """Query CODE graph/vector ONLY."""
        # Implementation follows existing query patterns from mcp/tools.py
        pass

    def _query_document_only(self, request: QueryRequest) -> QueryResponse:
        """Query DOCUMENT graph/vector ONLY."""
        pass

    def _query_both_merged(self, request: QueryRequest) -> QueryResponse:
        """Query BOTH graphs, merge results."""
        pass

    def _validate_code_against_doc(self, request: QueryRequest) -> QueryResponse:
        """Validate CODE against DOCUMENT specs."""
        from .validation.code_vs_doc import CodeVsDocValidator
        validator = CodeVsDocValidator(self.code_graph, self.doc_graph)
        report = validator.validate(request.question)
        return QueryResponse(
            answer=validator.generate_summary(report),
            sources=[],
            mode=request.mode,
            validation_report=report,
        )

    def _validate_doc_against_code(self, request: QueryRequest) -> QueryResponse:
        """Validate DOCUMENT against actual CODE."""
        from .validation.doc_vs_code import DocVsCodeValidator
        validator = DocVsCodeValidator(self.code_graph, self.doc_graph)
        report = validator.validate(request.question)
        return QueryResponse(
            answer=validator.generate_summary(report),
            sources=[],
            mode=request.mode,
            validation_report=report,
        )
```

---

## 7. Document Ingestion Pipeline

### 7.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Document Ingestion Pipeline                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────────┐                                                   │
│  │  ALL Documents   │                                                   │
│  │  (No filtering)  │                                                   │
│  └────────┬─────────┘                                                   │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 1: Extract Content (Deterministic)                         │    │
│  │  - Text extraction (format-specific)                            │    │
│  │  - Header/section detection                                     │    │
│  │  - Code block extraction                                        │    │
│  │  - Reference extraction (function/class names)                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 2: Store in Document Graph (Uniform Schema)                │    │
│  │  - Document node with minimal metadata                          │    │
│  │  - Section nodes for each header                                │    │
│  │  - Code reference relationships (NOT validated)                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 3: Generate Embeddings (Reuse embeddings/ module)         │    │
│  │  - Use existing EmbeddingProvider pattern                       │    │
│  │  - Store in vector backend (Memgraph or Qdrant)                 │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Step 4: NO Pre-Validation                                       │    │
│  │  - Validation happens at QUERY TIME only                        │    │
│  │  - Saves 60-80% on LLM costs                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Extractor Interface (Following EmbeddingProvider Pattern)

```python
# codebase_rag/document/extractors/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class ExtractedSection:
    """Represents a document section."""
    title: str
    level: int                      # Header level (1-6)
    start_line: int
    end_line: int
    content: str
    subsections: list['ExtractedSection']


@dataclass
class ExtractedDocument:
    """Represents an extracted document."""
    path: str
    file_type: str
    content: str
    sections: list[ExtractedSection]
    code_blocks: list[str]
    code_references: list[str]      # Extracted function/class names
    word_count: int
    modified_date: str


class BaseDocumentExtractor(ABC):
    """
    Abstract base class for document extractors.

    Pattern follows EmbeddingProvider from codebase_rag/embeddings/base.py:
    - Registry pattern in __init__.py
    - Factory function get_extractor_for_file()
    - Stateless extraction
    """

    __slots__ = ("_config",)

    def __init__(self, **config: str | int | None) -> None:
        self._config: dict[str, str | int | None] = config

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """List of supported file extensions (e.g., ['.md', '.rst'])."""
        pass

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Extract content from document.

        Args:
            file_path: Path to document file

        Returns:
            ExtractedDocument with all extracted content
        """
        pass

    def _extract_code_references(self, content: str) -> list[str]:
        """
        Extract potential function/class references from content.

        Uses deterministic patterns (regex), not LLM.
        """
        import re

        patterns = [
            r'`([a-zA-Z_][\w]*(?:\.[\w]+)*)\(\)`',  # Function calls
            r'class `(\w+)`',                         # Class references
            r'method `(\w+)`',                        # Method references
        ]

        refs = []
        for pattern in patterns:
            matches = re.findall(pattern, content)
            refs.extend(matches)

        return list(set(refs))[:50]  # Limit to 50 refs
```

### 7.3 Extractor Registry

```python
# codebase_rag/document/extractors/__init__.py

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseDocumentExtractor, ExtractedDocument

# Registry pattern following codebase_rag/embeddings/__init__.py
_EXTRACTOR_REGISTRY: dict[str, type['BaseDocumentExtractor']] = {}


def _register_extractor(cls: type['BaseDocumentExtractor']) -> None:
    """Register an extractor for all its supported extensions."""
    instance = cls()
    for ext in instance.supported_extensions:
        _EXTRACTOR_REGISTRY[ext.lower()] = cls


def get_extractor_for_file(file_path: Path) -> 'BaseDocumentExtractor | None':
    """Get appropriate extractor for file type."""
    ext = file_path.suffix.lower()
    extractor_cls = _EXTRACTOR_REGISTRY.get(ext)
    return extractor_cls() if extractor_cls else None


# Import and register extractors
from .markdown_extractor import MarkdownExtractor
from .base import BaseDocumentExtractor, ExtractedDocument, ExtractedSection

_register_extractor(MarkdownExtractor)

# Optional extractors (import if dependencies available)
try:
    from .pdf_extractor import PDFExtractor
    _register_extractor(PDFExtractor)
except ImportError:
    pass

try:
    from .docx_extractor import DocxExtractor
    _register_extractor(DocxExtractor)
except ImportError:
    pass


__all__ = [
    "get_extractor_for_file",
    "BaseDocumentExtractor",
    "ExtractedDocument",
    "ExtractedSection",
    "MarkdownExtractor",
]
```

### 7.4 Markdown Extractor

```python
# codebase_rag/document/extractors/markdown_extractor.py

import re
from pathlib import Path
from datetime import UTC, datetime

from .base import BaseDocumentExtractor, ExtractedDocument, ExtractedSection


class MarkdownExtractor(BaseDocumentExtractor):
    """
    Extractor for Markdown files (.md, .rst, .txt).
    """

    @property
    def supported_extensions(self) -> list[str]:
        return ['.md', '.rst', '.txt']

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract content from Markdown file."""

        content = file_path.read_text(encoding='utf-8')
        modified_date = datetime.fromtimestamp(
            file_path.stat().st_mtime, UTC
        ).isoformat()

        # Extract sections (headers)
        sections = self._extract_sections(content)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)

        # Extract code references
        code_references = self._extract_code_references(content)

        return ExtractedDocument(
            path=str(file_path),
            file_type=file_path.suffix,
            content=content,
            sections=sections,
            code_blocks=code_blocks,
            code_references=code_references,
            word_count=len(content.split()),
            modified_date=modified_date
        )

    def _extract_sections(self, content: str) -> list[ExtractedSection]:
        """Extract header hierarchy from Markdown."""
        pattern = r'^(#{1,6})\s+(.+)$'
        sections = []

        lines = content.split('\n')
        current_section = None

        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if match:
                # Save previous section
                if current_section:
                    current_section.end_line = i - 1
                    sections.append(current_section)

                # Start new section
                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = ExtractedSection(
                    title=title,
                    level=level,
                    start_line=i,
                    end_line=len(lines),  # Will be updated
                    content='',
                    subsections=[]
                )

        return sections

    def _extract_code_blocks(self, content: str) -> list[str]:
        """Extract code blocks from Markdown."""
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.findall(pattern, content, re.DOTALL)
        return [m[1] for m in matches]
```

### 7.5 Document Updater (Following GraphUpdater Pattern)

```python
# codebase_rag/document/document_updater.py

from pathlib import Path
from datetime import UTC, datetime
from loguru import logger

from ..services.graph_service import MemgraphIngestor
from ..embeddings import get_embedding_provider
from .extractors import get_extractor_for_file, ExtractedDocument


class DocumentGraphUpdater:
    """
    Handles document graph ingestion and updates.

    Pattern follows GraphUpdater from codebase_rag/graph_updater.py
    """

    def __init__(
        self,
        host: str,
        port: int,
        repo_path: Path,
        batch_size: int = 1000,
    ):
        self.host = host
        self.port = port
        self.repo_path = repo_path
        self.batch_size = batch_size

    def run(self, force: bool = False) -> None:
        """
        Ingest all documents into document graph.

        Args:
            force: If True, re-index all documents (ignore cache)
        """
        with MemgraphIngestor(
            host=self.host,
            port=self.port,
            batch_size=self.batch_size,
        ) as ingestor:

            # Ensure document graph schema
            ingestor.ensure_constraints()

            # Collect all document files
            documents = self._collect_documents()

            logger.info(f"Indexing {len(documents)} documents...")

            # Process each document
            for doc_path in documents:
                self._process_document(doc_path, ingestor, force)

            # Flush remaining buffers
            ingestor.flush_all()

            logger.info("Document indexing complete")

    def _collect_documents(self) -> list[Path]:
        """Collect all eligible document files."""
        documents = []

        for ext in ['.md', '.rst', '.txt']:  # Extend with PDF, DOCX
            documents.extend(self.repo_path.rglob(f'*{ext}'))

        # Filter out excluded paths
        excluded = {'.git', 'node_modules', '__pycache__', '.venv'}
        documents = [
            d for d in documents
            if not any(excl in str(d) for excl in excluded)
        ]

        return documents

    def _process_document(
        self,
        file_path: Path,
        ingestor: MemgraphIngestor,
        force: bool,
    ) -> None:
        """Process single document."""

        # Find appropriate extractor
        extractor = get_extractor_for_file(file_path)
        if not extractor:
            logger.warning(f"No extractor for {file_path}")
            return

        # Extract content
        doc = extractor.extract(file_path)

        # Store in graph
        self._store_document(doc, ingestor)

        # Generate and store embedding (reuse embeddings module)
        embedding = self._generate_embedding(doc)
        self._store_embedding(doc.path, embedding, ingestor)

    def _generate_embedding(self, doc: ExtractedDocument) -> list[float]:
        """Generate embedding using existing provider system with semantic chunking."""
        from ..config import settings
        from .chunking import SemanticDocumentChunker

        provider = get_embedding_provider(
            provider=settings.EMBEDDING_PROVIDER,
            model_id=settings.EMBEDDING_MODEL,
        )

        # Use semantic chunking instead of hardcoded character limits
        chunker = SemanticDocumentChunker(max_tokens=512)
        chunks = list(chunker.chunk_document(doc))

        if not chunks:
            # Fallback for empty documents
            return provider.embed("")

        # Embed all chunks and return the first (or aggregate as needed)
        embeddings = provider.embed_batch([c.content for c in chunks])
        return embeddings[0]  # Primary embedding; store others in Chunk nodes

    def _store_document(
        self,
        doc: ExtractedDocument,
        ingestor: MemgraphIngestor,
    ) -> None:
        """Store document and sections in graph."""

        # Create document node
        ingestor.ensure_node_batch(
            'Document',
            {
                'path': doc.path,
                'file_type': doc.file_type,
                'section_count': len(doc.sections),
                'code_block_count': len(doc.code_blocks),
                'code_references': doc.code_references,
                'word_count': doc.word_count,
                'modified_date': doc.modified_date,
                'indexed_at': datetime.now(UTC).isoformat(),
            }
        )

        # Create section nodes
        for section in doc.sections:
            section_qn = f"{doc.path}#{section.title}"

            ingestor.ensure_node_batch(
                'Section',
                {
                    'qualified_name': section_qn,
                    'title': section.title,
                    'level': section.level,
                    'start_line': section.start_line,
                    'end_line': section.end_line,
                    'content_snippet': section.content[:500],
                }
            )

            # Link section to document
            ingestor.ensure_relationship_batch(
                ('Document', 'path', doc.path),
                'CONTAINS_SECTION',
                ('Section', 'qualified_name', section_qn),
            )
```

### 7.6 Semantic Chunking for Documents

Documents should be chunked semantically (by section/topic) rather than by character count for better retrieval quality.

```python
# codebase_rag/document/chunking.py

from dataclasses import dataclass
from typing import Iterator
import tiktoken

@dataclass
class DocumentChunk:
    """A semantically coherent document chunk."""
    content: str
    section_title: str
    start_line: int
    end_line: int
    token_count: int
    document_path: str


class SemanticDocumentChunker:
    """
    Chunks documents by semantic boundaries (headers/sections).

    CRITICAL: Don't use hardcoded character limits like 8000.
    Use token-aware chunking with section boundaries.
    """

    MAX_CHUNK_TOKENS = 512  # Per chunk limit
    OVERLAP_TOKENS = 50     # Overlap for context continuity
    ENCODING_MODEL = "cl100k_base"  # GPT-4 encoding

    def __init__(self, max_tokens: int = 512):
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding(self.ENCODING_MODEL)

    def chunk_document(self, doc: ExtractedDocument) -> Iterator[DocumentChunk]:
        """
        Chunk document by semantic boundaries.

        Strategy:
        1. Each section is a potential chunk boundary
        2. If section exceeds max_tokens, split by paragraph
        3. Maintain overlap between chunks
        """
        for section in doc.sections:
            yield from self._chunk_section(section, doc.path)

    def _chunk_section(self, section: ExtractedSection, doc_path: str) -> Iterator[DocumentChunk]:
        """Chunk a single section, respecting token limits."""
        tokens = self.encoder.encode(section.content)

        if len(tokens) <= self.max_tokens:
            # Section fits in one chunk
            yield DocumentChunk(
                content=section.content,
                section_title=section.title,
                start_line=section.start_line,
                end_line=section.end_line,
                token_count=len(tokens),
                document_path=doc_path,
            )
        else:
            # Split by paragraphs within section
            yield from self._split_by_paragraphs(section, doc_path)

    def _split_by_paragraphs(self, section: ExtractedSection, doc_path: str) -> Iterator[DocumentChunk]:
        """Split large sections by paragraph boundaries."""
        paragraphs = section.content.split('\n\n')
        current_chunk = []
        current_tokens = 0
        start_line = section.start_line

        for para in paragraphs:
            para_tokens = len(self.encoder.encode(para))

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                # Flush current chunk
                yield DocumentChunk(
                    content='\n\n'.join(current_chunk),
                    section_title=section.title,
                    start_line=start_line,
                    end_line=start_line + len('\n\n'.join(current_chunk).split('\n')),
                    token_count=current_tokens,
                    document_path=doc_path,
                )
                current_chunk = []
                current_tokens = 0

            current_chunk.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_chunk:
            yield DocumentChunk(
                content='\n\n'.join(current_chunk),
                section_title=section.title,
                start_line=start_line,
                end_line=section.end_line,
                token_count=current_tokens,
                document_path=doc_path,
            )
```

### 7.7 Async Extractor Interface

Extractors MUST support async operations to avoid blocking the event loop for large files.

```python
# codebase_rag/document/extractors/base.py (async extension)

class BaseDocumentExtractor(ABC):
    """Abstract base class for document extractors - now with async support."""

    @abstractmethod
    async def extract_async(self, file_path: Path) -> ExtractedDocument:
        """
        Async extraction for large files.

        REQUIRED: All extractors must implement async variant.
        Sync extract() can delegate to async for small files.
        """
        pass

    @abstractmethod
    async def extract_batch_async(self, file_paths: list[Path]) -> list[ExtractedDocument]:
        """
        Batch extraction with concurrent processing.

        Uses asyncio.gather for parallel extraction.
        """
        pass

    async def extract_async_with_error_boundary(
        self, file_path: Path
    ) -> ExtractedDocument | ExtractionError:
        """
        Extraction with error boundary - never crashes pipeline.

        Returns either extracted document or structured error.
        """
        try:
            return await self.extract_async(file_path)
        except Exception as e:
            return ExtractionError(
                path=str(file_path),
                error_type=type(e).__name__,
                message=str(e),
                recoverable=self._is_recoverable(e),
            )

    def _is_recoverable(self, error: Exception) -> bool:
        """Determine if error is recoverable (skip file) vs fatal (stop pipeline)."""
        # Malformed file = recoverable, permission denied = fatal
        return isinstance(error, (ValueError, UnicodeDecodeError))
```

### 7.8 Error Handling Infrastructure

Pipeline must have comprehensive error handling with dead letter queue for failed documents.

```python
# codebase_rag/document/error_handling.py

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from enum import Enum
import json

class ErrorType(Enum):
    """Classification of extraction errors."""
    MALFORMED_FILE = "malformed_file"       # Corrupted file content
    MISSING_DEPENDENCY = "missing_dependency" # PyPDF2 not installed
    FILE_TOO_LARGE = "file_too_large"        # Exceeds DOC_MAX_FILE_SIZE_MB
    PERMISSION_DENIED = "permission_denied"
    ENCODING_ERROR = "encoding_error"
    UNKNOWN = "unknown"


@dataclass
class ExtractionError:
    """Structured error for failed document extraction."""
    path: str
    error_type: ErrorType
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    recoverable: bool = True
    retry_count: int = 0
    max_retries: int = 3


class DeadLetterQueue:
    """
    Dead letter queue for failed document extractions.

    Failed documents are logged and can be retried later.
    """

    def __init__(self, queue_path: Path):
        self.queue_path = queue_path
        self.queue_path.mkdir(parents=True, exist_ok=True)

    def enqueue(self, error: ExtractionError) -> None:
        """Add failed document to dead letter queue."""
        error_file = self.queue_path / f"{Path(error.path).name}.error.json"
        error_file.write_text(json.dumps(error.__dict__, indent=2))

    def get_pending(self) -> list[ExtractionError]:
        """Get all pending errors for retry."""
        errors = []
        for error_file in self.queue_path.glob("*.error.json"):
            data = json.loads(error_file.read_text())
            errors.append(ExtractionError(**data))
        return errors

    async def retry_with_backoff(self, extractor: BaseDocumentExtractor) -> dict[str, bool]:
        """
        Retry all pending errors with exponential backoff.

        Returns dict mapping path -> success status.
        """
        import asyncio

        results = {}
        for error in self.get_pending():
            if error.retry_count >= error.max_retries:
                continue  # Skip exhausted retries

            # Exponential backoff: 2^retry_count seconds
            delay = 2 ** error.retry_count
            await asyncio.sleep(delay)

            try:
                await extractor.extract_async(Path(error.path))
                results[error.path] = True
                # Remove from queue on success
                (self.queue_path / f"{hashlib.sha256(error.path.encode()).hexdigest()[:16]}.error.json").unlink()
            except Exception:
                error.retry_count += 1
                self.enqueue(error)  # Update retry count
                results[error.path] = False

        return results
```

### 7.9 Content Versioning for Incremental Updates

Documents must be tracked with content fingerprints for incremental updates (avoid full re-indexing).

```python
# codebase_rag/document/versioning.py

import hashlib
from dataclasses import dataclass
from pathlib import Path
from datetime import UTC, datetime

@dataclass
class DocumentVersion:
    """Version tracking for incremental updates."""
    path: str
    content_hash: str           # SHA-256 of content
    modified_date: str          # File modification time
    indexed_at: str             # Last indexing timestamp
    section_hashes: list[str]   # Hash of each section


class ContentVersionTracker:
    """
    Track document versions for incremental updates.

    Strategy:
    1. Compute content hash at extraction
    2. Compare with stored hash
    3. Only re-index changed documents/sections
    """

    HASH_ALGORITHM = "sha256"

    def compute_hash(self, content: str) -> str:
        """Compute content fingerprint."""
        return hashlib.sha256(content.encode()).hexdigest()

    def compute_section_hashes(self, sections: list[ExtractedSection]) -> list[str]:
        """Compute hashes for each section."""
        return [self.compute_hash(s.content) for s in sections]

    def needs_reindex(
        self,
        file_path: Path,
        stored_version: DocumentVersion | None
    ) -> tuple[bool, list[int]]:
        """
        Determine if document needs re-indexing and which sections changed.

        Returns: (needs_full_reindex, changed_section_indices)
        """
        current_hash = self.compute_hash(file_path.read_text())

        if stored_version is None:
            return True, []  # New document

        if current_hash != stored_version.content_hash:
            # Content changed - identify which sections
            # (Full reindex for now; section-level diff is future work)
            return True, []

        return False, []  # No changes

    def store_version(self, doc: ExtractedDocument, ingestor: MemgraphIngestor) -> None:
        """Store version metadata in document node."""
        version = DocumentVersion(
            path=doc.path,
            content_hash=self.compute_hash(doc.content),
            modified_date=doc.modified_date,
            indexed_at=datetime.now(UTC).isoformat(),
            section_hashes=self.compute_section_hashes(doc.sections),
        )

        ingestor.ensure_node_batch(
            'Document',
            {
                'path': doc.path,
                'content_hash': version.content_hash,
                'indexed_at': version.indexed_at,
                'section_hashes': version.section_hashes,
                # ... other fields ...
            }
        )
```

---

## 8. Validation System

### 8.1 Validation Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Validation System                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Validation Engine (Bidirectional)                               │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │  ┌──────────────────┐          ┌──────────────────┐             │    │
│  │  │  Code → Doc      │          │  Doc → Code      │             │    │
│  │  │  Validation      │          │  Validation      │             │    │
│  │  ├──────────────────┤          ├──────────────────┤             │    │
│  │  │ • Extract claims │          │ • Extract claims │             │    │
│  │  │   from docs      │          │   from docs      │             │    │
│  │  │ • Verify in code │          │ • Verify in code │             │    │
│  │  │ • Report gaps    │          │ • Report drift   │             │    │
│  │  └──────────────────┘          └──────────────────┘             │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  Validation is ON-DEMAND only (not pre-computed at index time)          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Code → Document Validation

```python
# codebase_rag/shared/validation/code_vs_doc.py

from ..query_router import ValidationReport, ValidationResult
from ...services.graph_service import MemgraphIngestor


class CodeVsDocValidator:
    """
    Validate CODE against DOCUMENT specifications.

    Document is SOURCE OF TRUTH.
    """

    def __init__(self, code_graph: MemgraphIngestor, doc_graph: MemgraphIngestor):
        self.code_graph = code_graph
        self.doc_graph = doc_graph

    def validate(self, document_path: str) -> ValidationReport:
        """
        Validate code implementation against document spec.

        Use cases:
        - API spec compliance (OpenAPI, gRPC proto)
        - Regulatory requirements (HIPAA, GDPR)
        - Contract verification
        """
        # Extract requirements from document
        requirements = self._extract_requirements(document_path)

        # Verify each requirement in code
        results = []
        for req in requirements:
            implemented = self._verify_in_code(req)
            results.append(ValidationResult(
                element=req.description,
                status="VALID" if implemented else "MISSING",
                direction="CODE_VS_DOC",
                suggestion=None if implemented else f"Implement {req.description}"
            ))

        return ValidationReport(
            total=len(results),
            passed=sum(1 for r in results if r.status == "VALID"),
            failed=sum(1 for r in results if r.status == "MISSING"),
            direction="CODE_VS_DOC",
            results=results,
            accuracy_score=sum(1 for r in results if r.status == "VALID") / len(results)
        )

    def _extract_requirements(self, document_path: str) -> list:
        """Extract requirements from specification document."""
        # Use LLM for this (requires semantic understanding)
        # Only called during validation, not indexing
        pass

    def _verify_in_code(self, requirement) -> bool:
        """Verify requirement exists in code graph."""
        pass

    def generate_summary(self, report: ValidationReport) -> str:
        """Generate human-readable summary."""
        pass
```

### 8.3 Document → Code Validation

```python
# codebase_rag/shared/validation/doc_vs_code.py

from ..query_router import ValidationReport, ValidationResult
from ...services.graph_service import MemgraphIngestor


class DocVsCodeValidator:
    """
    Validate DOCUMENT against actual CODE.

    Code is SOURCE OF TRUTH.
    """

    def __init__(self, code_graph: MemgraphIngestor, doc_graph: MemgraphIngestor):
        self.code_graph = code_graph
        self.doc_graph = doc_graph

    def validate(self, document_path: str) -> ValidationReport:
        """
        Validate document accuracy against code.

        Use cases:
        - API documentation accuracy
        - Tutorial correctness
        - Finding outdated references
        """
        # Extract claims from document
        claims = self._extract_claims(document_path)

        # Verify each claim against code
        results = []
        for claim in claims:
            code_confirms = self._verify_claim(claim)
            results.append(ValidationResult(
                element=claim.description,
                status="ACCURATE" if code_confirms else "OUTDATED",
                direction="DOC_VS_CODE",
                suggestion=self._suggest_fix(claim) if not code_confirms else None
            ))

        return ValidationReport(
            total=len(results),
            passed=sum(1 for r in results if r.status == "ACCURATE"),
            failed=sum(1 for r in results if r.status == "OUTDATED"),
            direction="DOC_VS_CODE",
            results=results,
            accuracy_score=sum(1 for r in results if r.status == "ACCURATE") / len(results)
        )

    def _extract_claims(self, document_path: str) -> list:
        """Extract factual claims from document."""
        # Use LLM for this (requires semantic understanding)
        # Only called during validation, not indexing
        pass

    def _verify_claim(self, claim) -> bool:
        """Verify claim against code graph."""
        pass

    def _suggest_fix(self, claim) -> str:
        """Suggest fix for outdated claim."""
        # Use LLM to generate suggestion based on actual code
        pass
```

### 8.4 Validation Trigger API

On-demand validation requires a clear trigger mechanism with cost estimation before execution.

```python
# codebase_rag/shared/validation/api.py

from dataclasses import dataclass
from enum import Enum
from typing import Literal

class ValidationTriggerMode(Enum):
    """How validation is triggered."""
    MANUAL = "manual"           # User explicitly requests
    QUERY_FLAG = "query_flag"   # Query with validate=true flag
    SCHEDULED = "scheduled"     # Periodic validation (optional)


@dataclass
class ValidationRequest:
    """User request for validation."""
    document_path: str
    mode: Literal["CODE_VS_DOC", "DOC_VS_CODE"]
    scope: Literal["all", "sections", "claims"] = "all"
    max_cost_usd: float = 0.50   # User's cost budget
    dry_run: bool = False        # Estimate cost without running


@dataclass
class CostEstimate:
    """Cost estimation before validation runs."""
    estimated_llm_calls: int
    estimated_tokens: int
    estimated_cost_usd: float
    exceeds_budget: bool
    breakdown: dict[str, float]  # Per-component cost estimate


@dataclass
class ValidationTriggerResult:
    """Result of validation trigger request."""
    accepted: bool
    cost_estimate: CostEstimate | None
    validation_id: str | None    # Unique ID for tracking
    message: str                 # User-facing message


class ValidationTriggerAPI:
    """
    API for triggering on-demand validation.

    Flow:
    1. User requests validation with budget
    2. System estimates cost (graph queries only, no LLM)
    3. If within budget, user confirms → validation executes
    4. Results cached keyed by: hash(query + graph_state)
    """

    COST_PER_LLM_CALL_USD = 0.01  # Approximate cost per extraction call
    TOKENS_PER_CLAIM = 500        # Average tokens per claim extraction

    async def request_validation(self, request: ValidationRequest) -> ValidationTriggerResult:
        """
        Request validation with cost estimation.

        IMPORTANT: Never run validation without cost estimate first.
        """
        # Step 1: Estimate cost without running LLM
        estimate = await self._estimate_cost(request)

        # Step 2: Check budget
        if estimate.exceeds_budget:
            return ValidationTriggerResult(
                accepted=False,
                cost_estimate=estimate,
                validation_id=None,
                message=f"Estimated cost ${estimate.estimated_cost_usd:.2f} exceeds budget ${request.max_cost_usd:.2f}",
            )

        # Step 3: If dry_run, return estimate only
        if request.dry_run:
            return ValidationTriggerResult(
                accepted=False,
                cost_estimate=estimate,
                validation_id=None,
                message=f"Cost estimate: ${estimate.estimated_cost_usd:.2f} ({estimate.estimated_llm_calls} LLM calls)",
            )

        # Step 4: Generate validation ID and queue execution
        validation_id = self._generate_id(request)
        await self._queue_validation(validation_id, request)

        return ValidationTriggerResult(
            accepted=True,
            cost_estimate=estimate,
            validation_id=validation_id,
            message=f"Validation queued. Estimated cost: ${estimate.estimated_cost_usd:.2f}",
        )

    async def _estimate_cost(self, request: ValidationRequest) -> CostEstimate:
        """
        Estimate validation cost using graph queries only.

        NO LLM calls during estimation - use graph statistics.
        """
        # Get document statistics from graph
        doc_stats = await self._get_document_stats(request.document_path)

        # Estimate based on scope
        if request.scope == "all":
            estimated_claims = doc_stats.get("estimated_claims", 10)
        elif request.scope == "sections":
            estimated_claims = doc_stats.get("section_count", 5) * 2
        else:
            estimated_claims = 3  # Single claim validation

        llm_calls = estimated_claims * 2  # Extract + verify per claim
        tokens = estimated_claims * self.TOKENS_PER_CLAIM * 2
        cost_usd = llm_calls * self.COST_PER_LLM_CALL_USD + tokens * 0.00001  # Token cost

        return CostEstimate(
            estimated_llm_calls=llm_calls,
            estimated_tokens=tokens,
            estimated_cost_usd=cost_usd,
            exceeds_budget=cost_usd > request.max_cost_usd,
            breakdown={
                "claim_extraction": estimated_claims * self.COST_PER_LLM_CALL_USD,
                "verification": estimated_claims * self.COST_PER_LLM_CALL_USD,
            },
        )

    async def _get_document_stats(self, document_path: str) -> dict:
        """Get document statistics from graph (no LLM)."""
        # Query graph for: section_count, code_reference_count, word_count
        # These are stored at index time, not computed here
        pass

    def _generate_id(self, request: ValidationRequest) -> str:
        """Generate unique validation ID."""
        import hashlib
        content = f"{request.document_path}:{request.mode}:{request.scope}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _queue_validation(self, validation_id: str, request: ValidationRequest):
        """Queue validation for execution."""
        # In production: use Redis/Celery for async queue
        pass
```

### 8.5 Code Graph Hash Computation

The code graph hash is required for cache invalidation when the code graph changes.

```python
# codebase_rag/shared/validation/graph_hash.py

import hashlib
from ...services.graph_service import MemgraphIngestor

class CodeGraphHashComputer:
    """
    Compute deterministic hash of code graph state.

    Used for cache invalidation when code changes.
    """

    async def compute_graph_hash(self, graph: MemgraphIngestor) -> str:
        """
        Compute hash from graph statistics.

        Strategy: Hash node counts and modification timestamps.
        """
        stats = await graph.execute("""
            MATCH (f:Function)
            WITH count(f) as func_count
            MATCH (c:Class)
            WITH func_count, count(c) as class_count
            MATCH (m:Method)
            RETURN func_count, class_count, count(m) as method_count
        """)

        if not stats:
            return "empty_graph"

        content = f"functions:{stats[0]['func_count']}:classes:{stats[0]['class_count']}:methods:{stats[0]['method_count']}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
```

### 8.6 Result Caching Strategy

Validation results should be cached to avoid redundant LLM calls for unchanged content.

```python
# codebase_rag/shared/validation/cache.py

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

@dataclass
class ValidationCacheKey:
    """Cache key for validation results."""
    document_path: str
    document_hash: str        # Content hash at validation time
    code_graph_hash: str      # Graph state hash
    mode: str
    scope: str


@dataclass
class CachedValidation:
    """Cached validation result."""
    key: ValidationCacheKey
    report: ValidationReport
    cached_at: datetime
    expires_at: datetime      # TTL for cache


class ValidationCache:
    """
    Cache validation results to avoid redundant LLM calls.

    Cache invalidation:
    - Document content changes → invalidate all for that document
    - Code graph changes → invalidate all CODE_VS_DOC results
    - TTL expires → lazy refresh
    """

    DEFAULT_TTL_HOURS = 24
    MAX_CACHE_SIZE = 1000

    def __init__(self, backend: dict | None = None):
        self._cache: dict[str, CachedValidation] = backend or {}
        # Structured index for O(1) document invalidation
        self._document_index: dict[str, set[str]] = {}  # document_path -> cache_keys

    def compute_key(
        self,
        document_path: str,
        document_hash: str,
        code_graph_hash: str,
        mode: str,
        scope: str,
    ) -> str:
        """Compute cache key from validation parameters."""
        content = f"{document_path}|{document_hash}|{code_graph_hash}|{mode}|{scope}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, key: str) -> CachedValidation | None:
        """Get cached validation if not expired."""
        cached = self._cache.get(key)
        if cached and cached.expires_at > datetime.now(UTC):
            return cached
        return None

    def set(self, key: str, report: ValidationReport, document_path: str, ttl_hours: int = None) -> None:
        """Cache validation result with TTL and document index."""
        ttl = ttl_hours or self.DEFAULT_TTL_HOURS
        now = datetime.now(UTC)

        self._cache[key] = CachedValidation(
            key=key,
            report=report,
            cached_at=now,
            expires_at=now + timedelta(hours=ttl),
        )

        # Track document -> keys mapping for O(1) invalidation
        if document_path not in self._document_index:
            self._document_index[document_path] = set()
        self._document_index[document_path].add(key)

        # Evict old entries if over limit
        if len(self._cache) > self.MAX_CACHE_SIZE:
            self._evict_oldest()

    def invalidate_document(self, document_path: str) -> int:
        """Invalidate all cached validations for a document using structured index."""
        keys_to_remove = self._document_index.get(document_path, set())
        for k in keys_to_remove:
            self._cache.pop(k, None)
        self._document_index.pop(document_path, None)
        return len(keys_to_remove)

    def invalidate_code_graph(self, new_graph_hash: str) -> int:
        """Invalidate all CODE_VS_DOC results when code graph changes."""
        keys_to_remove = []
        for key, cached in self._cache.items():
            if hasattr(cached, 'key') and hasattr(cached.key, 'mode'):
                if cached.key.mode == "CODE_VS_DOC":
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            self._cache.pop(key, None)
            # Clean up document index
            for doc_path, keys in self._document_index.items():
                keys.discard(key)

        return len(keys_to_remove)

    def _evict_oldest(self) -> None:
        """Evict oldest cached entries."""
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].cached_at
        )
        for key, _ in sorted_entries[:100]:  # Remove 100 oldest
            self._cache.pop(key, None)
            # Clean up document index
            for doc_path, keys in self._document_index.items():
                keys.discard(key)
```

---

## 9. Real-Time Updater

### 9.1 Unified Watcher Architecture

```python
# codebase_rag/realtime_updater.py (extended)

import asyncio
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

from .graph_updater import GraphUpdater
from .document.document_updater import DocumentGraphUpdater
from .config import settings


class CodebaseEventHandler(FileSystemEventHandler):
    """
    Watches entire codebase for BOTH code and document changes.
    Routes updates to appropriate graph updater.
    """

    def __init__(
        self,
        code_updater: GraphUpdater,
        doc_updater: DocumentGraphUpdater,
        repo_path: Path,
    ):
        self.code_updater = code_updater
        self.doc_updater = doc_updater
        self.repo_path = repo_path

        # File type classification
        self.code_extensions = {'.py', '.js', '.ts', '.rs', '.go', '.java', '.cpp', '.c'}
        self.doc_extensions = {'.md', '.rst', '.txt', '.pdf', '.docx'}

        # Debounce timer
        self._debounce_timers: dict[Path, asyncio.TimerHandle] = {}

    def _classify_file(self, file_path: Path) -> str:
        """Determine if file is code, document, or other."""
        suffix = file_path.suffix.lower()

        if suffix in self.code_extensions:
            return "code"
        elif suffix in self.doc_extensions:
            return "document"
        elif file_path.name in ['pyproject.toml', 'package.json', 'Cargo.toml']:
            return "dependency"
        elif 'docs/' in str(file_path) or 'documentation/' in str(file_path):
            return "document"
        else:
            return "other"

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        file_type = self._classify_file(file_path)

        # Debounce rapid changes
        if file_path in self._debounce_timers:
            self._debounce_timers[file_path].cancel()

        loop = asyncio.get_event_loop()
        timer = loop.call_later(
            settings.REALTIME_DEBOUNCE_SECONDS,
            lambda: self._process_change(file_path, file_type)
        )
        self._debounce_timers[file_path] = timer

    def _process_change(self, file_path: Path, file_type: str):
        """Process file change after debounce."""
        # Use new event loop for async operations from sync callback
        loop = asyncio.new_event_loop()
        try:
            if file_type == "code":
                loop.run_until_complete(self.code_updater.update_file(file_path))
            elif file_type == "document":
                loop.run_until_complete(self.doc_updater.update_file(file_path))
            elif file_type == "dependency":
                loop.run_until_complete(self.code_updater.update_dependencies(file_path))
        finally:
            loop.close()


def start_realtime_updater(repo_path: str):
    """Start real-time updater for both code and document graphs."""
    repo_path = Path(repo_path).resolve()

    # Initialize both updaters
    code_updater = GraphUpdater(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        repo_path=repo_path,
    )

    doc_updater = DocumentGraphUpdater(
        host=settings.DOC_MEMGRAPH_HOST,
        port=settings.DOC_MEMGRAPH_PORT,
        repo_path=repo_path,
    )

    # Set up file watcher
    event_handler = CodebaseEventHandler(code_updater, doc_updater, repo_path)
    observer = Observer()
    observer.schedule(event_handler, repo_path, recursive=True)

    # Start watching
    observer.start()

    try:
        while True:
            asyncio.run(asyncio.sleep(1))
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
```

---

## 10. Configuration

### 10.1 Environment Variables

```bash
# .env.example (extended)

# ─────────────────────────────────────────────────────────
# CODE GRAPHRAG (Existing)
# ─────────────────────────────────────────────────────────
MEMGRAPH_HOST=localhost
MEMGRAPH_PORT=7687
MEMGRAPH_USERNAME=cgr
MEMGRAPH_PASSWORD=cgr
VECTOR_STORE_BACKEND=memgraph

# ─────────────────────────────────────────────────────────
# DOCUMENT GRAPHRAG (NEW)
# ─────────────────────────────────────────────────────────
DOC_MEMGRAPH_HOST=localhost
DOC_MEMGRAPH_PORT=7688
DOC_MEMGRAPH_USERNAME=cgr
DOC_MEMGRAPH_PASSWORD=cgr
DOC_VECTOR_STORE_BACKEND=memgraph

# Document-specific
DOC_SUPPORTED_EXTENSIONS=.md,.rst,.txt,.pdf,.docx
DOC_ENABLE_PDF_EXTRACTION=true
DOC_MAX_FILE_SIZE_MB=50

# ─────────────────────────────────────────────────────────
# SHARED (Existing)
# ─────────────────────────────────────────────────────────
TARGET_REPO_PATH=/path/to/repo

ORCHESTRATOR_PROVIDER=google
ORCHESTRATOR_MODEL=gemini-2.5-pro
ORCHESTRATOR_API_KEY=your-api-key

CYPHER_PROVIDER=google
CYPHER_MODEL=gemini-2.5-flash
CYPHER_API_KEY=your-api-key

# Embedding (reused for documents)
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=microsoft/unixcoder-base
EMBEDDING_DIMENSION=768
```

### 10.2 Configuration Class Extension

```python
# codebase_rag/config.py (additions)

class AppConfig(BaseSettings):
    # ... existing config ...

    # Document GraphRAG (NEW)
    DOC_MEMGRAPH_HOST: str = "localhost"
    DOC_MEMGRAPH_PORT: int = 7688
    DOC_MEMGRAPH_USERNAME: str | None = None
    DOC_MEMGRAPH_PASSWORD: str | None = None
    DOC_VECTOR_STORE_BACKEND: str = "memgraph"

    DOC_SUPPORTED_EXTENSIONS: list[str] = Field(default=['.md', '.rst', '.txt', '.pdf', '.docx'])
    DOC_ENABLE_PDF_EXTRACTION: bool = True
    DOC_MAX_FILE_SIZE_MB: int = Field(default=50, gt=0)
    DOC_EXTRACTION_TIMEOUT_SECONDS: int = Field(default=30, gt=0)  # Timeout for async extraction

    # Real-time updater (extended)
    REALTIME_DEBOUNCE_SECONDS: int = Field(default=2, gt=0)
    REALTIME_BATCH_SIZE: int = Field(default=100, gt=0)

    @field_validator("DOC_SUPPORTED_EXTENSIONS", mode="before")
    @classmethod
    def parse_doc_extensions(cls, v: str | list[str]) -> list[str]:
        """Parse comma-separated env var to list."""
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",") if ext.strip()]
        return v

    @field_validator("DOC_VECTOR_STORE_BACKEND")
    @classmethod
    def validate_doc_vector_backend(cls, v: str) -> str:
        """Validate document vector backend is supported."""
        allowed = {"memgraph", "qdrant"}
        if v.lower() not in allowed:
            raise ValueError(f"DOC_VECTOR_STORE_BACKEND must be one of: {allowed}")
        return v.lower()
```

---

## 11. CLI & MCP Interface

### 11.1 CLI Commands (Following Existing Pattern)

```python
# codebase_rag/cli.py (additions)

import typer
from .shared.query_router import QueryMode, QueryRequest, QueryRouter

@app.command(help="Query code graph ONLY")
def query_code(
    question: str = typer.Argument(...),
    top_k: int = typer.Option(5, "--top-k"),
):
    """Query the CODE graph/vector exclusively."""
    request = QueryRequest(question=question, mode=QueryMode.CODE_ONLY, top_k=top_k)
    response = query_router.query(request)
    _print_response(response)


@app.command(help="Query document graph ONLY")
def query_docs(
    question: str = typer.Argument(...),
    top_k: int = typer.Option(5, "--top-k"),
):
    """Query the DOCUMENT graph/vector exclusively."""
    request = QueryRequest(question=question, mode=QueryMode.DOCUMENT_ONLY, top_k=top_k)
    response = query_router.query(request)
    _print_response(response)


@app.command(help="Query BOTH code and document graphs")
def query_all(
    question: str = typer.Argument(...),
    top_k: int = typer.Option(5, "--top-k"),
):
    """Query BOTH graphs and merge results."""
    request = QueryRequest(question=question, mode=QueryMode.BOTH_MERGED, top_k=top_k)
    response = query_router.query(request)
    _print_response(response)


@app.command(help="Validate code against document specs")
def validate_code(
    document_path: str = typer.Argument(...),
    scope: str = typer.Option("all", "--scope"),
):
    """Validate CODE against DOCUMENT specifications."""
    request = QueryRequest(
        question=f"Validate code against {document_path}",
        mode=QueryMode.CODE_VS_DOC,
        validate=True,
        scope=scope,  # Now uses the scope parameter
    )
    response = query_router.query(request)
    _print_validation_report(response.validation_report)


@app.command(help="Validate documents against actual code")
def validate_docs(
    document_path: str = typer.Argument(...),
    scope: str = typer.Option("all", "--scope"),
):
    """Validate DOCUMENTS against actual CODE."""
    request = QueryRequest(
        question=f"Validate {document_path} against code",
        mode=QueryMode.DOC_VS_CODE,
        validate=True,
        scope=scope,  # Now uses the scope parameter
    )
    response = query_router.query(request)
    _print_validation_report(response.validation_report)


@app.command(help="Index documents into document graph")
def index_docs(
    repo_path: str = typer.Argument(...),
    force: bool = typer.Option(False, "--force", "-f"),
):
    """Index documents into the document graph."""
    from .document.document_updater import DocumentGraphUpdater
    updater = DocumentGraphUpdater(
        host=settings.DOC_MEMGRAPH_HOST,
        port=settings.DOC_MEMGRAPH_PORT,
        repo_path=Path(repo_path),
    )
    updater.run(force=force)


@app.command(help="Watch repository for changes (code + documents)")
def watch(repo_path: str | None = typer.Option(None, "--repo-path")):
    """Start real-time updater for both code and document graphs."""
    from .realtime_updater import start_realtime_updater
    repo_path = repo_path or settings.TARGET_REPO_PATH
    start_realtime_updater(repo_path)
```

### 11.2 MCP Tools (Extending MCPToolsRegistry)

```python
# codebase_rag/mcp/tools.py (additions to MCPToolsRegistry)

# Add to __init__:

self._tools[cs.MCPToolName.QUERY_DOCUMENT_GRAPH] = ToolMetadata(
    name=cs.MCPToolName.QUERY_DOCUMENT_GRAPH,
    description="Query the DOCUMENT graph/vector ONLY.",
    input_schema=MCPInputSchema(
        type=cs.MCPSchemaType.OBJECT,
        properties={
            cs.MCPParamName.NATURAL_LANGUAGE_QUERY: MCPInputSchemaProperty(
                type=cs.MCPSchemaType.STRING,
                description="Natural language query about documentation",
            )
        },
        required=[cs.MCPParamName.NATURAL_LANGUAGE_QUERY],
    ),
    handler=self.query_document_graph,
    returns_json=True,
)

self._tools[cs.MCPToolName.QUERY_BOTH_GRAPHS] = ToolMetadata(
    name=cs.MCPToolName.QUERY_BOTH_GRAPHS,
    description="Query BOTH code and document graphs, merge results.",
    input_schema=MCPInputSchema(
        type=cs.MCPSchemaType.OBJECT,
        properties={
            cs.MCPParamName.NATURAL_LANGUAGE_QUERY: MCPInputSchemaProperty(
                type=cs.MCPSchemaType.STRING,
                description="Natural language query",
            )
        },
        required=[cs.MCPParamName.NATURAL_LANGUAGE_QUERY],
    ),
    handler=self.query_both_graphs,
    returns_json=True,
)

self._tools[cs.MCPToolName.VALIDATE_CODE_AGAINST_SPEC] = ToolMetadata(
    name=cs.MCPToolName.VALIDATE_CODE_AGAINST_SPEC,
    description="Validate CODE against DOCUMENT specifications.",
    input_schema=MCPInputSchema(
        type=cs.MCPSchemaType.OBJECT,
        properties={
            cs.MCPParamName.SPEC_DOCUMENT_PATH: MCPInputSchemaProperty(
                type=cs.MCPSchemaType.STRING,
                description="Path to specification document",
            )
        },
        required=[cs.MCPParamName.SPEC_DOCUMENT_PATH],
    ),
    handler=self.validate_code_against_spec,
    returns_json=True,
)

self._tools[cs.MCPToolName.VALIDATE_DOC_AGAINST_CODE] = ToolMetadata(
    name=cs.MCPToolName.VALIDATE_DOC_AGAINST_CODE,
    description="Validate DOCUMENT against actual CODE.",
    input_schema=MCPInputSchema(
        type=cs.MCPSchemaType.OBJECT,
        properties={
            cs.MCPParamName.DOCUMENT_PATH: MCPInputSchemaProperty(
                type=cs.MCPSchemaType.STRING,
                description="Path to document to validate",
            )
        },
        required=[cs.MCPParamName.DOCUMENT_PATH],
    ),
    handler=self.validate_document_against_code,
    returns_json=True,
)

# Handler implementations:

async def query_document_graph(self, natural_language_query: str, top_k: int = 5) -> dict:
    """Query the DOCUMENT graph/vector ONLY."""
    from ..shared.query_router import QueryRequest, QueryMode
    from dataclasses import asdict
    request = QueryRequest(question=natural_language_query, mode=QueryMode.DOCUMENT_ONLY, top_k=top_k)
    response = self._query_router.query(request)
    return asdict(response)  # Use asdict() for dataclass serialization

async def query_both_graphs(self, natural_language_query: str, top_k: int = 5) -> dict:
    """Query BOTH code and document graphs, merge results."""
    from ..shared.query_router import QueryRequest, QueryMode
    from dataclasses import asdict
    request = QueryRequest(question=natural_language_query, mode=QueryMode.BOTH_MERGED, top_k=top_k)
    response = self._query_router.query(request)
    return asdict(response)  # Use asdict() for dataclass serialization

async def validate_code_against_spec(self, spec_document_path: str, scope: str = "all") -> dict:
    """Validate CODE against DOCUMENT specifications."""
    from ..shared.query_router import QueryRequest, QueryMode
    from dataclasses import asdict
    request = QueryRequest(
        question=f"Validate code against {spec_document_path}",
        mode=QueryMode.CODE_VS_DOC,
        validate=True,
        scope=scope,
    )
    response = self._query_router.query(request)
    return asdict(response)  # Use asdict() for dataclass serialization

async def validate_document_against_code(self, document_path: str, scope: str = "all") -> dict:
    """Validate DOCUMENT against actual CODE."""
    from ..shared.query_router import QueryRequest, QueryMode
    from dataclasses import asdict
    request = QueryRequest(
        question=f"Validate {document_path} against code",
        mode=QueryMode.DOC_VS_CODE,
        validate=True,
        scope=scope,
    )
    response = self._query_router.query(request)
    return asdict(response)  # Use asdict() for dataclass serialization
```

### 11.3 Constants Additions

```python
# codebase_rag/constants.py (additions)

class MCPToolName(StrEnum):
    # ... existing tools ...
    QUERY_DOCUMENT_GRAPH = "query_document_graph"
    QUERY_BOTH_GRAPHS = "query_both_graphs"
    VALIDATE_CODE_AGAINST_SPEC = "validate_code_against_spec"
    VALIDATE_DOC_AGAINST_CODE = "validate_doc_against_code"
    INDEX_DOCUMENTS = "index_documents"


class MCPParamName(StrEnum):
    # ... existing params ...
    SPEC_DOCUMENT_PATH = "spec_document_path"
    DOCUMENT_PATH = "document_path"
```

---

## 12. Implementation Phases

### Phase 1: Foundation + Test Infrastructure (Week 1-2)

| Task | Description | Effort |
|------|-------------|--------|
| Add DOC_* config | Extend config.py with document graph settings | 0.5 days |
| Document graph docker | Add memgraph-doc container to docker-compose | 0.5 days |
| Document graph schema | Create init.cypherl for document graph (including Chunk nodes) | 1 day |
| **Test infrastructure** | **conftest.py, fixtures, mock utilities** | **1 day** |
| Extractor base class | BaseDocumentExtractor with async support | 1 day |
| Markdown extractor | .md, .rst extraction with error handling | 2 days |
| **Security: Input validation** | **Validate file paths, prevent path traversal** | **0.5 days** |

**Milestone:** Document graph infrastructure + basic test framework ready

### Phase 2: Document Ingestion + Unit Tests (Week 3-5 → 4 weeks)

| Task | Description | Effort |
|------|-------------|--------|
| Document updater | DocumentGraphUpdater with incremental support | 2 days |
| **Namespace isolation implementation** | **namespace_config.py + prefix logic** | **1 day** |
| **Semantic chunker integration** | **Wire SemanticDocumentChunker into embedding pipeline** | **1.5 days** |
| **Chunk embedding storage** | **Store all chunk embeddings in Chunk nodes** | **1 day** |
| Content versioning | Hash-based change detection with extractor_version | 1 day |
| **Error handling infrastructure** | **Dead letter queue, ExtractionError, recovery** | **1.5 days** |
| **Unit tests for ingestion** | **Edge cases, error paths, encoding handling** | **2 days** |
| CLI index-docs command | Index documents CLI | 1 day |
| PDF/DOCX extractors | Optional format support (conditional import) | **5 days** (increased for edge cases) |
| **Performance testing** | **Large file processing, memory usage benchmarks** | **1 day** |

**Milestone:** Documents can be indexed with comprehensive unit test coverage

### Phase 3: Query Router + Integration Tests (Week 6-7)

| Task | Description | Effort |
|------|-------------|--------|
| QueryMode enum | Define 5 query modes (fix CODE_VS_DOC naming) | 0.5 days |
| Query router | Route queries based on mode with isolation verification | 2.5 days |
| **Integration tests** | **Test routing with both graphs, namespace-prefix mode** | **2 days** |
| CLI query commands | Add query-code, query-docs, query-all | 1 day |
| MCP document tools | Expose document query modes | 1 day |
| Source attribution | Clear source labeling in responses | 1 day |
| **Security: Query validation** | **Prevent Cypher injection, validate inputs** | **0.5 days** |

**Milestone:** All 5 query modes functional with integration test coverage

### Phase 4: Validation System + E2E Tests (Week 8-10 → 10.5 weeks)

| Task | Description | Effort |
|------|-------------|--------|
| Validation engine | Base validation with LLM orchestration | 2 days |
| **Cost estimation API** | **ValidationTriggerAPI with provider-agnostic pricing** | **1.5 days** |
| **Result caching** | **ValidationCache with proper mode-based invalidation** | **1.5 days** |
| Code→Doc validation | Validate code against specs | 2 days |
| Doc→Code validation | Validate docs against code | 2 days |
| **LLM service integration** | **Provider-specific pricing, response handling** | **1 day** |
| **E2E validation tests** | **Full validation pipeline with LLM mocking** | **2 days** |
| CLI validation commands | validate-code, validate-docs | 1 day |

**Milestone:** Bidirectional validation complete with end-to-end test coverage

### Phase 5: Real-Time Updater + Performance Tests (Week 11)

| Task | Description | Effort |
|------|-------------|--------|
| Unified watcher | Extend existing realtime_updater with async handling | 3 days |
| File classification | Route to appropriate updater | 1 day |
| Debounce mechanism | Prevent duplicate processing | 0.5 days |
| **Performance testing** | **Benchmark document processing speeds** | **1 day** |
| End-to-end real-time testing | File event simulation | 1 day |

**Milestone:** Real-time updates for both graphs with performance benchmarks

### Phase 6: Documentation & Release (Week 12)

| Task | Description | Effort |
|------|-------------|--------|
| **Security: Final audit** | **Bandit scan, dependency check, penetration test** | **1 day** |
| **Comprehensive documentation** | **Update README, API docs, migration guide** | **1.5 days** |
| **Release preparation** | **Version bump, changelog, final validation** | **1 day** |
| Migration testing | Backward compatibility, single-to-dual graph migration | 1 day |

**Milestone:** Production-ready release with complete documentation

---

## 13. Testing Strategy

### 13.1 Test Categories

| Category | Scope | Tools | Coverage Target |
|----------|-------|-------|-----------------|
| **Unit Tests** | Individual components | pytest | 85% average |
| **Integration Tests** | Component interactions | pytest + Docker + testcontainers | 75% average |
| **E2E Tests** | Full pipeline | pytest + real documents | 70% average |
| **Performance Tests** | Speed and resource usage | pytest-benchmark | N/A |
| **Security Tests** | Input validation | bandit, manual review | 100% critical paths |

### 13.2 Risk-Based Coverage Strategy

Instead of rigid percentage targets, use risk-based approach:

| Path Type | Coverage | Examples |
|-----------|----------|----------|
| **Critical paths** | 95%+ | Query routing, file classification, isolation boundaries |
| **Business logic** | 85%+ | Extractor logic, validation rules, chunking |
| **Error handling** | 70%+ | Exception paths, malformed files, missing dependencies |
| **External dependencies** | Mock | LLM calls, database connections |

### 13.3 Required Test Fixtures

```python
# codebase_rag/tests/conftest.py (additions)

@pytest.fixture
def temp_document_repo(tmp_path: Path) -> Path:
    """Create temporary repo with standard document structures."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "api.md").write_text("# API Reference\n## GET /users\n...")
    (docs / "guide.md").write_text("# User Guide\n...")
    return tmp_path

@pytest.fixture
def mock_document_ingestor() -> _MockIngestor:
    """Mock MemgraphIngestor for document graph."""
    return _MockIngestor(port=7688)

@pytest.fixture
def document_test_files(test_data_path: Path) -> dict[str, Path]:
    """Standardized test documents for different formats."""
    return {
        "markdown": test_data_path / "sample.md",
        "pdf": test_data_path / "sample.pdf",
        "docx": test_data_path / "sample.docx",
        "malformed": test_data_path / "corrupted.pdf",
    }

@pytest.fixture
def mock_llm_responses() -> dict[str, str]:
    """Mock LLM responses for validation testing."""
    return {
        "extract_requirements": "...",
        "verify_claim": "...",
    }
```

### 13.4 Comprehensive Test Scenarios

#### Extractor Edge Cases

```python
# Required test scenarios for all extractors

def test_extractor_handles_empty_file():
    """Empty documents should not crash pipeline."""

def test_extractor_handles_malformed_content():
    """Corrupted/malformed documents should raise ExtractionError."""

def test_extractor_respects_file_size_limits():
    """Large files should be truncated or rejected based on config."""

def test_extractor_handles_different_encodings():
    """UTF-8, UTF-16, Latin-1 should be handled appropriately."""

def test_extractor_async_does_not_block_event_loop():
    """Large file extraction should not block for >1 second."""

def test_extractor_batch_processing_is_concurrent():
    """Batch extraction should use asyncio.gather for parallelism."""
```

#### Query Router Isolation

```python
# Critical: Verify graph isolation

def test_code_only_never_queries_document_graph(mock_document_ingestor):
    """CODE_ONLY mode must be isolated from document graph."""
    # Verify mock_document_ingestor.execute() was never called

def test_document_only_never_queries_code_graph(mock_code_ingestor):
    """DOCUMENT_ONLY mode must be isolated from code graph."""

def test_both_merged_preserves_source_attribution():
    """Results must clearly indicate source graph for each item."""

def test_validation_modes_use_correct_authority():
    """CODE_VS_DOC uses docs as truth, DOC_VS_CODE uses code as truth."""
```

#### Real-time Updater

```python
# File watching tests

def test_file_classification_accuracy():
    """Files should be correctly classified as code vs document."""

def test_debounce_prevents_duplicate_processing():
    """Rapid file changes should be debounced properly."""

def test_watcher_handles_directory_creation():
    """New directories with documents should be processed."""

def test_watcher_respects_exclude_patterns():
    """Files in excluded directories (.git, node_modules) should be ignored."""
```

### 13.5 CI Pipeline Tiers

| Tier | Trigger | Tests | Max Duration |
|------|---------|-------|--------------|
| **Fast unit** | Every commit | Unit tests only | 2 minutes |
| **Integration** | PR merge | Unit + Integration | 10 minutes (parallel) |
| **E2E** | Nightly/releases | Full suite | 30 minutes |
| **Security** | Every commit | bandit + dependency check | 1 minute |
    """DOCUMENT_ONLY mode must NOT query code graph."""
    pass

def test_both_merged_has_clear_attribution():
    """BOTH_MERGED must label each source."""
    pass

def test_code_vs_doc_validation_direction():
    """CODE_VS_DOC must use doc as source of truth."""
    pass

def test_doc_vs_code_validation_direction():
    """DOC_VS_CODE must use code as source of truth."""
    pass
```

---

## 14. Migration Path

### 14.1 For Existing Users

```bash
# Step 0: PRE-MIGRATION - Backup existing code graph (CRITICAL)
docker exec memgraph-code mgbackup create /var/lib/memgraph/backup
# Alternative: memgraph-snapshot if available

# Step 1: Update to new version
git pull origin main

# Step 2: Update dependencies
uv sync

# Step 3: Add document graph config to .env
cat >> .env << EOF

# Document GraphRAG
DOC_MEMGRAPH_HOST=localhost
DOC_MEMGRAPH_PORT=7688
DOC_MEMGRAPH_USERNAME=cgr
DOC_MEMGRAPH_PASSWORD=cgr
EOF

# Step 4: Start document graph container
docker compose up -d memgraph-doc

# Step 5: Index documents (optional)
cgr index-docs /path/to/repo

# Step 6: Verify both graphs running
cgr doctor
```

### 14.1.1 Rollback Procedure

```bash
# If upgrade fails, rollback to previous version:
docker compose down
docker volume restore memgraph_code_data  # Restore from backup
git checkout previous-version
uv sync
docker compose up -d memgraph-code
```

### 14.1.2 Disable Document Graph (Optional)

```bash
# For users who don't need document features:
echo "DOC_ENABLED=false" >> .env
# Only code graph will be active
```

### 14.2 Backward Compatibility

| Feature | Compatibility | Notes |
|---------|---------------|-------|
| Existing CLI commands | ✅ Full | No breaking changes |
| Code graph queries | ✅ Full | Unchanged behavior |
| MCP tools | ✅ Full | Existing tools still work |
| Configuration | ✅ Full | New DOC_* vars are optional |

---

## 15. Appendix

### 15.1 Glossary

| Term | Definition |
|------|------------|
| **Code Graph** | Memgraph instance storing code structure (functions, classes, calls) |
| **Document Graph** | Memgraph instance storing document structure (sections, references) |
| **Query Mode** | Explicit routing instruction (CODE_ONLY, DOCUMENT_ONLY, etc.) |
| **Bidirectional Validation** | Validation in both directions (Code→Doc and Doc→Code) |
| **On-Demand Validation** | Validation at query time, not index time |

### 15.2 Decision Log

| Decision | Date | Rationale |
|----------|------|-----------|
| Separate modules, same codebase | 2026-03-29 | 40-50% code reuse (embedding, vector, MCP, config); novel concepts require new implementation |
| Parallel database instances | 2026-03-29 | Independent scaling, no schema conflicts; namespace-prefix fallback for simpler deployments |
| No pre-classification | 2026-03-29 | Error-prone, creates false confidence |
| On-demand validation | Saves 60-80% on LLM costs by only validating queried documents; calculation: 1000 docs × $0.01/pre-validation = $10 vs avg 100 queried × $0.01 = $1 (assuming 10% query coverage) | 2026-03-30
| Explicit query modes | 2026-03-29 | User specifies intent; not present in LightRAG or graphrag upstream |
| Reuse embeddings module | 2026-03-29 | Avoid duplication, maintain consistency |
| Follow existing patterns | 2026-03-29 | EmbeddingProvider, VectorBackend, MCPToolsRegistry |
| Async extractor interface | 2026-03-29 | Avoid blocking event loop for large documents |
| Semantic chunking | 2026-03-29 | Token-aware section-based chunks, not hardcoded character limits |
| Content versioning | 2026-03-29 | Hash-based change detection for incremental updates |
| Testing from day 1 | 2026-03-29 | Test infrastructure established in Phase 1, not deferred to Phase 6 |
| Risk-based coverage | 2026-03-29 | Different coverage targets for critical paths vs error handling |

### 15.3 References

- [Memgraph Documentation](https://memgraph.com/docs)
- [MCP Specification](https://modelcontextprotocol.io/)
- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Tree-sitter Documentation](https://tree-sitter.github.io/tree-sitter/)
- [LightRAG Memgraph Implementation](https://github.com/HKUDS/LightRAG/blob/main/lightrag/kg/memgraph_impl.py)
- [CocoIndex Flow Pattern](https://github.com/cocoindex-io/cocoindex/blob/main/python/cocoindex/flow.py)

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-01-15 | Architecture Team | Initial specification |
| 2.0 | 2026-03-29 | Architecture Team | Implementation-ready update: aligned with existing patterns, added specific file paths, code examples following existing codebase patterns |
| 2.1 | 2026-03-29 | Architecture Team + 10 Parallel Workers | Comprehensive review update: corrected code reuse estimate (40-50%), added namespace-prefix fallback (§5.4), semantic chunking (§7.6), async extractor interface (§7.7), error handling infrastructure (§7.8), content versioning (§7.9), validation trigger API (§8.4), result caching (§8.5), restructured implementation phases with testing from day 1 (§12), risk-based coverage strategy (§13) |
| 2.2 | 2026-03-30 | Architecture Team | **Critical fixes from parallel review:** Fixed vector index Cypher syntax (§5.2), fixed enum name mismatch CODE_VS_DOC (§6.1), integrated SemanticDocumentChunker into embedding pipeline (§7.5), fixed async/sync bug in retry_with_backoff (§7.8), fixed asyncio.run() in sync callback (§9.1), added CodeGraphHashComputer (§8.5), fixed cache invalidation with structured index (§8.6), substantiated 60-80% cost savings calculation (Decision Log) |
| 2.3 | 2026-03-30 | Architecture Team + 10 Round-Robin Workers | **10-worker parallel review updates:** Fixed enum inconsistency CODE_VS_DOC throughout (§6), added Chunk node type and vector index (§5.2), added content_hash/section_hashes/extractor_version to Document schema, replaced invalid array index with containment query, added backtick escaping for namespace labels (§5.4), extended timeline to 12 weeks, added missing tasks (chunk integration, cost estimation, result caching, error handling), integrated security testing earlier, added performance testing for ingestion |
| 2.4 | 2026-03-30 | Architecture Team + Round-Robin Workers (crush/opencode/qwen/goose/kimi/claude) | **Consolidated 10-worker review fixes:** Added workspace isolation pattern (§3.2), added protocol-based dependency decoupling (§4.2), added provider-agnostic pricing (§8.4), fixed cache invalidation bugs (§8.6), added path traversal prevention (§7.7), fixed DLQ filename collision bugs (§7.8), added asyncio timer patterns (§9.1), added missing test fixtures (§13.3), fixed QueryMode enum consistency (§11), added config field validators (§10.2), extended timeline realism (§12), added migration rollback procedures (§14) |
| 2.5 | 2026-03-30 | Architecture Team + 10 Round-Robin Workers | **Reference project alignment:** Added Appendix C with LightRAG workspace isolation pattern (C.1), DocStatusStorage pattern (C.2), Storage factory pattern (C.3), CocoIndex flow pattern (C.4), fixed dataclass serialization (asdict vs model_dump), added scope parameter to QueryRequest, fixed enum inconsistency in CLI/MCP handlers, added top_k to MCP tools |

---

## Appendix B: Consolidated Worker Review Fixes (v2.4)

### B.1 Architecture Fixes (Worker 0: LightRAG Alignment)

**Missing Components Identified:**

| Component | Status | Fix Required |
|-----------|--------|--------------|
| Workspace isolation | ❌ Missing | Add `workspace` field to Document schema; use single instance with namespace prefixes |
| DOC_STATUS_STORAGE | ❌ Missing | Add `document/kv_storage.py` for pipeline state tracking |
| Orchestrator class | ❌ Missing | Create `DocumentGraphRAG` class in `document/lightrag.py` |
| Entity extraction | ❌ Missing | Add `document/operate.py` for entity/relationship extraction |
| Storage factory | ❌ Missing | Add `document/storage_factory.py` following LightRAG pattern |

**Protocol-Based Dependency Decoupling:**

```python
# codebase_rag/document/protocols.py (NEW)
from typing import Protocol, runtime_checkable

@runtime_checkable
class GraphStorage(Protocol):
    """Protocol to avoid circular imports from graph_service."""
    def query(self, cypher: str, params: dict) -> list: ...
    def ensure_node_batch(self, label: str, props: dict) -> None: ...
    def execute_write(self, cypher: str, params: dict) -> None: ...

# Update QueryRouter to use protocols
class QueryRouter:
    def __init__(
        self,
        code_graph: GraphStorage,  # Protocol, not MemgraphIngestor
        doc_graph: GraphStorage,   # Protocol
        ...
    ): ...
```

### B.2 Validation System Fixes (Worker 3)

**Signature Mismatch Fix:**

```python
# codebase_rag/shared/validation/code_vs_doc.py
def validate(
    self,
    query: str | None = None,
    document_path: str | None = None
) -> ValidationReport:
    """
    Validate code against document specs.

    Args:
        query: Natural language query (used to find relevant docs)
        document_path: Specific document to validate (optional)
    """
    if document_path:
        requirements = self._extract_requirements(document_path)
    elif query:
        doc_paths = self._find_relevant_docs(query)
        requirements = [self._extract_requirements(p) for p in doc_paths]
    else:
        raise ValueError("Either query or document_path required")
```

**Provider-Agnostic Pricing:**

```python
# codebase_rag/shared/validation/api.py
class ValidationTriggerAPI:
    def __init__(self, llm_provider: str = "google"):
        self._pricing = self._get_provider_pricing(llm_provider)

    def _get_provider_pricing(self, provider: str) -> dict:
        """Provider-specific token pricing (USD per 1K tokens)."""
        PRICING_TABLE = {
            "openai": {"input": 0.005, "output": 0.015},
            "google": {"input": 0.00025, "output": 0.0005},
            "ollama": {"input": 0.0, "output": 0.0},
        }
        return PRICING_TABLE.get(provider, PRICING_TABLE["google"])
```

**Division by Zero Fix:**

```python
# Use max(total, 1) to prevent crash on empty validation
accuracy_score=passed / max(total, 1)
```

### B.3 Cache Invalidation Fix (Worker 3)

```python
# codebase_rag/shared/validation/cache.py
class ValidationCache:
    def set(
        self,
        key: str,
        report: ValidationReport,
        document_path: str,
        mode: str,  # Store mode directly in cache dict
        ttl_hours: int = None
    ) -> None:
        self._cache[key] = {
            "report": report,
            "cached_at": datetime.now(UTC),
            "expires_at": datetime.now(UTC) + timedelta(hours=ttl or 24),
            "document_path": document_path,
            "mode": mode,
        }

    def invalidate_code_graph(self) -> int:
        """Invalidate all CODE_VS_DOC results."""
        keys_to_remove = [
            k for k, v in self._cache.items()
            if v.get("mode") == "CODE_VS_DOC"
        ]
        for key in keys_to_remove:
            self._cache.pop(key, None)
        return len(keys_to_remove)
```

### B.4 Security Fixes (Worker 8)

**Path Traversal Prevention:**

```python
# codebase_rag/document/extractors/base.py
def _validate_path(self, file_path: Path, repo_root: Path) -> Path:
    """Validate path is within repo boundaries."""
    resolved = file_path.resolve()
    repo_resolved = repo_root.resolve()
    if not str(resolved).startswith(str(repo_resolved)):
        raise SecurityError(f"Path traversal detected: {file_path}")
    # Add symlink check
    if resolved.is_symlink():
        resolved = self._validate_path(resolved.resolve(), repo_root)
    return resolved
```

**DLQ Filename Collision Fix:**

```python
# codebase_rag/document/error_handling.py
def _safe_error_filename(self, path: str) -> str:
    """Generate unique, safe filename for error file."""
    path_hash = hashlib.sha256(path.encode()).hexdigest()[:16]
    safe_name = re.sub(r'[^\w\-]', '_', Path(path).name)[:50]
    return f"{path_hash}_{safe_name}.error.json"

def enqueue(self, error: ExtractionError) -> None:
    filename = self._safe_error_filename(error.path)
    error_file = self.queue_path / filename
    # Use file locking
    with open(error_file, 'w') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(error.__dict__, f, indent=2)
```

**Error Classification Expansion:**

```python
RECOVERABLE_ERRORS = (
    ValueError, UnicodeDecodeError,
    OSError, IOError,
    json.JSONDecodeError,
)

FATAL_ERRORS = (
    PermissionError,
    MemoryError,
)

def _is_recoverable(self, error: Exception) -> bool:
    return isinstance(error, self.RECOVERABLE_ERRORS)
```

### B.5 Real-Time Updater Async Fixes (Worker 7)

**Async Timer Pattern:**

```python
# codebase_rag/realtime_updater.py
import asyncio

class CodeChangeEventHandler(FileSystemEventHandler):
    def __init__(...):
        self._debounce_timers: dict[str, asyncio.TimerHandle] = {}
        self._loop = asyncio.get_event_loop()

    def on_modified(self, event):
        if event.is_directory or not self._is_relevant(event.src_path):
            return

        rel_path = str(Path(event.src_path).relative_to(self.repo_path))

        with self.lock:
            if rel_path in self._debounce_timers:
                self._debounce_timers[rel_path].cancel()

            delay = min(self.debounce_seconds, self.max_wait_seconds)
            timer = self._loop.call_later(
                delay,
                self._process_debounced_change,
                rel_path
            )
            self._debounce_timers[rel_path] = timer
```

**File Classification and Routing:**

```python
def _classify_file(self, file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in self.code_extensions:
        return "code"
    elif suffix in self.doc_extensions or 'docs/' in str(file_path):
        return "document"
    elif file_path.name in ['pyproject.toml', 'package.json', 'Cargo.toml']:
        return "dependency"
    return "other"
```

### B.6 Query Mode Fixes (Worker 4)

**Enum Consistency Fix:**

```python
# Replace CODE_VALIDATED_AGAINST_DOC → CODE_VS_DOC everywhere
# Replace DOC_VALIDATED_AGAINST_CODE → DOC_VS_CODE everywhere
```

**Router Fallback:**

```python
# codebase_rag/shared/query_router.py
def __init__(...):
    self.doc_vector = doc_vector or get_vector_backend(config_key="DOC")
```

### B.7 Configuration Fixes (Worker 5)

**List Parsing Validator:**

```python
# codebase_rag/config.py
@field_validator("DOC_SUPPORTED_EXTENSIONS", mode="before")
@classmethod
def parse_doc_extensions(cls, v: str | list[str]) -> list[str]:
    """Parse comma-separated env var to list."""
    if isinstance(v, str):
        return [ext.strip().lower() for ext in v.split(",") if ext.strip()]
    return v

@field_validator("DOC_VECTOR_STORE_BACKEND")
@classmethod
def validate_doc_vector_backend(cls, v: str) -> str:
    """Validate document vector backend is supported."""
    allowed = {"memgraph", "qdrant"}
    if v.lower() not in allowed:
        raise ValueError(f"DOC_VECTOR_STORE_BACKEND must be one of: {allowed}")
    return v.lower()
```

### B.8 Testing Strategy Additions (Worker 6)

**Missing Fixtures:**

```python
# codebase_rag/tests/conftest.py

@pytest.fixture
def namespace_prefix_config():
    """Fixture for namespace-prefix isolation mode testing."""
    return {"ISOLATION_MODE": "namespace", "CODE_NAMESPACE": "code", "DOC_NAMESPACE": "doc"}

@pytest.fixture
def incremental_update_scenario(tmp_path):
    """Fixture simulating document content changes."""
    doc_path = tmp_path / "test.md"
    doc_path.write_text("# Initial content")
    initial_hash = hashlib.sha256(doc_path.read_text().encode()).hexdigest()
    doc_path.write_text("# Updated content")
    updated_hash = hashlib.sha256(doc_path.read_text().encode()).hexdigest()
    return {"path": doc_path, "initial_hash": initial_hash, "updated_hash": updated_hash}

@pytest.fixture
def validation_cache_with_expired_entries():
    """Fixture with mixed expired/fresh cache entries."""
    ...
```

**Critical Edge Cases:**

```python
def test_path_traversal_prevention():
    """File paths with '../' or absolute paths should be rejected."""
    pass

def test_concurrent_document_update_during_validation():
    """Validation should handle concurrent modifications safely."""
    pass

def test_extractor_version_compatibility():
    """Documents indexed with old extractor versions should be re-indexed."""
    pass

def test_llm_cost_estimation_accuracy():
    """Cost estimates should accurately reflect actual LLM usage."""
    pass
```

### B.9 Implementation Phase Fixes (Worker 9)

**Missing Tasks:**

| Phase | Task | Effort |
|-------|------|--------|
| Phase 1 | Namespace isolation implementation | 1 day |
| Phase 2 | Dual vector backend support for documents | 1 day |
| Phase 4 | LLM service integration for validation | 1 day |

**Timeline Adjustments:**

| Phase | Current | Adjusted | Reason |
|-------|---------|----------|--------|
| Phase 2 | 3 weeks | 4 weeks | PDF/DOCX edge cases |
| Phase 4 | 3 weeks | 3.5 weeks | LLM integration buffer |

**Migration Additions:**

```bash
# Pre-migration: Backup existing code graph
docker exec memgraph-code memgraph-snapshot create

# Rollback procedure if upgrade fails
docker compose down
docker volume restore memgraph_code_data
git checkout previous-version
```

---

**END OF SPECIFICATION**
---

## Appendix C: Reference Project Patterns (v2.5)

### C.1 LightRAG Workspace Isolation Pattern

**Source:** `/home/zealy/github/HKUDS/LightRAG/lightrag/kg/memgraph_impl.py`

LightRAG implements multi-tenant workspace isolation using Memgraph label prefixes. This pattern should be adopted for Document GraphRAG:

```python
# LightRAG pattern for workspace isolation
class MemgraphStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        # Priority: 1) MEMGRAPH_WORKSPACE env 2) user arg 3) default 'base'
        memgraph_workspace = os.environ.get("MEMGRAPH_WORKSPACE")
        if memgraph_workspace and memgraph_workspace.strip():
            workspace = memgraph_workspace
        if not workspace or not str(workspace).strip():
            workspace = "base"
        super().__init__(namespace=namespace, workspace=workspace, ...)

    def _get_workspace_label(self) -> str:
        """Return sanitized workspace label safe for Cypher queries.

        Escapes backticks by doubling them to prevent Cypher injection.
        """
        workspace = self.workspace.strip()
        if not workspace:
            return "base"
        return workspace.replace("`", "``")
```

**Key Adoption Points:**

| Pattern | LightRAG | CGR Adoption |
|---------|----------|--------------|
| Workspace via env var | `MEMGRAPH_WORKSPACE` | `DOC_WORKSPACE` |
| Default workspace | `"base"` | `"default"` |
| Label escaping | `workspace.replace("\`", "\`\`")` | Already in §5.4 |
| Priority chain | env → arg → default | Same |

### C.2 LightRAG Document Status Storage

**Source:** `/home/zealy/github/HKUDS/LightRAG/lightrag/base.py`

LightRAG tracks document processing status with `DocStatusStorage`. This is critical for incremental updates:

```python
# LightRAG DocStatus pattern
class DocStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"

@dataclass
class DocProcessingStatus:
    """Status tracking for document pipeline."""
    doc_id: str
    status: DocStatus
    file_path: str
    chunks_list: list[str]
    chunks_count: int
    content_hash: str
    last_updated: datetime
```

**CGR Implementation:**

```python
# codebase_rag/document/status.py (NEW)
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class DocStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"

@dataclass
class DocProcessingStatus:
    """Document pipeline status tracking."""
    doc_id: str
    status: DocStatus
    file_path: str
    content_hash: str
    section_count: int
    error_message: str | None = None
    last_updated: datetime | None = None
```

### C.3 LightRAG Storage Factory Pattern

**Source:** `/home/zealy/github/HKUDS/LightRAG/lightrag/kg/__init__.py`

LightRAG uses a registry-based storage factory:

```python
# LightRAG storage factory pattern
STORAGES = {
    "NetworkXStorage": ".graph.networkx_impl",
    "Neo4JStorage": ".graph.neo4j_impl",
    "MemgraphStorage": ".graph.memgraph_impl",
    "QdrantStorage": ".vector.qdrant_impl",
    # ...
}

def lazy_external_import(module_path: str, object_name: str):
    """Lazily import storage implementation."""
    module = importlib.import_module(module_path, package=__name__)
    return getattr(module, object_name)
```

**CGR Adoption:** Already implemented in `codebase_rag/embeddings/__init__.py` - reuse this pattern for document extractors.

### C.4 CocoIndex Flow Pattern

**Source:** `/home/zealy/github/cocoindex-io/cocoindex/python/cocoindex/flow.py`

CocoIndex uses a declarative flow pattern for document processing:

```python
# CocoIndex pattern: Lazy evaluation with type safety
class _DataSliceState:
    """Lazy data slice with on-demand evaluation."""
    _lazy_lock: Lock | None = None
    _data_slice: _engine.DataSlice | None = None
    _data_slice_creator: Callable | None = None

    @property
    def engine_data_slice(self) -> _engine.DataSlice:
        if self._lazy_lock is None:
            return self._data_slice
        with self._lazy_lock:
            if self._data_slice is None:
                self._data_slice = self._data_slice_creator(None)
            return self._data_slice
```

**Adoption for CGR:** Use lazy evaluation for document processing to avoid memory bloat on large documents.

### C.5 Updated Module Structure (v2.5)

Based on LightRAG patterns, add these modules:

```
codebase_rag/document/
├── __init__.py
├── protocols.py           # NEW: GraphStorage, VectorStorage protocols
├── status.py              # NEW: DocStatus, DocProcessingStatus
├── storage_factory.py     # NEW: Lazy storage loading
├── extractors/            # (existing)
├── document_updater.py    # (existing)
└── chunking.py            # (existing)
```

### C.6 Summary of v2.5 Changes

| Change | Location | Description |
|--------|----------|-------------|
| Enum consistency | §11 | Fixed `CODE_VALIDATED_AGAINST_DOC` → `CODE_VS_DOC` |
| Dataclass serialization | §11.2 | Changed `model_dump()` → `asdict()` |
| QueryRequest.scope | §6.2 | Added `scope` field for validation |
| MCP top_k parameter | §11.2 | Added `top_k` to document query tools |
| LightRAG patterns | Appendix C | Workspace isolation, DocStatus, Storage factory |
| CocoIndex patterns | Appendix C | Lazy evaluation flow pattern |
