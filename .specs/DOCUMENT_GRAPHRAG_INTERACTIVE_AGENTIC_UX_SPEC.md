# Document GraphRAG Integration with Interactive Agentic UX - Implementation Specification

**Version:** 1.2
**Status:** Implementation-Ready (with noted gaps)
**Last Updated:** 2026-04-01
**Author:** Code-Graph-RAG Architecture Team

---

## Executive Summary

This specification defines the **implementation-ready architecture** for integrating Document GraphRAG with the existing interactive agentic UX for CLI. It provides concrete code patterns, integration points, and step-by-step implementation guidance.

### Key Integration Points

| Component | Existing | Document GraphRAG Addition | Integration Status |
|-----------|----------|---------------------------|-------------------|
| **Query Router** | `query_graph` tool | `query_document_graph`, `query_both_graphs` | ✅ Implemented with vector search |
| **Validation** | N/A | `validate_code_against_spec`, `validate_doc_against_code` | ✅ Implemented with cost estimation |
| **Document Indexing** | N/A | `index_documents` CLI command | ✅ Implemented (`DocumentGraphUpdater` exists) |
| **Agent Tools** | 10 code tools | 5 document tools | ✅ All 5 registered with agent |
| **Session Context** | Code-only context | Code + document context | ✅ Implemented in SessionState |

### Implementation Status (Verified 2026-04-01)

| Module | Path | Status | Notes |
|--------|------|--------|-------|
| `QueryRouter` | `codebase_rag/shared/query_router.py` | ✅ Implemented | All 5 modes with vector search integration |
| `ValidationTriggerAPI` | `codebase_rag/shared/validation/api.py` | ✅ Implemented | Cost estimation with provider pricing |
| `ValidationCache` | `codebase_rag/shared/validation/cache.py` | ✅ Implemented | TTL, invalidation, `invalidate_all_code_dependent()` |
| `CodeVsDocValidator` | `codebase_rag/shared/validation/code_vs_doc.py` | ✅ Implemented | Regex-based requirement extraction |
| `DocVsCodeValidator` | `codebase_rag/shared/validation/doc_vs_code.py` | ✅ Implemented | Regex-based claim extraction |
| `DocumentGraphUpdater` | `codebase_rag/document/document_updater.py` | ✅ Implemented | Fully functional, `run()` returns stats dict |
| `document_semantic_search` | `codebase_rag/document/tools/document_search.py` | ✅ Implemented | Uses `vector_search.search` with `'doc_embeddings'` index |
| MCP Document Tools | `codebase_rag/mcp/tools.py` | ✅ Implemented | All 5 tools registered with QueryRouter |
| CLI Commands | `codebase_rag/cli.py` | ✅ Implemented | All 5 commands working |
| Agent Document Tools | `codebase_rag/tools/document_*.py` | ✅ Implemented | 5 tools: query, both, validate_code, validate_doc, index |

---

## 1. Architecture Overview

### 1.1 System Integration Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLI Agentic UX with Document Support            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  USER INPUT (Natural Language)                                          │
│  "Does our code implement all endpoints in the API spec?"              │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Agent Loop (main.py:_run_agent_response_loop)                  │   │
│  │  • Receives user question                                       │   │
│  │  • Routes to appropriate tools based on intent                  │   │
│  │  • Handles tool approvals for write operations                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ├──────────────────────────┬──────────────────────────┐        │
│         ▼                          ▼                          ▼        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │ Code Tools      │    │ Document Tools  │    │ Validation      │    │
│  │ • query_graph   │    │ • query_        │    │ Tools           │    │
│  │ • read_file     │    │   document_graph│    │ • validate_     │    │
│  │ • replace_code  │    │ • query_        │    │   code_against_ │    │
│  │ • semantic_     │    │   both_graphs   │    │   spec          │    │
│  │   search        │    │ • index_        │    │ • validate_     │    │
│  │                 │    │   documents     │    │   doc_against_  │    │
│  │                 │    │ • read_         │    │   code          │    │
│  │                 │    │   document      │    │                 │    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│         │                          │                          │        │
│         └──────────────────────────┼──────────────────────────┘        │
│                                    ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Query Router (shared/query_router.py)                          │   │
│  │  • Routes based on QueryMode enum                               │   │
│  │  • CODE_ONLY | DOCUMENT_ONLY | BOTH_MERGED                     │   │
│  │  • CODE_VS_DOC | DOC_VS_CODE                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ├──────────────────────────┬──────────────────────────┐        │
│         ▼                          ▼                          ▼        │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    │
│  │ Code Graph      │    │ Document Graph  │    │ Validation      │    │
│  │ Memgraph:7687   │    │ Memgraph:7688   │    │ Engine          │    │
│  │ • Functions     │    │ • Documents     │    │ • LLM-based     │    │
│  │ • Classes       │    │ • Sections      │    │ • Cost est.     │    │
│  │ • Calls         │    │ • Chunks        │    │ • Caching       │    │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow for Document Queries

```
User Question: "How do I use the authentication API?"
         │
         ▼
┌─────────────────────────────────────────┐
│  Agent receives question                │
│  (Pydantic AI run())                    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Agent decides to use query_document_   │
│  graph tool (based on training/prompt)  │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  query_document_graph tool handler      │
│  (mcp/tools.py or tools/document_       │
│   query.py)                             │
│  • Creates QueryRequest                 │
│  • mode=DOCUMENT_ONLY                   │
│  • question="How do I use auth API?"    │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  QueryRouter.query()                    │
│  • Routes to _query_document_only()     │
│  • Queries Document graph only          │
│  • Returns QueryResponse                │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Tool returns results to Agent          │
│  • Formatted as string                  │
│  • Includes source attribution          │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│  Agent generates final response         │
│  • Synthesizes document results         │
│  • Presents to user                     │
└─────────────────────────────────────────┘
```

---

## 2. Query Router Integration

### 2.1 QueryMode Enum (Already Defined)

**Location:** `codebase_rag/shared/query_router.py`

```python
class QueryMode(StrEnum):
    CODE_ONLY = "code_only"           # Existing code graph only
    DOCUMENT_ONLY = "document_only"   # Document graph only
    BOTH_MERGED = "both_merged"       # Query both, merge results
    CODE_VS_DOC = "code_vs_doc"       # Validate code against docs
    DOC_VS_CODE = "doc_vs_code"       # Validate docs against code
```

### 2.2 QueryRequest/Response Dataclasses (Already Defined)

**Location:** `codebase_rag/shared/query_router.py`

```python
@dataclass
class QueryRequest:
    question: str
    mode: QueryMode
    validate: bool = False
    include_metadata: bool = True
    top_k: int = 5
    scope: str = "all"  # For validation: "all", "sections", "claims"


@dataclass
class QueryResponse:
    answer: str
    sources: list[Source]
    mode: QueryMode
    validation_report: ValidationReport | None = None
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "sources": [s.to_dict() for s in self.sources],
            "mode": self.mode.value,
            "validation_report": self.validation_report.to_dict()
            if self.validation_report else None,
            "warnings": self.warnings,
        }
```

### 2.3 QueryRouter Class (Stub Exists - Needs Implementation)

**Location:** `codebase_rag/shared/query_router.py`

**Current Status:** Methods exist but return stub responses

**Implementation Required:**

```python
class QueryRouter:
    """Routes queries to appropriate graph(s) based on EXPLICIT mode."""

    def __init__(
        self,
        code_graph: MemgraphIngestor | None = None,
        doc_graph: MemgraphIngestor | None = None,
        code_vector: VectorBackend | None = None,
        doc_vector: VectorBackend | None = None,
    ):
        self.code_graph = code_graph
        self.doc_graph = doc_graph
        self.code_vector = code_vector
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

    def _query_document_only(self, request: QueryRequest) -> QueryResponse:
        """
        Query DOCUMENT graph/vector ONLY.
        
        Implementation:
        1. Generate embedding for question
        2. Query document vector index for similar chunks
        3. Retrieve full sections/documents
        4. Format results with source attribution
        """
        if not self.doc_graph:
            return QueryResponse(
                answer="Document graph is not available.",
                sources=[],
                mode=request.mode,
                warnings=["Document graph connection not configured"],
            )

        # Step 1: Generate embedding
        from ..embeddings import get_embedding_provider
        from ..config import settings
        
        provider = get_embedding_provider(
            provider=settings.EMBEDDING_PROVIDER,
            model_id=settings.EMBEDDING_MODEL,
        )
        question_embedding = provider.embed(request.question)

        # Step 2: Vector similarity search using Memgraph native vector_search module
        # Note: Parameter order is (index_name, limit, query_vector) per Memgraph v3.0+
        # IMPORTANT: Use actual index name 'doc_embeddings' (config.py:370), not 'chunk_embedding_index'
        # Output fields per Memgraph spec: node, distance, similarity
        # Note: node is already bound from YIELD, no need for MATCH (node:Chunk)
        chunks_query = """
        CALL vector_search.search('doc_embeddings', $top_k, $embedding)
        YIELD node, distance, similarity
        WITH node, distance, similarity
        WHERE node.workspace = $workspace
        MATCH (node)-[:BELONGS_TO_SECTION]->(s:Section)
        MATCH (s)<-[:CONTAINS_SECTION]-(d:Document)
        RETURN node.content as chunk_content, node.start_line, node.end_line,
               s.title as section_title, s.qualified_name as section_qn,
               d.path as doc_path, d.file_type, similarity
        ORDER BY similarity DESC
        """
        
        results = self.doc_graph.fetch_all(chunks_query, {
            'embedding': question_embedding,
            'top_k': request.top_k,
            'workspace': 'default',  # Configurable
        })

        # Step 3: Format response
        if not results:
            return QueryResponse(
                answer="No relevant documents found.",
                sources=[],
                mode=request.mode,
            )

        # Build answer from chunks
        answer_parts = ["**Relevant Documentation:**\n"]
        sources = []
        
        for i, result in enumerate(results, 1):
            answer_parts.append(f"\n{i}. **{result['section_title']}** ({Path(result['doc_path']).name})")
            answer_parts.append(f"   {result['chunk_content'][:200]}...")
            
            sources.append(Source(
                type="document",
                path=result['doc_path'],
                node_type="Chunk",
                qualified_name=result['section_qn'],
                line_range=(result['start_line'], result['end_line']),
            ))

        return QueryResponse(
            answer="\n".join(answer_parts),
            sources=sources,
            mode=request.mode,
        )

    def _query_both_merged(self, request: QueryRequest) -> QueryResponse:
        """Query BOTH graphs, merge results with clear attribution."""
        # Query both graphs in parallel
        code_response = self._query_code_only(request)
        doc_response = self._query_document_only(request)

        # Merge responses
        merged_sources = code_response.sources + doc_response.sources
        merged_answer = (
            f"**Code Results:**\n{code_response.answer}\n\n"
            f"**Document Results:**\n{doc_response.answer}"
        )

        return QueryResponse(
            answer=merged_answer,
            sources=merged_sources,
            mode=request.mode,
            warnings=code_response.warnings + doc_response.warnings,
        )
```

---

## 3. Document-Aware Tool System

### 3.1 New Tools Required

**Note:** ✅ Document search infrastructure already exists in `codebase_rag/document/tools/document_search.py`

> **Architecture Note (Review Finding):** The codebase has TWO tool patterns:
> 1. **MCP Tools** (`codebase_rag/mcp/tools.py`): Uses `MCPToolsRegistry` with `ToolMetadata` + handler methods. Document tools (query_document_graph, validate_code_against_spec, etc.) are registered here as stubs.
> 2. **Pydantic AI Tools** (`codebase_rag/tools/*.py`): Uses `@Tool` decorator factory pattern for code tools (semantic_search, replace_code, etc.).
>
> The spec proposes Pydantic AI factory pattern. Choose pattern based on target integration:
> - For MCP server → use MCPToolsRegistry pattern
> - For direct agent → use Pydantic AI Tool factories

**Location:** `codebase_rag/tools/document_query.py` (NEW FILE - needs creation)

**Recommended Return Type Pattern:** Follow existing `QueryGraphData` pattern from `mcp/tools.py`:

```python
class DocumentQueryData:
    """Structured response for document queries."""
    answer: str
    sources: list[dict]
    mode: str
    warnings: list[str]
```

```python
"""Document GraphRAG tools for interactive agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Tool

from ..shared.query_router import QueryMode, QueryRequest, QueryRouter

if TYPE_CHECKING:
    from ..services.graph_service import MemgraphIngestor


def create_query_document_graph_tool(
    query_router: QueryRouter,
) -> Tool:
    """
    Create query_document_graph tool for Pydantic AI agent.
    
    Usage: Query the DOCUMENT graph/vector ONLY.
    Example: "How do I use the authentication API?"
    """
    @Tool(
        name="query_document_graph",
        description="Query the DOCUMENT graph/vector ONLY. Use for questions about documentation, tutorials, guides, or API docs. Returns relevant document sections and chunks.",
    )
    async def query_document_graph(
        natural_language_query: str,
        top_k: int = 5,
    ) -> str:
        """Query document graph with natural language."""
        from dataclasses import asdict
        
        request = QueryRequest(
            question=natural_language_query,
            mode=QueryMode.DOCUMENT_ONLY,
            top_k=top_k,
        )
        response = query_router.query(request)
        
        # Format response for agent
        result = asdict(response)
        return f"Query Results:\n{result['answer']}\n\nSources: {len(result['sources'])} document(s) found"
    
    return query_document_graph


def create_query_both_graphs_tool(
    query_router: QueryRouter,
) -> Tool:
    """
    Create query_both_graphs tool for Pydantic AI agent.
    
    Usage: Query BOTH code and document graphs, merge results.
    Example: "Tell me everything about authentication"
    """
    @Tool(
        name="query_both_graphs",
        description="Query BOTH code and document graphs, merge results. Use for comprehensive searches spanning code and documentation. Results are labeled with their source.",
    )
    async def query_both_graphs(
        natural_language_query: str,
        top_k: int = 5,
    ) -> str:
        """Query both graphs with merged results."""
        from dataclasses import asdict
        
        request = QueryRequest(
            question=natural_language_query,
            mode=QueryMode.BOTH_MERGED,
            top_k=top_k,
        )
        response = query_router.query(request)
        
        result = asdict(response)
        return f"Merged Results:\n{result['answer']}\n\nSources: {len(result['sources'])} total (code + document)"
    
    return query_both_graphs
```

### 3.2 Validation Tools

**Location:** `codebase_rag/tools/document_validation.py` (NEW FILE)

```python
"""Document validation tools for interactive agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Tool

from ..shared.query_router import QueryMode, QueryRequest, QueryRouter
from ..shared.validation.api import ValidationTriggerAPI, ValidationRequest

if TYPE_CHECKING:
    from ..services.graph_service import MemgraphIngestor


def create_validate_code_against_spec_tool(
    query_router: QueryRouter,
    validation_api: ValidationTriggerAPI,
) -> Tool:
    """
    Create validate_code_against_spec tool.
    
    Usage: Validate CODE against DOCUMENT specifications.
    Document is SOURCE OF TRUTH.
    Example: "Does code implement all endpoints in OpenAPI spec?"
    """
    @Tool(
        name="validate_code_against_spec",
        description="Validate CODE against DOCUMENT specifications. Checks if the implementation matches the specification document. Returns validation report with discrepancies.",
    )
    async def validate_code_against_spec(
        spec_document_path: str,
        scope: str = "all",
        max_cost_usd: float = 0.50,
        dry_run: bool = False,
    ) -> str:
        """Validate code against specification document."""
        from dataclasses import asdict
        
        # Step 1: Request validation with cost estimation
        validation_request = ValidationRequest(
            document_path=spec_document_path,
            mode="CODE_VS_DOC",
            scope=scope,
            max_cost_usd=max_cost_usd,
            dry_run=dry_run,
        )
        
        trigger_result = await validation_api.request_validation(validation_request)
        
        if not trigger_result.accepted:
            return f"Validation not executed: {trigger_result.message}"
        
        # Step 2: Execute validation via QueryRouter
        request = QueryRequest(
            question=f"Validate code against {spec_document_path}",
            mode=QueryMode.CODE_VS_DOC,
            scope=scope,
        )
        response = query_router.query(request)
        
        result = asdict(response)
        report = result['validation_report']
        
        if report:
            return (
                f"Validation Report:\n"
                f"Direction: {report['direction']}\n"
                f"Total: {report['total']}, Passed: {report['passed']}, Failed: {report['failed']}\n"
                f"Accuracy: {report['accuracy_score']:.1%}\n\n"
                f"Results:\n{result['answer']}"
            )
        else:
            return f"Validation failed: {result['warnings']}"
    
    return validate_code_against_spec


def create_validate_doc_against_code_tool(
    query_router: QueryRouter,
    validation_api: ValidationTriggerAPI,
) -> Tool:
    """
    Create validate_doc_against_code tool.
    
    Usage: Validate DOCUMENT against actual CODE.
    Code is SOURCE OF TRUTH.
    Example: "Is docs/api.md still accurate?"
    """
    @Tool(
        name="validate_doc_against_code",
        description="Validate DOCUMENT against actual CODE. Checks if documentation accurately reflects the current code. Identifies outdated or incorrect documentation.",
    )
    async def validate_doc_against_code(
        document_path: str,
        scope: str = "all",
        max_cost_usd: float = 0.50,
        dry_run: bool = False,
    ) -> str:
        """Validate documentation against code."""
        from dataclasses import asdict
        
        # Step 1: Request validation with cost estimation
        validation_request = ValidationRequest(
            document_path=document_path,
            mode="DOC_VS_CODE",
            scope=scope,
            max_cost_usd=max_cost_usd,
            dry_run=dry_run,
        )
        
        trigger_result = await validation_api.request_validation(validation_request)
        
        if not trigger_result.accepted:
            return f"Validation not executed: {trigger_result.message}"
        
        # Step 2: Execute validation via QueryRouter
        request = QueryRequest(
            question=f"Validate {document_path} against code",
            mode=QueryMode.DOC_VS_CODE,
            scope=scope,
        )
        response = query_router.query(request)
        
        result = asdict(response)
        report = result['validation_report']
        
        if report:
            return (
                f"Validation Report:\n"
                f"Direction: {report['direction']}\n"
                f"Total: {report['total']}, Passed: {report['passed']}, Failed: {report['failed']}\n"
                f"Accuracy: {report['accuracy_score']:.1%}\n\n"
                f"Results:\n{result['answer']}"
            )
        else:
            return f"Validation failed: {result['warnings']}"
    
    return validate_doc_against_code
```

### 3.3 Document Indexing Tool

**Location:** `codebase_rag/tools/document_index.py` (NEW FILE)

```python
"""Document indexing tool for interactive agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_ai import Tool
from loguru import logger

if TYPE_CHECKING:
    from ..document.document_updater import DocumentGraphUpdater


def create_index_documents_tool(
    repo_path: Path,
    doc_updater_factory,  # Callable that creates DocumentGraphUpdater
) -> Tool:
    """
    Create index_documents tool.
    
    Usage: Index documents into the document graph.
    Example: "Index all markdown files in the docs folder"
    """
    @Tool(
        name="index_documents",
        description="Index documents into the document graph. Parses and ingests markdown, PDF, DOCX, and text files. Creates Document, Section, and Chunk nodes with embeddings.",
    )
    async def index_documents(
        force: bool = False,
        clean: bool = False,
    ) -> str:
        """Index documents into document graph."""
        try:
            # Create updater instance
            updater = doc_updater_factory()
            
            # Run indexing
            stats = updater.run(force=force)
            
            # Format results
            return (
                f"Document Indexing Complete:\n"
                f"  Total documents: {stats['total_documents']}\n"
                f"  Indexed: {stats['indexed']}\n"
                f"  Skipped: {stats['skipped']}\n"
                f"  Failed: {stats['failed']}\n"
                f"  Sections created: {stats['sections_created']}\n"
                f"  Chunks created: {stats['chunks_created']}"
            )
        except Exception as e:
            logger.exception("Document indexing failed")
            return f"Document indexing failed: {type(e).__name__}: {e}"
    
    return index_documents
```

---

## 4. Validation Workflows

### 4.1 Validation Architecture

**Location:** `codebase_rag/shared/validation/`

```
validation/
├── __init__.py
├── api.py              # ValidationTriggerAPI (cost estimation)
├── cache.py            # ValidationCache (result caching)
├── validator.py        # Base validation engine
├── code_vs_doc.py      # Code→Doc validation
└── doc_vs_code.py      # Doc→Code validation
```

### 4.2 Cost Estimation API (Already Implemented)

**Location:** `codebase_rag/shared/validation/api.py`

**Status:** ✅ Implemented with `_estimate_cost()` method

**Implementation Details:**
```python
class ValidationTriggerAPI:
    """
    API for triggering on-demand validation with cost estimation.

    Flow:
    1. User requests validation with budget
    2. System estimates cost (graph queries only, no LLM)
    3. If within budget, user confirms → validation executes
    4. Results cached keyed by: hash(query + graph_state)

    IMPORTANT: Never run validation without cost estimate first.
    """

    COST_PER_LLM_CALL_USD = 0.01  # Approximate cost per extraction call
    TOKENS_PER_CLAIM = 500  # Average tokens per claim extraction

    # Provider-specific token pricing (USD per 1K tokens)
    PRICING_TABLE: dict[str, dict[str, float]] = {
        "openai": {"input": 0.005, "output": 0.015},
        "google": {"input": 0.00025, "output": 0.0005},
        "ollama": {"input": 0.0, "output": 0.0},
        "local": {"input": 0.0, "output": 0.0},
    }

    async def _estimate_cost(self, request: ValidationRequest) -> CostEstimate:
        """
        Estimate validation cost using graph queries only.

        NO LLM calls during estimation - use graph statistics.

        Formula:
        - llm_calls = estimated_claims * 2 (extract + verify)
        - tokens = estimated_claims * TOKENS_PER_CLAIM * 2
        - cost = input_cost + output_cost + call_cost
        """
        # Get document statistics from graph
        doc_stats = await self._get_document_stats(request.document_path)

        # Estimate based on scope
        if request.scope == "all":
            estimated_claims = doc_stats.get("estimated_claims", 10)
        elif request.scope == "sections":
            estimated_claims = doc_stats.get("total_section_count", 5) * 2
        else:
            estimated_claims = 3  # Single claim validation

        llm_calls = estimated_claims * 2
        tokens = estimated_claims * self.TOKENS_PER_CLAIM * 2

        # Calculate cost based on provider pricing
        input_cost = (tokens * 0.5) * self._pricing["input"] / 1000
        output_cost = (tokens * 0.5) * self._pricing["output"] / 1000
        call_cost = llm_calls * self.COST_PER_LLM_CALL_USD
        cost_usd = input_cost + output_cost + call_cost

        return CostEstimate(
            estimated_llm_calls=llm_calls,
            estimated_tokens=tokens,
            estimated_cost_usd=cost_usd,
            exceeds_budget=cost_usd > request.max_cost_usd,
            breakdown={
                "claim_extraction": estimated_claims * self.COST_PER_LLM_CALL_USD,
                "verification": estimated_claims * self.COST_PER_LLM_CALL_USD,
                "token_costs": input_cost + output_cost,
            },
        )
```

### 4.3 Result Caching (Already Implemented)

**Location:** `codebase_rag/shared/validation/cache.py`

**Status:** ✅ Implemented with `ValidationCache` class

**Implementation Details:**
```python
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

    def invalidate_document(self, document_path: str) -> int:
        """Invalidate all cached validations for a document."""
        keys_to_remove = self._document_index.get(document_path, set())
        for k in keys_to_remove:
            self._cache.pop(k, None)
        self._document_index.pop(document_path, None)
        return len(keys_to_remove)

    def invalidate_code_graph(self) -> int:
        """Invalidate all CODE_VS_DOC results when code graph changes."""
        keys_to_remove = [
            k for k, v in self._cache.items()
            if v.mode == "CODE_VS_DOC"
        ]
        for key in keys_to_remove:
            self._cache.pop(key, None)
        return len(keys_to_remove)

    def _evict_oldest(self) -> None:
        """Evict oldest cached entries when over MAX_CACHE_SIZE."""
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].cached_at,
        )
        for key, _ in sorted_entries[:100]:
            self._cache.pop(key, None)
```

**Note:** The cache also tracks `DOC_VS_CODE` entries. When code graph changes, both `CODE_VS_DOC` AND `DOC_VS_CODE` entries should be invalidated since both depend on code state. Consider adding:
```python
def invalidate_both_directions(self) -> int:
    """Invalidate both CODE_VS_DOC and DOC_VS_CODE when code graph changes."""
    keys_to_remove = [
        k for k, v in self._cache.items()
        if v.mode in ("CODE_VS_DOC", "DOC_VS_CODE")
    ]
    for key in keys_to_remove:
        self._cache.pop(key, None)
    return len(keys_to_remove)
```

---

## 5. Interactive CLI Commands

### 5.1 CLI Command Extensions

**Location:** `codebase_rag/cli.py` (Already has stubs - needs integration)

```python
@app.command(name=ch.CLICommandName.QUERY_DOCS, help=ch.CMD_QUERY_DOCS)
def query_docs(
    query: str = typer.Argument(..., help=ch.HELP_QUERY),
    top_k: int = typer.Option(5, "--top-k", "-k", help=ch.HELP_TOP_K),
) -> None:
    """Query the document graph using natural language."""
    from ..shared.query_router import QueryRequest, QueryMode, QueryRouter
    from ..services.graph_service import MemgraphIngestor
    from dataclasses import asdict

    _info(style(f"Querying document graph: {query}", cs.Color.CYAN))

    try:
        # Initialize document graph connection
        with MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
        ) as doc_graph:
            query_router = QueryRouter(doc_graph=doc_graph)

            request = QueryRequest(
                question=query,
                mode=QueryMode.DOCUMENT_ONLY,
                top_k=top_k,
            )
            response = query_router.query(request)

            # Display response
            result = asdict(response)
            app_context.console.print(
                Panel(
                    result['answer'],
                    title=f"Document Query (Mode: {response.mode.value})",
                    border_style="cyan",
                )
            )

            if result['sources']:
                app_context.console.print(
                    f"\n[bold]Sources:[/bold] {len(result['sources'])} document(s)"
                )

            # Display warnings if any
            if result.get('warnings'):
                app_context.console.print(
                    f"[yellow]Warnings:[/yellow] {', '.join(result['warnings'])}"
                )
    except Exception as e:
        app_context.console.print(style(f"Query failed: {e}", cs.Color.RED))
        raise typer.Exit(1) from e


@app.command(name=ch.CLICommandName.VALIDATE_SPEC, help=ch.CMD_VALIDATE_SPEC)
def validate_spec(
    spec_path: str = typer.Argument(..., help=ch.HELP_SPEC_PATH),
    scope: str = typer.Option("all", "--scope", "-s", help=ch.HELP_SCOPE),
    max_cost: float = typer.Option(0.50, "--max-cost", "-c", help=ch.HELP_MAX_COST),
    dry_run: bool = typer.Option(False, "--dry-run", help=ch.HELP_DRY_RUN),
) -> None:
    """Validate code against a specification document."""
    from ..shared.query_router import QueryRequest, QueryMode, QueryRouter
    from ..shared.validation.api import ValidationTriggerAPI, ValidationRequest
    from ..services.graph_service import MemgraphIngestor
    from dataclasses import asdict
    import asyncio

    _info(style(f"Validating code against spec: {spec_path}", cs.Color.CYAN))

    try:
        # Initialize both graphs
        with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST,
            port=settings.MEMGRAPH_PORT,
        ) as code_graph, MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
        ) as doc_graph:
            query_router = QueryRouter(code_graph=code_graph, doc_graph=doc_graph)
            validation_api = ValidationTriggerAPI()

            # Step 1: Cost estimation (async, run in sync context)
            validation_request = ValidationRequest(
                document_path=spec_path,
                mode="CODE_VS_DOC",
                scope=scope,
                max_cost_usd=max_cost,
                dry_run=dry_run,
            )

            trigger_result = asyncio.run(validation_api.request_validation(validation_request))

            if not trigger_result.accepted:
                app_context.console.print(
                    Panel(
                        trigger_result.message,
                        title="Validation Not Executed",
                        border_style="yellow",
                    )
                )
                return

            # Step 2: Execute validation
            request = QueryRequest(
                question=f"Validate code against {spec_path}",
                mode=QueryMode.CODE_VS_DOC,
                scope=scope,
            )
            response = query_router.query(request)

            result = asdict(response)
            report = result['validation_report']

            if report:
                app_context.console.print(
                    Panel(
                        result['answer'],
                        title=(
                            f"Validation Report: {report['passed']}/{report['total']} "
                            f"({report['accuracy_score']:.1%} accurate)"
                        ),
                        border_style="green" if report['accuracy_score'] > 0.8 else "yellow",
                    )
                )
    except Exception as e:
        app_context.console.print(style(f"Validation failed: {e}", cs.Color.RED))
        raise typer.Exit(1) from e
```

---

## 6. Session Management for Documents

### 6.1 Enhanced Session Context

**Location:** `codebase_rag/main.py`

**Recommendation:** Cache document graph state to avoid DB query on every user input.

```python
def get_session_context() -> str:
    """
    Build context from session log for continuity.

    Enhanced to include document query context.
    """
    if app_context.session.log_file and app_context.session.log_file.exists():
        content = app_context.session.log_file.read_text(encoding="utf-8")

        # Add document graph state indicator (cached, not queried every time)
        doc_context = _get_document_graph_context_cached()

        return (
            f"{cs.SESSION_CONTEXT_START}\n"
            f"{doc_context}\n"
            f"{content}"
            f"{cs.SESSION_CONTEXT_END}"
        )
    return ""


def _get_document_graph_context_cached() -> str:
    """Get cached document graph state for context.

    CRITICAL: Do NOT query DB on every user input.
    Use cached state from app_context.session.
    """
    # Check cached state first
    if hasattr(app_context.session, 'doc_graph_available'):
        if app_context.session.doc_graph_available:
            return f"[Document Graph: {app_context.session.doc_count} documents indexed]"
        else:
            return "[Document Graph: Not available]"

    # First check - cache the result
    try:
        with MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
            connect_timeout=2.0,  # Don't block on slow/unavailable graph
        ) as doc_graph:
            result = doc_graph.fetch_all("""
                MATCH (d:Document)
                RETURN count(d) as doc_count
            """)
            if result:
                count = result[0]['doc_count']
                app_context.session.doc_graph_available = True
                app_context.session.doc_count = count
                return f"[Document Graph: {count} documents indexed]"
    except Exception:
        app_context.session.doc_graph_available = False
        return "[Document Graph: Not available]"

    return "[Document Graph: Not connected]"
```

**Session State Extension:**

```python
@dataclass
class SessionContext:
    log_file: Path | None = None
    cancelled: bool = False
    confirm_edits: bool = True
    # NEW document state (cached, not queried every input)
    doc_graph_available: bool = False
    doc_count: int = 0
    doc_graph_checked: bool = False
```

---

## 7. Implementation Phases

### Phase 1: Query Router Implementation (Week 1-2)

**Prerequisites:** ✅ Core modules already exist (`shared/query_router.py`, `shared/validation/api.py`, `shared/validation/cache.py`)

| Task | Description | Effort | Status |
|------|-------------|--------|--------|
| Implement `_query_document_only()` | Vector search on Chunk nodes (fix syntax: `vector_search.search`) | 2 days | 🔄 Needs vector integration |
| Implement `_query_both_merged()` | Merge code + doc results | 1 day | ✅ Framework exists |
| Add workspace isolation | Support multi-tenant queries | 1 day | ❌ |
| Unit tests for QueryRouter | Test all 5 query modes | 2 days | ❌ |

### Phase 2: Document Tools (Week 3-4)

**Prerequisites:** ✅ Validation API already implemented - no circular dependency

| Task | Description | Effort | Status |
|------|-------------|--------|--------|
| Create `document_query.py` | query_document_graph, query_both_graphs tools | 2 days | ❌ |
| Create `document_validation.py` | validate_code_against_spec, validate_doc_against_code | 3 days | ❌ |
| Create `document_index.py` | index_documents tool | 1 day | ❌ |
| Register tools with agent | Update `_initialize_services_and_agent()` | 1 day | ❌ |
| Integration tests | Test tools in agent loop | 2 days | ❌ |

### Phase 3: Validation Engine (Week 5-7)

**Prerequisites:** ✅ Framework exists (`shared/validation/code_vs_doc.py`, `shared/validation/doc_vs_code.py`)

| Task | Description | Effort | Status |
|------|-------------|--------|--------|
| Implement CodeVsDocValidator.validate() | LLM-based code→doc validation logic | 3 days | 🔄 Partial |
| Implement DocVsCodeValidator.validate() | LLM-based doc→code validation logic | 3 days | 🔄 Partial |
| Add bidirectional cache invalidation | Invalidate both modes on code change | 1 day | ❌ |
| E2E validation tests | Full validation pipeline | 2 days | ❌ |

### Phase 4: CLI Integration (Week 8)

**Prerequisites:** ✅ CLI commands exist as stubs, need QueryRouter integration

| Task | Description | Effort | Status |
|------|-------------|--------|--------|
| Update CLI commands | Integrate QueryRouter into CLI | 2 days | 🔄 Stubs exist |
| Add interactive prompts | Cost confirmation, scope selection | 1 day | ❌ |
| Update help text | Document new commands | 1 day | ✅ Done |
| Manual testing | Test all CLI commands | 1 day | ❌ |

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Location:** `codebase_rag/tests/shared/test_query_router.py`

```python
import pytest
from codebase_rag.shared.query_router import QueryRouter, QueryMode, QueryRequest


class TestQueryRouter:
    """Test QueryRouter query mode routing."""

    def test_document_only_queries_doc_graph(self, mock_doc_graph):
        """DOCUMENT_ONLY mode must query document graph only."""
        router = QueryRouter(doc_graph=mock_doc_graph)
        request = QueryRequest(
            question="How do I use auth?",
            mode=QueryMode.DOCUMENT_ONLY,
        )
        response = router.query(request)
        
        assert response.mode == QueryMode.DOCUMENT_ONLY
        assert len(response.sources) > 0
        assert all(s.type == "document" for s in response.sources)

    def test_code_only_never_queries_doc_graph(self, mock_code_graph, mock_doc_graph):
        """CODE_ONLY mode must NOT query document graph."""
        router = QueryRouter(code_graph=mock_code_graph, doc_graph=mock_doc_graph)
        request = QueryRequest(
            question="What functions call auth?",
            mode=QueryMode.CODE_ONLY,
        )
        response = router.query(request)
        
        # Verify mock_doc_graph.execute() was never called
        mock_doc_graph.execute.assert_not_called()

    def test_both_merged_has_clear_attribution(self, mock_code_graph, mock_doc_graph):
        """BOTH_MERGED must label each source."""
        router = QueryRouter(code_graph=mock_code_graph, doc_graph=mock_doc_graph)
        request = QueryRequest(
            question="Tell me about auth",
            mode=QueryMode.BOTH_MERGED,
        )
        response = router.query(request)
        
        code_sources = [s for s in response.sources if s.type == "code"]
        doc_sources = [s for s in response.sources if s.type == "document"]
        
        assert len(code_sources) > 0 or len(doc_sources) > 0
```

### 8.2 Integration Tests

**Location:** `codebase_rag/tests/integration/test_document_agent.py`

> **WARNING (Review Finding):** Current test coverage has significant gaps:
> - No integration tests connecting to actual Memgraph instances (ports 7687/7688)
> - Validation logic (CodeVsDocValidator, DocVsCodeValidator) has TODO implementations
> - No tests for vector search functionality
> - No error handling tests for network failures or timeouts
>
> Add these tests before production deployment.

```python
import pytest
import asyncio
from codebase_rag.main import _initialize_services_and_agent


@pytest.mark.integration
class TestDocumentAgentIntegration:
    """Test document tools integrated with agent."""

    def test_agent_uses_query_document_graph_tool(
        self, temp_repo_with_docs, mock_llm
    ):
        """Agent should use query_document_graph for doc questions."""
        # Setup agent with document tools
        with connect_memgraph() as code_graph:
            with connect_memgraph(port=7688) as doc_graph:
                query_router = QueryRouter(code_graph=code_graph, doc_graph=doc_graph)
                rag_agent, _ = _initialize_services_and_agent(
                    str(temp_repo_with_docs),
                    code_graph,
                    query_router=query_router,
                )

                # Ask document question (async run in sync test)
                response = asyncio.run(
                    rag_agent.run("How do I use the authentication API?")
                )

                # Verify agent used document tools
                assert "query_document_graph" in str(response.new_messages())

    def test_validation_workflow_with_cost_estimation(
        self, temp_repo_with_docs, mock_llm
    ):
        """Validation should estimate cost before executing."""
        # Setup validation API
        validation_api = ValidationTriggerAPI()

        # Request validation with budget (async run in sync test)
        request = ValidationRequest(
            document_path="docs/api-spec.md",
            mode="CODE_VS_DOC",
            max_cost_usd=0.50,
            dry_run=True,  # Estimate only
        )

        result = asyncio.run(validation_api.request_validation(request))

        assert result.cost_estimate is not None
        assert result.cost_estimate.estimated_cost_usd < 0.50
```

---

## 9. Configuration

### 9.1 Environment Variables

**Location:** `.env.example`

```bash
# ─────────────────────────────────────────────────────────
# DOCUMENT GRAPHRAG (VERIFIED CONFIG)
# ─────────────────────────────────────────────────────────
DOC_MEMGRAPH_HOST=localhost
DOC_MEMGRAPH_PORT=7688
DOC_MEMGRAPH_USERNAME=  # Optional, defaults to None
DOC_MEMGRAPH_PASSWORD=  # Optional, defaults to None
DOC_VECTOR_STORE_BACKEND=memgraph

# Document-specific (VERIFIED - see config.py:361-372)
DOC_SUPPORTED_EXTENSIONS=[".md", ".rst", ".txt", ".pdf", ".docx"]  # JSON array format
DOC_MAX_FILE_SIZE_MB=50
DOC_MEMGRAPH_VECTOR_CAPACITY=100000
DOC_MEMGRAPH_VECTOR_INDEX_NAME=doc_embeddings  # ACTUAL VALUE (not 'chunk_embedding_index')

# Validation (PENDING - not yet in config.py, add when implementing Phase 3)
# VALIDATION_CACHE_TTL_HOURS=24  # TODO: Add to AppConfig
# VALIDATION_MAX_CACHE_SIZE=1000  # TODO: Add to AppConfig
# VALIDATION_DEFAULT_MAX_COST_USD=0.50  # TODO: Add to AppConfig
```

### 9.2 Config Class Extension

**Location:** `codebase_rag/config.py`

```python
class AppConfig(BaseSettings):
    # ... existing config ...

    # Document GraphRAG (VERIFIED - see config.py:351-379)
    DOC_MEMGRAPH_HOST: str = "localhost"
    DOC_MEMGRAPH_PORT: int = 7688
    DOC_MEMGRAPH_USERNAME: str | None = None  # Optional
    DOC_MEMGRAPH_PASSWORD: str | None = None  # Optional
    DOC_MEMGRAPH_MEMORY_LIMIT: str = "2GB"
    DOC_LAB_PORT: int = 3001
    DOC_VECTOR_STORE_BACKEND: str = "memgraph"

    DOC_SUPPORTED_EXTENSIONS: list[str] = Field(
        default=[".md", ".rst", ".txt", ".pdf", ".docx"]
    )
    DOC_ENABLE_PDF_EXTRACTION: bool = True
    DOC_MAX_FILE_SIZE_MB: int = Field(default=50, gt=0)
    DOC_EXTRACTION_TIMEOUT_SECONDS: int = Field(default=30, gt=0)
    DOC_ENABLED: bool = True

    # Document vector settings (VERIFIED)
    DOC_MEMGRAPH_VECTOR_INDEX_NAME: str = "doc_embeddings"  # ACTUAL VALUE
    DOC_MEMGRAPH_VECTOR_CAPACITY: int = Field(default=100000, gt=0)
    DOC_VECTOR_SEARCH_TOP_K: int = Field(default=5, gt=0)

    # Validation (PENDING - add in Phase 3)
    # VALIDATION_CACHE_TTL_HOURS: int = Field(default=24, gt=0)
    # VALIDATION_MAX_CACHE_SIZE: int = Field(default=1000, gt=0)
    # VALIDATION_DEFAULT_MAX_COST_USD: float = Field(default=0.50, gt=0)
```

---

## 10. Appendix: Quick Start Implementation Guide

### Step 1: Implement QueryRouter Methods

```bash
# Edit: codebase_rag/shared/query_router.py
# Implement these methods:
# - _query_document_only()
# - _query_both_merged()
# - _validate_code_against_doc()
# - _validate_doc_against_code()
```

### Step 2: Create Document Tools

```bash
# Create: codebase_rag/tools/document_query.py
# Create: codebase_rag/tools/document_validation.py
# Create: codebase_rag/tools/document_index.py
```

### Step 3: Register Tools with Agent

```python
# Edit: codebase_rag/main.py:_initialize_services_and_agent()

# Add after existing tool creation:
query_router = QueryRouter(code_graph=ingestor, doc_graph=doc_ingestor)
document_query_tool = create_query_document_graph_tool(query_router)
document_both_tool = create_query_both_graphs_tool(query_router)
validation_code_tool = create_validate_code_against_spec_tool(query_router, validation_api)
validation_doc_tool = create_validate_doc_against_code_tool(query_router, validation_api)
index_docs_tool = create_index_documents_tool(repo_path, doc_updater_factory)

# Add to tools list:
rag_agent = create_rag_orchestrator(
    tools=[
        # ... existing tools ...
        document_query_tool,
        document_both_tool,
        validation_code_tool,
        validation_doc_tool,
        index_docs_tool,
    ]
)
```

### Step 4: Update CLI Commands

```bash
# Edit: codebase_rag/cli.py
# Replace stub implementations with actual QueryRouter integration
```

### Step 5: Test End-to-End

```bash
# 1. Start both Memgraph instances
# Note: Service names are 'memgraph' (code) and 'memgraph-doc' (document)
# See docker-compose.yaml for actual configuration
docker compose up -d memgraph memgraph-doc

# 2. Index documents
cgr index-docs --repo-path /path/to/repo

# 3. Test document query
cgr query-docs "How do I use authentication?"

# 4. Start interactive CLI
cgr start --repo-path /path/to/repo

# 5. Ask document questions in agent
> "How do I use the authentication API?"
> "Validate docs/api.md against code"
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-30 | Architecture Team | Initial implementation-ready specification |
| 1.1 | 2026-04-01 | Claude (Review) | Fixed critical issues: Memgraph vector syntax (`vector_search.search`), async/sync patterns in CLI, added existing implementation status table, updated phase dependencies, fixed test examples |
| 1.2 | 2026-04-01 | Claude (10-Worker Review) | Comprehensive review with 10 parallel workers. Fixed: vector index name (`doc_embeddings`), docker compose service names (`memgraph` vs `memgraph-code`), CLI command syntax (`--repo-path`), config vars verification, tool pattern architecture note, test coverage warning, accurate implementation status markers |

---

**END OF SPECIFICATION**
