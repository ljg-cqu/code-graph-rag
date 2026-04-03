"""Query router for Document GraphRAG.

Routes queries to appropriate graph(s) based on EXPLICIT mode.
User MUST specify mode — no automatic guessing.

Query Modes:
- CODE_ONLY: Query code graph only
- DOCUMENT_ONLY: Query document graph only
- BOTH_MERGED: Query both, merge results
- CODE_VS_DOC: Validate code against docs (doc is truth)
- DOC_VS_CODE: Validate docs against code (code is truth)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from loguru import logger

if TYPE_CHECKING:
    from ..services.graph_service import MemgraphIngestor
    from ..vector_backend import VectorBackend


class QueryMode(StrEnum):
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


@dataclass
class QueryRequest:
    """Explicit query request with mode specification."""

    question: str
    mode: QueryMode
    validate: bool = False  # Enable cross-validation
    include_metadata: bool = True  # Include source info
    top_k: int = 5  # Results per graph
    scope: str = "all"  # Validation scope: "all", "sections", "claims"


@dataclass
class Source:
    """Source attribution for query results."""

    type: Literal["code", "document"]
    path: str
    node_type: str | None = None  # Function, Class, Section, etc.
    qualified_name: str | None = None
    line_range: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.type,
            "path": self.path,
            "node_type": self.node_type,
            "qualified_name": self.qualified_name,
            "line_range": self.line_range,
        }


@dataclass
class ValidationResult:
    """Single validation result."""

    element: str  # Function name, section title, etc.
    status: Literal["VALID", "OUTDATED", "MISSING", "ACCURATE"]
    direction: Literal["CODE_VS_DOC", "DOC_VS_CODE"]
    suggestion: str | None = None  # Suggested fix

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "element": self.element,
            "status": self.status,
            "direction": self.direction,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationReport:
    """Validation report for bidirectional validation."""

    total: int
    passed: int
    failed: int
    direction: Literal["CODE_VS_DOC", "DOC_VS_CODE"]
    results: list[ValidationResult] = field(default_factory=list)
    accuracy_score: float = 0.0  # passed / total

    def __post_init__(self) -> None:
        """Calculate accuracy score."""
        if self.total > 0:
            self.accuracy_score = self.passed / self.total
        else:
            self.accuracy_score = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "direction": self.direction,
            "results": [r.to_dict() for r in self.results],
            "accuracy_score": self.accuracy_score,
        }


@dataclass
class QueryResponse:
    """Query response with clear source attribution."""

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
            if self.validation_report
            else None,
            "warnings": self.warnings,
        }


class QueryRouter:
    """
    Routes queries to appropriate graph(s) based on EXPLICIT mode.

    Pattern follows existing MCPToolsRegistry from codebase_rag/mcp/tools.py
    """

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
        self.current_mode: QueryMode = QueryMode.CODE_ONLY  # For in-chat mode switching

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
        """
        Query CODE graph/vector ONLY.

        CRITICAL: Document graph must NOT be touched.
        """
        if not self.code_graph:
            return QueryResponse(
                answer="Code graph is not available.",
                sources=[],
                mode=request.mode,
                warnings=["Code graph connection not configured"],
            )

        logger.info(f"Querying code graph: {request.question}")

        # Query code graph using semantic search if vector backend available
        sources: list[Source] = []
        answer_parts: list[str] = []

        if self.code_vector and self.code_graph:
            # Use vector similarity search for semantic queries
            try:
                from ..embeddings import get_embedding_provider
                from ..config import settings

                config = settings.active_embedding_config
                provider = get_embedding_provider(
                    provider=config.provider,
                    model_id=config.model_id,
                )
                query_embedding = provider.embed(request.question)

                # VectorBackend.search() returns (node_id, similarity) tuples
                backend_results = self.code_vector.search(
                    query_embedding=query_embedding,
                    top_k=request.top_k,
                )

                # Fetch node details from graph using node_ids
                if backend_results:
                    node_ids = [nid for nid, _ in backend_results]
                    similarity_map = {nid: sim for nid, sim in backend_results}

                    # Query to get function/method details by node IDs
                    node_query = """
                    MATCH (n)
                    WHERE id(n) IN $node_ids AND (n:Function OR n:Method OR n:Class)
                    OPTIONAL MATCH (m:Module)-[:DEFINES]->(n)
                    RETURN
                        id(n) as node_id,
                        n.qualified_name as qualified_name,
                        n.name as name,
                        labels(n)[0] as node_type,
                        coalesce(n.file_path, m.path) as file_path,
                        n.start_line as start_line,
                        n.end_line as end_line
                    """
                    results = self.code_graph.fetch_all(node_query, {"node_ids": node_ids})

                    for result in results:
                        result["similarity"] = similarity_map.get(result.get("node_id", 0), 0.0)
                else:
                    results = []

                for result in results:
                    sources.append(Source(
                        type="code",
                        path=result.get("file_path", "unknown"),
                        node_type=result.get("node_type", "Function"),
                        qualified_name=result.get("qualified_name"),
                        line_range=(result.get("start_line", 0), result.get("end_line", 0)),
                    ))
                    answer_parts.append(
                        f"- **{result.get('qualified_name', 'unknown')}** "
                        f"({result.get('node_type', 'Function')}) in {result.get('file_path', 'unknown')} "
                        f"[Similarity: {result.get('similarity', 0.0):.2f}]"
                    )

            except Exception as e:
                logger.warning(f"Vector search failed, falling back to text search: {e}")

        if not sources and self.code_graph:
            # Fallback: Query code graph using keyword-based search
            # Simple text search in function/class names
            keyword_query = """
            MATCH (n)
            WHERE (n:Function OR n:Class OR n:Method)
              AND (n.name CONTAINS $keyword OR n.qualified_name CONTAINS $keyword)
            RETURN n.name as name, n.qualified_name as qualified_name,
                   n.file_path as file_path, n.start_line as start_line,
                   n.end_line as end_line, labels(n) as labels
            LIMIT $limit
            """
            # Extract keyword from question (simple approach)
            keyword = request.question.split()[0] if request.question else ""
            try:
                results = self.code_graph.fetch_all(keyword_query, {
                    "keyword": keyword,
                    "limit": request.top_k,
                })

                # Ensure results is iterable
                if results is None:
                    results = []
                elif not hasattr(results, '__iter__'):
                    results = []

                for result in results:
                    node_type = result.get("labels", ["Unknown"])[0]
                    sources.append(Source(
                        type="code",
                        path=result.get("file_path", "unknown"),
                        node_type=node_type,
                        qualified_name=result.get("qualified_name"),
                        line_range=(result.get("start_line", 0), result.get("end_line", 0)),
                    ))
                    answer_parts.append(
                        f"- **{result.get('qualified_name', result.get('name', 'unknown'))}** "
                        f"({node_type}) in {result.get('file_path', 'unknown')}"
                    )
            except Exception as e:
                logger.warning(f"Code graph query failed: {e}")

        if not sources:
            return QueryResponse(
                answer=f"No relevant code found for: {request.question}",
                sources=[],
                mode=request.mode,
            )

        return QueryResponse(
            answer=f"**Code Results:**\n\n" + "\n".join(answer_parts),
            sources=sources,
            mode=request.mode,
        )

    def _query_document_only(self, request: QueryRequest) -> QueryResponse:
        """
        Query DOCUMENT graph/vector ONLY.

        CRITICAL: Code graph must NOT be touched.
        """
        if not self.doc_graph:
            return QueryResponse(
                answer="Document graph is not available.",
                sources=[],
                mode=request.mode,
                warnings=["Document graph connection not configured"],
            )

        logger.info(f"Querying document graph: {request.question}")

        from pathlib import Path
        from ..document.tools.document_search import document_semantic_search
        from ..config import settings

        sources: list[Source] = []
        answer_parts: list[str] = ["**Relevant Documentation:**\n"]

        try:
            # Use semantic search for document queries
            results = document_semantic_search(
                query=request.question,
                ingestor=self.doc_graph,
                vector_backend=self.doc_vector,
                workspace="default",
                limit=request.top_k,
                min_similarity=0.5,
            )

            if not results:
                return QueryResponse(
                    answer=f"No relevant documents found for: {request.question}",
                    sources=[],
                    mode=request.mode,
                )

            for i, result in enumerate(results, 1):
                doc_path = result.get("document_path", "unknown")
                section_title = result.get("section_title", "Unknown Section")
                content = result.get("content", "")
                similarity = result.get("similarity", 0.0)

                # Truncate content for display
                content_preview = content[:200] + "..." if len(content) > 200 else content

                answer_parts.append(
                    f"\n{i}. **{section_title}** ({Path(doc_path).name}) [Similarity: {similarity:.2f}]"
                )
                answer_parts.append(f"   {content_preview}")

                sources.append(Source(
                    type="document",
                    path=doc_path,
                    node_type="Chunk",
                    qualified_name=result.get("section_qn") or result.get("chunk_qn"),
                    line_range=(result.get("chunk_start_line", 0), result.get("chunk_start_line", 0)),
                ))

        except Exception as e:
            logger.error(f"Document semantic search failed: {e}")
            return QueryResponse(
                answer=f"Document search failed: {e}",
                sources=[],
                mode=request.mode,
                warnings=[f"Search error: {e}"],
            )

        return QueryResponse(
            answer="\n".join(answer_parts),
            sources=sources,
            mode=request.mode,
        )

    def _query_both_merged(self, request: QueryRequest) -> QueryResponse:
        """
        Query BOTH graphs, merge results.

        CRITICAL: Each result must have clear source attribution.
        """
        code_response = self._query_code_only(request)
        doc_response = self._query_document_only(request)

        # Merge responses
        merged_sources = code_response.sources + doc_response.sources
        merged_answer = f"**Code Results:**\n{code_response.answer}\n\n**Document Results:**\n{doc_response.answer}"

        return QueryResponse(
            answer=merged_answer,
            sources=merged_sources,
            mode=request.mode,
            warnings=code_response.warnings + doc_response.warnings,
        )

    def _validate_code_against_doc(self, request: QueryRequest) -> QueryResponse:
        """
        Validate CODE against DOCUMENT specifications.

        Document is SOURCE OF TRUTH.
        """
        from .validation.code_vs_doc import CodeVsDocValidator

        if not self.code_graph or not self.doc_graph:
            return QueryResponse(
                answer="Validation requires both code and document graphs.",
                sources=[],
                mode=request.mode,
                warnings=["Missing graph connections"],
            )

        validator = CodeVsDocValidator(self.code_graph, self.doc_graph)
        report = validator.validate(request.question, scope=request.scope)

        return QueryResponse(
            answer=validator.generate_summary(report),
            sources=[],
            mode=request.mode,
            validation_report=report,
        )

    def _validate_doc_against_code(self, request: QueryRequest) -> QueryResponse:
        """
        Validate DOCUMENT against actual CODE.

        Code is SOURCE OF TRUTH.
        """
        from .validation.doc_vs_code import DocVsCodeValidator

        if not self.code_graph or not self.doc_graph:
            return QueryResponse(
                answer="Validation requires both code and document graphs.",
                sources=[],
                mode=request.mode,
                warnings=["Missing graph connections"],
            )

        validator = DocVsCodeValidator(self.code_graph, self.doc_graph)
        report = validator.validate(request.question, scope=request.scope)

        return QueryResponse(
            answer=validator.generate_summary(report),
            sources=[],
            mode=request.mode,
            validation_report=report,
        )


__all__ = [
    "QueryMode",
    "QueryRequest",
    "QueryResponse",
    "QueryRouter",
    "Source",
    "ValidationResult",
    "ValidationReport",
]