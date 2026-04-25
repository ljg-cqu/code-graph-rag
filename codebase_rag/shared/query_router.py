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
    from ..orchestrator.llm_query_planner import QueryPlan
    from ..services import QueryProtocol
    from ..vector_backend import VectorBackend


def _coerce_str(value: object, default: str = "") -> str:
    return value if isinstance(value, str) else default


def _coerce_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def _coerce_str_list(value: object, default: list[str] | None = None) -> list[str]:
    fallback = default or []
    if not isinstance(value, list):
        return fallback
    return [item for item in value if isinstance(item, str)]


def _format_path_result(path) -> str:
    lines = [
        "Found path between concepts:",
        "",
        path.formatted_path,
        "",
        f"Path length: {path.path_length} hops",
    ]
    return "\n".join(lines)


def _format_related_concepts(related) -> str:
    if not related:
        return "No related concepts found."
    lines = ["Related concepts:"]
    for name, rel_type, strength in related:
        lines.append(f"  - {name} ({rel_type}, strength: {strength:.2f})")
    return "\n".join(lines)


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
    validate: bool = False
    include_metadata: bool = True
    top_k: int = 5
    scope: str = "all"
    use_orchestrator: bool = False
    plan: QueryPlan | None = None
    forced: bool = False


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
        code_graph: QueryProtocol | None = None,
        doc_graph: QueryProtocol | None = None,
        code_vector: VectorBackend | None = None,
        doc_vector: VectorBackend | None = None,
        concept_graph: QueryProtocol | None = None,
    ):
        self.code_graph = code_graph
        self.doc_graph = doc_graph
        self.concept_graph = concept_graph
        self._code_vector = code_vector
        self._doc_vector = doc_vector
        self.current_mode: QueryMode = QueryMode.CODE_ONLY  # For in-chat mode switching

    @property
    def code_vector(self) -> VectorBackend | None:
        """Lazy initialization of code vector backend."""
        if self._code_vector is None:
            from ..vector_backend import get_shared_backend

            self._code_vector = get_shared_backend()
        return self._code_vector

    @property
    def doc_vector(self) -> VectorBackend | None:
        """Lazy initialization of document vector backend - only when doc_graph is available."""
        if self._doc_vector is None and self.doc_graph is not None:
            from ..vector_backend import get_shared_backend_for_documents

            self._doc_vector = get_shared_backend_for_documents()
        return self._doc_vector

    def query(self, request: QueryRequest) -> QueryResponse:
        """Route query based on EXPLICIT mode."""
        if request.mode == QueryMode.CODE_ONLY:
            return self._query_code_only(request)

        elif request.mode == QueryMode.DOCUMENT_ONLY:
            import asyncio

            try:
                _ = asyncio.get_running_loop()
                # We're in sync context but document query is async
                # Loop is running; we can't block here
                # Fall back to legacy behavior
                return self._query_document_only_legacy(request)
            except RuntimeError:
                pass
            return asyncio.run(self._query_document_only_async(request))

        elif request.mode == QueryMode.BOTH_MERGED:
            return self._query_both_merged(request)

        elif request.mode == QueryMode.CODE_VS_DOC:
            return self._validate_code_against_doc(request)

        elif request.mode == QueryMode.DOC_VS_CODE:
            return self._validate_doc_against_code(request)

        else:
            raise ValueError(f"Unknown query mode: {request.mode}")

    async def query_async(self, request: QueryRequest) -> QueryResponse:
        """Async version of query() for use in async contexts."""
        if request.mode == QueryMode.CODE_ONLY:
            return self._query_code_only(request)

        elif request.mode == QueryMode.DOCUMENT_ONLY:
            return await self._query_document_only_async(request)

        elif request.mode == QueryMode.BOTH_MERGED:
            return self._query_both_merged(request)

        elif request.mode == QueryMode.CODE_VS_DOC:
            return self._validate_code_against_doc(request)

        elif request.mode == QueryMode.DOC_VS_CODE:
            return self._validate_doc_against_code(request)

        else:
            raise ValueError(f"Unknown query mode: {request.mode}")

    @staticmethod
    def _normalize_reference_list(value: object) -> list[str]:
        if not isinstance(value, list | tuple):
            return []
        return [item for item in value if isinstance(item, str) and item]

    def _fetch_document_results(self, request: QueryRequest) -> list[dict]:
        from ..document.tools.document_search import document_semantic_search

        return document_semantic_search(
            query=request.question,
            ingestor=self.doc_graph,
            vector_backend=self.doc_vector,
            workspace="default",
            limit=request.top_k,
            min_similarity=0.5,
        )

    def _build_document_response(
        self,
        request: QueryRequest,
        results: list[dict],
    ) -> QueryResponse:
        from pathlib import Path

        if not results:
            return QueryResponse(
                answer=f"No relevant documents found for: {request.question}",
                sources=[],
                mode=request.mode,
            )

        sources: list[Source] = []
        answer_parts: list[str] = ["**Relevant Documentation:**\n"]

        for i, result in enumerate(results, 1):
            doc_path = result.get("document_path", "unknown")
            section_title = result.get("section_title", "Unknown Section")
            content = result.get("content", "")
            similarity = result.get("similarity", 0.0)
            content_preview = content[:200] + "..." if len(content) > 200 else content
            resolved_refs = self._normalize_reference_list(
                result.get("resolved_code_references")
            )

            answer_parts.append(
                f"\n{i}. **{section_title}** ({Path(doc_path).name}) [Similarity: {similarity:.2f}]"
            )
            answer_parts.append(f"   {content_preview}")
            if resolved_refs:
                answer_parts.append(f"   References: {', '.join(resolved_refs[:3])}")

            chunk_start_line = result.get("chunk_start_line", 0)
            chunk_end_line = result.get("chunk_end_line", chunk_start_line)
            if not isinstance(chunk_start_line, int):
                chunk_start_line = 0
            if not isinstance(chunk_end_line, int):
                chunk_end_line = chunk_start_line

            sources.append(
                Source(
                    type="document",
                    path=doc_path,
                    node_type="Chunk",
                    qualified_name=result.get("section_qn") or result.get("chunk_qn"),
                    line_range=(chunk_start_line, chunk_end_line),
                )
            )

        return QueryResponse(
            answer="\n".join(answer_parts),
            sources=sources,
            mode=request.mode,
        )

    def _build_document_reference_context(
        self,
        document_results: list[dict],
    ) -> tuple[str, list[Source], list[str]]:
        if not self.code_graph:
            return "", [], []

        qualified_names: list[str] = []
        seen: set[str] = set()

        for result in document_results:
            for reference in self._normalize_reference_list(
                result.get("resolved_code_references")
            ):
                if reference in seen:
                    continue
                seen.add(reference)
                qualified_names.append(reference)

        if not qualified_names:
            return "", [], []

        try:
            rows = self.code_graph.fetch_all(
                """
                MATCH (n)
                WHERE n.qualified_name IN $qualified_names
                RETURN n.qualified_name AS qualified_name,
                       coalesce(n.path, 'unknown') AS file_path,
                       coalesce(n.start_line, 0) AS start_line,
                       coalesce(n.end_line, 0) AS end_line,
                       labels(n) AS labels
                ORDER BY n.qualified_name
                """,
                {"qualified_names": qualified_names[:20]},
            )
        except Exception as e:
            return "", [], [f"Cross-graph reference resolution failed: {e}"]

        lines: list[str] = []
        sources: list[Source] = []

        for row in rows:
            qualified_name = _coerce_str(row.get("qualified_name"), "unknown")
            file_path = _coerce_str(row.get("file_path"), "unknown")
            labels = _coerce_str_list(row.get("labels"), ["Unknown"])
            node_type = labels[0] if labels else "Unknown"
            start_line = _coerce_int(row.get("start_line"), 0)
            end_line = _coerce_int(row.get("end_line"), start_line)

            lines.append(f"- **{qualified_name}** ({node_type}) in {file_path}")
            sources.append(
                Source(
                    type="code",
                    path=file_path,
                    node_type=node_type,
                    qualified_name=qualified_name,
                    line_range=(start_line, end_line),
                )
            )

        if not lines:
            return "", [], []

        return "\n".join(lines), sources, []

    def _query_code_only(self, request: QueryRequest) -> QueryResponse:
        """Query CODE graph/vector ONLY.

        CRITICAL: Document graph must NOT be touched.
        """
        if self.current_mode == QueryMode.DOCUMENT_ONLY and not request.forced:
            logger.warning("Code query called in DOCUMENT_ONLY mode, returning empty")
            return QueryResponse(
                answer="Code queries are disabled in DOCUMENT_ONLY mode.",
                sources=[],
                mode=request.mode,
                warnings=["Code queries disabled in DOCUMENT_ONLY mode"],
            )

        if not self.code_graph:
            return QueryResponse(
                answer="Code graph is not available.",
                sources=[],
                mode=request.mode,
                warnings=["Code graph connection not configured"],
            )

        logger.info(f"Querying code graph: {request.question}")

        if request.use_orchestrator:
            return self._query_code_with_orchestrator(request)

        return self._query_code_legacy(request)

    def _query_code_with_orchestrator(
        self,
        request: QueryRequest,
    ) -> QueryResponse:
        """Query code using QueryMethodOrchestrator for multi-method retrieval.

        Uses the synchronous wrapper for backward compatibility.
        For async contexts, use query_async() instead.
        """
        import asyncio

        from ..retrieval import QueryMethodOrchestrator

        assert self.code_graph is not None, "code_graph must be available"
        orchestrator = QueryMethodOrchestrator(
            code_graph=self.code_graph,
            code_vector=self.code_vector,
        )

        # Use async version in event loop aware manner
        try:
            _ = asyncio.get_running_loop()
            # We're in an async context, but this method is sync
            # Use the synchronous wrapper
            combined = orchestrator.execute(
                query=request.question,
                top_k=request.top_k,
                max_methods=3,
            )
        except RuntimeError:
            # No event loop running, safe to use asyncio.run
            combined = asyncio.run(
                orchestrator.execute_async(
                    query=request.question,
                    top_k=request.top_k,
                    max_methods=3,
                )
            )

        return self._build_orchestrator_response(request, combined)

    async def _query_code_with_orchestrator_async(
        self,
        request: QueryRequest,
    ) -> QueryResponse:
        """Async version for use in async contexts (MCP server, pydantic-ai)."""
        from ..retrieval import QueryMethodOrchestrator

        assert self.code_graph is not None, "code_graph must be available"
        orchestrator = QueryMethodOrchestrator(
            code_graph=self.code_graph,
            code_vector=self.code_vector,
        )

        combined = await orchestrator.execute_async(
            query=request.question,
            top_k=request.top_k,
            max_methods=3,
        )

        return self._build_orchestrator_response(request, combined)

    def _build_orchestrator_response(
        self,
        request: QueryRequest,
        combined,
    ) -> QueryResponse:
        """Build QueryResponse from orchestrator results."""
        sources: list[Source] = []
        answer_parts: list[str] = ["**Code Results (multi-method):**\n"]

        # Add integrity warnings if present
        integrity_warnings = getattr(combined, "integrity_warnings", [])
        if integrity_warnings:
            answer_parts.append("⚠️ **Integrity Warnings:**")
            for w in integrity_warnings[:3]:  # Limit to first 3
                answer_parts.append(f"  - [{w.severity}] {w.item}: {w.issue}")
            answer_parts.append("")

        for item in combined.items:
            start_line = item.get("start_line")
            end_line = item.get("end_line")
            line_range = None
            if start_line and end_line:
                line_range = (start_line, end_line)

            sources.append(
                Source(
                    type="code",
                    path=item.get("file_path", "unknown"),
                    node_type=item.get("type", "Unknown"),
                    qualified_name=item.get("qualified_name"),
                    line_range=line_range,
                )
            )
            score = item.get("combined_score", 0)
            methods = ", ".join(item.get("found_by_methods", ["unknown"]))
            answer_parts.append(
                f"- **{item.get('qualified_name', 'unknown')}** "
                f"({item.get('type', 'Unknown')}) "
                f"[Score: {score:.2f}, Methods: {methods}]"
            )

        warnings = list(combined.warnings)
        if integrity_warnings:
            warnings.extend(
                f"[Integrity] {w.item}: {w.issue}" for w in integrity_warnings
            )

        return QueryResponse(
            answer="\n".join(answer_parts),
            sources=sources,
            mode=request.mode,
            warnings=warnings,
        )

    def _query_code_legacy(self, request: QueryRequest) -> QueryResponse:
        """Query code using legacy hybrid retrieval + keyword fallback."""
        assert self.code_graph is not None, "code_graph must be available"
        sources: list[Source] = []
        answer_parts: list[str] = []
        warnings: list[str] = []

        if self.code_vector:
            try:
                from ..memgraph_advanced import (
                    HybridSearchResult,
                    create_hybrid_retriever,
                )

                retriever = create_hybrid_retriever(self.code_graph)

                results: list[HybridSearchResult] = retriever.search(
                    query=request.question,
                    top_k=request.top_k,
                )

                for result in results:
                    sources.append(
                        Source(
                            type="code",
                            path=result.file_path,
                            node_type=result.node_type,
                            qualified_name=result.qualified_name,
                            line_range=(
                                result.start_line,
                                result.end_line,
                            ),
                        )
                    )
                    answer_parts.append(
                        f"- **{result.qualified_name}** "
                        f"({result.node_type}) in {result.file_path} "
                        f"[Score: {result.combined_score:.2f} (vector: {result.vector_score:.2f}, text: {result.text_score:.2f}, graph: {result.graph_score:.2f})]"
                    )

            except Exception as e:
                warning_msg = (
                    f"Hybrid retrieval failed, falling back to basic search: {e}"
                )
                logger.warning(warning_msg)
                warnings.append(warning_msg)

        if not sources and self.code_graph:
            keyword_query = """
            MATCH (n)
            WHERE labels(n)[0] IN ['Function', 'Class', 'Method', 'Enum', 'Type',
                                    'Union', 'Interface', 'Contract', 'Library']
              AND ANY(kw IN $keywords WHERE
                  n.name CONTAINS kw OR n.qualified_name CONTAINS kw)
            RETURN n.name as name, n.qualified_name as qualified_name,
                   n.path as file_path, n.start_line as start_line,
                   n.end_line as end_line, labels(n) as labels
            LIMIT $limit
            """
            # Use LLM-extracted entities only; no keyword fallback per LLM-First spec
            keywords = (
                request.plan.expected_entities[:3]
                if request.plan and request.plan.expected_entities
                else []
            )
            try:
                keyword_results = self.code_graph.fetch_all(
                    keyword_query,
                    {
                        "keywords": keywords,
                        "limit": request.top_k,
                    },
                )

                for result in keyword_results:
                    labels = _coerce_str_list(result.get("labels"), ["Unknown"])
                    node_type = labels[0] if labels else "Unknown"
                    file_path = _coerce_str(result.get("file_path"), "unknown")
                    qualified_name = _coerce_str(result.get("qualified_name")) or None
                    start_line = _coerce_int(result.get("start_line"), 0)
                    end_line = _coerce_int(result.get("end_line"), 0)
                    display_name = qualified_name or _coerce_str(
                        result.get("name"), "unknown"
                    )
                    sources.append(
                        Source(
                            type="code",
                            path=file_path,
                            node_type=node_type,
                            qualified_name=qualified_name,
                            line_range=(
                                start_line,
                                end_line,
                            ),
                        )
                    )
                    answer_parts.append(
                        f"- **{display_name}** ({node_type}) in {file_path}"
                    )
            except Exception as e:
                logger.warning(f"Code graph query failed: {e}")

        if not sources:
            return QueryResponse(
                answer=f"No relevant code found for: {request.question}",
                sources=[],
                mode=request.mode,
                warnings=warnings,
            )

        return QueryResponse(
            answer="**Code Results:**\n\n" + "\n".join(answer_parts),
            sources=sources,
            mode=request.mode,
            warnings=warnings,
        )

    def _query_document_only_legacy(self, request: QueryRequest) -> QueryResponse:
        """Legacy synchronous document query for backward compatibility."""
        if self.current_mode == QueryMode.CODE_ONLY and not request.forced:
            logger.warning("Document query called in CODE_ONLY mode, returning empty")
            return QueryResponse(
                answer="Document queries are disabled in CODE_ONLY mode.",
                sources=[],
                mode=request.mode,
                warnings=["Document queries disabled in CODE_ONLY mode"],
            )

        if not self.doc_graph:
            return QueryResponse(
                answer="Document graph is not available.",
                sources=[],
                mode=request.mode,
                warnings=["Document graph connection not configured"],
            )

        if not self.doc_vector:
            return QueryResponse(
                answer="Document vector backend is not available.",
                sources=[],
                mode=request.mode,
                warnings=["Document vector storage not configured"],
            )

        logger.info(f"Querying document graph: {request.question}")

        try:
            results = self._fetch_document_results(request)
            return self._build_document_response(request, results)

        except Exception as e:
            logger.error(f"Document semantic search failed: {e}")
            return QueryResponse(
                answer=f"Document search failed: {e}",
                sources=[],
                mode=request.mode,
                warnings=[f"Search error: {e}"],
            )

    async def _query_document_only_async(self, request: QueryRequest) -> QueryResponse:
        """Query DOCUMENT graph/vector with optional graph traversal."""
        if self.current_mode == QueryMode.CODE_ONLY and not request.forced:
            return QueryResponse(
                answer="Document queries are disabled in CODE_ONLY mode.",
                sources=[],
                mode=request.mode,
                warnings=["Document queries disabled in CODE_ONLY mode"],
            )

        if not self.doc_graph:
            return QueryResponse(
                answer="Document graph is not available.",
                sources=[],
                mode=request.mode,
                warnings=["Document graph unavailable"],
            )

        if not self.doc_vector:
            return QueryResponse(
                answer="Document vector backend is not available.",
                sources=[],
                mode=request.mode,
                warnings=["Document vector backend unavailable"],
            )

        logger.info(f"Querying document graph: {request.question}")

        try:
            # Use LLM-provided plan for routing decisions
            from ..orchestrator.llm_query_planner import QueryIntent

            if request.plan and request.plan.intent == QueryIntent.DOC_GRAPH_TRAVERSAL:
                from ..document.graph_algorithms import DocumentGraphAlgorithms

                workspace = getattr(self, "workspace", "default")
                graph = self.concept_graph if self.concept_graph is not None else self.doc_graph
                algo = DocumentGraphAlgorithms(graph, workspace=workspace)
                concepts = request.plan.expected_entities

                if len(concepts) >= 2:
                    path = await algo.find_shortest_path(concepts[0], concepts[1])
                    if path:
                        return QueryResponse(
                            answer=_format_path_result(path),
                            sources=[],
                            mode=request.mode,
                        )
                elif len(concepts) == 1:
                    related = await algo.find_related_concepts(concepts[0])
                    if related:
                        return QueryResponse(
                            answer=_format_related_concepts(related),
                            sources=[],
                            mode=request.mode,
                        )

            # Default: semantic/vector search
            results = self._fetch_document_results(request)
            return self._build_document_response(request, results)

        except Exception as e:
            return QueryResponse(
                answer=f"Document search failed: {e}",
                sources=[],
                mode=request.mode,
                warnings=[str(e)],
            )

    def _query_document_only(self, request: QueryRequest) -> QueryResponse:
        """Synchronous wrapper for backward compatibility."""
        return self._query_document_only_legacy(request)

    def _query_both_merged(self, request: QueryRequest) -> QueryResponse:
        """
        Query BOTH graphs, merge results.

        CRITICAL: Each result must have clear source attribution.
        """
        code_response = self._query_code_only(request)
        doc_response = self._query_document_only(request)
        cross_reference_summary = ""
        cross_reference_sources: list[Source] = []
        cross_reference_warnings: list[str] = []

        if self.code_graph and self.doc_graph and self.doc_vector:
            try:
                document_results = self._fetch_document_results(request)
                (
                    cross_reference_summary,
                    cross_reference_sources,
                    cross_reference_warnings,
                ) = self._build_document_reference_context(document_results)
            except Exception as e:
                cross_reference_warnings.append(
                    f"Cross-graph reference expansion failed: {e}"
                )

        # Merge responses
        merged_sources = (
            code_response.sources + doc_response.sources + cross_reference_sources
        )
        merged_answer = (
            f"**Code Results:**\n{code_response.answer}\n\n"
            f"**Document Results:**\n{doc_response.answer}"
        )
        if cross_reference_summary:
            merged_answer += (
                "\n\n**Document References Resolved In Code:**\n"
                f"{cross_reference_summary}"
            )

        return QueryResponse(
            answer=merged_answer,
            sources=merged_sources,
            mode=request.mode,
            warnings=(
                code_response.warnings
                + doc_response.warnings
                + cross_reference_warnings
            ),
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
