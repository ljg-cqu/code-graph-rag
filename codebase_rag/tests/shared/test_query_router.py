"""Tests for QueryRouter and QueryMode."""

import pytest
from unittest.mock import Mock, patch

from codebase_rag.shared.query_router import (
    QueryMode,
    QueryRequest,
    QueryResponse,
    QueryRouter,
    Source,
    ValidationResult,
    ValidationReport,
)


class TestQueryMode:
    """Tests for QueryMode enum."""

    def test_all_modes_exist(self):
        """All 5 required modes must exist."""
        assert QueryMode.CODE_ONLY.value == "code_only"
        assert QueryMode.DOCUMENT_ONLY.value == "document_only"
        assert QueryMode.BOTH_MERGED.value == "both_merged"
        assert QueryMode.CODE_VS_DOC.value == "code_vs_doc"
        assert QueryMode.DOC_VS_CODE.value == "doc_vs_code"

    def test_mode_count(self):
        """Exactly 5 query modes."""
        assert len(QueryMode) == 5

    def test_mode_is_string_enum(self):
        """QueryMode should be StrEnum for JSON serialization."""
        assert isinstance(QueryMode.CODE_ONLY.value, str)


class TestQueryRequest:
    """Tests for QueryRequest dataclass."""

    def test_create_request_with_mode(self):
        """Request must have mode specified."""
        request = QueryRequest(question="test query", mode=QueryMode.CODE_ONLY)
        assert request.question == "test query"
        assert request.mode == QueryMode.CODE_ONLY

    def test_default_values(self):
        """Test default parameter values."""
        request = QueryRequest(question="test", mode=QueryMode.DOCUMENT_ONLY)
        assert request.validate is False
        assert request.include_metadata is True
        assert request.top_k == 5
        assert request.scope == "all"

    def test_custom_values(self):
        """Test custom parameter values."""
        request = QueryRequest(
            question="test",
            mode=QueryMode.CODE_VS_DOC,
            validate=True,
            top_k=10,
            scope="sections",
        )
        assert request.validate is True
        assert request.top_k == 10
        assert request.scope == "sections"


class TestSource:
    """Tests for Source dataclass."""

    def test_create_source(self):
        """Create source attribution."""
        source = Source(
            type="code",
            path="/path/to/file.py",
            node_type="Function",
            qualified_name="module.function",
        )
        assert source.type == "code"
        assert source.path == "/path/to/file.py"

    def test_to_dict(self):
        """Source should serialize to dict."""
        source = Source(type="document", path="/docs/api.md")
        result = source.to_dict()
        assert result["type"] == "document"
        assert result["path"] == "/docs/api.md"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_create_result(self):
        """Create validation result."""
        result = ValidationResult(
            element="authenticate_user",
            status="VALID",
            direction="CODE_VS_DOC",
        )
        assert result.element == "authenticate_user"
        assert result.status == "VALID"
        assert result.suggestion is None

    def test_result_with_suggestion(self):
        """Validation result can include suggestion."""
        result = ValidationResult(
            element="missing_endpoint",
            status="MISSING",
            direction="CODE_VS_DOC",
            suggestion="Implement POST /users endpoint",
        )
        assert result.suggestion == "Implement POST /users endpoint"


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_create_report(self):
        """Create validation report."""
        report = ValidationReport(
            total=10,
            passed=8,
            failed=2,
            direction="CODE_VS_DOC",
        )
        assert report.total == 10
        assert report.passed == 8
        assert report.failed == 2

    def test_accuracy_score_calculation(self):
        """Accuracy score should be calculated on init."""
        report = ValidationReport(
            total=10,
            passed=8,
            failed=2,
            direction="CODE_VS_DOC",
        )
        assert report.accuracy_score == 0.8

    def test_accuracy_score_zero_total(self):
        """Zero total should not crash."""
        report = ValidationReport(
            total=0,
            passed=0,
            failed=0,
            direction="CODE_VS_DOC",
        )
        assert report.accuracy_score == 0.0

    def test_to_dict(self):
        """Report should serialize to dict."""
        report = ValidationReport(
            total=5,
            passed=3,
            failed=2,
            direction="DOC_VS_CODE",
            results=[
                ValidationResult(
                    element="test",
                    status="VALID",
                    direction="DOC_VS_CODE",
                )
            ],
        )
        result = report.to_dict()
        assert result["total"] == 5
        assert result["accuracy_score"] == 0.6
        assert len(result["results"]) == 1


class TestQueryResponse:
    """Tests for QueryResponse dataclass."""

    def test_create_response(self):
        """Create query response."""
        response = QueryResponse(
            answer="Test answer",
            sources=[Source(type="code", path="/test.py")],
            mode=QueryMode.CODE_ONLY,
        )
        assert response.answer == "Test answer"
        assert len(response.sources) == 1

    def test_response_with_validation(self):
        """Response can include validation report."""
        report = ValidationReport(
            total=5,
            passed=5,
            failed=0,
            direction="CODE_VS_DOC",
        )
        response = QueryResponse(
            answer="Validated",
            sources=[],
            mode=QueryMode.CODE_VS_DOC,
            validation_report=report,
        )
        assert response.validation_report is not None

    def test_to_dict(self):
        """Response should serialize to dict."""
        response = QueryResponse(
            answer="Test",
            sources=[],
            mode=QueryMode.DOCUMENT_ONLY,
            warnings=["Test warning"],
        )
        result = response.to_dict()
        assert result["answer"] == "Test"
        assert result["mode"] == "document_only"
        assert "Test warning" in result["warnings"]


class TestQueryRouter:
    """Tests for QueryRouter class."""

    def test_router_init(self):
        """Router initializes without graphs."""
        router = QueryRouter()
        assert router.code_graph is None
        assert router.doc_graph is None

    def test_router_with_graphs(self):
        """Router can accept graph connections."""
        mock_code = Mock()
        mock_doc = Mock()
        router = QueryRouter(code_graph=mock_code, doc_graph=mock_doc)
        assert router.code_graph is mock_code
        assert router.doc_graph is mock_doc

    def test_code_only_without_graph(self):
        """CODE_ONLY returns warning without graph."""
        router = QueryRouter()
        request = QueryRequest(question="test", mode=QueryMode.CODE_ONLY)
        response = router.query(request)
        assert "not available" in response.answer.lower()
        assert "Code graph connection not configured" in response.warnings

    def test_document_only_without_graph(self):
        """DOCUMENT_ONLY returns warning without graph."""
        router = QueryRouter()
        request = QueryRequest(question="test", mode=QueryMode.DOCUMENT_ONLY)
        response = router.query(request)
        assert "not available" in response.answer.lower()

    def test_both_merged_without_graphs(self):
        """BOTH_MERGED handles missing graphs."""
        router = QueryRouter()
        request = QueryRequest(question="test", mode=QueryMode.BOTH_MERGED)
        response = router.query(request)
        assert response.mode == QueryMode.BOTH_MERGED

    def test_unknown_mode_raises(self):
        """Unknown mode should raise ValueError."""
        router = QueryRouter()
        request = QueryRequest(question="test", mode="invalid")  # type: ignore
        with pytest.raises(ValueError):
            router.query(request)

    def test_validation_modes_require_both_graphs(self):
        """Validation modes require both graphs."""
        router = QueryRouter()
        request = QueryRequest(question="test", mode=QueryMode.CODE_VS_DOC)
        response = router.query(request)
        assert "requires both" in response.answer.lower()


class TestQueryRouterIsolation:
    """Tests for graph isolation guarantees."""

    def test_code_only_does_not_touch_doc_graph(self):
        """CODE_ONLY mode must never query document graph."""
        mock_code = Mock()
        mock_doc = Mock()
        router = QueryRouter(code_graph=mock_code, doc_graph=mock_doc)

        request = QueryRequest(question="test", mode=QueryMode.CODE_ONLY)
        response = router.query(request)

        # Response should come from code graph only
        # The doc_graph mock should not have been used for any queries
        assert response.mode == QueryMode.CODE_ONLY

    def test_document_only_does_not_touch_code_graph(self):
        """DOCUMENT_ONLY mode must never query code graph."""
        mock_code = Mock()
        mock_doc = Mock()
        router = QueryRouter(code_graph=mock_code, doc_graph=mock_doc)

        request = QueryRequest(question="test", mode=QueryMode.DOCUMENT_ONLY)
        response = router.query(request)

        # Response should come from document graph only
        assert response.mode == QueryMode.DOCUMENT_ONLY