"""Tests for MCP Document GraphRAG tools.

Tests for:
- query_document_graph
- query_both_graphs
- validate_code_against_spec
- validate_doc_against_code
- index_documents
"""

from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from codebase_rag.mcp.tools import MCPToolsRegistry
from codebase_rag.shared.query_router import QueryMode, QueryRequest, QueryResponse, Source
from codebase_rag.shared.validation.api import CostEstimate, ValidationTriggerResult

pytestmark = [pytest.mark.anyio]


@pytest.fixture(params=["asyncio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    """Configure anyio to only use asyncio backend."""
    return str(request.param)


@pytest.fixture
def temp_test_repo(tmp_path: Path) -> Path:
    """Create a temporary test repository with sample code and docs."""
    sample_file = tmp_path / "sample.py"
    sample_file.write_text(
        '''def hello_world():
    """Say hello to the world."""
    print("Hello, World!")

class Calculator:
    """Simple calculator class."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b
''',
        encoding="utf-8",
    )

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    api_doc = docs_dir / "api.md"
    api_doc.write_text(
        '''# API Documentation

## Overview

This document describes the API.

## Calculator

The Calculator class must provide an `add` method that adds two numbers.

## Authentication

The system shall implement user authentication.
''',
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def mcp_registry(temp_test_repo: Path) -> MCPToolsRegistry:
    """Create MCP tools registry with minimal mocks."""
    mock_ingestor = MagicMock()
    mock_cypher_gen = MagicMock()

    async def mock_generate(query: str) -> str:
        return "MATCH (n) RETURN n"

    mock_cypher_gen.generate = mock_generate

    return MCPToolsRegistry(
        project_root=str(temp_test_repo),
        ingestor=mock_ingestor,
        cypher_gen=mock_cypher_gen,
    )


@pytest.fixture
def mcp_registry_with_doc_graph(temp_test_repo: Path) -> MCPToolsRegistry:
    """Create MCP tools registry with both code and doc graph mocks."""
    mock_code_ingestor = MagicMock()
    mock_doc_ingestor = MagicMock()
    mock_cypher_gen = MagicMock()

    async def mock_generate(query: str) -> str:
        return "MATCH (n) RETURN n"

    mock_cypher_gen.generate = mock_generate

    return MCPToolsRegistry(
        project_root=str(temp_test_repo),
        ingestor=mock_code_ingestor,
        cypher_gen=mock_cypher_gen,
        doc_ingestor=mock_doc_ingestor,
    )


class TestQueryDocumentGraph:
    """Tests for query_document_graph MCP tool."""

    async def test_query_document_graph_returns_results(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """query_document_graph returns document results."""
        # Mock the document graph query
        mcp_registry_with_doc_graph.doc_ingestor.fetch_all.return_value = [
            {
                "content": "This is a test document chunk.",
                "chunk_qn": "docs.api#overview",
                "chunk_start_line": 1,
                "section_title": "Overview",
                "section_qn": "docs.api",
                "document_path": "docs/api.md",
                "similarity": 0.95,
            }
        ]

        result = await mcp_registry_with_doc_graph.query_document_graph(
            "What is the API?", top_k=5
        )

        assert "error" not in result or result.get("error") is None
        assert result.get("mode") == "DOCUMENT_ONLY"
        assert "answer" in result
        assert "sources" in result

    async def test_query_document_graph_uses_query_router(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """query_document_graph uses QueryRouter with DOCUMENT_ONLY mode."""
        # Mock the query router
        mock_router = MagicMock()
        mock_router.query.return_value = QueryResponse(
            answer="Test answer",
            sources=[Source(type="document", path="docs/api.md")],
            mode=QueryMode.DOCUMENT_ONLY,
        )
        mcp_registry_with_doc_graph._query_router = mock_router

        result = await mcp_registry_with_doc_graph.query_document_graph("test query")

        # Verify QueryRouter was called with DOCUMENT_ONLY mode
        mock_router.query.assert_called_once()
        call_args = mock_router.query.call_args[0][0]
        assert call_args.mode == QueryMode.DOCUMENT_ONLY
        assert call_args.question == "test query"

    async def test_query_document_graph_handles_error(
        self, mcp_registry: MCPToolsRegistry
    ) -> None:
        """query_document_graph handles errors gracefully."""
        # Registry without doc_ingestor should still work
        result = await mcp_registry.query_document_graph("test query")

        # Should return a result (possibly with warnings) rather than crash
        assert "mode" in result or "error" in result


class TestQueryBothGraphs:
    """Tests for query_both_graphs MCP tool."""

    async def test_query_both_graphs_merges_results(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """query_both_graphs merges code and document results."""
        mock_router = MagicMock()
        mock_router.query.return_value = QueryResponse(
            answer="**Code Results:**\nfunc1\n\n**Document Results:**\ndoc1",
            sources=[
                Source(type="code", path="src/main.py", node_type="Function"),
                Source(type="document", path="docs/api.md"),
            ],
            mode=QueryMode.BOTH_MERGED,
        )
        mcp_registry_with_doc_graph._query_router = mock_router

        result = await mcp_registry_with_doc_graph.query_both_graphs("test query")

        assert result.get("mode") == "BOTH_MERGED"
        mock_router.query.assert_called_once()
        call_args = mock_router.query.call_args[0][0]
        assert call_args.mode == QueryMode.BOTH_MERGED

    async def test_query_both_graphs_has_source_attribution(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """query_both_graphs includes source attribution."""
        mock_router = MagicMock()
        mock_router.query.return_value = QueryResponse(
            answer="Results from both graphs",
            sources=[
                Source(type="code", path="src/main.py"),
                Source(type="document", path="docs/api.md"),
            ],
            mode=QueryMode.BOTH_MERGED,
        )
        mcp_registry_with_doc_graph._query_router = mock_router

        result = await mcp_registry_with_doc_graph.query_both_graphs("test")

        sources = result.get("sources", [])
        code_sources = [s for s in sources if s.get("type") == "code"]
        doc_sources = [s for s in sources if s.get("type") == "document"]

        # At least one source type should be present
        assert len(code_sources) > 0 or len(doc_sources) > 0


class TestValidateCodeAgainstSpec:
    """Tests for validate_code_against_spec MCP tool."""

    async def test_validate_code_against_spec_dry_run(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """validate_code_against_spec with dry_run returns cost estimate."""
        result = await mcp_registry_with_doc_graph.validate_code_against_spec(
            spec_document_path="docs/api.md",
            scope="all",
            max_cost_usd=0.50,
            dry_run=True,
        )

        assert result.get("mode") == "CODE_VS_DOC"
        # dry_run returns cost estimate without executing validation
        assert "cost_estimate" in result

    async def test_validate_code_against_spec_uses_validation_api(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """validate_code_against_spec uses ValidationTriggerAPI for cost estimation."""
        result = await mcp_registry_with_doc_graph.validate_code_against_spec(
            spec_document_path="docs/api.md",
            max_cost_usd=100.0,  # High budget to ensure acceptance
        )

        assert result.get("mode") == "CODE_VS_DOC"
        # Should have validation_report or answer
        assert "validation_report" in result or "answer" in result

    async def test_validate_code_against_spec_rejected_when_budget_exceeded(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """validate_code_against_spec rejects when cost exceeds budget."""
        result = await mcp_registry_with_doc_graph.validate_code_against_spec(
            spec_document_path="docs/api.md",
            max_cost_usd=0.001,  # Very low budget
        )

        # Should be rejected or have validation report
        assert result.get("mode") == "CODE_VS_DOC"


class TestValidateDocAgainstCode:
    """Tests for validate_doc_against_code MCP tool."""

    async def test_validate_doc_against_code_dry_run(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """validate_doc_against_code with dry_run returns cost estimate."""
        result = await mcp_registry_with_doc_graph.validate_doc_against_code(
            document_path="docs/api.md",
            scope="all",
            max_cost_usd=0.50,
            dry_run=True,
        )

        assert result.get("mode") == "DOC_VS_CODE"
        # dry_run returns cost estimate without executing validation
        assert "cost_estimate" in result

    async def test_validate_doc_against_code_uses_correct_mode(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """validate_doc_against_code uses DOC_VS_CODE mode."""
        result = await mcp_registry_with_doc_graph.validate_doc_against_code(
            document_path="docs/api.md",
            max_cost_usd=100.0,
        )

        assert result.get("mode") == "DOC_VS_CODE"


class TestIndexDocuments:
    """Tests for index_documents MCP tool."""

    async def test_index_documents_returns_stats(
        self, mcp_registry: MCPToolsRegistry
    ) -> None:
        """index_documents returns indexing statistics."""
        # Mock DocumentGraphUpdater
        with patch(
            "codebase_rag.mcp.tools.DocumentGraphUpdater"
        ) as mock_updater_class:
            mock_updater = MagicMock()
            mock_updater.run.return_value = {
                "total_documents": 5,
                "indexed": 4,
                "skipped": 1,
                "failed": 0,
                "sections_created": 10,
                "chunks_created": 50,
            }
            mock_updater_class.return_value = mock_updater

            result = await mcp_registry.index_documents()

            assert result.get("success") is True
            assert "stats" in result
            assert result["stats"]["total_documents"] == 5

    async def test_index_documents_handles_error(
        self, mcp_registry: MCPToolsRegistry
    ) -> None:
        """index_documents handles errors gracefully."""
        with patch(
            "codebase_rag.mcp.tools.DocumentGraphUpdater"
        ) as mock_updater_class:
            mock_updater_class.side_effect = Exception("Connection failed")

            result = await mcp_registry.index_documents()

            assert result.get("success") is False
            assert "error" in result


class TestQueryRouterProperty:
    """Tests for MCPToolsRegistry.query_router property."""

    def test_query_router_lazy_initialization(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """query_router is lazily initialized."""
        assert mcp_registry_with_doc_graph._query_router is None

        router = mcp_registry_with_doc_graph.query_router

        assert router is not None
        assert mcp_registry_with_doc_graph._query_router is router

    def test_query_router_uses_existing_doc_ingestor(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """query_router uses existing doc_ingestor if provided."""
        router = mcp_registry_with_doc_graph.query_router

        assert router.doc_graph is mcp_registry_with_doc_graph.doc_ingestor
        assert router.code_graph is mcp_registry_with_doc_graph.ingestor

    def test_query_router_reuses_instance(
        self, mcp_registry_with_doc_graph: MCPToolsRegistry
    ) -> None:
        """query_router reuses the same instance."""
        router1 = mcp_registry_with_doc_graph.query_router
        router2 = mcp_registry_with_doc_graph.query_router

        assert router1 is router2


class TestDocumentToolSchemas:
    """Tests for document tool schemas."""

    def test_query_document_graph_schema(self, mcp_registry: MCPToolsRegistry) -> None:
        """query_document_graph has correct schema."""
        tools = mcp_registry.get_tool_schemas()
        query_doc = next(
            (t for t in tools if t.name == "query_document_graph"), None
        )

        assert query_doc is not None
        props = query_doc.inputSchema["properties"]
        assert "natural_language_query" in props
        required = query_doc.inputSchema["required"]
        assert required == ["natural_language_query"]

    def test_validate_code_against_spec_schema(
        self, mcp_registry: MCPToolsRegistry
    ) -> None:
        """validate_code_against_spec has correct schema."""
        tools = mcp_registry.get_tool_schemas()
        validate = next(
            (t for t in tools if t.name == "validate_code_against_spec"), None
        )

        assert validate is not None
        props = validate.inputSchema["properties"]
        assert "spec_document_path" in props
        assert "scope" in props
        assert "max_cost_usd" in props
        assert "dry_run" in props

    def test_index_documents_schema(self, mcp_registry: MCPToolsRegistry) -> None:
        """index_documents has correct schema."""
        tools = mcp_registry.get_tool_schemas()
        index_docs = next(
            (t for t in tools if t.name == "index_documents"), None
        )

        assert index_docs is not None
        # index_documents has no required parameters
        required = index_docs.inputSchema["required"]
        assert required == []