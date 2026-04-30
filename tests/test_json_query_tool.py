"""Tests for JSON graph query tool output formatting."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag import constants as cs
from codebase_rag.json_queries import JsonEntityResult
from codebase_rag.tools.json_query import create_query_json_graph_tool


class TestJsonQueryToolOutput:
    """Test JSON graph query tool output formatting."""

    @pytest.fixture
    def tool(self):
        """Return the query_json_graph tool function."""
        pydantic_tool = create_query_json_graph_tool()
        return pydantic_tool.function

    @pytest.fixture
    def mock_engine_results(self):
        """Return mock JsonEntityResult objects."""
        return [
            JsonEntityResult(
                unique_id="dataset::entity-001",
                name="Leadership",
                entity_type="Skill",
                entity_category="ABSTRACT_CONCEPT",
                entity_subtype="Competency",
                entity_emoji="🎯",
                emoji="⭐",
                description="The ability to guide and inspire teams toward common goals.",
                pagerank_score=0.85,
                combined_score=0.92,
                vector_score=0.95,
            ),
            JsonEntityResult(
                unique_id="dataset::entity-002",
                name="Strategic Planning",
                entity_type="Process",
                entity_category="EVENT_PROCESS",
                entity_subtype="",
                entity_emoji="📋",
                emoji="",
                description="A structured approach to defining direction and making decisions.",
                pagerank_score=0.72,
                combined_score=0.88,
                vector_score=0.90,
            ),
        ]

    @staticmethod
    @asynccontextmanager
    async def _patched_tool_context(results: list[JsonEntityResult]) -> AsyncGenerator[None, None]:
        """Patch engine, embedder, and ingestor for isolated tool testing."""
        mock_ingestor = MagicMock()
        mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
        mock_ingestor.__exit__ = MagicMock(return_value=False)

        with patch(
            "codebase_rag.tools.json_query.JsonGraphQueryEngine.semantic_search",
            return_value=results,
        ), patch(
            "codebase_rag.embedder.get_embedding_provider_instance",
            side_effect=Exception("no embeddings"),
        ), patch(
            "codebase_rag.json_ingestion._create_json_ingestor",
            return_value=mock_ingestor,
        ):
            yield

    @pytest.mark.asyncio
    async def test_output_includes_unique_id(self, tool, mock_engine_results):
        """Each result must include its unique_id for traceability."""
        async with self._patched_tool_context(mock_engine_results):
            result = await tool("leadership skills", top_k=10)

        assert cs.MSG_JSON_GRAPH_ENTITY_ID.format(unique_id="dataset::entity-001") in result
        assert cs.MSG_JSON_GRAPH_ENTITY_ID.format(unique_id="dataset::entity-002") in result

    @pytest.mark.asyncio
    async def test_output_includes_entity_name_and_category(self, tool, mock_engine_results):
        """Output must show entity name and MECE category."""
        async with self._patched_tool_context(mock_engine_results):
            result = await tool("leadership skills", top_k=10)

        assert "**⭐ Leadership**" in result
        assert "[ABSTRACT_CONCEPT 💡]" in result
        assert "**Strategic Planning**" in result
        assert "[EVENT_PROCESS ⏱️]" in result

    @pytest.mark.asyncio
    async def test_output_includes_similarity_score_when_available(self, tool, mock_engine_results):
        """Similarity score must appear when vector_score is present."""
        async with self._patched_tool_context(mock_engine_results):
            result = await tool("leadership skills", top_k=10)

        assert cs.MSG_JSON_GRAPH_ENTITY_SIMILARITY.format(score=0.92) in result
        assert cs.MSG_JSON_GRAPH_ENTITY_SIMILARITY.format(score=0.88) in result

    @pytest.mark.asyncio
    async def test_output_truncates_long_descriptions(self, tool):
        """Descriptions longer than 120 chars must be truncated with ellipsis."""
        long_description = "A" * 200
        results = [
            JsonEntityResult(
                unique_id="ds::long-desc",
                name="LongDescEntity",
                entity_type="Concept",
                entity_category="ABSTRACT_CONCEPT",
                entity_subtype="",
                entity_emoji="💡",
                emoji="",
                description=long_description,
                vector_score=0.0,
            )
        ]

        async with self._patched_tool_context(results):
            result = await tool("long description", top_k=10)

        assert "A" * 120 + "..." in result
        assert "A" * 121 not in result

    @pytest.mark.asyncio
    async def test_no_results_returns_json_specific_message(self, tool):
        """Empty results must return JSON-specific no-results message."""
        async with self._patched_tool_context([]):
            result = await tool("nonexistent entity", top_k=10)

        assert cs.MSG_JSON_GRAPH_NO_RESULTS.format(query="") in result.replace(
            "nonexistent entity", ""
        )
        assert "No functions match" not in result

    @pytest.mark.asyncio
    async def test_header_shows_correct_count(self, tool, mock_engine_results):
        """Result header must show the correct entity count."""
        async with self._patched_tool_context(mock_engine_results):
            result = await tool("leadership skills", top_k=10)

        assert cs.MSG_JSON_GRAPH_RESULT_HEADER.format(count=2) in result

    @pytest.mark.asyncio
    async def test_top_k_limits_results(self, tool, mock_engine_results):
        """Only top_k results must appear in output."""
        async with self._patched_tool_context(mock_engine_results):
            result = await tool("leadership skills", top_k=1)

        assert "dataset::entity-001" in result
        assert "dataset::entity-002" not in result

    @pytest.mark.asyncio
    async def test_error_returns_failure_message(self, tool):
        """Exceptions must be caught and returned as error string."""
        with patch(
            "codebase_rag.json_ingestion._create_json_ingestor",
            side_effect=Exception("connection refused"),
        ):
            result = await tool("test query", top_k=10)

        assert "JSON graph query failed" in result
        assert "connection refused" in result
