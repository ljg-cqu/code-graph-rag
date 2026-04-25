"""Unit tests for mode command warnings and auto-detection."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from codebase_rag import constants as cs
from codebase_rag.main import _determine_default_query_mode, _handle_mode_command
from codebase_rag.shared.query_router import QueryMode


class TestHandleModeCommandWarnings:
    """Test content-aware warnings in _handle_mode_command."""

    @pytest.fixture
    def mock_router(self):
        router = MagicMock()
        router.doc_graph = MagicMock()
        return router

    def test_switch_to_code_only_when_empty_warns(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode code_only",
            mock_router,
            QueryMode.DOCUMENT_ONLY,
            code_count=0,
            doc_count=5,
        )
        assert mode == QueryMode.DOCUMENT_ONLY
        assert "Code graph is empty" in msg
        assert "document_only" in msg

    def test_switch_to_code_only_when_both_empty_warns(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode code_only",
            mock_router,
            QueryMode.DOCUMENT_ONLY,
            code_count=0,
            doc_count=0,
        )
        assert mode == QueryMode.DOCUMENT_ONLY
        assert "Both graphs are empty" in msg

    def test_switch_to_document_only_when_empty_warns(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode document_only",
            mock_router,
            QueryMode.CODE_ONLY,
            code_count=10,
            doc_count=0,
        )
        assert mode == QueryMode.CODE_ONLY
        assert "Document graph is empty" in msg
        assert "code_only" in msg

    def test_switch_to_document_only_when_both_empty_warns(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode document_only",
            mock_router,
            QueryMode.CODE_ONLY,
            code_count=0,
            doc_count=0,
        )
        assert mode == QueryMode.CODE_ONLY
        assert "Both graphs are empty" in msg

    def test_switch_to_both_merged_missing_code_warns(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode both_merged",
            mock_router,
            QueryMode.CODE_ONLY,
            code_count=0,
            doc_count=5,
        )
        assert mode == QueryMode.CODE_ONLY
        assert "code graph is empty" in msg.lower()

    def test_switch_to_both_merged_missing_docs_warns(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode both_merged",
            mock_router,
            QueryMode.CODE_ONLY,
            code_count=10,
            doc_count=0,
        )
        assert mode == QueryMode.CODE_ONLY
        assert "documents graph is empty" in msg.lower()

    def test_switch_to_both_merged_missing_both_warns(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode both_merged",
            mock_router,
            QueryMode.CODE_ONLY,
            code_count=0,
            doc_count=0,
        )
        assert mode == QueryMode.CODE_ONLY
        assert "code" in msg.lower()
        assert "documents" in msg.lower()

    def test_valid_switch_succeeds(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode both_merged",
            mock_router,
            QueryMode.CODE_ONLY,
            code_count=10,
            doc_count=5,
        )
        assert mode == QueryMode.BOTH_MERGED
        assert "switched to" in msg.lower()
        assert mock_router.current_mode == QueryMode.BOTH_MERGED

    def test_no_arg_shows_current_mode(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode",
            mock_router,
            QueryMode.CODE_ONLY,
            code_count=0,
            doc_count=0,
        )
        assert mode == QueryMode.CODE_ONLY
        assert "code_only" in msg

    def test_help_arg_returns_usage(self, mock_router):
        mode, msg = _handle_mode_command(
            "/mode help",
            mock_router,
            QueryMode.CODE_ONLY,
            code_count=0,
            doc_count=0,
        )
        assert mode == QueryMode.CODE_ONLY
        assert "Available modes" in msg


class TestDetermineDefaultQueryMode:
    """Test auto-detection logic."""

    @pytest.fixture
    def mock_graph(self):
        graph = MagicMock()
        graph.fetch_all = MagicMock()
        return graph

    def test_document_only_when_only_docs_present(self, mock_graph):
        mock_graph.fetch_all.return_value = [{"count": 5}]
        mode = _determine_default_query_mode(None, mock_graph)
        assert mode == QueryMode.DOCUMENT_ONLY

    def test_code_only_when_only_code_present(self, mock_graph):
        mock_graph.fetch_all.return_value = [{"count": 10}]
        mode = _determine_default_query_mode(mock_graph, None)
        assert mode == QueryMode.CODE_ONLY

    def test_both_merged_when_both_present(self, mock_graph):
        mock_graph.fetch_all.return_value = [{"count": 10}]
        mode = _determine_default_query_mode(mock_graph, mock_graph)
        assert mode == QueryMode.BOTH_MERGED

    def test_code_only_fallback_when_empty(self):
        mode = _determine_default_query_mode(None, None)
        assert mode == QueryMode.CODE_ONLY

    def test_code_only_when_both_empty_even_if_both_default_configured(self):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                "codebase_rag.main.settings.CGR_DEFAULT_MODE_WHEN_BOTH",
                "both_merged",
            )
            mode = _determine_default_query_mode(None, None)

        assert mode == QueryMode.CODE_ONLY
