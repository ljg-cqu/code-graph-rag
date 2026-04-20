"""Unit tests for _display_graph_status."""
from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest

from codebase_rag import constants as cs
from codebase_rag.main import _display_graph_status


class TestDisplayGraphStatus:
    """Test repository type panel display."""

    @pytest.fixture
    def mock_console(self):
        return MagicMock()

    def test_document_only_repo_shows_panel(self, mock_console):
        _display_graph_status(0, 5, mock_console)
        assert mock_console.print.call_count == 1
        panel = mock_console.print.call_args[0][0]
        assert panel.title == cs.UI_DOC_REPO_DETECTED

    def test_code_only_repo_shows_panel(self, mock_console):
        _display_graph_status(10, 0, mock_console)
        assert mock_console.print.call_count == 1
        panel = mock_console.print.call_args[0][0]
        assert panel.title == cs.UI_CODE_REPO_DETECTED

    def test_mixed_repo_shows_panel(self, mock_console):
        _display_graph_status(10, 5, mock_console)
        assert mock_console.print.call_count == 1
        panel = mock_console.print.call_args[0][0]
        assert panel.title == cs.UI_MIXED_REPO_DETECTED

    def test_empty_repo_shows_nothing(self, mock_console):
        _display_graph_status(0, 0, mock_console)
        mock_console.print.assert_not_called()
