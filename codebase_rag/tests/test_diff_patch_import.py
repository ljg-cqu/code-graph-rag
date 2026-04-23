"""Tests for diff-match-patch import guard.

This module tests the graceful fallback when diff-match-patch is unavailable.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestDiffPatchImport:
    """Test diff-match-patch import guard."""

    def test_diff_patch_importable(self):
        """Test that diff_match_patch can be imported."""
        from codebase_rag.tools._diff_patch import DIFF_PATCH_AVAILABLE

        # This will be True if diff-match-patch is installed, False otherwise
        assert isinstance(DIFF_PATCH_AVAILABLE, bool)

    def test_file_editor_imports(self):
        """Test FileEditor can be imported without error."""
        from codebase_rag.tools.file_editor import FileEditor

        assert FileEditor is not None

    def test_language_tools_imports(self):
        """Test LanguageTools can be imported."""
        from codebase_rag.tools.language import cli

        assert cli is not None

    def test_get_diff_match_patch_raises_when_unavailable(self):
        """Test that get_diff_match_patch raises ImportError when unavailable."""
        from codebase_rag.tools._diff_patch import get_diff_match_patch

        # Mock the availability flag to False
        with patch(
            "codebase_rag.tools._diff_patch.DIFF_PATCH_AVAILABLE", False
        ), patch("codebase_rag.tools._diff_patch.diff_match_patch", None):
            with pytest.raises(ImportError) as exc_info:
                get_diff_match_patch()

            assert "diff-match-patch package is required" in str(exc_info.value)
            assert "pip install diff-match-patch" in str(exc_info.value)


class TestFileEditorWithUnavailableDiffPatch:
    """Test FileEditor behavior when diff-match-patch is unavailable."""

    def test_file_editor_initializes_without_dmp(self):
        """Test FileEditor initializes with dmp=None when unavailable."""
        from codebase_rag.tools.file_editor import FileEditor

        with patch(
            "codebase_rag.tools.file_editor.DIFF_PATCH_AVAILABLE", False
        ), patch("codebase_rag.tools.file_editor.get_diff_match_patch") as mock_get:
            mock_get.side_effect = ImportError("diff-match-patch not available")
            editor = FileEditor()
            assert editor.dmp is None

    def test_get_diff_returns_none_when_dmp_unavailable(self):
        """Test get_diff returns None when diff-match-patch unavailable."""
        from codebase_rag.tools.file_editor import FileEditor

        with patch(
            "codebase_rag.tools.file_editor.DIFF_PATCH_AVAILABLE", False
        ), patch("codebase_rag.tools.file_editor.get_diff_match_patch") as mock_get:
            mock_get.side_effect = ImportError("diff-match-patch not available")
            editor = FileEditor()
            result = editor.get_diff("test.py", "func", "new_code")
            assert result is None

    def test_apply_patch_returns_false_when_dmp_unavailable(self):
        """Test apply_patch_to_file returns False when diff-match-patch unavailable."""
        from codebase_rag.tools.file_editor import FileEditor

        with patch(
            "codebase_rag.tools.file_editor.DIFF_PATCH_AVAILABLE", False
        ), patch("codebase_rag.tools.file_editor.get_diff_match_patch") as mock_get:
            mock_get.side_effect = ImportError("diff-match-patch not available")
            editor = FileEditor()
            result = editor.apply_patch_to_file("test.py", "patch_text")
            assert result is False

    def test_replace_code_block_returns_false_when_dmp_unavailable(self):
        """Test replace_code_block returns False when diff-match-patch unavailable."""
        from codebase_rag.tools.file_editor import FileEditor

        with patch(
            "codebase_rag.tools.file_editor.DIFF_PATCH_AVAILABLE", False
        ), patch("codebase_rag.tools.file_editor.get_diff_match_patch") as mock_get:
            mock_get.side_effect = ImportError("diff-match-patch not available")
            editor = FileEditor()
            result = editor.replace_code_block("test.py", "old", "new")
            assert result is False
