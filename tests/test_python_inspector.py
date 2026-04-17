"""Unit tests for PythonObjectInspector tool."""
from __future__ import annotations

import asyncio
import builtins
import importlib
import inspect
from unittest.mock import Mock, patch

import pytest

from codebase_rag.tools.python_inspector import (
    PythonObjectInspector,
    _validate_object_path,
)


class TestValidateObjectPath:
    """Test path validation function."""

    def test_empty_path(self):
        assert _validate_object_path("") == "Empty object path."
        assert _validate_object_path("   ") == "Empty object path."

    def test_path_traversal(self):
        assert "traversal" in _validate_object_path("..").lower()
        assert "traversal" in _validate_object_path("module..attr").lower()
        assert "traversal" in _validate_object_path("module/../attr").lower()

    def test_absolute_paths(self):
        assert "absolute" in _validate_object_path("/etc/passwd").lower()
        assert "absolute" in _validate_object_path("\\windows\\path").lower()

    def test_blocked_dunders(self):
        assert "blocked" in _validate_object_path("os.__import__").lower()
        assert "blocked" in _validate_object_path("__builtins__").lower()
        assert "blocked" in _validate_object_path("module.__globals__").lower()

    def test_allowed_dunders(self):
        # These should pass validation (though may fail later with AttributeError)
        assert _validate_object_path("module.__init__") is None
        assert _validate_object_path("module.__main__") is None
        assert _validate_object_path("module.__version__") is None

    def test_invalid_characters(self):
        assert "invalid" in _validate_object_path("module-name.attr").lower()
        assert "invalid" in _validate_object_path("123module.attr").lower()
        assert "invalid" in _validate_object_path("module.attr-name").lower()

    def test_valid_paths(self):
        assert _validate_object_path("os") is None
        assert _validate_object_path("os.path") is None
        assert _validate_object_path("codebase_rag.config") is None
        assert _validate_object_path("codebase_rag.config.settings") is None


class TestPythonObjectInspector:
    """Test PythonObjectInspector class."""

    @pytest.fixture
    def inspector(self):
        return PythonObjectInspector(project_root=".")

    @pytest.mark.asyncio
    async def test_inspect_builtin_function(self, inspector):
        """Test inspecting a built-in function."""
        result = await inspector.inspect("len")
        assert result.error_message is None
        assert result.object_path == "len"
        assert result.object_type == "builtin_function"
        assert result.signature == "(obj, /)"
        assert result.is_builtin is True
        assert result.file_path is None  # Builtins have no source file
        assert result.docstring is not None

    @pytest.mark.asyncio
    async def test_inspect_module(self, inspector):
        """Test inspecting a module."""
        result = await inspector.inspect("os")
        assert result.error_message is None
        assert result.object_path == "os"
        assert result.object_type == "module"
        assert result.name == "os"
        assert result.file_path is not None
        assert "os.py" in result.file_path or "os/__init__.py" in result.file_path
        assert result.members is not None
        assert "path" in result.members

    @pytest.mark.asyncio
    async def test_inspect_module_attribute(self, inspector):
        """Test inspecting a module attribute (function)."""
        result = await inspector.inspect("os.path.join")
        assert result.error_message is None
        assert result.object_path == "os.path.join"
        assert result.object_type == "function"
        assert result.name == "join"
        assert result.signature is not None
        assert result.docstring is not None
        assert result.file_path is not None

    @pytest.mark.asyncio
    async def test_inspect_nonexistent_module(self, inspector):
        """Test inspecting a non-existent module."""
        result = await inspector.inspect("nonexistent_module_xyz")
        assert result.error_message is not None
        assert "Module not found" in result.error_message

    @pytest.mark.asyncio
    async def test_inspect_nonexistent_attribute(self, inspector):
        """Test inspecting a non-existent attribute in existing module."""
        result = await inspector.inspect("os.nonexistent_attr_xyz")
        assert result.error_message is not None
        assert "Object not found in module" in result.error_message

    @pytest.mark.asyncio
    async def test_inspect_project_module(self, inspector):
        """Test inspecting a module from the current project."""
        result = await inspector.inspect("codebase_rag.config")
        assert result.error_message is None
        assert result.object_type == "module"
        assert result.members is not None
        assert "settings" in result.members

    @pytest.mark.asyncio
    async def test_inspect_class(self, inspector):
        """Test inspecting a class."""
        # Use a simple built-in class
        result = await inspector.inspect("str")
        assert result.error_message is None
        assert result.object_type == "class"
        assert result.members is not None
        assert "lower" in result.members

    @pytest.mark.asyncio
    async def test_timeout(self, inspector):
        """Test timeout behavior."""
        with patch.object(inspector, "_inspect_sync", side_effect=lambda x: asyncio.sleep(20)):
            with patch("codebase_rag.tools.python_inspector.settings") as mock_settings:
                mock_settings.PYTHON_INSPECT_TIMEOUT = 0.01  # Very short timeout
                result = await inspector.inspect("os")
                assert result.error_message is not None
                assert "timed out" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_unexpected_error(self, inspector):
        """Test handling of unexpected errors."""
        with patch("importlib.import_module", side_effect=RuntimeError("Unexpected")):
            result = await inspector.inspect("os")
            assert result.error_message is not None
            assert "Unexpected error" in result.error_message

    def test_validate_object_path_inspector(self, inspector):
        """Test that inspector uses validation."""
        result = asyncio.run(inspector.inspect(".."))
        assert result.error_message is not None
        assert "traversal" in result.error_message.lower()


class TestPythonObjectInspectorIntegration:
    """Integration tests with real imports."""

    @pytest.mark.asyncio
    async def test_inspect_self(self):
        """Test inspecting the inspector itself."""
        inspector = PythonObjectInspector(project_root=".")
        result = await inspector.inspect("codebase_rag.tools.python_inspector.PythonObjectInspector")
        assert result.error_message is None
        assert result.object_type == "class"
        assert result.name == "PythonObjectInspector"
        assert result.file_path is not None
        assert "python_inspector.py" in result.file_path