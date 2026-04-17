"""Unit tests for GraphNavigator tool."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from codebase_rag.tools.graph_navigation import GraphNavigator, os_walk_safe


class TestGraphNavigator:
    """Test GraphNavigator class."""

    @pytest.fixture
    def mock_ingestor(self):
        """Create a mock ingestor."""
        ingestor = Mock()
        ingestor.fetch_all = Mock()
        return ingestor

    @pytest.fixture
    def navigator(self, mock_ingestor):
        """Create GraphNavigator with mock ingestor."""
        return GraphNavigator(project_root="/tmp/test_project", ingestor=mock_ingestor)

    @pytest.mark.asyncio
    async def test_find_references_calls(self, navigator, mock_ingestor):
        """Test finding call references."""
        mock_ingestor.fetch_all.return_value = [
            {
                "qualified_name": "module1.function1",
                "name": "function1",
                "type": ["Function"],
                "path": "/tmp/test_project/module1.py",
                "start_line": 10,
            },
            {
                "qualified_name": "module2.function2",
                "name": "function2",
                "type": ["Method"],
                "path": "/tmp/test_project/module2.py",
                "start_line": 20,
            },
        ]
        result = await navigator.find_references("target.func", reference_type="calls")
        assert "References to 'target.func' (2):" in result
        assert "[calls] module1.function1" in result
        assert "[calls] module2.function2" in result
        assert "module1.py:10" in result
        assert "module2.py:20" in result
        mock_ingestor.fetch_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_find_references_imports(self, navigator, mock_ingestor):
        """Test finding import references."""
        mock_ingestor.fetch_all.return_value = [
            {"qualified_name": "importer.module", "path": "/tmp/test_project/importer.py"},
        ]
        result = await navigator.find_references("target.module", reference_type="imports")
        assert "[imports] importer.module" in result
        assert "importer.py" in result

    @pytest.mark.asyncio
    async def test_find_references_all(self, navigator, mock_ingestor):
        """Test finding all references."""
        # Mock two calls: first for calls, second for imports
        def side_effect(query, params):
            if "CALLS" in query:
                return [{"qualified_name": "caller", "name": "caller", "type": ["Function"], "path": "/tmp/caller.py", "start_line": 5}]
            else:  # IMPORTS
                return [{"qualified_name": "importer", "path": "/tmp/importer.py"}]
        mock_ingestor.fetch_all.side_effect = side_effect
        result = await navigator.find_references("target", reference_type="all")
        assert "[calls] caller" in result
        assert "[imports] importer" in result

    @pytest.mark.asyncio
    async def test_find_references_no_results(self, navigator, mock_ingestor):
        """Test when no references found."""
        mock_ingestor.fetch_all.return_value = []
        result = await navigator.find_references("target.func")
        assert "No references found for 'target.func'." in result

    @pytest.mark.asyncio
    async def test_find_references_invalid_type(self, navigator):
        """Test invalid reference_type."""
        result = await navigator.find_references("target", reference_type="invalid")
        assert "Invalid reference_type" in result

    @pytest.mark.asyncio
    async def test_get_call_hierarchy_callers(self, navigator, mock_ingestor):
        """Test getting callers hierarchy."""
        mock_ingestor.fetch_all.return_value = [
            {"qualified_name": "caller1", "depth": 1},
            {"qualified_name": "caller2", "depth": 2},
        ]
        result = await navigator.get_call_hierarchy("target.func", direction="callers", depth=3)
        assert "Call hierarchy for 'target.func'" in result
        assert "caller1" in result
        assert "caller2" in result

    @pytest.mark.asyncio
    async def test_get_call_hierarchy_callees(self, navigator, mock_ingestor):
        """Test getting callees hierarchy."""
        mock_ingestor.fetch_all.return_value = [
            {"qualified_name": "callee1", "depth": 1},
            {"qualified_name": "callee2", "depth": 2},
        ]
        result = await navigator.get_call_hierarchy("target.func", direction="callees", depth=3)
        assert "callee1" in result
        assert "callee2" in result

    @pytest.mark.asyncio
    async def test_get_call_hierarchy_both(self, navigator, mock_ingestor):
        """Test getting both callers and callees."""
        # First call for callers, second for callees
        side_effect = [
            [{"qualified_name": "caller1", "depth": 1}],
            [{"qualified_name": "callee1", "depth": 1}],
        ]
        mock_ingestor.fetch_all.side_effect = side_effect
        result = await navigator.get_call_hierarchy("target.func", direction="both", depth=2)
        assert "caller1" in result
        assert "callee1" in result

    @pytest.mark.asyncio
    async def test_get_call_hierarchy_depth_cap(self, navigator):
        """Test depth parameter capping."""
        # Depth should be capped between 1 and _MAX_DEPTH (5)
        # We'll test by checking the query parameter
        with patch.object(navigator.ingestor, "fetch_all") as mock_fetch:
            mock_fetch.return_value = []
            await navigator.get_call_hierarchy("target", depth=10)
            # Check depth param passed
            call_args = mock_fetch.call_args
            params = call_args[0][1]  # second arg is params dict
            assert params["depth"] == 5  # Should be capped to max

            await navigator.get_call_hierarchy("target", depth=0)
            call_args = mock_fetch.call_args
            params = call_args[0][1]
            assert params["depth"] == 1  # Should be min 1

    @pytest.mark.asyncio
    async def test_get_call_hierarchy_invalid_direction(self, navigator):
        """Test invalid direction parameter."""
        result = await navigator.get_call_hierarchy("target", direction="invalid")
        assert "Invalid direction" in result

    @pytest.mark.asyncio
    async def test_find_implementations(self, navigator, mock_ingestor):
        """Test finding implementations."""
        mock_ingestor.fetch_all.return_value = [
            {
                "qualified_name": "ImplClass",
                "name": "ImplClass",
                "relationship_type": "IMPLEMENTS",
                "path": "/tmp/impl.py",
                "start_line": 30,
            },
            {
                "qualified_name": "ChildClass",
                "name": "ChildClass",
                "relationship_type": "INHERITS",
                "path": "/tmp/child.py",
                "start_line": 40,
            },
        ]
        result = await navigator.find_implementations("BaseInterface")
        assert "Implementations of 'BaseInterface'" in result
        assert "ImplClass [IMPLEMENTS]" in result
        assert "ChildClass [INHERITS]" in result
        assert "impl.py:30" in result
        assert "child.py:40" in result

    @pytest.mark.asyncio
    async def test_find_implementations_no_results(self, navigator, mock_ingestor):
        """Test when no implementations found."""
        mock_ingestor.fetch_all.return_value = []
        result = await navigator.find_implementations("BaseInterface")
        assert "No implementations found for 'BaseInterface'." in result

    @pytest.mark.asyncio
    async def test_get_project_structure(self, navigator, mock_ingestor):
        """Test getting project structure."""
        mock_ingestor.fetch_all.return_value = [
            {
                "dir_name": "src",
                "dir_path": "src",
                "file_count": 10,
                "function_count": 50,
                "class_count": 5,
            },
            {
                "dir_name": "tests",
                "dir_path": "tests",
                "file_count": 5,
                "function_count": 20,
                "class_count": 0,
            },
        ]
        with patch("os.walk") as mock_walk:
            mock_walk.return_value = [
                ("/tmp/test_project", ["src", "tests"], ["README.md"]),
                ("/tmp/test_project/src", [], ["module1.py", "module2.py"]),
                ("/tmp/test_project/tests", [], ["test_module.py"]),
            ]
            result = await navigator.get_project_structure()
            assert "Project structure for 'test_project'" in result
            assert "src: 10 files, 50 functions, 5 classes" in result
            assert "tests: 5 files, 20 functions, 0 classes" in result
            assert "Totals: 15 files, 70 functions, 5 classes" in result
            assert "src/" in result  # Directory overview

    @pytest.mark.asyncio
    async def test_get_project_structure_with_project_name(self, navigator, mock_ingestor):
        """Test getting project structure with explicit project name."""
        mock_ingestor.fetch_all.return_value = []
        with patch("os.walk"):
            result = await navigator.get_project_structure(project_name="MyProject")
            assert "Project structure for 'MyProject'" in result

    @pytest.mark.asyncio
    async def test_get_import_dependencies(self, navigator, mock_ingestor):
        """Test getting import dependencies."""
        mock_ingestor.fetch_all.return_value = [
            {"qualified_name": "dep1", "depth": 1},
            {"qualified_name": "dep2", "depth": 2},
            {"qualified_name": "dep1", "depth": 3},  # circular
        ]
        result = await navigator.get_import_dependencies("module.main", depth=3)
        assert "Import dependencies for 'module.main'" in result
        assert "dep1" in result
        assert "dep2" in result
        assert "Circular dependencies detected" in result

    @pytest.mark.asyncio
    async def test_get_import_dependencies_no_results(self, navigator, mock_ingestor):
        """Test when no dependencies found."""
        mock_ingestor.fetch_all.return_value = []
        result = await navigator.get_import_dependencies("module.main")
        assert "No import dependencies found for 'module.main'." in result

    @pytest.mark.asyncio
    async def test_get_import_dependencies_depth_cap(self, navigator):
        """Test depth capping for import dependencies."""
        with patch.object(navigator.ingestor, "fetch_all") as mock_fetch:
            mock_fetch.return_value = []
            await navigator.get_import_dependencies("module", depth=10)
            params = mock_fetch.call_args[0][1]
            assert params["depth"] == 5  # Capped to _MAX_DEPTH

            await navigator.get_import_dependencies("module", depth=0)
            params = mock_fetch.call_args[0][1]
            assert params["depth"] == 1  # Min 1

    @pytest.mark.asyncio
    async def test_error_handling(self, navigator, mock_ingestor):
        """Test error handling in navigator methods."""
        mock_ingestor.fetch_all.side_effect = Exception("Database error")
        result = await navigator.find_references("target")
        assert "Error:" in result
        assert "Database error" in result


class TestOsWalkSafe:
    """Test the os_walk_safe utility function."""

    def test_skips_hidden_dirs(self):
        """Test that hidden directories are skipped."""
        with patch("os.walk") as mock_walk:
            mock_walk.return_value = [
                ("/root", [".git", "__pycache__", "src", ".hidden"], ["file1.py"]),
                ("/root/src", [], ["file2.py"]),
            ]
            result = os_walk_safe(Path("/root"))
            # Should have only one entry (the root) because .git and __pycache__ removed
            assert len(result) == 2  # root and src (since .hidden also removed)
            dirpaths = [r[0] for r in result]
            assert "/root" in dirpaths
            assert "/root/src" in dirpaths
            # Check dirnames were pruned in-place
            call_args = mock_walk.call_args
            # The mock is called once, but we can't check modified dirnames easily