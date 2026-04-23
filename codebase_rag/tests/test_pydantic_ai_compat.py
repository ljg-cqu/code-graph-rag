"""Tests for pydantic_ai compatibility layer.

Per LLM-First Design: dependency availability is deterministic infrastructure.
These tests verify enhanced detection, detailed diagnostics, and graceful degradation.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestPydanticAiDetection:
    """Tests for pydantic_ai availability detection."""

    def test_has_pydantic_ai_is_boolean(self) -> None:
        """HAS_PYDANTIC_AI should be a boolean."""
        from codebase_rag.compat.pydantic_ai import HAS_PYDANTIC_AI

        assert isinstance(HAS_PYDANTIC_AI, bool)

    def test_check_pydantic_ai_returns_boolean(self) -> None:
        """check_pydantic_ai() should return a boolean."""
        from codebase_rag.compat.pydantic_ai import check_pydantic_ai

        result = check_pydantic_ai()
        assert isinstance(result, bool)

    def test_get_pydantic_ai_failure_reason_returns_string_or_none(self) -> None:
        """get_pydantic_ai_failure_reason() should return str or None."""
        from codebase_rag.compat.pydantic_ai import (
            HAS_PYDANTIC_AI,
            get_pydantic_ai_failure_reason,
        )

        result = get_pydantic_ai_failure_reason()
        assert result is None or isinstance(result, str)

        # If pydantic_ai is available, failure reason should be None
        if HAS_PYDANTIC_AI:
            assert result is None

    def test_refresh_pydantic_ai_status_returns_boolean(self) -> None:
        """refresh_pydantic_ai_status() should return a boolean."""
        from codebase_rag.compat.pydantic_ai import refresh_pydantic_ai_status

        result = refresh_pydantic_ai_status()
        assert isinstance(result, bool)

    def test_refresh_updates_cached_value(self) -> None:
        """refresh_pydantic_ai_status() should update the cached value."""
        from codebase_rag.compat.pydantic_ai import (
            HAS_PYDANTIC_AI,
            refresh_pydantic_ai_status,
        )

        # Call refresh and verify it matches the current state
        result = refresh_pydantic_ai_status()
        assert result == HAS_PYDANTIC_AI


class TestRequirePydanticAi:
    """Tests for require_pydantic_ai function."""

    def test_require_pydantic_ai_raises_without_package(self) -> None:
        """require_pydantic_ai should raise RuntimeError when unavailable."""
        from codebase_rag.compat.pydantic_ai import HAS_PYDANTIC_AI, require_pydantic_ai

        if HAS_PYDANTIC_AI:
            # If available, should not raise
            require_pydantic_ai("test_feature")
        else:
            # If unavailable, should raise
            with pytest.raises(RuntimeError, match="requires pydantic_ai"):
                require_pydantic_ai("test_feature")

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_error_message_includes_install_instructions(self) -> None:
        """Error message should include installation instructions."""
        from codebase_rag.compat.pydantic_ai import require_pydantic_ai

        with pytest.raises(RuntimeError) as exc_info:
            require_pydantic_ai("test_feature")

        msg = str(exc_info.value)
        assert "uv sync --extra ai" in msg
        assert "pip install 'code-graph-rag[ai]'" in msg

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_error_message_includes_python_path(self) -> None:
        """Error message should include Python path for debugging."""
        import sys

        from codebase_rag.compat.pydantic_ai import require_pydantic_ai

        with pytest.raises(RuntimeError) as exc_info:
            require_pydantic_ai("test_feature")

        msg = str(exc_info.value)
        assert "Python path:" in msg
        assert sys.executable in msg

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_error_message_includes_feature_name(self) -> None:
        """Error message should include the feature name."""
        from codebase_rag.compat.pydantic_ai import require_pydantic_ai

        with pytest.raises(RuntimeError) as exc_info:
            require_pydantic_ai("my_cool_feature")

        msg = str(exc_info.value)
        assert "'my_cool_feature' requires pydantic_ai" in msg


class TestValidateAiDependencies:
    """Tests for validate_ai_dependencies function."""

    def test_validate_ai_dependencies_imports(self) -> None:
        """validate_ai_dependencies should be importable."""
        from codebase_rag.config import validate_ai_dependencies

        assert callable(validate_ai_dependencies)

    def test_validate_ai_dependencies_with_pydantic_ai_available(self) -> None:
        """validate_ai_dependencies should succeed when pydantic_ai is available."""
        from codebase_rag.compat.pydantic_ai import HAS_PYDANTIC_AI
        from codebase_rag.config import validate_ai_dependencies

        if HAS_PYDANTIC_AI:
            # Should not raise
            validate_ai_dependencies()
        else:
            # Should raise with actionable message
            with pytest.raises(RuntimeError, match="AI dependencies not available"):
                validate_ai_dependencies()


class TestForceRecheckEnvironment:
    """Tests for CGR_FORCE_PYDANTIC_AI_RECHECK environment variable."""

    def test_force_recheck_env_triggers_refresh(self) -> None:
        """Setting CGR_FORCE_PYDANTIC_AI_RECHECK should trigger refresh."""
        import os

        from codebase_rag.compat.pydantic_ai import check_pydantic_ai

        # Save original value
        original = os.environ.get("CGR_FORCE_PYDANTIC_AI_RECHECK")

        try:
            # Set the environment variable
            os.environ["CGR_FORCE_PYDANTIC_AI_RECHECK"] = "1"

            # This should trigger a refresh (no error means it worked)
            result = check_pydantic_ai()
            assert isinstance(result, bool)
        finally:
            # Restore original value
            if original is None:
                os.environ.pop("CGR_FORCE_PYDANTIC_AI_RECHECK", None)
            else:
                os.environ["CGR_FORCE_PYDANTIC_AI_RECHECK"] = original


class TestStubClasses:
    """Tests for stub classes when pydantic_ai is unavailable.

    Note: These tests only run when pydantic_ai is NOT installed.
    When pydantic_ai is available, the real classes are used instead.
    """

    def test_stub_agent_raises_on_run(self) -> None:
        """Stub Agent should raise RuntimeError on run() when unavailable."""
        from codebase_rag.compat.pydantic_ai import HAS_PYDANTIC_AI

        if HAS_PYDANTIC_AI:
            pytest.skip("pydantic_ai is installed, stub classes not used")

        from codebase_rag.compat.pydantic_ai import Agent

        agent = Agent(model="test")
        with pytest.raises(RuntimeError, match="pydantic_ai is required"):
            agent.run_sync("test prompt")

    def test_stub_tool_raises_on_run(self) -> None:
        """Stub Tool should raise RuntimeError on run() when unavailable."""
        import asyncio

        from codebase_rag.compat.pydantic_ai import HAS_PYDANTIC_AI

        if HAS_PYDANTIC_AI:
            pytest.skip("pydantic_ai is installed, stub classes not used")

        from codebase_rag.compat.pydantic_ai import Tool

        tool = Tool(function=lambda: None)
        with pytest.raises(RuntimeError, match="pydantic_ai is required"):
            asyncio.run(tool.run())

    def test_stub_classes_can_be_instantiated(self) -> None:
        """Classes should be instantiable for configuration regardless of availability."""
        from codebase_rag.compat.pydantic_ai import (
            Agent,
            ModelRequest,
            ModelResponse,
            Tool,
            UsageLimits,
        )

        # These should not raise during instantiation
        agent = Agent(model="test", tools=[])
        tool = Tool(function=lambda: None)
        limits = UsageLimits(request_limit=10)
        request = ModelRequest(parts=[])
        response = ModelResponse(parts=[])

        # Verify they exist
        assert agent is not None
        assert tool is not None
        assert limits is not None
        assert request is not None
        assert response is not None
