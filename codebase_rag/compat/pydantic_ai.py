"""Pydantic AI compatibility shim with graceful degradation.

This module provides a compatibility layer for pydantic_ai, allowing the
codebase to import and use pydantic_ai classes even when the package is
not installed. When pydantic_ai is unavailable, stub classes are provided
that raise informative errors when used.

Usage:
    from codebase_rag.compat.pydantic_ai import Agent, Tool, HAS_PYDANTIC_AI

    if HAS_PYDANTIC_AI:
        agent = Agent(model="gpt-4")
    else:
        # Graceful degradation path
        pass

LLM-First Design Note:
    When pydantic_ai itself is missing, the LLM agent framework is unavailable.
    Therefore, graceful degradation messages are static/deterministic, not
    LLM-generated. The LLM-First principle applies to feature behavior, not
    to dependency availability.

    Dependency availability is deterministic infrastructure - we check explicitly
    rather than assuming availability. The LLM uses the result, but doesn't decide it.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

# Environment variable to force cache refresh
_FORCE_RECHECK_ENV = "CGR_FORCE_PYDANTIC_AI_RECHECK"

# Track detailed failure reason for diagnostics
_PYDANTIC_AI_FAILURE_REASON: str | None = None


def _detect_pydantic_ai() -> tuple[bool, str | None]:
    """Detect pydantic-ai availability with detailed diagnostics.

    LLM-First Design Note:
        This is deterministic infrastructure - we check explicitly rather than
        assuming availability. The LLM uses the result, but doesn't decide it.

    Returns:
        Tuple of (is_available, failure_reason)
    """
    # First check: Can we find the package spec?
    spec = importlib.util.find_spec("pydantic_ai")
    if spec is None:
        return False, "pydantic_ai package not found in Python path"

    # Second check: Can we import the main module?
    try:
        import pydantic_ai as _pa
    except ImportError as e:
        return False, f"Failed to import pydantic_ai: {e}"
    except Exception as e:
        return False, f"Unexpected error importing pydantic_ai: {type(e).__name__}: {e}"

    # Third check: Can we import required submodules?
    required_modules = [
        "pydantic_ai.agent",
        "pydantic_ai.tools",
        "pydantic_ai.models",
        "pydantic_ai.usage",
        "pydantic_ai.exceptions",
        "pydantic_ai.messages",
    ]

    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            return False, f"Failed to import {module_name}: {e}"
        except Exception as e:
            return False, f"Unexpected error importing {module_name}: {type(e).__name__}: {e}"

    # Fourth check: Verify version compatibility (optional, degrades gracefully)
    try:
        from packaging import version as _version

        version = getattr(_pa, "__version__", None)
        if version:
            min_version = "1.70.0"
            if _version.parse(version) < _version.parse(min_version):
                return (
                    False,
                    f"pydantic_ai version {version} is too old (minimum: {min_version})",
                )
    except ImportError:
        # packaging not available - skip version check but log debug info
        from loguru import logger

        logger.debug("packaging module not available, skipping version check")
    except Exception:
        # Version check failed but import succeeded - warn but don't fail
        pass

    return True, None


# Perform detection at module load
_HAS_PYDANTIC_AI, _PYDANTIC_AI_FAILURE_REASON = _detect_pydantic_ai()

# Public constant for backward compatibility
HAS_PYDANTIC_AI = _HAS_PYDANTIC_AI


def refresh_pydantic_ai_status() -> bool:
    """Re-check pydantic-ai availability, bypassing cached result.

    Use this when you suspect the environment changed after module load
    (e.g., after installing packages in a running process).

    Returns:
        True if pydantic-ai is now available, False otherwise.
    """
    global HAS_PYDANTIC_AI, _HAS_PYDANTIC_AI, _PYDANTIC_AI_FAILURE_REASON

    # Re-run detection
    _HAS_PYDANTIC_AI, _PYDANTIC_AI_FAILURE_REASON = _detect_pydantic_ai()
    HAS_PYDANTIC_AI = _HAS_PYDANTIC_AI

    return _HAS_PYDANTIC_AI


def check_pydantic_ai() -> bool:
    """Check if pydantic_ai is available with full functionality.

    Honors CGR_FORCE_PYDANTIC_AI_RECHECK environment variable for debugging.

    Returns:
        True if pydantic_ai is installed and available, False otherwise.
    """
    if os.environ.get(_FORCE_RECHECK_ENV):
        return refresh_pydantic_ai_status()
    return _HAS_PYDANTIC_AI


def get_pydantic_ai_failure_reason() -> str | None:
    """Get the detailed reason for pydantic_ai unavailability.

    Returns:
        Failure reason string if unavailable, None if available.
    """
    return _PYDANTIC_AI_FAILURE_REASON


def require_pydantic_ai(feature_name: str) -> None:
    """Raise informative error if pydantic_ai is required but unavailable.

    Args:
        feature_name: Name of the feature requiring pydantic_ai

    Raises:
        RuntimeError: If pydantic_ai is not installed, with detailed diagnostics.
    """
    if not HAS_PYDANTIC_AI:
        # Build detailed error message
        lines = [
            f"'{feature_name}' requires pydantic_ai.",
            "",
            "Installation:",
            "  uv sync --extra ai",
            "  pip install 'code-graph-rag[ai]'",
        ]

        # Add diagnostic information
        if _PYDANTIC_AI_FAILURE_REASON:
            lines.extend(
                [
                    "",
                    "Diagnostic information:",
                    f"  {_PYDANTIC_AI_FAILURE_REASON}",
                ]
            )

        # Add Python path hint
        lines.extend(
            [
                "",
                "Python path:",
                f"  {sys.executable}",
            ]
        )

        # Add environment hint if in subprocess
        if os.environ.get("CGR_SUBPROCESS"):
            lines.extend(
                [
                    "",
                    "Note: Running in subprocess. Ensure environment is inherited.",
                ]
            )

        raise RuntimeError("\n".join(lines))


# Import actual classes if available, otherwise define stubs
if _HAS_PYDANTIC_AI:
    from pydantic_ai import Agent as _Agent
    from pydantic_ai import ApprovalRequired as _ApprovalRequired
    from pydantic_ai import RunContext as _RunContext
    from pydantic_ai import Tool as _Tool
    from pydantic_ai.exceptions import ModelHTTPError as _ModelHTTPError
    from pydantic_ai.messages import (
        ModelMessage as _ModelMessage,
    )
    from pydantic_ai.messages import (
        ModelRequest as _ModelRequest,
    )
    from pydantic_ai.messages import (
        ModelResponse as _ModelResponse,
    )
    from pydantic_ai.messages import (
        ToolCallPart as _ToolCallPart,
    )
    from pydantic_ai.messages import (
        ToolReturnPart as _ToolReturnPart,
    )
    from pydantic_ai.messages import (
        UserPromptPart as _UserPromptPart,
    )
    from pydantic_ai.models import Model as _Model
    from pydantic_ai.usage import UsageLimits as _UsageLimits

else:
    # Stub RunResult for Agent.run() return value
    class _RunResult:
        """Stub RunResult when pydantic_ai is unavailable."""

        def __init__(self) -> None:
            self.data: Any = ""
            self._messages: list[object] = []
            self._usage = object()

        def all_messages(self) -> list[object]:
            return self._messages

        def usage(self) -> object:
            return self._usage

    # Stub Tool class
    class _Tool:
        """Stub Tool when pydantic_ai is unavailable.

        Allows Tool objects to be created for registration, but raises
        RuntimeError on actual execution.
        """

        def __init__(
            self,
            function: Callable[..., object],
            name: str | None = None,
            description: str | None = None,
            takes_ctx: bool = False,
            max_retries: int | None = None,
        ) -> None:
            self.function = function
            self.name = name or getattr(function, "__name__", "tool")
            self.description = description or ""
            self.takes_ctx = takes_ctx
            self.max_retries = max_retries

        async def run(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError(
                "pydantic_ai is required for tool execution. "
                "Install with: uv sync --extra ai"
            )

    # Stub Agent class
    class _Agent:
        """Stub Agent when pydantic_ai is unavailable.

        Allows Agent objects to be created for configuration, but raises
        RuntimeError on actual execution.
        """

        def __init__(
            self,
            model: Any = None,
            *,
            system_prompt: str | Sequence[str] = (),
            name: str | None = None,
            result_type: Any = None,
            tools: Sequence[Any] = (),
            retries: int = 1,
            result_tool_name: str = "final_result",
            result_tool_description: str | None = None,
            defer_model_check: bool = False,
            end_strategy: str = "early",
        ) -> None:
            self.model = model
            self.system_prompt = system_prompt
            self.name = name
            self.result_type = result_type
            self.tools = list(tools)
            self.retries = retries
            self._unavailable = True

        async def run(
            self,
            user_prompt: str | None = None,
            *,
            message_history: list[Any] | None = None,
            model: Any = None,
            usage_limits: Any = None,
            usage: Any = None,
        ) -> _RunResult:
            raise RuntimeError(
                "pydantic_ai is required for agent execution. "
                "Install with: uv sync --extra ai"
            )

        def run_sync(
            self,
            user_prompt: str | None = None,
            *,
            message_history: list[Any] | None = None,
            model: Any = None,
            usage_limits: Any = None,
            usage: Any = None,
        ) -> _RunResult:
            raise RuntimeError(
                "pydantic_ai is required for agent execution. "
                "Install with: uv sync --extra ai"
            )

    # Stub exception class
    class _ModelHTTPError(Exception):
        """Stub ModelHTTPError when pydantic_ai is unavailable."""

        pass

    # Stub message classes
    class _ModelMessage:
        """Stub ModelMessage when pydantic_ai is unavailable."""

        pass

    class _ModelRequest:
        """Stub ModelRequest when pydantic_ai is unavailable."""

        def __init__(
            self,
            parts: Sequence[Any] = (),
            instructions: str | None = None,
        ) -> None:
            self.parts = list(parts)
            self.instructions = instructions

    class _ModelResponse:
        """Stub ModelResponse when pydantic_ai is unavailable."""

        def __init__(
            self,
            parts: Sequence[Any] = (),
            model_name: str | None = None,
            timestamp: Any = None,
        ) -> None:
            self.parts = list(parts)
            self.model_name = model_name
            self.timestamp = timestamp

    class _UserPromptPart:
        """Stub UserPromptPart when pydantic_ai is unavailable."""

        def __init__(
            self,
            content: str,
            timestamp: Any = None,
        ) -> None:
            self.content = content
            self.timestamp = timestamp

    class _ToolCallPart:
        """Stub ToolCallPart when pydantic_ai is unavailable."""

        def __init__(
            self,
            tool_name: str,
            args: dict[str, Any] | str,
            tool_call_id: str | None = None,
        ) -> None:
            self.tool_name = tool_name
            self.args = args
            self.tool_call_id = tool_call_id

    class _ToolReturnPart:
        """Stub ToolReturnPart when pydantic_ai is unavailable."""

        def __init__(
            self,
            tool_name: str,
            content: Any,
            tool_call_id: str | None = None,
            timestamp: Any = None,
        ) -> None:
            self.tool_name = tool_name
            self.content = content
            self.tool_call_id = tool_call_id
            self.timestamp = timestamp

    # Stub Model class
    class _Model:
        """Stub Model when pydantic_ai is unavailable."""

        pass

    # Stub UsageLimits class
    class _UsageLimits:
        """Stub UsageLimits when pydantic_ai is unavailable."""

        def __init__(
            self,
            request_limit: int | None = None,
            request_tokens_limit: int | None = None,
            response_tokens_limit: int | None = None,
            total_tokens_limit: int | None = None,
        ) -> None:
            self.request_limit = request_limit
            self.request_tokens_limit = request_tokens_limit
            self.response_tokens_limit = response_tokens_limit
            self.total_tokens_limit = total_tokens_limit

    # Stub ApprovalRequired exception
    class _ApprovalRequired(Exception):
        """Stub ApprovalRequired when pydantic_ai is unavailable."""

        pass

    # Stub RunContext for tool context
    class _RunContext:
        """Stub RunContext when pydantic_ai is unavailable."""

        def __init__(
            self,
            *,
            deps: Any = None,
            model: Any = None,
            usage: Any = None,
            prompt: str | None = None,
        ) -> None:
            self.deps = deps
            self.model = model
            self.usage = usage
            self.prompt = prompt


# Public exports
Agent = _Agent
Tool = _Tool
ModelHTTPError = _ModelHTTPError
ModelMessage = _ModelMessage
ModelRequest = _ModelRequest
ModelResponse = _ModelResponse
UserPromptPart = _UserPromptPart
ToolCallPart = _ToolCallPart
ToolReturnPart = _ToolReturnPart
Model = _Model
UsageLimits = _UsageLimits
ApprovalRequired = _ApprovalRequired
RunContext = _RunContext


__all__ = [
    "Agent",
    "ApprovalRequired",
    "HAS_PYDANTIC_AI",
    "Model",
    "ModelHTTPError",
    "ModelMessage",
    "ModelRequest",
    "ModelResponse",
    "RunContext",
    "Tool",
    "ToolCallPart",
    "ToolReturnPart",
    "UsageLimits",
    "UserPromptPart",
    "check_pydantic_ai",
    "get_pydantic_ai_failure_reason",
    "refresh_pydantic_ai_status",
    "require_pydantic_ai",
]
