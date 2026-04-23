"""Compatibility shims for optional dependencies.

This module provides graceful degradation when optional dependencies
are not installed, allowing the core codebase to function with reduced
capabilities.
"""

from .pydantic_ai import (
    HAS_PYDANTIC_AI,
    Agent,
    ApprovalRequired,
    Model,
    ModelHTTPError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    Tool,
    ToolCallPart,
    ToolReturnPart,
    UsageLimits,
    UserPromptPart,
    check_pydantic_ai,
    require_pydantic_ai,
)

__all__ = [
    "HAS_PYDANTIC_AI",
    "Agent",
    "ApprovalRequired",
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
    "require_pydantic_ai",
]
