"""Mocks for pydantic_ai classes when unavailable or for testing.

Provides mock implementations that match the pydantic_ai API for testing
agent orchestration without requiring the actual pydantic_ai package.
"""

from __future__ import annotations

import asyncio
from typing import Any


class MockTool:
    """Mock Tool for testing without pydantic_ai."""

    def __init__(
        self,
        function: Any = None,
        name: str | None = None,
        description: str | None = None,
        takes_ctx: bool = False,
        max_retries: int | None = None,
    ) -> None:
        self.function = function or (lambda: None)
        self.name = name or (getattr(function, "__name__", "mock_tool") if function else "mock_tool")
        self.description = description or ""
        self.takes_ctx = takes_ctx
        self.max_retries = max_retries

    async def run(self, *args: object, **kwargs: object) -> Any:
        """Execute the tool function."""
        if asyncio.iscoroutinefunction(self.function):
            return await self.function(*args, **kwargs)
        return self.function(*args, **kwargs)


class MockAgent:
    """Mock Agent for testing without pydantic_ai.

    Provides a configurable mock that returns preset responses and tracks
    calls for test assertions. Implements the key pydantic_ai Agent methods:
    - run() -> RunResult with .data, .all_messages(), .usage
    - run_sync() -> RunResult
    """

    def __init__(
        self,
        model: Any = None,
        *,
        system_prompt: str | None = None,
        name: str | None = None,
        output_type: Any = None,
        tools: list[Any] | None = None,
        retries: int = 1,
        model_settings: Any = None,
        response_data: Any = "mock response",
        **kwargs: object,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.name = name
        self.output_type = output_type
        self.tools = tools or []
        self.retries = retries
        self.model_settings = model_settings
        self.response_data = response_data
        self.run_count = 0
        self.last_prompt: str | None = None
        self.last_message_history: list[Any] | None = None

    async def run(
        self,
        user_prompt: str | None = None,
        *,
        message_history: list[Any] | None = None,
        model: Any = None,
        usage_limits: Any = None,
        usage: Any = None,
        model_settings: Any = None,
    ) -> MockRunResult:
        """Mock async run that returns a preset result."""
        self.run_count += 1
        self.last_prompt = user_prompt
        self.last_message_history = message_history
        self.last_model_settings = model_settings

        result = MockRunResult(data=self.response_data)
        return result

    def run_sync(
        self,
        user_prompt: str | None = None,
        *,
        message_history: list[Any] | None = None,
        model: Any = None,
        usage_limits: Any = None,
        usage: Any = None,
        model_settings: Any = None,
    ) -> MockRunResult:
        """Mock sync run that returns a preset result."""
        self.run_count += 1
        self.last_prompt = user_prompt
        self.last_message_history = message_history
        self.last_model_settings = model_settings

        return MockRunResult(data=self.response_data)


class MockRunResult:
    """Mock RunResult for Agent.run() return value.

    Implements the pydantic_ai RunResult interface:
    - .data: The result data
    - .all_messages(): List of messages in the conversation
    - .usage(): Usage statistics
    """

    def __init__(
        self,
        data: Any = None,
        messages: list[Any] | None = None,
    ) -> None:
        self.data = data
        self._messages = messages or []

    def all_messages(self) -> list[Any]:
        """Return all messages in the conversation."""
        return self._messages

    def usage(self) -> MockUsage:
        """Return usage statistics."""
        return MockUsage()


class MockUsage:
    """Mock usage statistics."""

    def __init__(
        self,
        request_tokens: int = 0,
        response_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self.request_tokens = request_tokens
        self.response_tokens = response_tokens
        self.total_tokens = total_tokens


# Fixture factory for pytest
def create_mock_agent_fixture(response_data: Any = "mock response") -> MockAgent:
    """Create a MockAgent with configurable response.

    Args:
        response_data: Data to return from run()

    Returns:
        Configured MockAgent instance
    """
    return MockAgent(response_data=response_data)
