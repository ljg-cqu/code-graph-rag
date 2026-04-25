from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from prompt_toolkit.key_binding import KeyBindings

from codebase_rag.main import (
    InputCancelled,
    get_multiline_input,
    get_multiline_input_async,
)

pytestmark = [pytest.mark.anyio]


@pytest.fixture(params=["asyncio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_get_multiline_input_uses_simple_prompt_toolkit_path() -> None:
    with patch("codebase_rag.main.prompt", return_value="  hello  ") as prompt_mock:
        result = get_multiline_input()

    assert result == "hello"
    args, kwargs = prompt_mock.call_args
    assert args
    assert kwargs["multiline"] is False
    assert isinstance(kwargs["key_bindings"], KeyBindings)
    assert kwargs["wrap_lines"] is True


def test_get_multiline_input_does_not_enable_completion_ui() -> None:
    with patch("codebase_rag.main.prompt", return_value="hello") as prompt_mock:
        get_multiline_input()

    _, kwargs = prompt_mock.call_args
    assert "completer" not in kwargs
    assert "prompt_continuation" not in kwargs


async def test_get_multiline_input_async_uses_sync_prompt_path() -> None:
    with patch(
        "codebase_rag.main.asyncio.to_thread",
        new=AsyncMock(return_value="hello"),
    ) as to_thread_mock:
        result = await get_multiline_input_async()

    assert result == "hello"
    to_thread_mock.assert_awaited_once_with(get_multiline_input, "Ask a question")


async def test_get_multiline_input_async_preserves_input_cancelled() -> None:
    with patch(
        "codebase_rag.main.asyncio.to_thread",
        new=AsyncMock(side_effect=InputCancelled),
    ):
        with pytest.raises(InputCancelled):
            await get_multiline_input_async()