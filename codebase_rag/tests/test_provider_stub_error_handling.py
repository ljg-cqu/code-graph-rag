from __future__ import annotations

from unittest.mock import patch

import pytest

from codebase_rag.constants import GoogleProviderType
from codebase_rag.providers.base import (
    AnthropicProvider,
    AzureOpenAIProvider,
    GoogleProvider,
    OllamaProvider,
    OpenAIProvider,
)


class TestProviderStubErrorHandling:
    """Test provider error handling when pydantic_ai is not installed.

    Per LLM-First Design: dependency availability is deterministic infrastructure.
    These tests verify that missing pydantic_ai fails fast with actionable messages,
    not with cryptic TypeError from object() instantiation.
    """

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_openai_create_model_raises_without_pydantic_ai(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(RuntimeError, match="requires pydantic_ai"):
            provider.create_model("gpt-4o")

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_google_create_model_raises_without_pydantic_ai(self) -> None:
        provider = GoogleProvider(
            api_key="test-key", provider_type=GoogleProviderType.GLA
        )
        with pytest.raises(RuntimeError, match="requires pydantic_ai"):
            provider.create_model("gemini-2.5-pro")

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_ollama_create_model_raises_without_pydantic_ai(self) -> None:
        provider = OllamaProvider(endpoint="http://localhost:11434/v1")
        with pytest.raises(RuntimeError, match="requires pydantic_ai"):
            provider.create_model("llama3.2")

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_anthropic_create_model_raises_without_pydantic_ai(self) -> None:
        provider = AnthropicProvider(api_key="test-key")
        with pytest.raises(RuntimeError, match="requires pydantic_ai"):
            provider.create_model("claude-opus")

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_azure_create_model_raises_without_pydantic_ai(self) -> None:
        provider = AzureOpenAIProvider(
            api_key="test-key",
            endpoint="https://myresource.openai.azure.com",
        )
        with pytest.raises(RuntimeError, match="requires pydantic_ai"):
            provider.create_model("gpt-4o")

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_error_message_includes_install_instructions(self) -> None:
        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(RuntimeError) as exc_info:
            provider.create_model("gpt-4o")
        msg = str(exc_info.value)
        assert "uv sync --extra ai" in msg
        assert "pip install 'code-graph-rag[ai]'" in msg

    @patch("codebase_rag.compat.pydantic_ai.HAS_PYDANTIC_AI", False)
    def test_error_message_names_the_provider(self) -> None:
        provider = GoogleProvider(
            api_key="test-key", provider_type=GoogleProviderType.GLA
        )
        with pytest.raises(RuntimeError) as exc_info:
            provider.create_model("gemini-2.5-pro")
        assert "'google' requires pydantic_ai" in str(exc_info.value)
