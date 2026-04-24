"""Tests verifying the architectural boundary between chat and embedding models."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.config import EmbeddingConfig, ModelConfig
from codebase_rag.embeddings import get_embedding_provider
from codebase_rag.services.llm import _create_chat_model


class TestChatModelFactory:
    """Verify _create_chat_model only accepts ModelConfig."""

    def test_accepts_model_config(self) -> None:
        """ModelConfig should be accepted by chat factory."""
        chat_config = ModelConfig(provider="openai", model_id="gpt-4o")
        with patch("codebase_rag.services.llm.get_provider_from_config") as mock_get:
            mock_provider = MagicMock()
            mock_provider.create_model.return_value = MagicMock()
            mock_get.return_value = mock_provider

            result = _create_chat_model(chat_config)

            mock_get.assert_called_once_with(chat_config)
            mock_provider.create_model.assert_called_once_with("gpt-4o")
            assert result is mock_provider.create_model.return_value

    def test_rejects_embedding_config_type(self) -> None:
        """EmbeddingConfig must not be accepted by chat factory at runtime."""
        embed_config = EmbeddingConfig(provider="local", model_id="test-model")
        # The type system prevents this; runtime behavior is undefined
        # because EmbeddingConfig lacks the interface ModelConfig provides
        with pytest.raises((TypeError, AttributeError)):
            _create_chat_model(embed_config)  # type: ignore[arg-type]


class TestEmbeddingProviderFactory:
    """Verify get_embedding_provider only accepts EmbeddingConfig."""

    def test_accepts_embedding_config(self) -> None:
        """EmbeddingConfig should be accepted by embedding factory."""
        embed_config = EmbeddingConfig(provider="local", model_id="microsoft/unixcoder-base")
        with patch("codebase_rag.embeddings.get_embedding_provider_class") as mock_get_cls:
            mock_provider = MagicMock()
            mock_provider.return_value.dimension = 768
            mock_get_cls.return_value = mock_provider

            result = get_embedding_provider(config=embed_config)

            mock_get_cls.assert_called_once_with("local")
            assert result is mock_provider.return_value

    def test_rejects_model_config_type(self) -> None:
        """ModelConfig must not be accepted by embedding factory at runtime."""
        chat_config = ModelConfig(provider="openai", model_id="gpt-4o")
        # The type system prevents this; runtime behavior is undefined
        # because ModelConfig lacks the to_provider_kwargs method EmbeddingConfig provides
        with pytest.raises((TypeError, AttributeError)):
            get_embedding_provider(config=chat_config)  # type: ignore[arg-type]


class TestConceptExtractorUsesChatPath:
    """Verify LLMConceptExtractor uses the chat model configuration path."""

    def test_uses_orchestrator_config(self) -> None:
        """Extractor must reference active_orchestrator_config (ModelConfig)."""
        from codebase_rag.document.concept_extraction import LLMConceptExtractor

        extractor = LLMConceptExtractor()
        assert extractor.agent is None

        with (
            patch("codebase_rag.config.settings") as mock_settings,
            patch("codebase_rag.services.llm._create_chat_model") as mock_create,
            patch("codebase_rag.compat.pydantic_ai.Agent") as mock_agent_cls,
        ):
            mock_orchestrator_config = ModelConfig(provider="openai", model_id="gpt-4o")
            mock_settings.active_orchestrator_config = mock_orchestrator_config
            mock_create.return_value = MagicMock()

            result = extractor._initialize_agent()

            assert result is True
            mock_create.assert_called_once_with(mock_orchestrator_config)
            mock_agent_cls.assert_called_once()
            call_kwargs = mock_agent_cls.call_args.kwargs
            assert call_kwargs["model"] is mock_create.return_value
            assert call_kwargs["output_type"] is not None


class TestConfigTypeSeparation:
    """Verify ModelConfig and EmbeddingConfig are distinct incompatible types."""

    def test_model_config_has_no_dimension(self) -> None:
        """ModelConfig does not have a dimension field — embedding-specific."""
        chat_config = ModelConfig(provider="openai", model_id="gpt-4o")
        assert not hasattr(chat_config, "dimension")

    def test_embedding_config_has_dimension(self) -> None:
        """EmbeddingConfig has a dimension field for vector size."""
        embed_config = EmbeddingConfig(provider="local", model_id="test")
        assert hasattr(embed_config, "dimension")
        assert embed_config.dimension is None  # Defaults to auto-detect

    def test_embedding_config_has_fallback_fields(self) -> None:
        """EmbeddingConfig has fallback fields not present in ModelConfig."""
        embed_config = EmbeddingConfig(provider="local", model_id="test")
        assert hasattr(embed_config, "fallback_to_local")
        assert hasattr(embed_config, "fallback_model")

        chat_config = ModelConfig(provider="openai", model_id="gpt-4o")
        assert not hasattr(chat_config, "fallback_to_local")
        assert not hasattr(chat_config, "fallback_model")

    def test_model_config_has_thinking_budget(self) -> None:
        """ModelConfig has thinking_budget for reasoning models."""
        chat_config = ModelConfig(provider="anthropic", model_id="claude-3-7-sonnet")
        assert hasattr(chat_config, "thinking_budget")

        embed_config = EmbeddingConfig(provider="local", model_id="test")
        assert not hasattr(embed_config, "thinking_budget")
