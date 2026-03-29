"""Unit tests for embedding providers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.embeddings import (
    get_embedding_provider,
    get_embedding_provider_class,
    _EMBEDDING_PROVIDER_REGISTRY,
)
from codebase_rag.embeddings.base import EmbeddingProvider
from codebase_rag.exceptions import (
    EmbeddingAuthenticationError,
    EmbeddingConnectionError,
    EmbeddingProviderNotFoundError,
)


class TestProviderRegistry:
    """Tests for the provider registry."""

    def test_registry_has_local_provider(self) -> None:
        """Local provider should always be registered."""
        assert "local" in _EMBEDDING_PROVIDER_REGISTRY

    def test_registry_has_openai_provider(self) -> None:
        """OpenAI provider should be registered."""
        assert "openai" in _EMBEDDING_PROVIDER_REGISTRY

    def test_registry_has_google_provider(self) -> None:
        """Google provider should be registered."""
        assert "google" in _EMBEDDING_PROVIDER_REGISTRY

    def test_registry_has_ollama_provider(self) -> None:
        """Ollama provider should be registered."""
        assert "ollama" in _EMBEDDING_PROVIDER_REGISTRY

    def test_get_provider_class_unknown_raises(self) -> None:
        """Unknown provider should raise EmbeddingProviderNotFoundError."""
        with pytest.raises(EmbeddingProviderNotFoundError) as exc_info:
            get_embedding_provider_class("unknown_provider")
        assert "Unknown embedding provider" in str(exc_info.value)


class TestLocalEmbeddingProvider:
    """Tests for the local embedding provider."""

    def test_provider_name(self) -> None:
        """Provider name should be 'local'."""
        provider = get_embedding_provider("local", "microsoft/unixcoder-base")
        assert provider.provider_name.value == "local"

    def test_default_dimension(self) -> None:
        """Default dimension for UniXcoder should be 768."""
        provider = get_embedding_provider("local", "microsoft/unixcoder-base")
        assert provider.dimension == 768

    def test_custom_dimension(self) -> None:
        """Should accept custom dimension override."""
        provider = get_embedding_provider("local", "microsoft/unixcoder-base", dimension=1024)
        assert provider.dimension == 1024

    def test_known_model_dimensions(self) -> None:
        """Known models should have correct dimensions."""
        test_cases = [
            ("microsoft/unixcoder-base", 768),
            ("sentence-transformers/all-MiniLM-L6-v2", 384),
            ("BAAI/bge-small-en-v1.5", 384),
            ("BAAI/bge-large-en-v1.5", 1024),
        ]
        for model_id, expected_dim in test_cases:
            provider = get_embedding_provider("local", model_id)
            assert provider.dimension == expected_dim, f"Failed for {model_id}"

    def test_unknown_model_defaults_to_768(self) -> None:
        """Unknown models should default to 768."""
        provider = get_embedding_provider("local", "unknown-model-xyz")
        assert provider.dimension == 768


class TestOpenAIEmbeddingProvider:
    """Tests for the OpenAI embedding provider."""

    def test_provider_name(self) -> None:
        """Provider name should be 'openai'."""
        provider = get_embedding_provider(
            "openai", "text-embedding-3-small", api_key="test-key"
        )
        assert provider.provider_name.value == "openai"

    def test_default_dimension(self) -> None:
        """Default dimension for text-embedding-3-small should be 1536."""
        provider = get_embedding_provider(
            "openai", "text-embedding-3-small", api_key="test-key"
        )
        assert provider.dimension == 1536

    def test_text_embedding_3_large_dimension(self) -> None:
        """text-embedding-3-large should have dimension 3072."""
        provider = get_embedding_provider(
            "openai", "text-embedding-3-large", api_key="test-key"
        )
        assert provider.dimension == 3072

    def test_validate_config_missing_api_key_raises(self) -> None:
        """Missing API key should raise EmbeddingAuthenticationError."""
        provider = get_embedding_provider("openai", "text-embedding-3-small", api_key=None)
        with pytest.raises(EmbeddingAuthenticationError):
            provider.validate_config()

    def test_validate_config_empty_api_key_raises(self) -> None:
        """Empty API key should raise EmbeddingAuthenticationError."""
        provider = get_embedding_provider("openai", "text-embedding-3-small", api_key="")
        with pytest.raises(EmbeddingAuthenticationError):
            provider.validate_config()

    def test_validate_config_whitespace_api_key_raises(self) -> None:
        """Whitespace-only API key should raise EmbeddingAuthenticationError."""
        provider = get_embedding_provider("openai", "text-embedding-3-small", api_key="   ")
        with pytest.raises(EmbeddingAuthenticationError):
            provider.validate_config()

    def test_validate_config_with_valid_key(self) -> None:
        """Valid API key should pass validation."""
        provider = get_embedding_provider(
            "openai", "text-embedding-3-small", api_key="sk-test-key"
        )
        # Should not raise
        provider.validate_config()

    def test_custom_endpoint(self) -> None:
        """Should accept custom endpoint."""
        provider = get_embedding_provider(
            "openai",
            "text-embedding-3-small",
            api_key="test-key",
            endpoint="https://custom.openai.com/v1/embeddings",
        )
        assert provider._endpoint == "https://custom.openai.com/v1/embeddings"


class TestOllamaEmbeddingProvider:
    """Tests for the Ollama embedding provider."""

    def test_provider_name(self) -> None:
        """Provider name should be 'ollama'."""
        provider = get_embedding_provider("ollama", "nomic-embed-text")
        assert provider.provider_name.value == "ollama"

    def test_default_dimension(self) -> None:
        """Default dimension for nomic-embed-text should be 768."""
        provider = get_embedding_provider("ollama", "nomic-embed-text")
        assert provider.dimension == 768

    def test_mxbai_embed_large_dimension(self) -> None:
        """mxbai-embed-large should have dimension 1024."""
        provider = get_embedding_provider("ollama", "mxbai-embed-large")
        assert provider.dimension == 1024

    def test_default_endpoint(self) -> None:
        """Default endpoint should be localhost:11434."""
        provider = get_embedding_provider("ollama", "nomic-embed-text")
        assert "localhost:11434" in provider._endpoint

    def test_custom_endpoint(self) -> None:
        """Should accept custom endpoint."""
        provider = get_embedding_provider(
            "ollama", "nomic-embed-text", endpoint="http://custom:11434/api/embeddings"
        )
        assert provider._endpoint == "http://custom:11434/api/embeddings"

    def test_keep_alive_config(self) -> None:
        """Should accept keep_alive configuration."""
        provider = get_embedding_provider(
            "ollama", "nomic-embed-text", keep_alive="10m"
        )
        assert provider._keep_alive == "10m"

    @patch("httpx.Client.get")
    def test_validate_config_connection_error(self, mock_get: MagicMock) -> None:
        """Connection error should raise EmbeddingConnectionError."""
        mock_get.side_effect = Exception("Connection refused")
        provider = get_embedding_provider("ollama", "nomic-embed-text")
        with pytest.raises(EmbeddingConnectionError):
            provider.validate_config()


class TestGoogleEmbeddingProvider:
    """Tests for the Google embedding provider."""

    def test_provider_name(self) -> None:
        """Provider name should be 'google'."""
        provider = get_embedding_provider(
            "google", "text-embedding-004", api_key="test-key"
        )
        assert provider.provider_name.value == "google"

    def test_default_dimension(self) -> None:
        """Default dimension for text-embedding-004 should be 768."""
        provider = get_embedding_provider(
            "google", "text-embedding-004", api_key="test-key"
        )
        assert provider.dimension == 768

    def test_gla_provider_type(self) -> None:
        """GLA provider type should be default."""
        provider = get_embedding_provider(
            "google", "text-embedding-004", api_key="test-key", provider_type="gla"
        )
        assert provider._provider_type == "gla"

    def test_vertex_provider_type(self) -> None:
        """Vertex provider type should be accepted."""
        provider = get_embedding_provider(
            "google",
            "text-embedding-004",
            api_key="test-key",
            provider_type="vertex",
            project_id="test-project",
        )
        assert provider._provider_type == "vertex"

    def test_validate_config_gla_missing_api_key_raises(self) -> None:
        """Missing API key for GLA should raise EmbeddingAuthenticationError."""
        provider = get_embedding_provider(
            "google", "text-embedding-004", api_key=None, provider_type="gla"
        )
        with pytest.raises(EmbeddingAuthenticationError):
            provider.validate_config()

    def test_validate_config_vertex_missing_project_raises(self) -> None:
        """Missing project_id for Vertex should raise EmbeddingAuthenticationError."""
        provider = get_embedding_provider(
            "google",
            "text-embedding-004",
            api_key="test-key",
            provider_type="vertex",
            project_id=None,
        )
        with pytest.raises(EmbeddingAuthenticationError):
            provider.validate_config()

    def test_validate_config_unknown_provider_type_raises(self) -> None:
        """Unknown provider type should raise EmbeddingAuthenticationError."""
        provider = get_embedding_provider(
            "google",
            "text-embedding-004",
            api_key="test-key",
            provider_type="unknown",
        )
        with pytest.raises(EmbeddingAuthenticationError) as exc_info:
            provider.validate_config()
        assert "Unknown Google provider type" in str(exc_info.value)

    def test_region_config(self) -> None:
        """Should accept region configuration."""
        provider = get_embedding_provider(
            "google",
            "text-embedding-004",
            api_key="test-key",
            region="europe-west1",
        )
        assert provider._region == "europe-west1"


class TestProviderFactory:
    """Tests for the provider factory function."""

    def test_factory_creates_local_provider(self) -> None:
        """Factory should create LocalEmbeddingProvider."""
        from codebase_rag.embeddings.local import LocalEmbeddingProvider

        provider = get_embedding_provider("local", "microsoft/unixcoder-base")
        assert isinstance(provider, LocalEmbeddingProvider)

    def test_factory_creates_openai_provider(self) -> None:
        """Factory should create OpenAIEmbeddingProvider."""
        from codebase_rag.embeddings.openai import OpenAIEmbeddingProvider

        provider = get_embedding_provider(
            "openai", "text-embedding-3-small", api_key="test-key"
        )
        assert isinstance(provider, OpenAIEmbeddingProvider)

    def test_factory_creates_ollama_provider(self) -> None:
        """Factory should create OllamaEmbeddingProvider."""
        from codebase_rag.embeddings.ollama import OllamaEmbeddingProvider

        provider = get_embedding_provider("ollama", "nomic-embed-text")
        assert isinstance(provider, OllamaEmbeddingProvider)

    def test_factory_creates_google_provider(self) -> None:
        """Factory should create GoogleEmbeddingProvider."""
        from codebase_rag.embeddings.google import GoogleEmbeddingProvider

        provider = get_embedding_provider(
            "google", "text-embedding-004", api_key="test-key"
        )
        assert isinstance(provider, GoogleEmbeddingProvider)

    def test_factory_passes_dimension(self) -> None:
        """Factory should pass dimension to provider."""
        provider = get_embedding_provider(
            "openai", "text-embedding-3-small", api_key="test-key", dimension=512
        )
        assert provider.dimension == 512

    def test_factory_unknown_provider_raises(self) -> None:
        """Factory should raise for unknown provider."""
        with pytest.raises(EmbeddingProviderNotFoundError):
            get_embedding_provider("unknown", "model-id")


class TestEmbeddingProviderBase:
    """Tests for the EmbeddingProvider base class."""

    def test_model_id_attribute(self) -> None:
        """Provider should have model_id attribute."""
        provider = get_embedding_provider("local", "microsoft/unixcoder-base")
        assert provider.model_id == "microsoft/unixcoder-base"

    def test_dimension_attribute(self) -> None:
        """Provider should have dimension attribute."""
        provider = get_embedding_provider("local", "microsoft/unixcoder-base")
        assert hasattr(provider, "dimension")

    def test_get_config(self) -> None:
        """Provider should store config and allow retrieval."""
        # Use Ollama which has keep_alive as a valid config parameter
        provider = get_embedding_provider(
            "ollama", "nomic-embed-text", keep_alive="10m"
        )
        assert provider.get_config("keep_alive") == "10m"
        assert provider.get_config("nonexistent", "default") == "default"