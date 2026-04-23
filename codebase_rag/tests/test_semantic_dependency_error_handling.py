from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from codebase_rag.config import settings
from codebase_rag.embedder import (
    clear_embedding_cache,
    embed_code,
    embed_code_batch,
    get_embedding_provider_instance,
)
from codebase_rag.exceptions import EmbeddingGenerationError


class TestSemanticDependencyErrorHandling:
    """Test semantic dependency error handling when torch/transformers are missing.

    Per LLM-First Design: dependency availability is deterministic infrastructure.
    These tests verify that missing torch/transformers fails fast with actionable
    messages, not truncated or unclear errors.
    """

    @patch.dict(sys.modules, {"torch": None}, clear=False)
    def test_check_local_embedding_available_returns_false_on_import_error(
        self,
    ) -> None:
        from codebase_rag.embeddings.local import check_local_embedding_available

        available, error = check_local_embedding_available()
        assert not available
        assert "torch and transformers" in error
        assert "uv sync --extra semantic" in error
        assert "pip install 'code-graph-rag[semantic]'" in error
        assert "EMBEDDING_PROVIDER=openai or ollama" in error

    def test_check_local_embedding_available_returns_true_when_installed(
        self,
    ) -> None:
        from codebase_rag.embeddings.local import check_local_embedding_available

        available, error = check_local_embedding_available()
        if available:
            assert error is None
        else:
            pytest.skip("torch/transformers not installed in this environment")

    @patch.dict(sys.modules, {"torch": None}, clear=False)
    def test_local_provider_validate_config_raises(self) -> None:
        from codebase_rag.embeddings.local import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(model_id="microsoft/unixcoder-base")
        with pytest.raises(EmbeddingGenerationError, match="torch and transformers"):
            provider.validate_config()

    @patch.dict(sys.modules, {"torch": None}, clear=False)
    def test_get_embedding_provider_instance_raises(self) -> None:
        clear_embedding_cache()
        with patch.object(settings, "EMBEDDING_PROVIDER", "local"):
            with pytest.raises(RuntimeError, match="torch and transformers"):
                get_embedding_provider_instance()
        clear_embedding_cache()

    @patch.dict(sys.modules, {"torch": None}, clear=False)
    def test_embed_code_raises_without_dependencies(self) -> None:
        with pytest.raises(RuntimeError, match="Local embedding requires"):
            embed_code("x = 1")

    @patch.dict(sys.modules, {"torch": None}, clear=False)
    def test_embed_code_batch_raises_without_dependencies(self) -> None:
        with pytest.raises(RuntimeError, match="Local embedding requires"):
            embed_code_batch(["x = 1"])
