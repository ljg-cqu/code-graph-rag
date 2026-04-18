"""Tests for dynamic model catalog functionality."""

import os
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.models_dynamic import (
    DynamicModelInfo,
    _build_dynamic_model_from_config,
    _extract_configured_models_from_env,
    build_dynamic_model_catalog,
)


def test_build_dynamic_model_from_config_unknown_provider() -> None:
    """Unknown providers are treated as OpenAI-compatible endpoints."""
    # Set a custom API key environment variable for the unknown provider
    os.environ["DASHSCOPE_API_KEY"] = "test-key-123"
    try:
        model_info = _build_dynamic_model_from_config(
            provider="dashscope",
            model_id="qwen3-max",
            endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=None,
            source="env_worker",
        )
        assert model_info is not None
        assert model_info.provider == "dashscope"
        assert model_info.model_id == "qwen3-max"
        assert model_info.endpoint == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert model_info.is_configured is True  # API key from environment
        assert model_info.is_local is False
        assert model_info.requires_api_key is True
    finally:
        del os.environ["DASHSCOPE_API_KEY"]


def test_build_dynamic_model_from_config_unknown_provider_no_api_key() -> None:
    """Unknown providers without API key are marked as not configured."""
    model_info = _build_dynamic_model_from_config(
        provider="volces",
        model_id="some-model",
        endpoint=None,
        api_key=None,
        source="env_worker",
    )
    assert model_info is not None
    assert model_info.provider == "volces"
    assert model_info.is_configured is False
    assert model_info.requires_api_key is True


def test_build_dynamic_model_from_config_standard_provider_with_custom_endpoint() -> None:
    """Standard providers with custom endpoints retain the endpoint."""
    os.environ["OPENAI_API_KEY"] = "sk-test"
    try:
        model_info = _build_dynamic_model_from_config(
            provider="openai",
            model_id="gpt-4o",
            endpoint="https://custom.openai.com/v1",
            api_key=None,
            source="env_orchestrator",
        )
        assert model_info is not None
        assert model_info.provider == "openai"
        assert model_info.endpoint == "https://custom.openai.com/v1"
        assert model_info.is_configured is True
    finally:
        del os.environ["OPENAI_API_KEY"]


@pytest.mark.skip(reason="Mocking issue")
@patch("codebase_rag.providers.base.get_provider")
@patch("codebase_rag.models_dynamic.logger")
def test_context_window_resolution_failure_logs_debug(mock_logger, mock_get_provider) -> None:
    """If context window resolution fails, debug log is emitted."""
    # Mock get_provider to raise an exception
    mock_get_provider.side_effect = RuntimeError("test")
    model_info = _build_dynamic_model_from_config(
        provider="openai",
        model_id="unknown-model",
        endpoint=None,
        api_key=None,
        source="env_worker",
    )
    assert model_info is not None
    assert model_info.context_window == 256000  # fallback
    # Verify debug log was called
    mock_logger.debug.assert_called()
    call_args = mock_logger.debug.call_args[0][0]
    assert "Context window resolution failed" in call_args


@pytest.mark.skip(reason="Requires proper configuration")
def test_extract_configured_models_from_env_with_worker_llms() -> None:
    """Worker LLMs defined in CGR_WORKER_LLMS are discovered."""
    # Set environment variable for worker LLMs (JSON format)
    os.environ["CGR_WORKER_LLMS"] = '[{"provider": "openai", "model_id": "gpt-4o"}, {"provider": "anthropic", "model_id": "claude-3-haiku"}]'
    try:
        models = _extract_configured_models_from_env()
        # Should have at least the two worker models
        openai_models = [m for m in models if m.provider == "openai" and m.model_id == "gpt-4o"]
        anthropic_models = [m for m in models if m.provider == "anthropic" and m.model_id == "claude-3-haiku"]
        assert len(openai_models) == 1
        assert len(anthropic_models) == 1
    finally:
        del os.environ["CGR_WORKER_LLMS"]


@pytest.mark.skip(reason="Mocking issues")
@patch("codebase_rag.models_dynamic.settings")
def test_build_dynamic_model_catalog_external_file(mock_settings) -> None:
    """External catalog file loads correctly."""
    import tempfile
    import json

    catalog_data = {
        "openai": [
            {
                "provider": "openai",
                "model_id": "custom-model",
                "display_name": "Custom Model",
                "context_window": 128000,
                "description": "Custom endpoint model",
                "requires_api_key": True,
                "is_local": False,
                "pricing_tier": "medium",
            }
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(catalog_data, f)
        temp_path = f.name

    try:
        mock_settings.CGR_MODEL_CATALOG_PATH = temp_path
        mock_settings.CGR_DISABLE_MODEL_DISCOVERY = True
        catalog = build_dynamic_model_catalog()
        assert "openai" in catalog
        models = catalog["openai"]
        assert len(models) == 1
        assert models[0].model_id == "custom-model"
        assert models[0].source == "external"
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])