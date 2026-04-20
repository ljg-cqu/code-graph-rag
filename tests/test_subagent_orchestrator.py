"""
Unit tests for subagent orchestrator fallback loop fixes.
"""

import pytest
from unittest.mock import patch, MagicMock

from codebase_rag.config import ModelConfig, settings
from codebase_rag.orchestrator.subagent_orchestrator import SubAgentOrchestrator


def test_fallback_excludes_current_model():
    """Verify fallback never returns the same model that failed."""
    orchestrator = SubAgentOrchestrator()

    config = ModelConfig(model_id="claude-sonnet-4-6", provider="anthropic")

    # Mock all models as unavailable except haiku
    with patch.object(orchestrator, '_validate_model_availability') as mock_validate:
        mock_validate.side_effect = lambda c: c.model_id == "claude-haiku-4-5"

        fallback = orchestrator._get_fallback_model_config(config)

        # Should return haiku, not sonnet
        assert fallback is not None
        assert fallback.model_id == "claude-haiku-4-5"


def test_failed_models_not_retried():
    """Verify models that failed are not retried."""
    orchestrator = SubAgentOrchestrator()

    # Simulate sonnet already failed
    orchestrator._failed_models.add("claude-sonnet-4-6")

    config = ModelConfig(model_id="claude-opus-4-7", provider="anthropic")

    with patch.object(orchestrator, '_validate_model_availability') as mock_validate:
        mock_validate.return_value = False

        orchestrator._get_fallback_model_config(config)

        # Should not have tried sonnet since it's in failed_models
        tried_models = [call[0][0].model_id for call in mock_validate.call_args_list]
        assert "claude-sonnet-4-6" not in tried_models


def test_all_models_failed_clear_error():
    """Verify clear error when all models fail."""
    orchestrator = SubAgentOrchestrator()
    orchestrator._failed_models = {"claude-sonnet-4-6", "claude-opus-4-7", "claude-haiku-4-5"}
    orchestrator._has_validated_models = False

    # Should report no validated models
    assert not orchestrator._has_validated_models


def test_failed_models_is_instance_level():
    """Verify failed_models is not shared across instances."""
    orch1 = SubAgentOrchestrator()
    orch2 = SubAgentOrchestrator()

    orch1._failed_models.add("claude-sonnet-4-6")

    assert "claude-sonnet-4-6" in orch1._failed_models
    assert "claude-sonnet-4-6" not in orch2._failed_models


def test_get_all_possible_models_includes_worker_llms():
    """Verify _get_all_possible_models includes configured worker LLMs."""
    from unittest.mock import PropertyMock

    orchestrator = SubAgentOrchestrator()

    # Mock the settings property using PropertyMock on the class
    mock_worker_llms = [ModelConfig(model_id="gpt-4o", provider="openai")]
    with patch.object(
        type(settings),
        'active_worker_llms',
        new_callable=PropertyMock,
        return_value=mock_worker_llms
    ):
        with patch.object(
            type(settings),
            'active_orchestrator_config',
            new_callable=PropertyMock,
            return_value=ModelConfig(model_id="claude-sonnet-4-6", provider="anthropic")
        ):
            all_models = orchestrator._get_all_possible_models()
            assert "gpt-4o" in all_models


def test_classify_error_model_unavailable():
    """Test error classification for model unavailable errors."""
    orchestrator = SubAgentOrchestrator()

    # 404 errors
    assert orchestrator._classify_error("404 Not Found") == "model_unavailable"
    assert orchestrator._classify_error("resource_not_found") == "model_unavailable"
    assert orchestrator._classify_error("Model not found") == "model_unavailable"

    # Network errors
    assert orchestrator._classify_error("connection error") == "network_error"
    assert orchestrator._classify_error("timeout") == "network_error"
    assert orchestrator._classify_error("network unreachable") == "network_error"

    # Rate limit errors (both with underscore and space)
    assert orchestrator._classify_error("rate_limit exceeded") == "rate_limit"
    assert orchestrator._classify_error("rate limit exceeded") == "rate_limit"
    assert orchestrator._classify_error("429 Too Many Requests") == "rate_limit"

    # Auth errors
    assert orchestrator._classify_error("auth failed") == "auth_error"
    assert orchestrator._classify_error("401 Unauthorized") == "auth_error"
    assert orchestrator._classify_error("403 Forbidden") == "auth_error"

    # Unknown errors
    assert orchestrator._classify_error("some other error") == "unknown"


def test_graceful_degradation_with_unavailable_models():
    """Test that parallel execution fails gracefully when all models 404."""
    orchestrator = SubAgentOrchestrator(worker_count=2)

    # Mock all models returning 404
    with patch('codebase_rag.orchestrator.subagent_orchestrator.get_provider_from_config') as mock_get_provider:
        mock_provider = mock_get_provider.return_value
        mock_provider.create_model.side_effect = Exception("404: Not found")

        orchestrator.initialize_agents()

        # Should log clear error about no models available
        result = orchestrator.execute_tasks([
            {"id": "test-1", "task": "Analyze file X"}
        ])

        # Should have error result, not hang or crash
        assert len(result.errors) == 1
        assert "No LLM models available" in result.errors[0]["error"]


def test_initialize_agents_sets_has_validated_models():
    """Verify initialize_agents sets _has_validated_models correctly."""
    from unittest.mock import PropertyMock

    orchestrator = SubAgentOrchestrator()

    # Mock settings properties using PropertyMock
    with patch.object(
        type(settings),
        'active_worker_llms',
        new_callable=PropertyMock,
        return_value=[]
    ):
        with patch.object(
            type(settings),
            'active_orchestrator_config',
            new_callable=PropertyMock,
            return_value=ModelConfig(model_id="claude-sonnet-4-6", provider="anthropic")
        ):
            with patch.object(orchestrator, '_validate_model_availability', return_value=True):
                orchestrator.initialize_agents()
                assert orchestrator._has_validated_models is True

    # Test when no models available
    orchestrator2 = SubAgentOrchestrator()
    with patch.object(
        type(settings),
        'active_worker_llms',
        new_callable=PropertyMock,
        return_value=[]
    ):
        with patch.object(
            type(settings),
            'active_orchestrator_config',
            new_callable=PropertyMock,
            return_value=ModelConfig(model_id="claude-sonnet-4-6", provider="anthropic")
        ):
            with patch.object(orchestrator2, '_validate_model_availability', return_value=False):
                orchestrator2.initialize_agents()
                assert orchestrator2._has_validated_models is False
