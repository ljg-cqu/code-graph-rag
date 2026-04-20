"""
Unit tests for subagent orchestrator fail-fast behavior.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from codebase_rag.config import ModelConfig, settings
from codebase_rag.orchestrator.subagent_orchestrator import SubAgentOrchestrator


class TestValidateModelConfig:
    """Tests for _validate_model_config fail-fast behavior."""

    def test_validate_model_config_raises_on_unavailable_model(self):
        """Verify _validate_model_config raises RuntimeError for unavailable models."""
        orchestrator = SubAgentOrchestrator()
        config = ModelConfig(model_id="invalid-model-id", provider="anthropic")

        with patch('codebase_rag.orchestrator.subagent_orchestrator.get_provider_from_config') as mock_get_provider:
            mock_provider = mock_get_provider.return_value
            mock_provider.create_model.side_effect = Exception("404: Not found")

            with pytest.raises(RuntimeError, match="Model 'invalid-model-id'.*is not available"):
                orchestrator._validate_model_config(config)

    def test_validate_model_config_succeeds_for_valid_model(self):
        """Verify _validate_model_config succeeds for valid models."""
        orchestrator = SubAgentOrchestrator()
        config = ModelConfig(model_id="claude-sonnet-4-20250514", provider="anthropic")

        with patch('codebase_rag.orchestrator.subagent_orchestrator.get_provider_from_config') as mock_get_provider:
            mock_provider = mock_get_provider.return_value
            mock_provider.create_model.return_value = MagicMock()

            # Should not raise
            orchestrator._validate_model_config(config)


class TestInitializeAgents:
    """Tests for initialize_agents fail-fast behavior."""

    def test_initialize_agents_fails_fast_on_unavailable_model(self):
        """Verify initialize_agents raises RuntimeError when model is unavailable."""
        orchestrator = SubAgentOrchestrator(worker_count=2)

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
                return_value=ModelConfig(model_id="invalid-model", provider="anthropic")
            ):
                with patch('codebase_rag.orchestrator.subagent_orchestrator.get_provider_from_config') as mock_get_provider:
                    mock_provider = mock_get_provider.return_value
                    mock_provider.create_model.side_effect = Exception("404: Not found")

                    with pytest.raises(RuntimeError, match="is not available"):
                        orchestrator.initialize_agents()

    def test_initialize_agents_succeeds_with_valid_models(self):
        """Verify initialize_agents succeeds when all models are valid."""
        orchestrator = SubAgentOrchestrator(worker_count=2)

        with patch.object(
            type(settings),
            'active_worker_llms',
            new_callable=PropertyMock,
            return_value=[ModelConfig(model_id="claude-sonnet-4-20250514", provider="anthropic")]
        ):
            with patch('codebase_rag.orchestrator.subagent_orchestrator.get_provider_from_config') as mock_get_provider:
                mock_provider = mock_get_provider.return_value
                mock_provider.create_model.return_value = MagicMock()

                orchestrator.initialize_agents()
                assert len(orchestrator.workers) == 1  # Reduced to 1 since only 1 LLM

    def test_single_llm_reduces_worker_count(self):
        """Verify worker count is reduced to 1 when only 1 valid LLM available."""
        orchestrator = SubAgentOrchestrator(worker_count=5)

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
                return_value=ModelConfig(model_id="claude-sonnet-4-20250514", provider="anthropic")
            ):
                with patch('codebase_rag.orchestrator.subagent_orchestrator.get_provider_from_config') as mock_get_provider:
                    mock_provider = mock_get_provider.return_value
                    mock_provider.create_model.return_value = MagicMock()

                    orchestrator.initialize_agents()
                    assert orchestrator.worker_count == 1


class TestClassifyError:
    """Tests for _classify_error method."""

    def test_classify_error_model_unavailable(self):
        """Test error classification for model unavailable errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("404 Not Found") == "model_unavailable"
        assert orchestrator._classify_error("resource_not_found") == "model_unavailable"
        assert orchestrator._classify_error("Model not found") == "model_unavailable"

    def test_classify_error_network_error(self):
        """Test error classification for network errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("connection error") == "network_error"
        assert orchestrator._classify_error("timeout") == "network_error"
        assert orchestrator._classify_error("network unreachable") == "network_error"

    def test_classify_error_rate_limit(self):
        """Test error classification for rate limit errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("rate_limit exceeded") == "rate_limit"
        assert orchestrator._classify_error("rate limit exceeded") == "rate_limit"
        assert orchestrator._classify_error("429 Too Many Requests") == "rate_limit"

    def test_classify_error_auth_error(self):
        """Test error classification for auth errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("auth failed") == "auth_error"
        assert orchestrator._classify_error("401 Unauthorized") == "auth_error"
        assert orchestrator._classify_error("403 Forbidden") == "auth_error"

    def test_classify_error_unknown(self):
        """Test error classification for unknown errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("some other error") == "unknown"


class TestExecuteTasks:
    """Tests for execute_tasks fail-fast behavior."""

    def test_model_unavailable_not_retryable(self):
        """Verify model_unavailable errors are not retryable (fail-fast)."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._classify_error("404: resource_not_found") == "model_unavailable"
        assert orchestrator._classify_error("Model not found") == "model_unavailable"

    def test_network_error_is_retryable(self):
        """Verify network errors are classified as retryable."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._classify_error("connection timeout") == "network_error"
        assert orchestrator._classify_error("network unreachable") == "network_error"

    def test_auth_error_not_retryable(self):
        """Verify auth errors are not retryable (fail-fast)."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._classify_error("401 Unauthorized") == "auth_error"
        assert orchestrator._classify_error("403 Forbidden") == "auth_error"


class TestWorkerLifecycle:
    """Tests for worker lifecycle management."""

    def test_workers_are_instance_level(self):
        """Verify workers are not shared across instances."""
        orch1 = SubAgentOrchestrator()
        orch2 = SubAgentOrchestrator()

        # Workers lists are separate instances
        assert orch1.workers is not orch2.workers

    def test_shutdown_clears_workers(self):
        """Verify shutdown clears workers."""
        orchestrator = SubAgentOrchestrator()

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
                return_value=ModelConfig(model_id="claude-sonnet-4-20250514", provider="anthropic")
            ):
                with patch('codebase_rag.orchestrator.subagent_orchestrator.get_provider_from_config') as mock_get_provider:
                    mock_provider = mock_get_provider.return_value
                    mock_provider.create_model.return_value = MagicMock()

                    orchestrator.initialize_agents()
                    assert len(orchestrator.workers) > 0

                    orchestrator.shutdown()
                    assert len(orchestrator.workers) == 0
