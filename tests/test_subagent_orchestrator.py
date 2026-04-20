"""
Unit tests for subagent orchestrator fail-fast behavior.
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock

from codebase_rag.config import ModelConfig, settings
from codebase_rag.orchestrator.subagent_orchestrator import SubAgentOrchestrator, SubagentErrorType


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
    """Tests for _classify_error method with SubagentErrorType constants."""

    def test_classify_error_model_not_found(self):
        """Test error classification for model not found errors (404 with model mention)."""
        orchestrator = SubAgentOrchestrator()

        # 404 errors with "model" and "not found"/"unknown"/"invalid" should be MODEL_NOT_FOUND
        assert orchestrator._classify_error("404 Model not found") == SubagentErrorType.MODEL_NOT_FOUND
        assert orchestrator._classify_error("404 Model unknown") == SubagentErrorType.MODEL_NOT_FOUND
        assert orchestrator._classify_error("Invalid model", status_code=404) == SubagentErrorType.MODEL_NOT_FOUND

    def test_classify_error_endpoint_not_found(self):
        """Test error classification for endpoint not found errors."""
        orchestrator = SubAgentOrchestrator()

        # 404 errors with "endpoint" should be ENDPOINT_NOT_FOUND
        assert orchestrator._classify_error("404 Endpoint not found") == SubagentErrorType.ENDPOINT_NOT_FOUND
        assert orchestrator._classify_error("Invalid URL", status_code=404) == SubagentErrorType.ENDPOINT_NOT_FOUND

    def test_classify_error_resource_unavailable(self):
        """Test error classification for generic 404 resource unavailable."""
        orchestrator = SubAgentOrchestrator()

        # Generic 404s without model/endpoint context should be RESOURCE_UNAVAILABLE
        assert orchestrator._classify_error("404 Not Found") == SubagentErrorType.RESOURCE_UNAVAILABLE
        assert orchestrator._classify_error("404 resource_not_found") == SubagentErrorType.RESOURCE_UNAVAILABLE

    def test_classify_error_network_error(self):
        """Test error classification for network errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("connection error") == SubagentErrorType.NETWORK_ERROR
        assert orchestrator._classify_error("network unreachable") == SubagentErrorType.NETWORK_ERROR

    def test_classify_error_timeout(self):
        """Test error classification for timeout errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("timeout") == SubagentErrorType.TIMEOUT
        assert orchestrator._classify_error("Request timeout") == SubagentErrorType.TIMEOUT

    def test_classify_error_rate_limit(self):
        """Test error classification for rate limit errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("rate_limit exceeded") == SubagentErrorType.RATE_LIMIT
        assert orchestrator._classify_error("429 Too Many Requests") == SubagentErrorType.RATE_LIMIT

    def test_classify_error_auth_error(self):
        """Test error classification for auth errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("auth failed") == SubagentErrorType.AUTH_ERROR
        assert orchestrator._classify_error("401 Unauthorized") == SubagentErrorType.AUTH_ERROR
        assert orchestrator._classify_error("403 Forbidden") == SubagentErrorType.AUTH_ERROR

    def test_classify_error_unknown(self):
        """Test error classification for unknown errors."""
        orchestrator = SubAgentOrchestrator()

        assert orchestrator._classify_error("some other error") == SubagentErrorType.UNKNOWN


class TestShouldRetry:
    """Tests for _should_retry method."""

    def test_model_not_found_never_retry(self):
        """Verify MODEL_NOT_FOUND errors are never retried (configuration error)."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._should_retry(SubagentErrorType.MODEL_NOT_FOUND, 0, 3) is False
        assert orchestrator._should_retry(SubagentErrorType.MODEL_NOT_FOUND, 2, 3) is False

    def test_endpoint_not_found_never_retry(self):
        """Verify ENDPOINT_NOT_FOUND errors are never retried (configuration error)."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._should_retry(SubagentErrorType.ENDPOINT_NOT_FOUND, 0, 3) is False

    def test_auth_error_never_retry(self):
        """Verify AUTH_ERROR errors are never retried (configuration error)."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._should_retry(SubagentErrorType.AUTH_ERROR, 0, 3) is False

    def test_rate_limit_always_retry(self):
        """Verify RATE_LIMIT errors are always retried if under limit."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._should_retry(SubagentErrorType.RATE_LIMIT, 0, 3) is True
        assert orchestrator._should_retry(SubagentErrorType.RATE_LIMIT, 2, 3) is True
        assert orchestrator._should_retry(SubagentErrorType.RATE_LIMIT, 3, 3) is False

    def test_network_error_retryable(self):
        """Verify NETWORK_ERROR errors are retryable."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._should_retry(SubagentErrorType.NETWORK_ERROR, 0, 3) is True
        assert orchestrator._should_retry(SubagentErrorType.NETWORK_ERROR, 3, 3) is False

    def test_timeout_retryable(self):
        """Verify TIMEOUT errors are retryable."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._should_retry(SubagentErrorType.TIMEOUT, 0, 3) is True

    def test_resource_unavailable_limited_retry(self):
        """Verify RESOURCE_UNAVAILABLE gets fewer retries (half of max)."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._should_retry(SubagentErrorType.RESOURCE_UNAVAILABLE, 0, 4) is True
        assert orchestrator._should_retry(SubagentErrorType.RESOURCE_UNAVAILABLE, 1, 4) is True
        # max_retries // 2 = 2, so retry_count=2 fails (must be < 2)
        assert orchestrator._should_retry(SubagentErrorType.RESOURCE_UNAVAILABLE, 2, 4) is False


class TestExecuteTasks:
    """Tests for execute_tasks fail-fast behavior."""

    def test_model_not_found_not_retryable(self):
        """Verify MODEL_NOT_FOUND errors are not retryable (fail-fast)."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._classify_error("404: Model not found") == SubagentErrorType.MODEL_NOT_FOUND
        assert orchestrator._should_retry(SubagentErrorType.MODEL_NOT_FOUND, 0, 3) is False

    def test_network_error_is_retryable(self):
        """Verify network errors are classified as retryable."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._classify_error("connection timeout") == SubagentErrorType.NETWORK_ERROR
        assert orchestrator._should_retry(SubagentErrorType.NETWORK_ERROR, 0, 3) is True

    def test_auth_error_not_retryable(self):
        """Verify auth errors are not retryable (fail-fast)."""
        orchestrator = SubAgentOrchestrator()
        assert orchestrator._classify_error("401 Unauthorized") == SubagentErrorType.AUTH_ERROR
        assert orchestrator._should_retry(SubagentErrorType.AUTH_ERROR, 0, 3) is False


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
