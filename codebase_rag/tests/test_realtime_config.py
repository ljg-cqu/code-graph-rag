"""
Tests for realtime configuration settings and CLI defaults.
"""

import os
from unittest.mock import patch

import pytest

from codebase_rag.config import AppConfig, settings
from codebase_rag.main import RealtimeConfig


class TestRealtimeConfigDefaults:
    """Test that RealtimeConfig uses settings defaults."""

    def test_realtime_config_defaults_match_settings(self) -> None:
        """Verify RealtimeConfig fields default to the corresponding settings."""
        config = RealtimeConfig()
        assert config.enabled == settings.REALTIME_UPDATER_ENABLED
        assert config.debounce == settings.REALTIME_DEBOUNCE_SECONDS
        assert config.max_wait == settings.REALTIME_MAX_WAIT_SECONDS
        assert config.enable_code == settings.REALTIME_CODE_ENABLED
        assert config.enable_docs == settings.REALTIME_DOCS_ENABLED
        assert config.enable_json == settings.REALTIME_JSON_ENABLED

    def test_realtime_config_override(self) -> None:
        """Verify RealtimeConfig can be overridden with custom values."""
        config = RealtimeConfig(
            enabled=True,
            debounce=10.0,
            max_wait=60.0,
            enable_code=False,
            enable_docs=True,
            enable_json=True,
        )
        assert config.enabled is True
        assert config.debounce == 10.0
        assert config.max_wait == 60.0
        assert config.enable_code is False
        assert config.enable_docs is True
        assert config.enable_json is True


class TestEnvironmentVariableConfiguration:
    """Test that environment variables affect settings."""

    def test_realtime_updater_enabled_env(self) -> None:
        """CGR_REALTIME_UPDATER environment variable sets REALTIME_UPDATER_ENABLED."""
        with patch.dict(os.environ, {"CGR_REALTIME_UPDATER": "true"}):
            config = AppConfig()
            assert config.REALTIME_UPDATER_ENABLED is True
        with patch.dict(os.environ, {"CGR_REALTIME_UPDATER": "false"}):
            config = AppConfig()
            assert config.REALTIME_UPDATER_ENABLED is False

    def test_realtime_debounce_env(self) -> None:
        """CGR_REALTIME_DEBOUNCE environment variable sets REALTIME_DEBOUNCE_SECONDS."""
        with patch.dict(os.environ, {"CGR_REALTIME_DEBOUNCE": "15.0"}):
            config = AppConfig()
            assert config.REALTIME_DEBOUNCE_SECONDS == 15.0

    def test_realtime_max_wait_env(self) -> None:
        """CGR_REALTIME_MAX_WAIT environment variable sets REALTIME_MAX_WAIT_SECONDS."""
        with patch.dict(os.environ, {"CGR_REALTIME_MAX_WAIT": "45.0"}):
            config = AppConfig()
            assert config.REALTIME_MAX_WAIT_SECONDS == 45.0

    def test_realtime_code_env(self) -> None:
        """CGR_REALTIME_CODE environment variable sets REALTIME_CODE_ENABLED."""
        with patch.dict(os.environ, {"CGR_REALTIME_CODE": "false"}):
            config = AppConfig()
            assert config.REALTIME_CODE_ENABLED is False
        with patch.dict(os.environ, {"CGR_REALTIME_CODE": "true"}):
            config = AppConfig()
            assert config.REALTIME_CODE_ENABLED is True

    def test_realtime_docs_env(self) -> None:
        """CGR_REALTIME_DOCS environment variable sets REALTIME_DOCS_ENABLED."""
        with patch.dict(os.environ, {"CGR_REALTIME_DOCS": "true"}):
            config = AppConfig()
            assert config.REALTIME_DOCS_ENABLED is True

    def test_realtime_json_env(self) -> None:
        """CGR_REALTIME_JSON environment variable sets REALTIME_JSON_ENABLED."""
        with patch.dict(os.environ, {"CGR_REALTIME_JSON": "true"}):
            config = AppConfig()
            assert config.REALTIME_JSON_ENABLED is True

    def test_realtime_batch_size_env(self) -> None:
        """CGR_REALTIME_BATCH_SIZE environment variable sets REALTIME_BATCH_SIZE."""
        with patch.dict(os.environ, {"CGR_REALTIME_BATCH_SIZE": "500"}):
            config = AppConfig()
            assert config.REALTIME_BATCH_SIZE == 500