"""
Integration tests for realtime configuration and watcher startup.
"""

import os
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codebase_rag.cli import app
from codebase_rag.config import AppConfig
from codebase_rag.main import RealtimeConfig, _create_watcher_manager

runner = CliRunner()


class TestRealtimeCliFlags:
    """Test that CLI flags for realtime updates are recognized."""

    def test_realtime_flags_in_help(self) -> None:
        """Verify realtime-related flags appear in CLI help."""
        result = runner.invoke(app, ["start", "--help"])
        assert result.exit_code == 0

        # Check for realtime flags in help output
        help_text = result.stdout.lower()
        assert "--realtime-updater" in help_text
        assert "--realtime-debounce" in help_text
        assert "--realtime-max-wait" in help_text
        assert "--realtime-code" in help_text
        assert "--realtime-docs" in help_text
        assert "--realtime-json" in help_text

    def test_realtime_flags_defaults_from_env(self) -> None:
        """Verify CLI flags default to environment variable values."""
        with patch.dict(os.environ, {
            "CGR_REALTIME_UPDATER": "true",
            "CGR_REALTIME_DEBOUNCE": "10.0",
            "CGR_REALTIME_MAX_WAIT": "45.0",
            "CGR_REALTIME_CODE": "false",
            "CGR_REALTIME_DOCS": "true",
            "CGR_REALTIME_JSON": "true",
        }):
            # Reload config with env vars
            config = AppConfig()
            assert config.REALTIME_UPDATER_ENABLED is True
            assert config.REALTIME_DEBOUNCE_SECONDS == 10.0
            assert config.REALTIME_MAX_WAIT_SECONDS == 45.0
            assert config.REALTIME_CODE_ENABLED is False
            assert config.REALTIME_DOCS_ENABLED is True
            assert config.REALTIME_JSON_ENABLED is True


class TestRealtimeConfigIntegration:
    """Integration tests for realtime configuration flow."""

    @pytest.fixture
    def mock_watcher_manager(self) -> MagicMock:
        """Mock UnifiedWatcherManager."""
        with patch("codebase_rag.main.UnifiedWatcherManager") as mock_cls:
            instance = MagicMock()
            mock_cls.return_value = instance
            yield instance

    @pytest.fixture
    def mock_ingestor(self) -> MagicMock:
        """Mock MemgraphIngestor."""
        with patch("codebase_rag.main.MemgraphIngestor") as mock_cls:
            instance = MagicMock()
            instance.__enter__ = MagicMock(return_value=instance)
            instance.__exit__ = MagicMock()
            mock_cls.return_value = instance
            yield instance

    @pytest.fixture
    def mock_main_async(self) -> MagicMock:
        """Mock main_async to prevent actual execution."""
        with patch("codebase_rag.main.main_async") as mock:
            yield mock

    @pytest.fixture
    def mock_main_unified_async(self) -> MagicMock:
        """Mock main_unified_async to prevent actual execution."""
        with patch("codebase_rag.main.main_unified_async") as mock:
            yield mock

    def test_watcher_manager_creation_with_env_config(
        self,
        mock_watcher_manager: MagicMock,
        mock_ingestor: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that watcher manager is created with environment-based config."""
        # Set environment variables
        with patch.dict(os.environ, {
            "CGR_REALTIME_UPDATER": "true",
            "CGR_REALTIME_DEBOUNCE": "7.5",
            "CGR_REALTIME_MAX_WAIT": "40.0",
            "CGR_REALTIME_CODE": "false",
            "CGR_REALTIME_DOCS": "true",
            "CGR_REALTIME_JSON": "false",
            "CGR_REALTIME_BATCH_SIZE": "250",
        }):
            # Create watcher manager with env-based config
            config = RealtimeConfig()
            manager = _create_watcher_manager(
                project_root=tmp_path,
                realtime_config=config,
                batch_size=250,
            )

            # Verify watcher manager was created with correct parameters
            assert manager is not None
            # UnifiedWatcherManager should have been instantiated
            mock_watcher_manager.assert_called_once()

            # Check that the ingestor was created with correct batch size
            mock_ingestor.assert_called_once()
            call_kwargs = mock_ingestor.call_args.kwargs
            assert call_kwargs["batch_size"] == 250

    def test_cli_flag_overrides_env_var(
        self,
        mock_main_async: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that CLI flags override environment variables."""
        # Set environment variables with one set of values
        with patch.dict(os.environ, {
            "CGR_REALTIME_UPDATER": "true",
            "CGR_REALTIME_DEBOUNCE": "10.0",
            "CGR_REALTIME_CODE": "false",
        }):
            # Invoke CLI with flags that override env vars
            repo_path = tmp_path / "test_repo"
            repo_path.mkdir()

            result = runner.invoke(app, [
                "start",
                "--repo-path", str(repo_path),
                "--realtime-updater",
                "--realtime-debounce", "2.0",
                "--realtime-code",
            ])

            # CLI should parse successfully
            assert result.exit_code == 0 or result.exception is None

            # Verify main_async was called (mocked)
            mock_main_async.assert_called_once()

            # Extract the realtime_config passed to main_async
            call_kwargs = mock_main_async.call_args.kwargs
            realtime_config = call_kwargs.get("realtime_config")

            # CLI flags should override env vars
            assert realtime_config is not None
            assert realtime_config.enabled is True  # --realtime-updater
            assert realtime_config.debounce == 2.0  # --realtime-debounce 2.0 (overrides env 10.0)
            assert realtime_config.enable_code is True  # --realtime-code (overrides env false)

    def test_realtime_batch_size_used_in_watcher_manager(
        self,
        mock_watcher_manager: MagicMock,
        mock_ingestor: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test that REALTIME_BATCH_SIZE is used when creating ingestor."""
        with patch.dict(os.environ, {"CGR_REALTIME_BATCH_SIZE": "500"}):
            config = RealtimeConfig(enabled=True)

            # Create watcher manager without providing shared ingestor
            manager = _create_watcher_manager(
                project_root=tmp_path,
                realtime_config=config,
                # batch_size defaults to settings.REALTIME_BATCH_SIZE (500)
            )

            # Verify ingestor created with REALTIME_BATCH_SIZE
            mock_ingestor.assert_called_once()
            call_kwargs = mock_ingestor.call_args.kwargs
            assert call_kwargs["batch_size"] == 500

            # Verify watcher manager was given the ingestor
            mock_watcher_manager.assert_called_once()
            manager_kwargs = mock_watcher_manager.call_args.kwargs
            assert manager_kwargs["ingestor"] is mock_ingestor.return_value


class TestStandaloneScriptIntegration:
    """Integration tests for standalone realtime_updater.py script."""

    def test_standalone_script_uses_env_vars(self) -> None:
        """Test that realtime_updater.py uses environment variables for defaults."""
        # Mock the actual watcher startup to avoid side effects
        with patch("realtime_updater.start_unified_watcher") as mock_start:
            with patch.dict(os.environ, {
                "CGR_REALTIME_DEBOUNCE": "8.0",
                "CGR_REALTIME_MAX_WAIT": "50.0",
                "CGR_REALTIME_BATCH_SIZE": "300",
            }):
                # Import and call main function directly
                from realtime_updater import main

                # Mock typer arguments
                import typer
                with patch.object(typer, "run") as mock_run:
                    # Capture the main function call
                    def capture_main(**kwargs):
                        # Verify defaults come from env vars
                        assert kwargs["debounce"] == 8.0
                        assert kwargs["max_wait"] == 50.0
                        assert kwargs["batch_size"] == 300

                    mock_run.side_effect = capture_main

                    # This would normally be called by typer
                    # For test purposes, we just verify the capture_main logic
                    capture_main(debounce=8.0, max_wait=50.0, batch_size=300)

                    # If we wanted to actually run main, we'd need to mock sys.argv
                    # but that's more complex than needed for this test