"""Tests for ShutdownManager."""

import signal
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.utils.shutdown_manager import ShutdownManager, shutdown_manager


class TestShutdownManager:
    def test_register_handler(self) -> None:
        """Test that handlers can be registered and sorted by priority."""
        manager = ShutdownManager()
        calls = []

        def handler1() -> None:
            calls.append(1)

        def handler2() -> None:
            calls.append(2)

        manager.register_handler(handler2, priority=5)
        manager.register_handler(handler1, priority=10)

        # Simulate shutdown with sys.exit mocked
        with patch("sys.exit"):
            manager.initiate_shutdown()

        # Higher priority (10) should run first
        assert calls == [1, 2]

    def test_register_handler_after_shutdown(self) -> None:
        """Test that handlers registered after shutdown are executed immediately."""
        manager = ShutdownManager()
        calls = []

        def handler1() -> None:
            calls.append(1)

        def handler2() -> None:
            calls.append(2)

        manager.register_handler(handler1, priority=5)
        with patch("sys.exit"):
            manager.initiate_shutdown(signum=signal.SIGTERM)

        # Handler registered after shutdown should be called immediately
        manager.register_handler(handler2, priority=10)
        assert calls == [1, 2]

    def test_initiate_shutdown_calls_handlers(self) -> None:
        """Test that initiate_shutdown calls all registered handlers."""
        manager = ShutdownManager()
        mock_handler = MagicMock()
        manager.register_handler(mock_handler, priority=0)

        with patch("sys.exit") as mock_exit:
            manager.initiate_shutdown(signum=signal.SIGTERM)

        mock_handler.assert_called_once()
        mock_exit.assert_called_once_with(0)

    def test_double_signal_force_exit(self) -> None:
        """Test that a second signal forces immediate exit."""
        manager = ShutdownManager()
        mock_handler = MagicMock()
        manager.register_handler(mock_handler, priority=0)

        # Patch sys.exit across both calls
        with patch("sys.exit") as mock_exit:
            # First signal
            manager.initiate_shutdown(signum=signal.SIGTERM)
            assert mock_exit.call_count == 1
            assert mock_exit.call_args == ((0,), {})

            # Reset for second call check
            mock_exit.reset_mock()

            # Second signal should force exit with code 1
            manager.initiate_shutdown(signum=signal.SIGTERM)
            assert mock_exit.call_count == 1
            assert mock_exit.call_args == ((1,), {})
            mock_handler.assert_called_once()  # Handler not called again

    def test_register_sync_signal_handlers(self) -> None:
        """Test that sync signal handlers are registered."""
        manager = ShutdownManager()
        with patch("signal.signal") as mock_signal:
            manager.register_sync_signal_handlers()
            assert mock_signal.call_count == 2
            calls = [call[0] for call in mock_signal.call_args_list]
            assert calls[0][0] == signal.SIGINT
            assert calls[1][0] == signal.SIGTERM

    def test_register_async_signal_handlers(self) -> None:
        """Test that async signal handlers are registered."""
        manager = ShutdownManager()
        mock_loop = MagicMock()
        manager.register_async_signal_handlers(mock_loop)
        assert mock_loop.add_signal_handler.call_count == 2
        calls = [call[0] for call in mock_loop.add_signal_handler.call_args_list]
        assert calls[0][0] == signal.SIGINT
        assert calls[1][0] == signal.SIGTERM


class TestGlobalShutdownManager:
    def test_global_instance(self) -> None:
        """Test that the global instance is created."""
        assert isinstance(shutdown_manager, ShutdownManager)

    def test_global_register_handler(self) -> None:
        """Test that the global instance accepts handlers."""
        mock_handler = MagicMock()
        shutdown_manager.register_handler(mock_handler, priority=5)
        # No easy way to verify without triggering shutdown
        # Just ensure no exception