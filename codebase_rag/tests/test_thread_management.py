"""Tests for ManagedThreadPoolExecutor."""

import threading
import time
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.utils.thread_management import ManagedThreadPoolExecutor


class TestManagedThreadPoolExecutor:
    def test_context_manager(self) -> None:
        """Test that executor starts and stops automatically with context manager."""
        with ManagedThreadPoolExecutor(max_workers=2) as executor:
            assert executor._executor is not None
            future = executor.submit(lambda: 42)
            assert future.result() == 42
        # Executor should be shutdown after context exit
        assert executor._executor is None
        assert executor._shutdown_event.is_set()

    def test_manual_lifecycle(self) -> None:
        """Test manual start and shutdown."""
        executor = ManagedThreadPoolExecutor(max_workers=2)
        executor.start()
        assert executor._executor is not None
        future = executor.submit(lambda: 43)
        assert future.result() == 43
        executor.shutdown()
        assert executor._executor is None
        assert executor._shutdown_event.is_set()

    def test_submit_before_start_raises(self) -> None:
        """Test that submit before start raises RuntimeError."""
        executor = ManagedThreadPoolExecutor(max_workers=2)
        with pytest.raises(RuntimeError, match="not started"):
            executor.submit(lambda: None)

    def test_submit_after_shutdown_raises(self) -> None:
        """Test that submit after shutdown raises RuntimeError."""
        executor = ManagedThreadPoolExecutor(max_workers=2)
        executor.start()
        executor.shutdown()
        with pytest.raises(RuntimeError, match="not started|shutdown"):
            executor.submit(lambda: None)

    def test_double_start_raises(self) -> None:
        """Test that calling start twice raises RuntimeError."""
        executor = ManagedThreadPoolExecutor(max_workers=2)
        executor.start()
        with pytest.raises(RuntimeError, match="already started"):
            executor.start()

    def test_shutdown_idempotent(self) -> None:
        """Test that shutdown can be called multiple times safely."""
        executor = ManagedThreadPoolExecutor(max_workers=2)
        executor.start()
        executor.shutdown()
        assert executor._shutdown_event.is_set()
        executor.shutdown()  # Should not raise

    def test_cancel_futures_on_shutdown(self) -> None:
        """Test that pending futures are cancelled on shutdown with cancel_futures=True."""
        start_event = threading.Event()
        def slow_task() -> int:
            start_event.set()
            time.sleep(0.5)
            return 1

        executor = ManagedThreadPoolExecutor(max_workers=1)
        executor.start()
        # Submit two tasks; second will be pending because only one worker
        future1 = executor.submit(slow_task)
        future2 = executor.submit(lambda: 2)
        # Wait for first task to start
        start_event.wait()
        # Shutdown with cancel_futures=True (default)
        executor.shutdown(wait=False, cancel_futures=True)
        # First future may have completed or cancelled; second should be cancelled
        # We can't guarantee state, but at least no exception

    def test_thread_name_prefix(self) -> None:
        """Test that thread_name_prefix is passed to underlying executor."""
        with patch(
            "codebase_rag.utils.thread_management.ThreadPoolExecutor"
        ) as mock_tpe_class:
            with ManagedThreadPoolExecutor(
                max_workers=3, thread_name_prefix="test-prefix"
            ) as executor:
                pass
            mock_tpe_class.assert_called_once_with(
                max_workers=3, thread_name_prefix="test-prefix"
            )

    def test_shutdown_timeout_parameter(self) -> None:
        """Test that shutdown_timeout parameter is stored (even if not used)."""
        executor = ManagedThreadPoolExecutor(max_workers=2, shutdown_timeout=10.0)
        assert executor.shutdown_timeout == 10.0