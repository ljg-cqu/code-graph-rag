"""Managed thread pool with explicit lifecycle management.

Enhances ThreadPoolExecutor with:
- Explicit shutdown control with cancel_futures
- Thread-safe state checking
- Configurable shutdown timeout
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from loguru import logger


class ManagedThreadPoolExecutor:
    """Thread pool wrapper with explicit lifecycle management.

    Usage:
        # Context manager (recommended) - auto-starts and auto-shutdowns
        with ManagedThreadPoolExecutor(max_workers=10) as executor:
            future = executor.submit(task)

        # Manual lifecycle control
        executor = ManagedThreadPoolExecutor(max_workers=10)
        executor.start()  # Must call start() before submit()
        try:
            future = executor.submit(task)
        finally:
            executor.shutdown()
    """

    def __init__(
        self,
        max_workers: int,
        thread_name_prefix: str = "",
        shutdown_timeout: float = 30.0,
    ) -> None:
        self.max_workers = max_workers
        self.thread_name_prefix = thread_name_prefix
        self.shutdown_timeout = shutdown_timeout
        self._executor: Optional[ThreadPoolExecutor] = None
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        """Explicitly start the executor. Called automatically by __enter__."""
        with self._lock:
            if self._executor is not None:
                raise RuntimeError("Executor already started")
            if self._shutdown_event.is_set():
                raise RuntimeError("Executor is shutdown")
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix=self.thread_name_prefix,
            )
            logger.debug(
                f"Started ManagedThreadPoolExecutor with {self.max_workers} workers"
            )

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Submit a callable to be executed.

        Returns a Future representing the result.
        """
        with self._lock:
            if self._executor is None:
                raise RuntimeError("Executor not started. Use start() or context manager.")
            if self._shutdown_event.is_set():
                raise RuntimeError("Executor is shutdown")
            return self._executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True, cancel_futures: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for all pending futures to complete.
            cancel_futures: If True, cancel pending futures that have not started.
        """
        with self._lock:
            if self._shutdown_event.is_set():
                return
            self._shutdown_event.set()

            if self._executor is not None:
                executor = self._executor
                self._executor = None
                executor.shutdown(
                    wait=wait,
                    cancel_futures=cancel_futures,
                )
                logger.debug("ManagedThreadPoolExecutor shutdown complete")

    def __enter__(self) -> ManagedThreadPoolExecutor:
        """Context manager entry - automatically starts the executor."""
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - automatically shuts down the executor."""
        self.shutdown(wait=True, cancel_futures=True)