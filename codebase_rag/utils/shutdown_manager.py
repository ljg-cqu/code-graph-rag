"""Centralized shutdown coordinator.

Replaces multiple signal handler registrations with a single point
that coordinates cleanup across all components. Components register
cleanup handlers instead of signal handlers.
"""

from __future__ import annotations

import signal
import sys
import threading
from collections.abc import Callable
from typing import Any

from loguru import logger


class ShutdownManager:
    def __init__(self) -> None:
        self._handlers: list[tuple[int, Callable[[], None]]] = []
        self._lock = threading.Lock()
        self._shutdown_initiated = False

    def register_sync_signal_handlers(self) -> None:
        """Register synchronous signal handlers (signal.signal)."""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        logger.debug("Registered synchronous signal handlers")

    def register_async_signal_handlers(self, loop) -> None:
        """Register asynchronous signal handlers (asyncio loop)."""
        try:
            loop.add_signal_handler(signal.SIGINT, lambda: self.initiate_shutdown(signal.SIGINT))
            loop.add_signal_handler(signal.SIGTERM, lambda: self.initiate_shutdown(signal.SIGTERM))
            logger.debug("Registered asynchronous signal handlers")
        except (NotImplementedError, RuntimeError):
            # Windows or loop already closed
            pass

    def register_handler(self, handler: Callable[[], None], priority: int = 0) -> None:
        with self._lock:
            if not self._shutdown_initiated:
                self._handlers.append((priority, handler))
                self._handlers.sort(key=lambda x: -x[0])
            else:
                handler()

    def unregister_handler(self, handler: Callable[[], None]) -> None:
        with self._lock:
            self._handlers = [
                (p, h) for p, h in self._handlers if h is not handler
            ]

    def _signal_handler(self, signum: int, frame: Any) -> None:
        self.initiate_shutdown(signum)

    def initiate_shutdown(self, signum: int | None = None) -> None:
        with self._lock:
            if self._shutdown_initiated:
                logger.warning("Forced shutdown - second signal received")
                sys.exit(1)
                return  # Unreachable in normal execution, but aids testing
            self._shutdown_initiated = True
            handlers = list(self._handlers)

        logger.info(f"Initiating graceful shutdown (signal {signum})...")

        for priority, handler in handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Shutdown handler failed: {e}")

        logger.info("Graceful shutdown complete")
        sys.exit(0)


shutdown_manager = ShutdownManager()
