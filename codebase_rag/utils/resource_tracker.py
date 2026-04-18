"""Resource tracking context manager for guaranteed cleanup."""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any

from loguru import logger


class ResourceTracker:
    """Tracks resources and ensures cleanup in reverse registration order."""

    def __init__(self) -> None:
        self._resources: list[tuple[Any, Callable[[Any], None]]] = []
        self._lock = threading.Lock()

    def track(self, resource: Any, cleanup_func: Callable[[Any], None]) -> None:
        with self._lock:
            self._resources.append((resource, cleanup_func))

    def cleanup_all(self) -> None:
        with self._lock:
            resources_to_cleanup = list(reversed(self._resources))
            self._resources.clear()

        for resource, cleanup_func in resources_to_cleanup:
            try:
                cleanup_func(resource)
            except Exception as e:
                logger.warning(f"Failed to cleanup resource {resource}: {e}")


@contextmanager
def tracked_resources() -> Generator[ResourceTracker, None, None]:
    tracker = ResourceTracker()
    try:
        yield tracker
    finally:
        tracker.cleanup_all()
