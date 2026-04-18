"""Thread-safe atomic primitives for concurrent state management."""

from __future__ import annotations

import threading


class AtomicBoolean:
    """Thread-safe boolean with compare-and-set for safe state transitions."""

    def __init__(self, initial_value: bool = False) -> None:
        self._value = initial_value
        self._lock = threading.Lock()

    def get(self) -> bool:
        with self._lock:
            return self._value

    def set(self, value: bool) -> bool:
        with self._lock:
            old_value = self._value
            self._value = value
            return old_value

    def compare_and_set(self, expected: bool, new_value: bool) -> bool:
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False

    def get_and_set(self, new_value: bool) -> bool:
        with self._lock:
            old_value = self._value
            self._value = new_value
            return old_value
