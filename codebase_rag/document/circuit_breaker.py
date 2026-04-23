"""Circuit breaker for protecting external service calls.

Deterministic state machine with no LLM involvement.
Safe for asyncio: state transitions are synchronous and atomic
under the event loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto


class CircuitState(Enum):
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    window_size: int = 10


@dataclass
class CircuitBreaker:
    name: str
    config: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    recent_failures: list[bool] = field(default_factory=list)

    def can_execute(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.last_failure_time is not None:
                elapsed = (datetime.now(UTC) - self.last_failure_time).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True
            return False

        return True

    def record_success(self) -> None:
        self._record_result(True)

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._reset()

    def record_failure(self) -> None:
        self._record_result(False)
        self.last_failure_time = datetime.now(UTC)

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self._should_open():
                self.state = CircuitState.OPEN

    def _record_result(self, success: bool) -> None:
        self.recent_failures.append(success)
        if len(self.recent_failures) > self.config.window_size:
            self.recent_failures.pop(0)

    def _should_open(self) -> bool:
        if len(self.recent_failures) < self.config.window_size:
            return self.failure_count >= self.config.failure_threshold

        failure_rate = sum(1 for s in self.recent_failures if not s) / len(
            self.recent_failures
        )
        return failure_rate >= 0.5

    def _reset(self) -> None:
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.recent_failures.clear()

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN
