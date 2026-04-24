from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from loguru import logger

F = TypeVar("F", bound=Callable)


def retry_on_exception(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    if attempt >= max_attempts:
                        raise
                    delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)
                    if on_retry:
                        on_retry(e, attempt)
                    time.sleep(delay)
            return None

        return wrapper

    return decorator
