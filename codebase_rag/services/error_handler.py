"""Centralized error handling with LLM-generated guidance.

This module provides the ErrorHandler class that orchestrates the error
handling pipeline: Exception → Classification → Guidance.

Per LLM-First Design:
- Classification = Deterministic (via FailureClassification)
- Guidance = LLM-generated (via ErrorGuidance)
"""

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING, Callable, TypeVar

from ..exceptions import GraphQueryError
from .error_guidance import ErrorContext, ErrorGuidance, LLMErrorGuidance
from .failure_classifier import FailureClassification, FailureType

if TYPE_CHECKING:
    pass

T = TypeVar("T")


class ErrorHandler:
    """Centralized error handling with LLM-generated guidance.

    Orchestrates the error handling pipeline:
    1. Deterministic classification via classifier function
    2. Context extraction from error and operation
    3. LLM-generated or static guidance
    4. Rich exception wrapping

    Example:
        ```python
        from codebase_rag.services.error_handler import ErrorHandler
        from codebase_rag.services.failure_classifier import classify_memgraph_failure

        handler = ErrorHandler(classifier=classify_memgraph_failure)

        @handler.with_error_handling(operation_type="query")
        async def query_graph(cypher: str):
            # Errors automatically classified and guided
            return await execute_query(cypher)
        ```
    """

    def __init__(
        self,
        llm_guidance: LLMErrorGuidance | None = None,
        classifier: Callable[[Exception], FailureClassification] | None = None,
    ) -> None:
        """Initialize the error handler.

        Args:
            llm_guidance: Optional LLM guidance generator. If None, uses static guidance.
            classifier: Optional classification function. If None, uses default classifier.
        """
        self.llm_guidance = llm_guidance or LLMErrorGuidance()
        self.classifier = classifier or self._default_classifier

    def _default_classifier(self, error: Exception) -> FailureClassification:
        """Default classifier that creates a basic classification."""
        return FailureClassification(
            failure_type=FailureType.UNKNOWN,
            message=str(error),
            should_retry=False,
        )

    async def handle_error(
        self,
        error: Exception,
        operation_type: str,
        query: str | None = None,
        **context_kwargs: object,
    ) -> GraphQueryError:
        """Handle error through full pipeline: classify → guide → wrap.

        Args:
            error: The exception to handle
            operation_type: Type of operation that failed
            query: Optional query string for context
            **context_kwargs: Additional context fields

        Returns:
            GraphQueryError with classification and guidance
        """
        # Step 1: Deterministic classification
        classification = self.classifier(error)

        # Step 2: Build context
        context = ErrorContext.from_exception(
            error=error,
            operation_type=operation_type,
            classification=classification,
            **context_kwargs,  # type: ignore[arg-type]
        )

        # Step 3: Generate guidance (LLM or static)
        guidance = await self.llm_guidance.generate_guidance(
            error=error,
            context=context,
            classification=classification,
        )

        # Step 4: Wrap in rich exception
        return GraphQueryError(
            original_error=error,
            classification=classification,
            user_guidance=guidance,
            query=query,
        )

    def handle_sync(
        self,
        error: Exception,
        operation_type: str,
        query: str | None = None,
        **context_kwargs: object,
    ) -> GraphQueryError:
        """Synchronous error handling using static guidance only.

        Args:
            error: The exception to handle
            operation_type: Type of operation that failed
            query: Optional query string for context
            **context_kwargs: Additional context fields

        Returns:
            GraphQueryError with classification and static guidance
        """
        classification = self.classifier(error)
        context = ErrorContext.from_exception(
            error=error,
            operation_type=operation_type,
            classification=classification,
            **context_kwargs,  # type: ignore[arg-type]
        )
        guidance = self.llm_guidance._get_static_guidance(error, context, classification)
        return GraphQueryError(
            original_error=error,
            classification=classification,
            user_guidance=guidance,
            query=query,
        )

    def with_error_handling(
        self,
        operation_type: str,
        **context_kwargs: object,
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator for automatic error handling.

        Wraps a function to automatically handle exceptions through
        the error handling pipeline.

        Args:
            operation_type: Type of operation for context
            **context_kwargs: Additional context fields

        Returns:
            Decorator function

        Example:
            ```python
            @handler.with_error_handling(operation_type="vector_search")
            async def search_vectors(query: str):
                return await perform_search(query)
            ```
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            async def async_wrapper(*args: object, **kwargs: object) -> T:
                try:
                    return await func(*args, **kwargs)  # type: ignore[misc]
                except GraphQueryError:
                    raise  # Already handled
                except Exception as e:
                    raise await self.handle_error(
                        error=e,
                        operation_type=operation_type,
                        **context_kwargs,  # type: ignore[arg-type]
                    )

            @functools.wraps(func)
            def sync_wrapper(*args: object, **kwargs: object) -> T:
                try:
                    return func(*args, **kwargs)
                except GraphQueryError:
                    raise  # Already handled
                except Exception as e:
                    raise self.handle_sync(
                        error=e,
                        operation_type=operation_type,
                        **context_kwargs,  # type: ignore[arg-type]
                    )

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator


# Default handler instance with Memgraph classifier
def _get_memgraph_classifier():
    """Lazy import to avoid circular dependencies."""
    from .failure_classifier import classify_memgraph_failure
    return classify_memgraph_failure


default_handler = ErrorHandler(classifier=lambda e: _get_memgraph_classifier()(e))


__all__ = [
    "ErrorHandler",
    "default_handler",
]
