"""LLM-First error guidance system for Code-Graph-RAG.

Provides context-aware error messages and guidance using LLM interpretation
while keeping error classification deterministic.

Key Principle:
- Classification = Deterministic (fast, reliable, programmatic handling)
- Guidance = LLM-generated (contextual, user-friendly, adaptive)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, auto
from typing import TYPE_CHECKING

from loguru import logger

from ..prompts import build_error_guidance_prompt
from .failure_classifier import FailureClassification, FailureType

if TYPE_CHECKING:
    from collections.abc import Callable


class UserExpertiseLevel(Enum):
    """User expertise levels for message personalization."""

    BEGINNER = auto()
    INTERMEDIATE = auto()
    EXPERT = auto()


@dataclass
class ErrorContext:
    """Sanitized context for LLM error interpretation.

    Contains only safe, non-sensitive information about the error context.
    """

    operation_type: str  # "document_indexing", "query", "ingestion", "embedding"
    error_category: FailureType  # Deterministic classification
    embedding_provider: str | None = None  # "local", "openai", etc.
    graph_type: str = "code"  # "code" or "document"
    recent_changes: list[str] = field(default_factory=list)
    user_expertise: UserExpertiseLevel = UserExpertiseLevel.INTERMEDIATE
    previous_provider: str | None = None  # For provider switch context
    workspace: str | None = None

    @classmethod
    def from_exception(
        cls,
        error: Exception,
        operation_type: str,
        classification: FailureClassification | None = None,
        **kwargs: object,
    ) -> ErrorContext:
        """Create context from exception with automatic field extraction.

        Args:
            error: The exception to extract context from
            operation_type: Type of operation that failed
            classification: Optional pre-computed classification
            **kwargs: Additional context fields

        Returns:
            ErrorContext with extracted and provided fields
        """
        if classification is None:
            from .failure_classifier import classify_memgraph_failure

            classification = classify_memgraph_failure(error)

        return cls(
            operation_type=operation_type,
            error_category=classification.failure_type,
            **kwargs,  # type: ignore[arg-type]
        )

    @classmethod
    def from_config(
        cls,
        operation_type: str,
        config: dict[str, str | None],
    ) -> ErrorContext:
        """Create context from application configuration.

        Args:
            operation_type: Type of operation
            config: Configuration dictionary with optional keys:
                - embedding_provider
                - graph_type
                - user_expertise

        Returns:
            ErrorContext with configuration-based fields
        """
        expertise_str = config.get("user_expertise") or "intermediate"
        try:
            user_expertise = UserExpertiseLevel[expertise_str.upper()]
        except KeyError:
            user_expertise = UserExpertiseLevel.INTERMEDIATE

        return cls(
            operation_type=operation_type,
            error_category=FailureType.UNKNOWN,
            embedding_provider=config.get("embedding_provider"),
            graph_type=config.get("graph_type") or "code",
            user_expertise=user_expertise,
        )

    def to_llm_prompt(self) -> str:
        """Generate sanitized prompt context for LLM."""
        parts = [
            f"Operation: {self.operation_type}",
            f"Error Type: {self.error_category.name}",
            f"Graph Type: {self.graph_type}",
            f"User Level: {self.user_expertise.name.lower()}",
        ]

        if self.embedding_provider:
            parts.append(f"Embedding Provider: {self.embedding_provider}")
        if self.previous_provider and self.previous_provider != self.embedding_provider:
            parts.append(f"Previous Provider: {self.previous_provider}")
        if self.recent_changes:
            # Sanitize: limit to 3 most recent, truncate long entries
            safe_changes = [c[:100] for c in self.recent_changes[:3]]
            parts.append(f"Recent Changes: {', '.join(safe_changes)}")

        return "\n".join(parts)


@dataclass
class ErrorGuidance:
    """LLM-generated guidance for error resolution."""

    summary: str  # Brief headline (5-7 words)
    explanation: str  # What happened and why
    suggested_fix: str  # Specific steps to resolve
    severity: str = "blocking"  # "blocking", "warning", "info"
    code_example: str | None = None  # Shell command or config snippet
    doc_link: str | None = None  # Documentation URL
    should_retry: bool = False
    retry_after_seconds: int | None = None


# Static fallback guidance for common errors (used when LLM is unavailable)
STATIC_GUIDANCE: dict[str, ErrorGuidance] = {
    "TRANSIENT_NETWORK": ErrorGuidance(
        summary="Connection Issue",
        explanation="The graph database connection was interrupted. This is usually temporary.",
        suggested_fix="Wait a moment and try again. If the issue persists, check if Memgraph is running.",
        severity="warning",
        code_example="docker ps | grep memgraph",
        should_retry=True,
        retry_after_seconds=30,
    ),
    "CONNECTION_REFUSED": ErrorGuidance(
        summary="Graph Database Not Running",
        explanation="Cannot connect to the graph database. The Memgraph instance is not accessible.",
        suggested_fix="Start the Memgraph database before indexing documents.",
        severity="blocking",
        code_example="docker run -p 7688:7687 memgraph/memgraph:latest",
        should_retry=True,
        retry_after_seconds=5,
    ),
    "AUTHENTICATION_FAILURE": ErrorGuidance(
        summary="Authentication Failed",
        explanation="Could not authenticate with the graph database. Check your credentials.",
        suggested_fix="Verify MEMGRAPH_USERNAME and MEMGRAPH_PASSWORD in your .env file.",
        severity="blocking",
    ),
    "VECTOR_DIMENSION_MISMATCH": ErrorGuidance(
        summary="Embedding Dimension Mismatch",
        explanation="The embedding dimensions don't match the vector index configuration.",
        suggested_fix="Recreate the vector index with the correct dimensions for your embedding model.",
        severity="blocking",
        code_example="cgr-cli vector-index recreate --workspace default",
    ),
    "SYNTAX_ERROR": ErrorGuidance(
        summary="Query Syntax Error",
        explanation="The database query contains invalid syntax.",
        suggested_fix="Check the query syntax. Ensure it uses Memgraph-compatible Cypher.",
        severity="blocking",
    ),
    "EMBEDDING_ERROR": ErrorGuidance(
        summary="Embedding Generation Failed",
        explanation="Could not generate embeddings for the document content.",
        suggested_fix="Check your embedding provider configuration and API key.",
        severity="blocking",
    ),
    "IMPORT_TARGET_MISSING": ErrorGuidance(
        summary="Import Target Not in Graph",
        explanation="An import relationship references a module not present in the codebase. This is expected for external dependencies (standard library, third-party packages).",
        suggested_fix="For external dependencies, no action needed. For internal modules, check that the source file was successfully ingested.",
        severity="info",
        should_retry=False,
    ),
    "MGCLIENT_STATE_CORRUPTION": ErrorGuidance(
        summary="Connection State Error",
        explanation="The database connection encountered an internal state error. This is usually transient and can occur during high load or connection pool exhaustion.",
        suggested_fix="The system will automatically retry with a fresh connection. If this error persists, consider restarting Memgraph or increasing connection pool limits.",
        severity="warning",
        should_retry=True,
        retry_after_seconds=1,
    ),
}


class LLMErrorGuidance:
    """Generates user-facing error guidance via LLM interpretation."""

    def __init__(
        self,
        model_call: Callable | None = None,
        cache_ttl_seconds: int = 3600,
    ) -> None:
        """Initialize the guidance generator.

        Args:
            model_call: Optional async function to call the LLM.
                        If None, falls back to static guidance.
            cache_ttl_seconds: Cache TTL for generated guidance.
        """
        self.model_call = model_call
        self.cache_ttl = cache_ttl_seconds
        self._cache: dict[str, tuple[ErrorGuidance, float]] = {}

    def _cache_key(
        self,
        error: Exception,
        context: ErrorContext,
    ) -> str:
        """Generate cache key from error type and context."""
        # Hash error type + operation + provider (not full error message)
        key_data = (
            f"{type(error).__name__}:"
            f"{context.operation_type}:"
            f"{context.embedding_provider or 'none'}:"
            f"{context.error_category.name}"
        )
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_cached(self, key: str) -> ErrorGuidance | None:
        """Get cached guidance if still valid."""
        if key in self._cache:
            guidance, timestamp = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return guidance
        return None

    def _set_cache(self, key: str, guidance: ErrorGuidance) -> None:
        """Cache guidance with current timestamp."""
        self._cache[key] = (guidance, time.time())

    async def generate_guidance(
        self,
        error: Exception,
        context: ErrorContext,
        classification: FailureClassification,
    ) -> ErrorGuidance:
        """Generate user-facing guidance for an error.

        Args:
            error: The original exception
            context: Sanitized error context
            classification: Deterministic failure classification

        Returns:
            ErrorGuidance with user-friendly message and fix suggestions
        """
        # Check cache first
        cache_key = self._cache_key(error, context)
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"Using cached error guidance for {type(error).__name__}")
            return cached

        # Try LLM generation
        if self.model_call:
            try:
                guidance = await self._generate_with_llm(error, context, classification)
                self._set_cache(cache_key, guidance)
                return guidance
            except Exception as e:
                logger.warning(f"LLM guidance generation failed: {e}")

        # Fall back to static guidance
        guidance = self._get_static_guidance(error, context, classification)
        self._set_cache(cache_key, guidance)
        return guidance

    async def _generate_with_llm(
        self,
        error: Exception,
        context: ErrorContext,
        classification: FailureClassification,
    ) -> ErrorGuidance:
        """Generate guidance using LLM."""
        prompt = self._build_prompt(error, context, classification)

        # Call LLM (injected via model_call)
        response = await self.model_call(prompt)  # type: ignore

        # Parse response
        return ErrorGuidance(
            summary=response.get("summary", "Error Occurred"),
            explanation=response.get("explanation", str(error)[:200]),
            suggested_fix=response.get("suggested_fix", "Check the error details."),
            severity=response.get("severity", "blocking"),
            code_example=response.get("code_example"),
            doc_link=response.get("doc_link"),
            should_retry=classification.should_retry,
            retry_after_seconds=30 if classification.should_retry else None,
        )

    def _build_prompt(
        self,
        error: Exception,
        context: ErrorContext,
        classification: FailureClassification,
    ) -> str:
        """Build LLM prompt for error guidance generation."""
        return build_error_guidance_prompt(
            error_name=type(error).__name__,
            error_message=str(error),
            operation_type=context.operation_type,
            error_category=classification.failure_type.name,
            graph_type=context.graph_type,
            user_level=context.user_expertise.name.lower(),
            embedding_provider=context.embedding_provider,
            previous_provider=context.previous_provider,
            recent_changes=context.recent_changes,
            should_retry=classification.should_retry,
            recovery_action=classification.recovery_action,
        )

    def _get_static_guidance(
        self,
        error: Exception,
        context: ErrorContext,
        classification: FailureClassification,
    ) -> ErrorGuidance:
        """Get static fallback guidance based on error type."""
        # Map failure type to static guidance
        failure_type_name = classification.failure_type.name

        # Check for specific error patterns
        error_message = str(error).lower()

        if "connection refused" in error_message:
            return STATIC_GUIDANCE.get(
                "CONNECTION_REFUSED",
                self._default_guidance(error, classification),
            )

        if "dimension" in error_message or failure_type_name == "VECTOR_DIMENSION_MISMATCH":
            guidance = STATIC_GUIDANCE.get(
                "VECTOR_DIMENSION_MISMATCH",
                self._default_guidance(error, classification),
            )
            # Add context about provider switch if applicable
            if context.previous_provider and context.previous_provider != context.embedding_provider:
                guidance = ErrorGuidance(
                    summary=guidance.summary,
                    explanation=(
                        f"You switched from {context.previous_provider} to "
                        f"{context.embedding_provider} embeddings. {guidance.explanation}"
                    ),
                    suggested_fix=guidance.suggested_fix,
                    severity=guidance.severity,
                    code_example=guidance.code_example,
                    should_retry=guidance.should_retry,
                )
            return guidance

        # Use failure type for lookup
        if failure_type_name in STATIC_GUIDANCE:
            return STATIC_GUIDANCE[failure_type_name]

        return self._default_guidance(error, classification)

    def _default_guidance(
        self,
        error: Exception,
        classification: FailureClassification,
    ) -> ErrorGuidance:
        """Generate default guidance when no specific match found."""
        return ErrorGuidance(
            summary="Operation Failed",
            explanation=f"An error occurred: {type(error).__name__}",
            suggested_fix="Check the error details and try again.",
            severity="blocking",
            should_retry=classification.should_retry,
            retry_after_seconds=30 if classification.should_retry else None,
        )


class CachedGuidanceGenerator(LLMErrorGuidance):
    """Wrapper that adds caching to LLMErrorGuidance."""

    def __init__(
        self,
        llm_generator: LLMErrorGuidance,
        ttl_seconds: int = 3600,
    ) -> None:
        super().__init__(model_call=llm_generator.model_call, cache_ttl_seconds=ttl_seconds)
        self.llm = llm_generator

    async def generate_guidance(
        self,
        error: Exception,
        context: ErrorContext,
        classification: FailureClassification,
    ) -> ErrorGuidance:
        """Get guidance with caching."""
        cache_key = self._cache_key(error, context)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        guidance = await self.llm.generate_guidance(error, context, classification)
        self._set_cache(cache_key, guidance)
        return guidance


def format_user_error(
    guidance: ErrorGuidance,
    error: Exception,
    show_technical: bool = False,
) -> str:
    """Format error guidance for user display.

    Args:
        guidance: LLM-generated or static error guidance
        error: Original exception
        show_technical: Whether to include technical error details

    Returns:
        Formatted multi-line string for display
    """
    lines = [
        guidance.summary,
        "─" * 40,
        "",
        guidance.explanation,
        "",
        "To fix:",
        f"  {guidance.suggested_fix}",
    ]

    if guidance.code_example:
        lines.extend([
            "",
            f"  {guidance.code_example}",
        ])

    if guidance.doc_link:
        lines.extend([
            "",
            f"Documentation: {guidance.doc_link}",
        ])

    if show_technical:
        lines.extend([
            "",
            "Technical details:",
            f"  {type(error).__name__}: {error}",
        ])

    return "\n".join(lines)


__all__ = [
    "ErrorContext",
    "ErrorGuidance",
    "LLMErrorGuidance",
    "CachedGuidanceGenerator",
    "UserExpertiseLevel",
    "STATIC_GUIDANCE",
    "format_user_error",
]
