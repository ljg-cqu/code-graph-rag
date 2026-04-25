from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from codebase_rag.document.circuit_breaker import CircuitBreaker
    from codebase_rag.document.error_handling import DeadLetterQueue, ExtractionError


class ExtractedConcept(BaseModel):
    """A concept extracted from document content."""

    name: str = Field(..., description="Concept name")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names for this concept",
    )
    type: str | None = Field(
        default=None,
        description="Concept type: skill, framework, model, process, risk, principle, concept",
    )
    definition: str = Field(..., description="Brief definition")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence",
    )
    source_chunk_qn: str = Field(..., description="Source chunk qualified name")
    context: str = Field(default="", description="Surrounding context")


class ConceptRelationship(BaseModel):
    """A relationship between two concepts."""

    from_concept: str = Field(..., description="Source concept name")
    to_concept: str = Field(..., description="Target concept name")
    relationship_type: str = Field(
        ...,
        description="Type: RELATED_TO, IS_A, PART_OF, CAUSES",
    )
    strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relationship strength",
    )


class ExtractionResult(BaseModel):
    """Result of concept extraction from a chunk."""

    concepts: list[ExtractedConcept] = Field(default_factory=list)
    relationships: list[ConceptRelationship] = Field(default_factory=list)


class ConceptExtractor(Protocol):
    """Protocol for concept extraction strategies."""

    async def extract(self, chunk_content: str, chunk_qn: str) -> ExtractionResult:
        """Extract concepts and relationships from a text chunk.

        Args:
            chunk_content: The text content of the chunk.
            chunk_qn: Qualified name of the chunk for attribution.

        Returns:
            ExtractionResult with concepts and relationships.
        """
        ...


@dataclass
class ConceptExtractionStats:
    """Statistics for concept extraction session.

    Separate from document extraction stats to track concept-specific metrics.
    """

    total_chunks: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_concepts_extracted: int = 0
    total_relationships_extracted: int = 0
    errors_by_type: dict = field(default_factory=dict)
    dead_letter_queue_size: int = 0

    def record_success(self, result: ExtractionResult) -> None:
        self.successful_extractions += 1
        self.total_concepts_extracted += len(result.concepts)
        self.total_relationships_extracted += len(result.relationships)

    def record_failure(self, error) -> None:
        """Record a failure with an ExtractionError."""
        self.failed_extractions += 1
        self.errors_by_type[error.error_type.value] = (
            self.errors_by_type.get(error.error_type.value, 0) + 1
        )

    def to_summary(self) -> str:
        """Generate a summary report."""
        success_rate = (
            self.successful_extractions / self.total_chunks * 100
            if self.total_chunks > 0
            else 0
        )
        # Filter to only concept-related errors
        concept_errors = {
            k: v for k, v in self.errors_by_type.items() if k.startswith("concept_")
        }
        return (
            f"Concept Extraction Summary:\n"
            f"  Total chunks: {self.total_chunks}\n"
            f"  Successful: {self.successful_extractions} ({success_rate:.1f}%)\n"
            f"  Failed: {self.failed_extractions}\n"
            f"  Concepts extracted: {self.total_concepts_extracted}\n"
            f"  Relationships extracted: {self.total_relationships_extracted}\n"
            f"  Errors by type: {concept_errors}\n"
            f"  Queued for retry: {self.dead_letter_queue_size}"
        )


def calculate_adaptive_timeout(
    chunk_content: str,
    base_timeout: float = 30.0,
    max_timeout: float = 120.0,
    timeout_per_1k_chars: float = 10.0,
    timeout_per_code_block: float = 5.0,
    max_size_factor: float = 30.0,
    max_complexity_factor: float = 20.0,
) -> float:
    chunk_length = len(chunk_content)
    size_factor = min(chunk_length / 1000 * timeout_per_1k_chars, max_size_factor)
    code_block_count = chunk_content.count("```")
    complexity_factor = min(code_block_count * timeout_per_code_block, max_complexity_factor)
    timeout = base_timeout + size_factor + complexity_factor
    return min(timeout, max_timeout)


class LLMConceptExtractor:
    """LLM-based concept extraction implementation."""

    __slots__ = (
        "agent",
        "timeout",
        "max_timeout",
        "_initialization_failed",
        "_circuit_breaker",
        "_consecutive_timeouts",
        "_consecutive_timeouts_lock",
    )

    SYSTEM_PROMPT = """You are a concept extractor for technical documentation.
Given a document chunk, identify:
1. Key concepts mentioned (with definitions if available)
2. The type/category of each concept
3. Relationships between concepts (RELATED_TO, IS_A, PART_OF, CAUSES)

Concept types:
- skill: A learnable capability or competency
- framework: A structured approach, methodology, or system
- model: A conceptual representation or theoretical construct
- process: A systematic sequence of actions or operations
- risk: A potential negative outcome, pitfall, or concern
- principle: A fundamental truth, rule, or guideline
- concept: A general idea, notion, or abstract thought

Respond with JSON matching this structure:
{
  "concepts": [
    {
      "name": "concept name",
      "aliases": ["alternative name"],
      "type": "skill|framework|model|process|risk|principle|concept",
      "definition": "brief definition",
      "confidence": 0.9
    }
  ],
  "relationships": [
    {
      "from_concept": "source",
      "to_concept": "target",
      "relationship_type": "RELATED_TO",
      "strength": 0.8
    }
  ]
}

Rules:
- Only extract concepts that are clearly defined or important in the text
- Assign the most specific type that fits; use "concept" as fallback
- Confidence should reflect how clearly the concept is presented
- Relationship types must be one of: RELATED_TO, IS_A, PART_OF, CAUSES
- Strength reflects how explicitly the relationship is stated"""

    def __init__(
        self,
        timeout: float | None = None,
        max_timeout: float | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        from codebase_rag.config import settings
        from codebase_rag.document.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )

        self.agent = None
        self.timeout = timeout if timeout is not None else settings.DOC_CONCEPT_BASE_TIMEOUT
        self.max_timeout = max_timeout if max_timeout is not None else settings.DOC_CONCEPT_MAX_TIMEOUT
        self._initialization_failed = False
        self._consecutive_timeouts = 0
        self._consecutive_timeouts_lock = asyncio.Lock()
        if circuit_breaker is not None:
            self._circuit_breaker = circuit_breaker
        elif settings.CGR_CIRCUIT_BREAKER_ENABLED:
            self._circuit_breaker = CircuitBreaker(
                name="concept_extraction",
                config=CircuitBreakerConfig(
                    failure_threshold=settings.CGR_CIRCUIT_FAILURE_THRESHOLD,
                    success_threshold=settings.CGR_CIRCUIT_SUCCESS_THRESHOLD,
                    timeout_seconds=settings.CGR_CIRCUIT_TIMEOUT_SECONDS,
                    window_size=settings.CGR_CIRCUIT_WINDOW_SIZE,
                ),
            )
        else:
            self._circuit_breaker = None

    async def _probe_provider_health(self) -> bool:
        from codebase_rag.config import settings

        if not self._initialize_agent():
            return False
        if self._circuit_breaker is not None and not self._circuit_breaker.can_execute():
            return False
        try:
            await asyncio.wait_for(
                self.agent.run("Respond with the single word: OK"),
                timeout=settings.DOC_CONCEPT_PROBE_TIMEOUT,
            )
            return True
        except Exception:
            return False

    def _initialize_agent(self) -> bool:
        if self.agent is not None:
            return True
        if self._initialization_failed:
            return False

        try:
            from codebase_rag.compat.pydantic_ai import Agent
            from codebase_rag.config import settings
            from codebase_rag.services.llm import _create_chat_model

            config = settings.active_orchestrator_config
            llm = _create_chat_model(config)

            self.agent = Agent(
                model=llm,
                system_prompt=self.SYSTEM_PROMPT,
                output_type=ExtractionResult,
                retries=1,
            )
            return True
        except Exception as e:
            from loguru import logger

            logger.error(f"Failed to initialize concept extraction agent: {e}")
            self._initialization_failed = True
            return False

    async def extract(
        self,
        chunk_content: str,
        chunk_qn: str,
        timeout: float | None = None,
    ) -> ExtractionResult:
        if not self._initialize_agent():
            return ExtractionResult()

        if self._circuit_breaker is not None and not self._circuit_breaker.can_execute():
            from loguru import logger

            from codebase_rag.document import logs as doc_ls

            logger.debug(
                doc_ls.DOC_CONCEPT_CIRCUIT_BREAKER_OPEN.format(chunk_qn=chunk_qn)
            )
            return ExtractionResult()

        if timeout is None:
            timeout = calculate_adaptive_timeout(
                chunk_content,
                base_timeout=self.timeout,
                max_timeout=self.max_timeout,
            )

        try:
            result = await asyncio.wait_for(
                self.agent.run(chunk_content),
                timeout=timeout,
            )
            for concept in result.output.concepts:
                concept.source_chunk_qn = chunk_qn
            if self._circuit_breaker is not None:
                self._circuit_breaker.record_success()
            return result.output
        except TimeoutError:
            raise
        except Exception:
            if self._circuit_breaker is not None:
                self._circuit_breaker.record_failure()
            raise

    async def extract_with_retry(
        self,
        chunk_content: str,
        chunk_qn: str,
        dead_letter_queue: DeadLetterQueue | None = None,
    ) -> ExtractionResult:
        from loguru import logger

        from codebase_rag.config import settings
        from codebase_rag.document import logs as doc_ls
        from codebase_rag.document.error_handling import ErrorType

        if self._circuit_breaker is not None and not self._circuit_breaker.can_execute():
            return ExtractionResult()

        max_retries = settings.DOC_CONCEPT_EXTRACTION_MAX_RETRIES
        base_delay = settings.DOC_CONCEPT_EXTRACTION_RETRY_DELAY

        original_timeout = calculate_adaptive_timeout(
            chunk_content,
            base_timeout=self.timeout,
            max_timeout=self.max_timeout,
        )
        current_timeout = original_timeout

        last_error = None

        for attempt in range(max_retries + 1):
            try:
                result = await self.extract(chunk_content, chunk_qn, timeout=current_timeout)
                return result
            except Exception as e:
                error = classify_concept_extraction_error(e, chunk_content, chunk_qn)
                error.retry_count = attempt
                last_error = error

                if error.error_type == ErrorType.CONCEPT_QUOTA_EXCEEDED:
                    logger.error(
                        f"LLM quota exceeded. Reset at: {error.retry_after or 'unknown'}. "
                        f"Stopping all retries to preserve remaining quota."
                    )
                    break

                if not error.recoverable:
                    logger.error(f"Non-recoverable error: {error.error_type.value}")
                    break

                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)

                    if error.error_type == ErrorType.CONCEPT_TIMEOUT:
                        async with self._consecutive_timeouts_lock:
                            self._consecutive_timeouts += 1
                            if self._consecutive_timeouts >= settings.DOC_CONCEPT_CONSECUTIVE_TIMEOUT_THRESHOLD:
                                current_timeout = min(settings.DOC_CONCEPT_FAST_FAIL_TIMEOUT, current_timeout)
                            else:
                                current_timeout = min(
                                    current_timeout * settings.DOC_CONCEPT_TIMEOUT_RETRY_MULTIPLIER,
                                    original_timeout * settings.DOC_CONCEPT_TIMEOUT_RETRY_CAP_MULTIPLIER,
                                    self.max_timeout,
                                )
                        delay = max(
                            base_delay * settings.DOC_CONCEPT_TIMEOUT_RETRY_DELAY_MULTIPLIER,
                            delay,
                        )
                        if not await self._probe_provider_health():
                            logger.warning(
                                doc_ls.DOC_CONCEPT_PROVIDER_UNHEALTHY.format(chunk_qn=chunk_qn)
                            )
                            break
                    else:
                        async with self._consecutive_timeouts_lock:
                            self._consecutive_timeouts = 0

                    jitter = delay * random.uniform(0.0, 0.25)
                    delay = delay + jitter

                    if error.error_type == ErrorType.CONCEPT_RATE_LIMITED:
                        if error.retry_after:
                            from datetime import datetime
                            retry_dt = datetime.fromisoformat(error.retry_after)
                            delay = max(delay, (retry_dt - datetime.now()).total_seconds())

                    logger.warning(
                        doc_ls.DOC_CONCEPT_RETRY_ATTEMPT.format(
                            attempt=attempt + 1,
                            max_retries=max_retries + 1,
                            chunk_qn=chunk_qn,
                            delay=delay,
                            error_type=error.error_type.value,
                            timeout=current_timeout,
                        )
                    )
                    await asyncio.sleep(delay)

        if last_error and dead_letter_queue:
            dead_letter_queue.enqueue(last_error)
            logger.info(
                doc_ls.DOC_CONCEPT_RETRY_EXHAUSTED.format(chunk_qn=chunk_qn)
            )

        return ExtractionResult()


def _parse_quota_reset_time(message: str) -> datetime | None:
    """Parse quota reset time from API error message.

    Example: "It will reset at 2026-05-09 23:59:59 +0800 CST"

    Returns:
        datetime object or None if parsing fails
    """
    import re

    # Pattern: YYYY-MM-DD HH:MM:SS +/-HHMM TZ
    pattern = r"reset at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4})"
    match = re.search(pattern, message)

    if match:
        try:
            # Parse with timezone offset
            time_str = match.group(1)
            return datetime.fromisoformat(time_str.replace(" ", "T"))
        except ValueError:
            pass

    return None


def classify_concept_extraction_error(
    exc: Exception,
    chunk_content: str,
    chunk_qn: str,
) -> ExtractionError:
    """Classify a concept extraction error for better handling.

    LLM-First Design: Classification is deterministic Python logic.
    LLM interprets user intent; infrastructure handles error classification.

    Args:
        exc: The exception that occurred
        chunk_content: Content of the chunk being processed
        chunk_qn: Qualified name of the chunk

    Returns:
        ExtractionError with classification and context
    """
    from datetime import UTC, datetime

    from codebase_rag.document.error_handling import ErrorType, ExtractionError

    error_type = ErrorType.UNKNOWN
    retry_after: datetime | None = None
    is_recoverable = True

    exc_type = type(exc).__name__
    exc_message = str(exc) or "(empty error message)"
    exc_message_lower = exc_message.lower()

    # Check for ModelHTTPError attributes
    status_code = getattr(exc, "status_code", None)
    error_code = None
    reset_time = None

    # Extract structured error info from ModelHTTPError
    if hasattr(exc, "body") and isinstance(exc.body, dict):
        error_code = exc.body.get("code")
        # Parse reset time from message if available
        body_message = exc.body.get("message", "")
        reset_time = _parse_quota_reset_time(body_message)

    # Classification priority: specific to general

    # 1. Quota Exceeded (HTTP 429 with AccountQuotaExceeded, or quota in message)
    if status_code == 429 or "429" in exc_message:
        if error_code == "AccountQuotaExceeded" or "quota" in exc_message_lower:
            error_type = ErrorType.CONCEPT_QUOTA_EXCEEDED
            is_recoverable = False  # Cannot retry until quota resets
            retry_after = reset_time
        elif "rate" in exc_message_lower or error_code in ("RateLimitExceeded", "TooManyRequests"):
            error_type = ErrorType.CONCEPT_RATE_LIMITED
            # Rate limits are transient - recoverable with backoff
            is_recoverable = True
        else:
            # Generic 429 - treat as rate limited (conservative)
            error_type = ErrorType.CONCEPT_RATE_LIMITED
            is_recoverable = True
    # Also detect quota errors from message content (even without 429 status code)
    elif "quota" in exc_message_lower and ("exceed" in exc_message_lower or "limit" in exc_message_lower):
        error_type = ErrorType.CONCEPT_QUOTA_EXCEEDED
        is_recoverable = False
        retry_after = reset_time

    # 2. Timeout errors
    elif isinstance(exc, asyncio.TimeoutError):
        error_type = ErrorType.CONCEPT_TIMEOUT

    # 3. Authentication errors
    elif "auth" in exc_message_lower or "api key" in exc_message_lower:
        error_type = ErrorType.CONCEPT_AUTH_ERROR
        is_recoverable = False

    # 4. Context/token overflow
    elif "context" in exc_message_lower or "token" in exc_message_lower:
        error_type = ErrorType.CONCEPT_CONTEXT_OVERFLOW
        is_recoverable = False

    # 5. Network errors
    elif "connection" in exc_message_lower or "network" in exc_message_lower:
        error_type = ErrorType.CONCEPT_NETWORK_ERROR

    # 6. Parsing errors
    elif "json" in exc_message_lower or "parse" in exc_message_lower:
        error_type = ErrorType.CONCEPT_PARSING_ERROR

    # 7. Generic LLM errors (fallback)
    elif "llm" in exc_message_lower or "model" in exc_message_lower:
        error_type = ErrorType.CONCEPT_LLM_ERROR

    return ExtractionError(
        path=chunk_qn,
        error_type=error_type,
        message=exc_message,
        timestamp=datetime.now(UTC).isoformat(),
        recoverable=is_recoverable,
        retry_after=retry_after.isoformat() if retry_after else None,
        chunk_qn=chunk_qn,
        chunk_length=len(chunk_content),
        chunk_preview=chunk_content[:200] if len(chunk_content) > 200 else chunk_content,
        exception_type=exc_type,
    )


def get_user_facing_message(error: ExtractionError) -> str:
    """Generate user-friendly error message with actionable guidance.

    LLM-First Design: Deterministic messages for infrastructure errors.
    LLM enhances messaging for user context when needed.

    Args:
        error: The ExtractionError to generate a message for

    Returns:
        User-friendly error message with actionable guidance
    """
    from codebase_rag.document.error_handling import ErrorType

    messages = {
        ErrorType.CONCEPT_QUOTA_EXCEEDED: (
            "The LLM API monthly quota has been exceeded. "
            f"Quota resets at: {error.retry_after or 'end of billing period'}. "
            "Options: 1) Wait for quota reset, 2) Upgrade your API plan, "
            "3) Use a different LLM provider (set via EMBEDDING_PROVIDER)."
        ),
        ErrorType.CONCEPT_RATE_LIMITED: (
            "Too many requests to the LLM API. "
            "The system will automatically retry with exponential backoff. "
            "No action needed - extraction will continue shortly."
        ),
        ErrorType.CONCEPT_AUTH_ERROR: (
            "Authentication failed with the LLM API. "
            "Please check your API key configuration. "
            "Set the correct key via environment variable or config file."
        ),
        ErrorType.CONCEPT_CONTEXT_OVERFLOW: (
            "Document chunk is too large for the LLM context window. "
            "Consider reducing DOC_CHUNK_SIZE in configuration."
        ),
    }

    return messages.get(
        error.error_type,
        f"Concept extraction failed: {error.error_type.value}. "
        f"Details: {error.message[:100]}"
    )


def save_failed_chunk_for_debug(
    error: ExtractionError,
    chunk_content: str,
    debug_dir: Path | None = None,
) -> Path | None:
    """Save failed chunk content for debugging.

    LLM-First Design: This is infrastructure for debugging - deterministic
    file operations, no semantic decisions needed.

    Args:
        error: The ExtractionError from the failed extraction
        chunk_content: The full chunk content that failed
        debug_dir: Directory to save debug files (defaults to config setting)

    Returns:
        Path to the debug file, or None if debug mode is disabled
    """
    from loguru import logger

    from codebase_rag.config import settings

    if not settings.CGR_DEBUG_CONCEPT_EXTRACTION:
        return None

    if debug_dir is None:
        debug_dir = Path(settings.CGR_DEBUG_DIR)

    debug_dir.mkdir(parents=True, exist_ok=True)

    # Create safe filename from timestamp and error type
    safe_timestamp = error.timestamp.replace(":", "-").replace(".", "-")
    debug_file = debug_dir / f"failed_{error.error_type.value}_{safe_timestamp}.json"

    debug_data = {
        "error_type": error.error_type.value,
        "exception_type": error.exception_type,
        "exception_message": error.message,
        "chunk_qn": error.chunk_qn,
        "chunk_length": error.chunk_length,
        "chunk_content": chunk_content,
        "timestamp": error.timestamp,
        "recoverable": error.recoverable,
    }

    try:
        with open(debug_file, "w") as f:
            json.dump(debug_data, f, indent=2)

        logger.debug(f"Saved failed chunk debug data to {debug_file}")
        return debug_file
    except Exception as e:
        logger.warning(f"Failed to save debug data: {e}")
        return None
