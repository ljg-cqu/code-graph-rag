from __future__ import annotations

import asyncio
import fcntl
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

from pydantic import BaseModel, Field

from codebase_rag.document.error_handling import FATAL_ERRORS, ErrorType
from codebase_rag.utils.token_utils import count_tokens

if TYPE_CHECKING:
    from codebase_rag.document.circuit_breaker import CircuitBreaker
    from codebase_rag.document.error_handling import DeadLetterQueue, ExtractionError


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class ExtractedConcept(BaseModel):
    """A concept extracted from document content.

    Entity classification uses a 7-category MECE taxonomy (entity_category)
    with optional domain-specific sub-types (entity_subtype) and server-resolved
    emoji (entity_emoji). The old `type` field is deprecated and set equal to
    entity_category for backward compatibility.
    """

    name: str = Field(..., description="Concept name")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names for this concept",
    )
    type: str | None = Field(
        default=None,
        description="DEPRECATED — use entity_category. Set equal to entity_category for backward compat.",
    )
    definition: str = Field(..., description="Brief definition")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Extraction confidence",
    )
    context: str = Field(default="", description="Surrounding context")
    entity_category: str | None = Field(
        default=None,
        description="One of 7 canonical entity categories (CONCRETE_ENTITY, EVENT_PROCESS, etc.)",
    )
    entity_subtype: str | None = Field(
        default=None,
        description="Optional domain-specific sub-type (e.g. 'Container Image', 'Deployment')",
    )
    entity_emoji: str = Field(
        default="",
        description="Server-resolved emoji for the entity category",
    )


class ConceptRelationship(BaseModel):
    """A relationship between two concepts.

    The `verb` is the specific semantic predicate (e.g. "mitigates", "deployed-in").
    `category` is the canonical category (edge label in the graph).
    `emoji` is always server-resolved from category via CATEGORY_EMOJI_MAP.
    """

    from_concept: str = Field(..., description="Source concept name")
    to_concept: str = Field(..., description="Target concept name")
    verb: str = Field(
        ...,
        description="Specific verb describing how concepts relate",
    )
    category: str = Field(
        ...,
        description="Canonical category: HIERARCHICAL, COMPOSITIONAL, CONTEXTUAL, "
        "ATTRIBUTIVE, COMPARATIVE, SEQUENTIAL, CAUSAL, ANALOGICAL, or RELATED_TO",
    )
    emoji: str = Field(
        default="",
        description="Visual marker (server-resolved from category, not LLM-provided)",
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
    was_rebalanced: bool = Field(
        default=False,
        description="True if relationship diversity rebalancing was triggered",
    )


class ConceptExtractor(Protocol):
    """Protocol for concept extraction strategies."""

    async def extract(
        self,
        chunk_content: str,
        chunk_qn: str,
        max_tokens: int | None = None,
    ) -> ExtractionResult:
        """Extract concepts and relationships from a text chunk.

        Args:
            chunk_content: The text content of the chunk.
            chunk_qn: Qualified name of the chunk for attribution.
            max_tokens: Optional override for output token budget.

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
    skewed_chunks_count: int = 0

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
            f"  Skewed chunks: {self.skewed_chunks_count}\n"
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
    # Markdown-specific complexity factors for document-heavy chunks
    timeout_per_heading: float = 2.0,
    max_heading_factor: float = 15.0,
    timeout_per_list_item: float = 0.5,
    max_list_factor: float = 10.0,
) -> float:
    chunk_length = len(chunk_content)
    size_factor = min(chunk_length / 1000 * timeout_per_1k_chars, max_size_factor)
    code_block_count = chunk_content.count("```")
    complexity_factor = min(code_block_count * timeout_per_code_block, max_complexity_factor)

    # Markdown-specific heuristics for dense conceptual documents
    heading_count = chunk_content.count("#")
    heading_factor = min(heading_count * timeout_per_heading, max_heading_factor)

    list_item_count = (
        chunk_content.count("\n-") + chunk_content.count("\n*") + chunk_content.count("\n1.")
    )
    list_factor = min(list_item_count * timeout_per_list_item, max_list_factor)

    timeout = base_timeout + size_factor + complexity_factor + heading_factor + list_factor
    return min(timeout, max_timeout)


class DensityResult(NamedTuple):
    """Content density classification for adaptive token calculation."""

    category: str
    multiplier: float


DENSITY_PLAIN = "plain"
DENSITY_CODE = "code"
DENSITY_LIST = "list"
DENSITY_TABLE = "table"


def _estimate_content_density(content: str) -> DensityResult:
    """Estimate content density based on structural markers."""
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines == 0:
        return DensityResult(DENSITY_PLAIN, 1.0)

    table_lines = sum(
        1 for line in lines if "|" in line and line.strip().startswith("|")
    )
    list_lines = sum(
        1
        for line in lines
        if line.strip().startswith(("- ", "* ", "1. ", "2. "))
    )
    code_fence_lines = sum(
        1 for line in lines if line.strip().startswith("```")
    )

    table_ratio = table_lines / total_lines
    list_ratio = list_lines / total_lines

    if table_ratio > 0.3:
        return DensityResult(DENSITY_TABLE, 2.0)
    if list_ratio > 0.3:
        return DensityResult(DENSITY_LIST, 1.5)
    if code_fence_lines > 0:
        return DensityResult(DENSITY_CODE, 1.2)
    return DensityResult(DENSITY_PLAIN, 1.0)


def calculate_adaptive_max_tokens(
    chunk_content: str,
    base_tokens: int = 4096,
    min_tokens: int = 1024,
    max_tokens: int = 16384,
    tokens_per_char: float = 0.5,
) -> int:
    """Calculate adaptive max_tokens based on chunk characteristics."""
    char_count = len(chunk_content)
    estimated_tokens = int(char_count * tokens_per_char)
    density = _estimate_content_density(chunk_content)
    adaptive_tokens = int(estimated_tokens * density.multiplier)
    return max(min_tokens, min(max_tokens, adaptive_tokens))


def estimate_input_tokens(content: str) -> int:
    """Estimate input token count for concept extraction."""
    return count_tokens(content)


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
2. The entity category of each concept (7-category MECE taxonomy)
3. Relationships between concepts

Entity categories (7 MECE categories — every concept fits exactly one):

🧱 CONCRETE_ENTITY — Does it occupy physical space?
  Natural Object, Artifact, Substance, Organism, Body Part, Food/Consumable,
  Geographic Feature, Celestial Body
  Software-specific: Virtual Machine, Container Image, Data Center, Mobile Device, Peripheral

⏱️ EVENT_PROCESS — Does it unfold over time?
  Natural Event, Human Action, Process, Incident, Activity, State Change,
  Project/Initiative, Ritual/Routine
  Software-specific: Deployment, Build, Test Run, Incident, Migration

📨 INFORMATION_EXPRESSION — Is it a representation?
  Data, Signal, Symbol, Narrative, Code/Formula, Record/Document, Media
  Software-specific: API, Protocol, Config File, Log Stream, Source File

📏 PROPERTY_ATTRIBUTE — Is it a characteristic of something else?
  Physical Quality, Quantitative Measure, Mental State, Capability/Skill,
  Disposition, Relational Property, Evaluative Property
  Software-specific: SLI, Quality Attribute, Capacity Metric

🏗️ SYSTEM_STRUCTURE — Is it an organized collection?
  Natural System, Social System, Technological System, Network, Hierarchy,
  Framework, Market/Platform
  Software-specific: Distributed System, CI/CD Pipeline, Monorepo, Service Mesh

🎭 AGENT_ROLE — Does it exercise intention or fulfill a role?
  Individual, Collective, Institutional Agent, Non-Human Agent, Role/Position, Persona
  Software-specific: CI Bot, Service Account, End User, On-Call Engineer

💡 ABSTRACT_CONCEPT — Is it a pure idea with no physical form?
  Domain/Discipline, Theory/Model, Principle/Rule, Value/Ideal, Category/Class, Relation/Connection
  Software-specific: Design Pattern, Algorithm, Protocol Spec, Paradigm, SLA
  Academic/Cognitive-specific: Mental Model, Psychological Theory, Capability

💡 ABSTRACT_CONCEPT is the LAST RESORT — confirm the entity doesn't fit any of the
first 6 categories before using it. This mirrors RELATED_TO in the relationship taxonomy.

Examples by domain:
- "Critical Thinking" → EVENT_PROCESS (cognitive activity unfolding over time)
- "Intellectual Humility" → PROPERTY_ATTRIBUTE (characteristic of a person)
- "Educational Framework" → SYSTEM_STRUCTURE (pedagogical framework schema)
- "Case Studies" → INFORMATION_EXPRESSION (representation of knowledge)
- "Active Learning Methods" → EVENT_PROCESS (teaching activity over time)
- "Cognitive Bias" → PROPERTY_ATTRIBUTE (characteristic of thinking)
- "Working Memory" → PROPERTY_ATTRIBUTE (cognitive capacity)
- "Debate" → EVENT_PROCESS (structured discursive activity)

Relationships use an 8-category canonical taxonomy. Choose the most specific verb
that accurately describes how two concepts relate. The verb must belong to exactly
one of these categories:

🌳 HIERARCHICAL — What type/category?
  is-a, subtype-of, classifies-as, inherits-from, specializes, instance-of

🧩 COMPOSITIONAL — What parts make up this?
  part-of, comprises, contains, includes, consists-of, component-of

🎯 CONTEXTUAL — What context surrounds?
  located-in, situated-in, operates-within, deployed-in, occurs-within, is-bound-by

💭 ATTRIBUTIVE — What attributes describe?
  has-property, characterized-by, exhibits, possesses, features, requires

⚖️ COMPARATIVE — How do these compare?
  compares-to, contrasts-with, similar-to, different-from, supersedes, equivalent-to

⏩ SEQUENTIAL — What happens in order?
  precedes, follows, transitions-to, evolves-into, progresses-to

⚡ CAUSAL — What causes what?
  causes, produces, triggers, enables, prevents, depends-on, influences, mitigates, generates

🌉 ANALOGICAL — What is this similar to?
  analogous-to, corresponds-to, maps-to, parallels, resembles, mirrors

🔗 RELATED_TO — Use ONLY as a last resort when no other category fits.

Relationship diversity guidance: Diversify relationship categories. Aim for no single
category to exceed 30% of relationships in a chunk. If you find yourself using CAUSAL
repeatedly, pause and consider if COMPOSITIONAL, ATTRIBUTIVE, or COMPARATIVE might fit.

Respond with JSON matching this structure:
{
  "concepts": [
    {
      "name": "concept name",
      "aliases": ["alternative name"],
      "type": "CONCRETE_ENTITY",
      "entity_category": "CONCRETE_ENTITY",
      "entity_subtype": "Container Image",
      "entity_emoji": "🧱",
      "definition": "brief definition",
      "confidence": 0.9
    }
  ],
  "relationships": [
    {
      "from_concept": "source",
      "to_concept": "target",
      "verb": "mitigates",
      "category": "CAUSAL",
      "emoji": "⚡",
      "strength": 0.8
    }
  ]
}

Rules:
- Only extract concepts that are clearly defined or important in the text
- Every concept must have an entity_category from the 7 canonical categories
- entity_category must be the canonical NAME (e.g., "EVENT_PROCESS"), NEVER an emoji
- entity_subtype is optional but recommended; use domain-specific subtypes when available
- entity_emoji must match the category: 🧱⏱️📨📏🏗️🎭💡
- ABSTRACT_CONCEPT is a last resort, not a default
- The old type field should equal entity_category for backward compatibility
- Use the most precise relationship verb possible, even beyond the examples listed
- Every verb must fit into exactly one of the 8 categories above
- category must be one of: HIERARCHICAL, COMPOSITIONAL, CONTEXTUAL, ATTRIBUTIVE, COMPARATIVE, SEQUENTIAL, CAUSAL, ANALOGICAL, RELATED_TO
- emoji must match the category: 🌳🧩🎯💭⚖️⏩⚡🌉🔗
- RELATED_TO is a last resort, not a default
- Confidence/strength should reflect how clearly the concept/relationship is presented"""

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
            from codebase_rag.compat.pydantic_ai import Agent, ModelSettings
            from codebase_rag.config import settings
            from codebase_rag.services.llm import _create_chat_model

            config = settings.active_orchestrator_config
            llm = _create_chat_model(config)

            max_tokens = settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS

            self.agent = Agent(
                model=llm,
                system_prompt=self.SYSTEM_PROMPT,
                output_type=ExtractionResult,
                retries=1,
                model_settings=ModelSettings(max_tokens=max_tokens),
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
        max_tokens: int | None = None,
    ) -> ExtractionResult:
        from loguru import logger

        if not self._initialize_agent():
            return ExtractionResult()

        if self._circuit_breaker is not None and not self._circuit_breaker.can_execute():
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

        if max_tokens is None:
            from codebase_rag.config import settings

            max_tokens = calculate_adaptive_max_tokens(
                chunk_content,
                base_tokens=settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS,
                min_tokens=settings.DOC_CONCEPT_MIN_OUTPUT_TOKENS,
                max_tokens=settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS,
                tokens_per_char=settings.DOC_CONCEPT_TOKENS_PER_CHAR,
            )

        from codebase_rag.compat.pydantic_ai import ModelSettings

        model_settings = ModelSettings(max_tokens=max_tokens)

        try:
            result = await asyncio.wait_for(
                self.agent.run(chunk_content, model_settings=model_settings),
                timeout=timeout,
            )
            for concept in result.output.concepts:
                (
                    concept.entity_category,
                    concept.entity_subtype,
                    concept.entity_emoji,
                ) = resolve_entity_category(
                    concept.entity_category,
                    concept.entity_subtype,
                    concept.definition,
                )
                # Backward compat: old `type` field = entity_category
                concept.type = concept.entity_category
            for rel in result.output.relationships:
                rel.category, rel.emoji = resolve_category(rel.verb, rel.category)

            # D-8: Relationship category diversity check + rebalancing
            if result.output.relationships:
                is_skewed = _check_relationship_diversity(
                    result.output.relationships,
                    chunk_qn,
                    min_count_for_check=4,
                )
                if is_skewed:
                    result.output.was_rebalanced = True
                    result.output.relationships = await _rebalance_relationships(
                        result.output.relationships,
                        chunk_content,
                        chunk_qn,
                    )
                    # Apply hard cap as deterministic fallback
                    result.output.relationships = _apply_category_caps(
                        result.output.relationships
                    )

            # Confidence / strength filtering
            from codebase_rag.config import settings

            min_confidence = settings.DOC_CONCEPT_MIN_CONFIDENCE
            min_strength = getattr(settings, "DOC_CONCEPT_MIN_RELATIONSHIP_STRENGTH", 0.6)

            filtered_concepts = [
                c for c in result.output.concepts
                if c.confidence >= min_confidence
            ]
            if len(filtered_concepts) < len(result.output.concepts):
                logger.debug(
                    f"Filtered {len(result.output.concepts) - len(filtered_concepts)} "
                    f"low-confidence concepts for {chunk_qn}"
                )
            result.output.concepts = filtered_concepts

            filtered_rels = [
                r for r in result.output.relationships
                if r.strength >= min_strength
            ]
            if len(filtered_rels) < len(result.output.relationships):
                logger.debug(
                    f"Filtered {len(result.output.relationships) - len(filtered_rels)} "
                    f"low-strength relationships for {chunk_qn}"
                )
            result.output.relationships = filtered_rels

            if self._circuit_breaker is not None:
                self._circuit_breaker.record_success()
            return result.output
        except TimeoutError:
            raise
        except Exception as e:
            error = classify_concept_extraction_error(e, chunk_content, chunk_qn)
            if error.error_type not in FATAL_ERRORS and error.error_type != ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT:
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
        current_max_tokens = calculate_adaptive_max_tokens(
            chunk_content,
            base_tokens=settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS,
            min_tokens=settings.DOC_CONCEPT_MIN_OUTPUT_TOKENS,
            max_tokens=settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS,
            tokens_per_char=settings.DOC_CONCEPT_TOKENS_PER_CHAR,
        )

        last_error = None

        # Proactive split: avoid initial failure for predictable overflow cases
        estimated_tokens = estimate_input_tokens(chunk_content)
        input_limit = settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT
        if estimated_tokens > input_limit or _should_proactive_split(chunk_content):
            logger.info(
                doc_ls.DOC_CONCEPT_PROACTIVE_SPLIT.format(
                    chunk_qn=chunk_qn, tokens=estimated_tokens, limit=input_limit
                )
            )
            try:
                split_result = await self._recursive_split(
                    chunk_content, chunk_qn, current_timeout
                )
                if split_result.concepts or split_result.relationships:
                    logger.info(doc_ls.DOC_CONCEPT_SPLIT_SUCCESS.format(chunk_qn=chunk_qn))
                    return split_result
            except Exception as split_error:
                logger.warning(
                    doc_ls.DOC_CONCEPT_SPLIT_FAILED.format(
                        chunk_qn=chunk_qn, error=split_error
                    )
                )

        for attempt in range(max_retries + 1):
            try:
                result = await self.extract(
                    chunk_content, chunk_qn, timeout=current_timeout, max_tokens=current_max_tokens
                )
                async with self._consecutive_timeouts_lock:
                    self._consecutive_timeouts = 0
                if attempt > 0:
                    logger.debug(
                        doc_ls.DOC_CONCEPT_RETRY_SUCCESS.format(
                            attempt=attempt + 1, chunk_qn=chunk_qn
                        )
                    )
                return result
            except Exception as e:
                error = classify_concept_extraction_error(e, chunk_content, chunk_qn)
                error.retry_count = attempt
                last_error = error

                if error.error_type in (
                    ErrorType.CONCEPT_CONTEXT_OVERFLOW,
                    ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT,
                ):
                    logger.info(doc_ls.DOC_CONCEPT_SPLIT_ATTEMPT.format(chunk_qn=chunk_qn))
                    try:
                        split_result = await self._extract_with_splitting(
                            chunk_content, chunk_qn, current_timeout
                        )
                        if split_result.concepts or split_result.relationships:
                            logger.info(doc_ls.DOC_CONCEPT_SPLIT_SUCCESS.format(chunk_qn=chunk_qn))
                            return split_result
                    except Exception as split_error:
                        logger.warning(
                            doc_ls.DOC_CONCEPT_SPLIT_FAILED.format(
                                chunk_qn=chunk_qn, error=split_error
                            )
                        )

                    if error.error_type == ErrorType.CONCEPT_CONTEXT_OVERFLOW:
                        model_max = settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS
                        if current_max_tokens < model_max and attempt < max_retries:
                            current_max_tokens = min(current_max_tokens * 2, model_max)
                            logger.info(
                                doc_ls.DOC_CONCEPT_ADAPTIVE_RETRY.format(
                                    chunk_qn=chunk_qn, max_tokens=current_max_tokens
                                )
                            )
                            continue
                    else:
                        model_max = settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS
                        if current_max_tokens < model_max and attempt < max_retries:
                            current_max_tokens = min(current_max_tokens * 2, model_max)
                            logger.info(
                                doc_ls.DOC_CONCEPT_OUTPUT_ADAPTIVE_RETRY.format(
                                    chunk_qn=chunk_qn, max_tokens=current_max_tokens
                                )
                            )
                            continue

                if error.error_type == ErrorType.CONCEPT_QUOTA_EXCEEDED:
                    logger.error(
                        doc_ls.DOC_CONCEPT_QUOTA_EXCEEDED.format(
                            retry_after=error.retry_after or "unknown"
                        )
                    )
                    break

                if not error.recoverable:
                    logger.error(
                        doc_ls.DOC_CONCEPT_NON_RECOVERABLE.format(
                            error_type=error.error_type.value
                        )
                    )
                    break

                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)

                    if error.error_type == ErrorType.CONCEPT_TIMEOUT:
                        async with self._consecutive_timeouts_lock:
                            self._consecutive_timeouts += 1
                            if (
                                self._consecutive_timeouts >= settings.DOC_CONCEPT_CONSECUTIVE_TIMEOUT_THRESHOLD
                                and attempt == 0
                            ):
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

    async def _extract_with_splitting(
        self,
        chunk_content: str,
        chunk_qn: str,
        timeout: float,
    ) -> ExtractionResult:
        paragraphs = chunk_content.split("\n\n")

        if len(paragraphs) == 1:
            sentences = _SENTENCE_SPLIT_RE.split(chunk_content)
            if len(sentences) > 1:
                mid = len(sentences) // 2
                parts = [" ".join(sentences[:mid]), " ".join(sentences[mid:])]
            else:
                return ExtractionResult()
        else:
            mid = len(paragraphs) // 2
            parts = ["\n\n".join(paragraphs[:mid]), "\n\n".join(paragraphs[mid:])]

        results = []
        for i, part in enumerate(parts):
            if not part.strip():
                continue
            try:
                result = await self.extract(part, f"{chunk_qn}_part{i}", timeout=timeout)
                results.append(result)
            except Exception:
                continue

        return self._merge_extraction_results(results)

    async def _recursive_split(
        self,
        content: str,
        chunk_qn: str,
        timeout: float,
        depth: int = 0,
    ) -> ExtractionResult:
        from loguru import logger

        from codebase_rag.config import settings
        from codebase_rag.document.chunk_splitting import _split_chunk
        from codebase_rag.document import logs as doc_ls

        max_depth = settings.DOC_CONCEPT_MAX_SPLIT_DEPTH
        input_limit = settings.DOC_CONCEPT_INPUT_TOKEN_LIMIT
        min_size = settings.DOC_CONCEPT_MIN_CHUNK_SIZE

        if depth > 0:
            logger.info(
                doc_ls.DOC_CONCEPT_RECURSIVE_SPLIT.format(
                    depth=depth, chunk_qn=chunk_qn
                )
            )

        parts = _split_chunk(content)
        results: list[ExtractionResult] = []

        for i, part in enumerate(parts):
            if not part.strip():
                continue
            part_tokens = estimate_input_tokens(part)
            if part_tokens > input_limit and len(part) > min_size and depth < max_depth:
                sub_result = await self._recursive_split(
                    part, f"{chunk_qn}_part{i}", timeout, depth + 1
                )
                results.append(sub_result)
            else:
                try:
                    part_max_tokens = calculate_adaptive_max_tokens(
                        part,
                        base_tokens=settings.DOC_CONCEPT_EXTRACTION_MAX_TOKENS,
                        min_tokens=settings.DOC_CONCEPT_MIN_OUTPUT_TOKENS,
                        max_tokens=settings.DOC_CONCEPT_MAX_OUTPUT_TOKENS,
                        tokens_per_char=settings.DOC_CONCEPT_TOKENS_PER_CHAR,
                    )
                    result = await self.extract(
                        part,
                        f"{chunk_qn}_part{i}",
                        timeout=timeout,
                        max_tokens=part_max_tokens,
                    )
                    results.append(result)
                except Exception:
                    continue

        return self._merge_extraction_results(results)

    def _merge_extraction_results(self, results: list[ExtractionResult]) -> ExtractionResult:
        all_concepts = {}
        all_relationships = []

        for result in results:
            for concept in result.concepts:
                if concept.name not in all_concepts:
                    all_concepts[concept.name] = concept
            all_relationships.extend(result.relationships)

        seen_rels = set()
        unique_relationships = []
        for rel in all_relationships:
            key = (rel.from_concept, rel.to_concept, rel.verb)
            if key not in seen_rels:
                seen_rels.add(key)
                unique_relationships.append(rel)

        return ExtractionResult(
            concepts=list(all_concepts.values()),
            relationships=unique_relationships,
        )


VERB_REGISTRY: dict[str, str] = {
    # Hierarchical (🌳)
    "is-a": "HIERARCHICAL",
    "subtype-of": "HIERARCHICAL",
    "classifies-as": "HIERARCHICAL",
    "categorizes-under": "HIERARCHICAL",
    "inherits-from": "HIERARCHICAL",
    "specializes": "HIERARCHICAL",
    "generalizes-to": "HIERARCHICAL",
    "instance-of": "HIERARCHICAL",
    "supertype-of": "HIERARCHICAL",
    "descends-from": "HIERARCHICAL",
    "is-parent-of": "HIERARCHICAL",
    "falls-under": "HIERARCHICAL",
    "derives-from": "HIERARCHICAL",
    "derived-from": "HIERARCHICAL",  # Variant of derives-from
    # Compositional (🧩)
    "part-of": "COMPOSITIONAL",
    "comprises": "COMPOSITIONAL",
    "contains": "COMPOSITIONAL",
    "includes": "COMPOSITIONAL",
    "component-of": "COMPOSITIONAL",
    "constituent-of": "COMPOSITIONAL",
    "element-of": "COMPOSITIONAL",
    "member-of": "COMPOSITIONAL",
    "composed-of": "COMPOSITIONAL",
    "consists-of": "COMPOSITIONAL",
    "is-built-from": "COMPOSITIONAL",
    "aggregates": "COMPOSITIONAL",
    "is-formed-from": "COMPOSITIONAL",
    "incorporated-into": "COMPOSITIONAL",
    # Contextual (🎯)
    "located-in": "CONTEXTUAL",
    "situated-in": "CONTEXTUAL",
    "provides-context-for": "CONTEXTUAL",
    "framed-by": "CONTEXTUAL",
    "environment-of": "CONTEXTUAL",
    "setting-for": "CONTEXTUAL",
    "surrounds": "CONTEXTUAL",
    "contained-within": "CONTEXTUAL",
    "occurs-within": "CONTEXTUAL",
    "operates-within": "CONTEXTUAL",
    "exists-under": "CONTEXTUAL",
    "takes-place-in": "CONTEXTUAL",
    "is-bound-by": "CONTEXTUAL",
    "is-hosted-in": "CONTEXTUAL",
    "belongs-to": "CONTEXTUAL",
    "resolves-to": "CONTEXTUAL",
    "presented-as": "CONTEXTUAL",
    # Attributive (💭)
    "has-property": "ATTRIBUTIVE",
    "characterized-by": "ATTRIBUTIVE",
    "exhibits": "ATTRIBUTIVE",
    "possesses": "ATTRIBUTIVE",
    "displays": "ATTRIBUTIVE",
    "manifests": "ATTRIBUTIVE",
    "features": "ATTRIBUTIVE",
    "embodies": "ATTRIBUTIVE",
    "expresses": "ATTRIBUTIVE",
    "has-characteristic": "ATTRIBUTIVE",
    "bears": "ATTRIBUTIVE",
    "remains-lazy-for": "ATTRIBUTIVE",
    "characterizes": "ATTRIBUTIVE",
    "subject_to": "ATTRIBUTIVE",
    "quantified_by": "ATTRIBUTIVE",
    # Comparative (⚖️)
    "compares-to": "COMPARATIVE",
    "contrasts-with": "COMPARATIVE",
    "similar-to": "COMPARATIVE",
    "akin-to": "COMPARATIVE",
    "different-from": "COMPARATIVE",
    "equivalent-to": "COMPARATIVE",
    "comparable-to": "COMPARATIVE",
    "interchangeable-with": "COMPARATIVE",
    "opposite-of": "COMPARATIVE",
    "synonym-of": "COMPARATIVE",
    "antonym-of": "COMPARATIVE",
    "same-as": "COMPARATIVE",
    "related-to": "COMPARATIVE",
    "relates-to": "COMPARATIVE",  # Variant of related-to
    # Sequential (⏩)
    "precedes": "SEQUENTIAL",
    "follows": "SEQUENTIAL",
    "occurs-during": "SEQUENTIAL",
    "transitions-to": "SEQUENTIAL",
    "evolves-into": "SEQUENTIAL",
    "progresses-to": "SEQUENTIAL",
    "succeeds": "SEQUENTIAL",
    "sequences": "SEQUENTIAL",
    # Causal (⚡)
    "causes": "CAUSAL",
    "produces": "CAUSAL",
    "triggers": "CAUSAL",
    "prevents": "CAUSAL",
    "enables": "CAUSAL",
    "inhibits": "CAUSAL",
    "influences": "CAUSAL",
    "affects": "CAUSAL",
    "determines": "CAUSAL",
    "results-in": "CAUSAL",
    "creates": "CAUSAL",
    "destroys": "CAUSAL",
    "modifies": "CAUSAL",
    "amplifies": "CAUSAL",
    "reduces": "CAUSAL",
    "depends-on": "CAUSAL",
    "accelerates": "CAUSAL",
    "activates": "CAUSAL",
    "alleviates": "CAUSAL",
    "blocks": "CAUSAL",
    "boosts": "CAUSAL",
    "catalyzes": "CAUSAL",
    "constrains": "CAUSAL",
    "converts": "CAUSAL",
    "delays": "CAUSAL",
    "degrades": "CAUSAL",
    "drives": "CAUSAL",
    "eases": "CAUSAL",
    "enhances": "CAUSAL",
    "facilitates": "CAUSAL",
    "fosters": "CAUSAL",
    "generates": "CAUSAL",
    "impedes": "CAUSAL",
    "induces": "CAUSAL",
    "limits": "CAUSAL",
    "maintains": "CAUSAL",
    "motivates": "CAUSAL",
    "obstructs": "CAUSAL",
    "permits": "CAUSAL",
    "prolongs": "CAUSAL",
    "promotes": "CAUSAL",
    "anticipates": "CAUSAL",
    "correlates-with": "CAUSAL",
    "initiates": "CAUSAL",
    "terminates": "CAUSAL",
    "leads-to": "CAUSAL",
    "prepares-for": "CAUSAL",
    "builds": "CAUSAL",
    "detects": "CAUSAL",
    "corrects": "CAUSAL",
    "draws-from": "CAUSAL",
    "constrained-by": "CAUSAL",
    "demonstrated-by": "CAUSAL",
    "via": "CAUSAL",
    "reduces-load-on": "CAUSAL",
    "engages-in": "CAUSAL",
    "applied-to": "CAUSAL",
    "initiates-evaluation-with": "CAUSAL",
    "checks-for-bias-in": "CAUSAL",
    "requests-evidence-from": "CAUSAL",
    "provides-data-to": "CAUSAL",
    "proposes-judgment-to": "CAUSAL",
    "delivers-input-to": "CAUSAL",
    "confirms-or-flags": "CAUSAL",
    "re-evaluation-triggered": "CAUSAL",
    "structures": "CAUSAL",
    "validated_by": "CAUSAL",
    "influenced_by": "CAUSAL",
    # Analogical (🌉)
    "analogous-to": "ANALOGICAL",
    "corresponds-to": "ANALOGICAL",
    "maps-to": "ANALOGICAL",
    "parallels": "ANALOGICAL",
    "resembles": "ANALOGICAL",
    "mirrors": "ANALOGICAL",
    "symbolizes": "ANALOGICAL",
    "represents": "ANALOGICAL",
    "stands-for": "ANALOGICAL",
    "exemplifies": "ANALOGICAL",
    "illustrates": "ANALOGICAL",
    "metaphor-for": "ANALOGICAL",
    "isomorphic-to": "ANALOGICAL",
    "metaphorically-represents": "ANALOGICAL",
    # Domain: Software/Application
    "deployed-in": "CONTEXTUAL",
    "runs-on": "CONTEXTUAL",
    "hosted-by": "CONTEXTUAL",
    "connects-to": "CONTEXTUAL",
    "interfaces-with": "CONTEXTUAL",
    "consumes": "CONTEXTUAL",
    "provides": "CONTEXTUAL",
    "configured-with": "ATTRIBUTIVE",
    "secured-by": "ATTRIBUTIVE",
    "versioned-as": "ATTRIBUTIVE",
    "requires": "ATTRIBUTIVE",
    "exposes": "ATTRIBUTIVE",
    "supports": "ATTRIBUTIVE",
    "implements": "ATTRIBUTIVE",
    "integrates-with": "COMPOSITIONAL",
    "layer-in": "COMPOSITIONAL",
    "bundles": "COMPOSITIONAL",
    "encapsulates": "COMPOSITIONAL",
    "decomposes-into": "COMPOSITIONAL",
    "alternative-to": "COMPARATIVE",
    "predecessor-of": "COMPARATIVE",
    "successor-of": "COMPARATIVE",
    "replaces": "COMPARATIVE",
    "extends": "HIERARCHICAL",
    "supersedes": "COMPARATIVE",
    "superseded_by": "COMPARATIVE",
    "compatible-with": "COMPARATIVE",
    "processes": "SEQUENTIAL",
    "handles": "SEQUENTIAL",
    "validates": "SEQUENTIAL",
    "transforms": "SEQUENTIAL",
    "schedules": "SEQUENTIAL",
    "impacts": "CAUSAL",
    "resolves": "CAUSAL",
    "mitigates": "CAUSAL",
    "pattern-is": "ANALOGICAL",
    "models": "ANALOGICAL",
    "abstracts": "ANALOGICAL",
}

_logged_verb_warnings: set[tuple[str, str, str]] = set()


def _fuzzy_match_verb(verb: str, cutoff: float = 0.8) -> str | None:
    """Fuzzy match a verb against VERB_REGISTRY keys as a last resort.

    Uses difflib.get_close_matches to find approximate matches.
    Returns the resolved category if a close match is found, None otherwise.
    """
    import difflib

    matches = difflib.get_close_matches(verb, VERB_REGISTRY.keys(), n=1, cutoff=cutoff)
    if matches:
        return VERB_REGISTRY[matches[0]]
    return None


_LEARNED_VERBS: dict[str, str] | None = None


def _learned_verbs_path() -> Path | None:
    from codebase_rag.config import settings

    repo = getattr(settings, "TARGET_REPO_PATH", None)
    if repo:
        return Path(repo) / ".cgr" / "learned_verbs.json"
    return None


def _load_learned_verbs() -> dict[str, str]:
    global _LEARNED_VERBS
    if _LEARNED_VERBS is not None:
        return _LEARNED_VERBS
    path = _learned_verbs_path()
    if path and path.exists():
        with open(path) as f:
            data = json.load(f)
        _LEARNED_VERBS = {
            verb: entry["category"]
            for verb, entry in data.items()
            if entry.get("count", 0) >= 3
        }
    else:
        _LEARNED_VERBS = {}
    return _LEARNED_VERBS


def _learn_verb(verb: str, category: str) -> None:
    path = _learned_verbs_path()
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                data = json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    entry = data.get(
        verb,
        {"category": category, "count": 0, "first_seen": datetime.now(UTC).isoformat()},
    )
    entry["count"] += 1
    data[verb] = entry

    with open(path, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _verb_in_registry(verb: str) -> bool:
    """Check if a verb (or its hyphen-normalized form) exists in VERB_REGISTRY."""
    v = verb.lower().strip()
    return v in VERB_REGISTRY or v.replace("-", "_") in VERB_REGISTRY


def _should_log_verb_warning(
    verb: str,
    declared_category: str,
    registry_category: str,
) -> bool:
    """Return True if this exact conflict hasn't been logged yet this session.

    De-duplicates warnings for recurring verb+declared+registry conflicts
    to reduce log noise. The key includes the verb, the LLM-declared
    category, and the registry category so that different conflicts for
    the same verb are still logged.
    """
    key = (verb.lower().strip(), declared_category.upper().strip(), registry_category.upper().strip())
    if key in _logged_verb_warnings:
        return False
    _logged_verb_warnings.add(key)
    return True


def _should_proactive_split(chunk_content: str) -> bool:
    """Check if chunk has known structural patterns that cause timeouts.

    These patterns are orthogonal to token count and catch cases where
    dense structural content (long lines, tables) makes extraction slow
    even when the token total is under the input limit.

    Very long individual lines (tables, dense data) and extremely dense
    content with heavy punctuation are the main signals.
    """
    max_line_len = max((len(line) for line in chunk_content.split("\n")), default=0)
    if max_line_len > 2000:
        return True
    if len(chunk_content) > 15000 and chunk_content.count(",") > 50:
        return True
    return False


def resolve_category(verb: str, declared_category: str | None) -> tuple[str, str]:
    """Resolve a verb to its canonical category and emoji.

    5-step resolution:
    1. Learned registry (persistent, repo-scoped)
    2. Hardcoded VERB_REGISTRY
    3. Valid declared_category
    4. Fuzzy match
    5. Fallback to RELATED_TO

    Emoji is ALWAYS derived server-side from the resolved category via
    CATEGORY_EMOJI_MAP. LLM-provided emoji is ignored.

    Args:
        verb: The relationship verb (e.g. "mitigates", "is-a").
        declared_category: The category the LLM declared, or None.

    Returns:
        (category, emoji) tuple.
    """
    from loguru import logger

    from codebase_rag.constants import CATEGORY_EMOJI_MAP, DOC_CONCEPT_CATEGORIES
    from codebase_rag.document import logs as doc_ls

    verb_lower = verb.lower().strip()
    verb_normalized = verb_lower.replace("-", "_")
    learned = _load_learned_verbs()
    learned_category = learned.get(verb_lower) or learned.get(verb_normalized)
    registry_category = VERB_REGISTRY.get(verb_lower) or VERB_REGISTRY.get(verb_normalized)

    if learned_category is not None:
        category = learned_category
    elif registry_category is not None:
        if declared_category and registry_category == declared_category.upper():
            category = declared_category.upper()
        elif declared_category and declared_category.upper() in DOC_CONCEPT_CATEGORIES:
            category = registry_category
            if _should_log_verb_warning(verb, declared_category, registry_category):
                if verb != verb_normalized:
                    logger.warning(
                        f"Verb '{verb}' (normalized: '{verb_normalized}') registry override: "
                        f"LLM declared '{declared_category}', registry says '{registry_category}'"
                    )
                else:
                    logger.warning(
                        f"Verb '{verb}' registry override: LLM declared "
                        f"'{declared_category}', registry says '{registry_category}'"
                    )
            else:
                logger.debug(
                    doc_ls.DOC_CONCEPT_VERB_OVERRIDE_SUPPRESSED.format(
                        verb=verb,
                        declared_category=declared_category,
                        registry_category=registry_category,
                    )
                )
        else:
            category = registry_category
    elif declared_category and declared_category.upper() in DOC_CONCEPT_CATEGORIES:
        category = declared_category.upper()
        _learn_verb(verb_normalized, category)
        logger.info(
            f"Verb '{verb}' not in registry — using declared category '{category}'. "
            f"Learned for future authoritative resolution."
        )
    else:
        category = _fuzzy_match_verb(verb_normalized)
        if category:
            logger.debug(
                f"Verb '{verb}' not in registry — fuzzy-matched to "
                f"'{category}' category via registry verbs."
            )
        else:
            category = "RELATED_TO"
            if declared_category:
                logger.warning(
                    f"Verb '{verb}' fell back to RELATED_TO. "
                    f"LLM declared: '{declared_category}'. "
                    f"Consider adding to VERB_REGISTRY."
                )

    emoji = CATEGORY_EMOJI_MAP.get(category, "🔗")
    return category, emoji


ENTITY_SUBTYPE_REGISTRY: dict[str, str] = {
    # 🧱 Concrete Entity — Core (8)
    "Natural Object": "CONCRETE_ENTITY",
    "Artifact": "CONCRETE_ENTITY",
    "Substance": "CONCRETE_ENTITY",
    "Organism": "CONCRETE_ENTITY",
    "Body Part": "CONCRETE_ENTITY",
    "Food/Consumable": "CONCRETE_ENTITY",
    "Geographic Feature": "CONCRETE_ENTITY",
    "Celestial Body": "CONCRETE_ENTITY",
    # 🧱 Concrete Entity — Software (5)
    "Virtual Machine": "CONCRETE_ENTITY",
    "Container Image": "CONCRETE_ENTITY",
    "Data Center": "CONCRETE_ENTITY",
    "Mobile Device": "CONCRETE_ENTITY",
    "Peripheral": "CONCRETE_ENTITY",
    # 🧱 Concrete Entity — Business (5)
    "Facility": "CONCRETE_ENTITY",
    "Inventory": "CONCRETE_ENTITY",
    "Equipment": "CONCRETE_ENTITY",
    "Product": "CONCRETE_ENTITY",
    "Prototype": "CONCRETE_ENTITY",
    # 🧱 Concrete Entity — Scientific (6)
    "Particle": "CONCRETE_ENTITY",
    "Molecule": "CONCRETE_ENTITY",
    "Mineral": "CONCRETE_ENTITY",
    "Fossil": "CONCRETE_ENTITY",
    "Specimen": "CONCRETE_ENTITY",
    "Isotope": "CONCRETE_ENTITY",

    # ⏱️ Event/Process — Core (8)
    "Natural Event": "EVENT_PROCESS",
    "Human Action": "EVENT_PROCESS",
    "Process": "EVENT_PROCESS",
    "Incident": "EVENT_PROCESS",
    "Activity": "EVENT_PROCESS",
    "State Change": "EVENT_PROCESS",
    "Project/Initiative": "EVENT_PROCESS",
    "Ritual/Routine": "EVENT_PROCESS",
    # ⏱️ Event/Process — Software (4)
    "Deployment": "EVENT_PROCESS",
    "Build": "EVENT_PROCESS",
    "Test Run": "EVENT_PROCESS",
    "Migration": "EVENT_PROCESS",
    # ⏱️ Event/Process — Scientific (6)
    "Chemical Reaction": "EVENT_PROCESS",
    "Biological Process": "EVENT_PROCESS",
    "Mutation": "EVENT_PROCESS",
    "Observation": "EVENT_PROCESS",
    "Geological Event": "EVENT_PROCESS",
    "Astronomical Event": "EVENT_PROCESS",

    # 📨 Information/Expression — Core (7)
    "Data": "INFORMATION_EXPRESSION",
    "Signal": "INFORMATION_EXPRESSION",
    "Symbol": "INFORMATION_EXPRESSION",
    "Narrative": "INFORMATION_EXPRESSION",
    "Code/Formula": "INFORMATION_EXPRESSION",
    "Record/Document": "INFORMATION_EXPRESSION",
    "Media": "INFORMATION_EXPRESSION",
    # 📨 Information/Expression — Software (5)
    "API": "INFORMATION_EXPRESSION",
    "Protocol": "INFORMATION_EXPRESSION",
    "Config File": "INFORMATION_EXPRESSION",
    "Log Stream": "INFORMATION_EXPRESSION",
    "Source File": "INFORMATION_EXPRESSION",

    # 📏 Property/Attribute — Core (7)
    "Physical Quality": "PROPERTY_ATTRIBUTE",
    "Quantitative Measure": "PROPERTY_ATTRIBUTE",
    "Mental State": "PROPERTY_ATTRIBUTE",
    "Capability/Skill": "PROPERTY_ATTRIBUTE",
    "Disposition": "PROPERTY_ATTRIBUTE",
    "Relational Property": "PROPERTY_ATTRIBUTE",
    "Evaluative Property": "PROPERTY_ATTRIBUTE",
    # 📏 Property/Attribute — Software (3)
    "SLI": "PROPERTY_ATTRIBUTE",
    "Quality Attribute": "PROPERTY_ATTRIBUTE",
    "Capacity Metric": "PROPERTY_ATTRIBUTE",
    # 📏 Property/Attribute — Scientific (4)
    "Chemical Property": "PROPERTY_ATTRIBUTE",
    "Biological Trait": "PROPERTY_ATTRIBUTE",
    "Quantum State": "PROPERTY_ATTRIBUTE",
    "Ecological Indicator": "PROPERTY_ATTRIBUTE",

    # 🏗️ System/Structure — Core (7)
    "Natural System": "SYSTEM_STRUCTURE",
    "Social System": "SYSTEM_STRUCTURE",
    "Technological System": "SYSTEM_STRUCTURE",
    "Network": "SYSTEM_STRUCTURE",
    "Hierarchy": "SYSTEM_STRUCTURE",
    "Framework": "SYSTEM_STRUCTURE",
    "Market/Platform": "SYSTEM_STRUCTURE",
    # 🏗️ System/Structure — Software (5)
    "Distributed System": "SYSTEM_STRUCTURE",
    "CI/CD Pipeline": "SYSTEM_STRUCTURE",
    "Monorepo": "SYSTEM_STRUCTURE",
    "Service Mesh": "SYSTEM_STRUCTURE",
    "Feature Flag System": "SYSTEM_STRUCTURE",
    # 🏗️ System/Structure — Business (6)
    "Org Chart": "SYSTEM_STRUCTURE",
    "Holding Company": "SYSTEM_STRUCTURE",
    "Joint Venture": "SYSTEM_STRUCTURE",
    "Franchise": "SYSTEM_STRUCTURE",
    "Cooperative": "SYSTEM_STRUCTURE",
    "Supply Chain": "SYSTEM_STRUCTURE",
    # 🏗️ System/Structure — Scientific (4)
    "Biome": "SYSTEM_STRUCTURE",
    "Watershed": "SYSTEM_STRUCTURE",
    "Geological Formation": "SYSTEM_STRUCTURE",
    "Star System": "SYSTEM_STRUCTURE",

    # 🎭 Agent/Role — Core (6)
    "Individual": "AGENT_ROLE",
    "Collective": "AGENT_ROLE",
    "Institutional Agent": "AGENT_ROLE",
    "Non-Human Agent": "AGENT_ROLE",
    "Role/Position": "AGENT_ROLE",
    "Persona": "AGENT_ROLE",
    # 🎭 Agent/Role — Software (4)
    "CI Bot": "AGENT_ROLE",
    "Service Account": "AGENT_ROLE",
    "End User": "AGENT_ROLE",
    "On-Call Engineer": "AGENT_ROLE",
    # 🎭 Agent/Role — Business (6)
    "Stakeholder": "AGENT_ROLE",
    "Vendor/Supplier": "AGENT_ROLE",
    "Regulator": "AGENT_ROLE",
    "Board": "AGENT_ROLE",
    "Founder": "AGENT_ROLE",
    "Customer/Client": "AGENT_ROLE",

    # 💡 Abstract Concept — Core (6)
    "Domain/Discipline": "ABSTRACT_CONCEPT",
    "Theory/Model": "ABSTRACT_CONCEPT",
    "Principle/Rule": "ABSTRACT_CONCEPT",
    "Value/Ideal": "ABSTRACT_CONCEPT",
    "Category/Class": "ABSTRACT_CONCEPT",
    "Relation/Connection": "ABSTRACT_CONCEPT",
    # 💡 Abstract Concept — Software (5)
    "Design Pattern": "ABSTRACT_CONCEPT",
    "Algorithm": "ABSTRACT_CONCEPT",
    "Protocol Spec": "ABSTRACT_CONCEPT",
    "Paradigm": "ABSTRACT_CONCEPT",
    "SLA": "ABSTRACT_CONCEPT",
    # 💡 Abstract Concept — Business (5)
    "Business Model": "ABSTRACT_CONCEPT",
    "Strategy": "ABSTRACT_CONCEPT",
    "KPI": "ABSTRACT_CONCEPT",
    "Brand": "ABSTRACT_CONCEPT",
    "Moat": "ABSTRACT_CONCEPT",

    # 📏 Property/Attribute — Pedagogical & Cognitive (16)
    "Cognitive Capacity": "PROPERTY_ATTRIBUTE",
    "Cognitive Condition": "PROPERTY_ATTRIBUTE",
    "Cognitive Constraint": "PROPERTY_ATTRIBUTE",
    "Cognitive Limitation": "PROPERTY_ATTRIBUTE",
    "Cognitive Phenomenon": "EVENT_PROCESS",
    "Cognitive Skill": "PROPERTY_ATTRIBUTE",
    "Cognitive State": "PROPERTY_ATTRIBUTE",
    "Cognitive Trait": "PROPERTY_ATTRIBUTE",
    "Character Traits": "PROPERTY_ATTRIBUTE",
    "Educational Metric": "PROPERTY_ATTRIBUTE",
    "Evaluative Criteria": "PROPERTY_ATTRIBUTE",
    "Evaluative Measure": "PROPERTY_ATTRIBUTE",
    "Knowledge Classification": "PROPERTY_ATTRIBUTE",
    "Performance Metric": "PROPERTY_ATTRIBUTE",
    "Quality Benchmark": "PROPERTY_ATTRIBUTE",
    "Quantitative Threshold": "PROPERTY_ATTRIBUTE",

    # 🏗️ System/Structure — Pedagogical & Cognitive (15)
    "AI System": "SYSTEM_STRUCTURE",
    "Cognitive Architecture": "SYSTEM_STRUCTURE",
    "Cognitive Framework": "SYSTEM_STRUCTURE",
    "Cognitive Model": "SYSTEM_STRUCTURE",
    "Cognitive Subsystem": "SYSTEM_STRUCTURE",
    "Cognitive System": "SYSTEM_STRUCTURE",
    "Conceptual Framework": "SYSTEM_STRUCTURE",
    "Decision Framework": "SYSTEM_STRUCTURE",
    "Educational Framework": "SYSTEM_STRUCTURE",
    "Educational Organization": "SYSTEM_STRUCTURE",
    "Methodology Framework": "SYSTEM_STRUCTURE",
    "Pedagogical Framework": "SYSTEM_STRUCTURE",
    "Research Framework": "SYSTEM_STRUCTURE",
    "Research Institution": "SYSTEM_STRUCTURE",
    "Technological Influence": "SYSTEM_STRUCTURE",

    # 🏗️ System/Structure — Cognitive (1)
    "Cognitive Mechanism": "SYSTEM_STRUCTURE",

    # 📨 Information/Expression — Framework Components & Tools (2)
    "Framework Component": "SYSTEM_STRUCTURE",
    "Technology Tool": "SYSTEM_STRUCTURE",

    # ⏱️ Event/Process — Pedagogical & Cognitive (13)
    "AI Practice": "EVENT_PROCESS",
    "Analytical Process": "EVENT_PROCESS",
    "Behavioral Phenomenon": "EVENT_PROCESS",
    "Cognitive Activity": "EVENT_PROCESS",
    "Cognitive Process": "EVENT_PROCESS",
    "Cognitive Strategy": "EVENT_PROCESS",
    "Decision Process": "EVENT_PROCESS",
    "Educational Outcome": "EVENT_PROCESS",
    "Human-Computer Interaction": "EVENT_PROCESS",
    "Pedagogical Activity": "EVENT_PROCESS",
    "Teaching Activity": "EVENT_PROCESS",
    "Teaching Method": "EVENT_PROCESS",

    # 📨 Information Expression — Pedagogical & Cognitive (10)
    "Assessment Instrument": "INFORMATION_EXPRESSION",
    "Assessment Tool": "INFORMATION_EXPRESSION",
    "Decision Tool": "INFORMATION_EXPRESSION",
    "Evaluation Instrument": "INFORMATION_EXPRESSION",
    "Instructional Component": "INFORMATION_EXPRESSION",
    "Publication Type": "INFORMATION_EXPRESSION",
    "Research Evidence": "INFORMATION_EXPRESSION",
    "Research Finding": "INFORMATION_EXPRESSION",
    "Research Study": "INFORMATION_EXPRESSION",
    "Tool/Framework": "INFORMATION_EXPRESSION",

    # 🎭 Agent/Role — Pedagogical & Cognitive (3)
    "Organization": "AGENT_ROLE",
    "Research Organization": "AGENT_ROLE",
    "Researcher": "AGENT_ROLE",

    # 💡 Abstract Concept — Pedagogical & Cognitive (5)
    "Capability": "ABSTRACT_CONCEPT",
    "Foundational Concept": "ABSTRACT_CONCEPT",
    "Psychological Theory": "ABSTRACT_CONCEPT",
    "Mental Model": "ABSTRACT_CONCEPT",
    "Reasoning Method": "ABSTRACT_CONCEPT",

    # 🧱 Concrete Entity — Pedagogical & Cognitive (1)
    "Software Tool": "CONCRETE_ENTITY",

    # ═══════════════════════════════════════════════════════════════
    # JSON Entity Types — CTO Competency Framework & General
    # ═══════════════════════════════════════════════════════════════

    # 🎭 Agent/Role
    "Role": "AGENT_ROLE",

    # 💡 Abstract Concept
    "Mindset": "ABSTRACT_CONCEPT",
    "Competency": "ABSTRACT_CONCEPT",
    "AntiPattern": "ABSTRACT_CONCEPT",
    "MentalModel": "ABSTRACT_CONCEPT",

    # 🏗️ System/Structure
    "GovernanceConstruct": "SYSTEM_STRUCTURE",
    "Layer": "SYSTEM_STRUCTURE",

    # ⏱️ Event/Process
    "ProgressionStage": "EVENT_PROCESS",

    # 📏 Property/Attribute
    "SafetyBoundary": "PROPERTY_ATTRIBUTE",

    # 📨 Information Expression
    "GovernanceRule": "INFORMATION_EXPRESSION",
    "Reference": "INFORMATION_EXPRESSION",
    "FrameworkComponent": "SYSTEM_STRUCTURE",

    # 🧱 Concrete Entity
    "Tool": "CONCRETE_ENTITY",
}


def resolve_entity_category(
    declared_category: str | None,
    declared_subtype: str | None,
    definition: str = "",
) -> tuple[str, str | None, str]:
    """Resolve an entity to its canonical category, sub-type, and emoji.

    3-step resolution:
    1. Valid declared_category → use it
    2. Invalid/None category + subtype in ENTITY_SUBTYPE_REGISTRY → use registry
    3. Invalid/None category + no subtype match → ABSTRACT_CONCEPT (fallback with logging)

    Emoji is ALWAYS server-derived from ENTITY_CATEGORY_EMOJI_MAP.
    LLM-provided emoji is ignored.

    Args:
        declared_category: The entity category declared by the LLM, or None.
        declared_subtype: The entity sub-type declared by the LLM, or None.
        definition: The concept definition (for future disambiguation context).

    Returns:
        (category, subtype, emoji) tuple.
    """
    from loguru import logger

    from codebase_rag.constants import (
        DOC_ENTITY_CATEGORIES,
        ENTITY_CATEGORY_EMOJI_MAP,
    )

    category: str | None = None
    subtype: str | None = declared_subtype

    # Step 0: Emoji-only category (LLM occasionally puts emoji in entity_category field)
    _EMOJI_TO_ENTITY_CATEGORY: dict[str, str] = {
        v: k for k, v in ENTITY_CATEGORY_EMOJI_MAP.items()
    }
    if declared_category and declared_category in _EMOJI_TO_ENTITY_CATEGORY:
        category = _EMOJI_TO_ENTITY_CATEGORY[declared_category]

    # Step 1: Valid declared category (strip emoji/prefix characters first)
    if category is None and declared_category:
        normalized_category = declared_category
        # Strip leading non-alphanumeric characters (e.g., emoji prefixes like "🏗️ ")
        while normalized_category and not normalized_category[0].isalnum():
            normalized_category = normalized_category[1:]
        normalized_category = normalized_category.strip()
        if normalized_category.upper() in DOC_ENTITY_CATEGORIES:
            category = normalized_category.upper()

    # Step 2: Sub-type registry lookup
    if category is None and declared_subtype:
        if declared_subtype in ENTITY_SUBTYPE_REGISTRY:
            category = ENTITY_SUBTYPE_REGISTRY[declared_subtype]
            logger.debug(
                f"Entity category resolved from sub-type registry: "
                f"'{declared_subtype}' → '{category}'"
            )
        else:
            logger.info(
                f"Unknown entity_subtype '{declared_subtype}' — "
                f"consider adding to ENTITY_SUBTYPE_REGISTRY"
            )

    # Step 3: Fallback to ABSTRACT_CONCEPT
    if category is None:
        category = "ABSTRACT_CONCEPT"
        if declared_category:
            logger.debug(
                f"Entity category not determined — falling back to ABSTRACT_CONCEPT. "
                f"Declared category: '{declared_category}', subtype: '{declared_subtype}'"
            )
        else:
            logger.warning(
                f"Entity category not determined — falling back to ABSTRACT_CONCEPT. "
                f"Declared category: '{declared_category}', subtype: '{declared_subtype}'"
            )

    emoji = ENTITY_CATEGORY_EMOJI_MAP.get(category, "💡")
    return category, subtype, emoji


REBALANCE_PROMPT = """You are re-categorizing relationships extracted from a document chunk.
The relationships are too heavily weighted toward one category.

Re-assign each relationship to a more diverse category from:
HIERARCHICAL, COMPOSITIONAL, CONTEXTUAL, ATTRIBUTIVE, COMPARATIVE, SEQUENTIAL, CAUSAL, ANALOGICAL.

Rules:
- Keep from_concept, to_concept, and verb exactly the same.
- Only change category and emoji.
- Do not add or remove relationships.
- Do not invent new concepts.
"""


async def _rebalance_relationships(
    relationships: list[ConceptRelationship],
    chunk_content: str,
    chunk_qn: str,
) -> list[ConceptRelationship]:
    """If category skew detected, ask LLM to re-categorize NON-REGISTRY relationships."""
    from loguru import logger

    from codebase_rag.compat.pydantic_ai import Agent
    from codebase_rag.config import settings
    from codebase_rag.services.llm import _create_chat_model

    rebalancable = [
        r for r in relationships
        if not _verb_in_registry(r.verb)
    ]
    registry_fixed = [
        r for r in relationships
        if _verb_in_registry(r.verb)
    ]

    if not rebalancable:
        logger.debug(f"No non-registry relationships to rebalance for {chunk_qn}")
        return relationships

    try:
        llm = _create_chat_model(settings.active_orchestrator_config)
        rebalance_agent = Agent(
            model=llm,
            system_prompt=REBALANCE_PROMPT,
            output_type=list[ConceptRelationship],
            retries=1,
        )
        result = await rebalance_agent.run(
            f"Chunk context:\n{chunk_content[:2000]}\n\n"
            f"Relationships to rebalance (non-registry verbs only):\n"
            f"{json.dumps([r.model_dump() for r in rebalancable])}"
        )
        rebalanced = result.output
        validated = []
        for orig, reb in zip(rebalancable, rebalanced):
            if (
                reb.from_concept == orig.from_concept
                and reb.to_concept == orig.to_concept
                and reb.verb == orig.verb
            ):
                from codebase_rag.constants import CATEGORY_EMOJI_MAP

                reb.emoji = CATEGORY_EMOJI_MAP.get(reb.category, "🔗")
                validated.append(reb)
            else:
                validated.append(orig)
        logger.info(
            f"Rebalanced {len(rebalancable)} relationships for {chunk_qn}"
        )
        return registry_fixed + validated
    except Exception as e:
        logger.warning(
            f"Rebalancing failed for {chunk_qn}: {e}. Keeping original relationships."
        )
        return relationships


def _apply_category_caps(
    relationships: list[ConceptRelationship],
    max_ratio: float = 0.50,
) -> list[ConceptRelationship]:
    """Apply hard cap to ensure no single category exceeds max_ratio."""
    if not relationships:
        return []
    total = len(relationships)
    counts = Counter(r.category for r in relationships)
    capped = []
    for cat in counts:
        cat_rels = [r for r in relationships if r.category == cat]
        max_allowed = int(total * max_ratio)
        if len(cat_rels) > max_allowed:
            cat_rels = sorted(cat_rels, key=lambda r: r.strength, reverse=True)[:max_allowed]
        capped.extend(cat_rels)
    return sorted(capped, key=lambda r: (r.from_concept, r.to_concept, r.verb))


def _check_relationship_diversity(
    relationships: list[ConceptRelationship],
    chunk_qn: str,
    causal_threshold: float = 0.60,
    min_count_for_check: int = 4,
) -> bool:
    """Warn if relationship categories are overly skewed toward CAUSAL.

    Only checks skew when there are at least min_count_for_check relationships,
    as small samples have statistically meaningless skew.

    D-8 guardrail: CAUSAL dominance indicates prompt bias or lazy LLM
    categorization. Logs at WARNING level when skew exceeds threshold.

    Returns:
        True if skew detected and rebalancing should run.
    """
    from loguru import logger

    if not relationships:
        return False
    total = len(relationships)
    if total < min_count_for_check:
        return False
    counts = Counter(r.category for r in relationships)
    max_count = max(counts.values())
    max_ratio = max_count / total
    if max_ratio > causal_threshold:
        dominant = max(counts.keys(), key=lambda c: counts[c])
        logger.warning(
            f"Relationship category skew detected for {chunk_qn}: "
            f"{max_ratio:.0%} {dominant} ({max_count}/{total}). "
            f"Triggering rebalancing."
        )
        return True
    return False


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

    elif isinstance(exc, asyncio.TimeoutError):
        error_type = ErrorType.CONCEPT_TIMEOUT

    elif "auth" in exc_message_lower or "api key" in exc_message_lower:
        error_type = ErrorType.CONCEPT_AUTH_ERROR
        is_recoverable = False

    elif (
        "output token" in exc_message_lower
        or "maximum output" in exc_message_lower
        or "response limit" in exc_message_lower
    ):
        error_type = ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT
        is_recoverable = True

    elif "context" in exc_message_lower or "token" in exc_message_lower:
        if "before any response was generated" in exc_message_lower:
            error_type = ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT
            is_recoverable = True
        else:
            error_type = ErrorType.CONCEPT_CONTEXT_OVERFLOW
            is_recoverable = False

    elif "connection" in exc_message_lower or "network" in exc_message_lower:
        error_type = ErrorType.CONCEPT_NETWORK_ERROR

    elif "json" in exc_message_lower or "parse" in exc_message_lower:
        error_type = ErrorType.CONCEPT_PARSING_ERROR

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
        chunk_content=chunk_content,
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
            "Document chunk exceeded LLM context window. "
            "Automatic chunk splitting was attempted. "
            "If this persists, consider reducing chunk size or increasing "
            "DOC_CONCEPT_EXTRACTION_MAX_TOKENS in configuration."
        ),
        ErrorType.CONCEPT_OUTPUT_TOKEN_LIMIT: (
            "LLM output token limit was reached before generating a response. "
            "Automatic chunk splitting was attempted. "
            "If this persists, consider reducing chunk size or checking "
            "the configured model's output token capacity."
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
