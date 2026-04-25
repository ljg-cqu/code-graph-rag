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
    source_chunk_qn: str = Field(..., description="Source chunk qualified name")
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

💡 ABSTRACT_CONCEPT is the LAST RESORT — confirm the entity doesn't fit any of the
first 6 categories before using it. This mirrors RELATED_TO in the relationship taxonomy.

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
- entity_subtype is optional but recommended for software-domain concepts
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
                async with self._consecutive_timeouts_lock:
                    self._consecutive_timeouts = 0
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
    # Sequential (⏩)
    "precedes": "SEQUENTIAL",
    "follows": "SEQUENTIAL",
    "occurs-during": "SEQUENTIAL",
    "transitions-to": "SEQUENTIAL",
    "evolves-into": "SEQUENTIAL",
    "progresses-to": "SEQUENTIAL",
    "succeeds": "SEQUENTIAL",
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
    "provides": "CAUSAL",
    "configured-with": "ATTRIBUTIVE",
    "secured-by": "ATTRIBUTIVE",
    "versioned-as": "ATTRIBUTIVE",
    "requires": "ATTRIBUTIVE",
    "exposes": "ATTRIBUTIVE",
    "supports": "CAUSAL",
    "implements": "COMPOSITIONAL",
    "integrates-with": "COMPOSITIONAL",
    "bundles": "COMPOSITIONAL",
    "encapsulates": "COMPOSITIONAL",
    "decomposes-into": "COMPOSITIONAL",
    "layer-in": "COMPOSITIONAL",
    "alternative-to": "COMPARATIVE",
    "predecessor-of": "COMPARATIVE",
    "successor-of": "COMPARATIVE",
    "replaces": "COMPARATIVE",
    "extends": "HIERARCHICAL",
    "supersedes": "COMPARATIVE",
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


def resolve_category(verb: str, declared_category: str | None) -> tuple[str, str]:
    """Resolve a verb to its canonical category and emoji.

    4-step resolution:
    1. Verb in registry AND matches declared → use declared
    2. Verb in registry AND mismatches declared → use registry (authoritative)
    3. Verb NOT in registry → use declared_category, log for registry expansion
    4. Verb NOT in registry AND no declared → fuzzy match, fallback to RELATED_TO

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

    verb_lower = verb.lower().strip()
    registry_category = VERB_REGISTRY.get(verb_lower)

    if registry_category is not None:
        if declared_category and registry_category == declared_category.upper():
            category = declared_category.upper()
        else:
            if declared_category and registry_category != declared_category.upper():
                logger.debug(
                    f"Verb '{verb}' registry override: LLM declared "
                    f"'{declared_category}', registry says '{registry_category}'"
                )
            category = registry_category
    elif declared_category and declared_category.upper() in DOC_CONCEPT_CATEGORIES:
        category = declared_category.upper()
        logger.info(
            f"Verb '{verb}' not in registry — using declared category '{category}'. "
            f"Consider adding to VERB_REGISTRY for future authoritative resolution."
        )
    else:
        category = _fuzzy_match_verb(verb_lower)
        if category:
            logger.info(
                f"Verb '{verb}' not in registry — fuzzy-matched to "
                f"'{category}' category via registry verbs."
            )
        else:
            category = "RELATED_TO"

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
}


def resolve_entity_category(
    declared_category: str | None,
    declared_subtype: str | None,
    definition: str = "",
) -> tuple[str, str | None, str]:
    """Resolve an entity to its canonical category, sub-type, and emoji.

    4-step resolution:
    1. Valid declared_category → use it
    2. Invalid/None category + subtype in ENTITY_SUBTYPE_REGISTRY → use registry
    3. Invalid/None category + no subtype match → ABSTRACT_CONCEPT (last resort)
    4. ABSTRACT_CONCEPT as result → log debug

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

    # Step 1: Valid declared category
    if declared_category and declared_category.upper() in DOC_ENTITY_CATEGORIES:
        category = declared_category.upper()
    # Step 2: Sub-type registry lookup
    elif declared_subtype and declared_subtype in ENTITY_SUBTYPE_REGISTRY:
        category = ENTITY_SUBTYPE_REGISTRY[declared_subtype]
        logger.debug(
            f"Entity category resolved from sub-type registry: "
            f"'{declared_subtype}' → '{category}'"
        )
    # Step 3: Fallback to ABSTRACT_CONCEPT
    else:
        category = "ABSTRACT_CONCEPT"
        logger.warning(
            f"Entity category not determined — falling back to ABSTRACT_CONCEPT. "
            f"Declared category: '{declared_category}', subtype: '{declared_subtype}'"
        )

    # Step 4: Debug-log if ABSTRACT_CONCEPT is the result (routine classification)
    if category == "ABSTRACT_CONCEPT":
        logger.debug(
            f"Entity classified as ABSTRACT_CONCEPT (last resort). "
            f"Verify other 6 categories were ruled out."
        )

    emoji = ENTITY_CATEGORY_EMOJI_MAP.get(category, "💡")
    return category, subtype, emoji


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
