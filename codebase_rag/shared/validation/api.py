"""Validation trigger API with cost estimation.

On-demand validation requires a clear trigger mechanism with cost
estimation before execution to prevent unexpected LLM costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .cache import ValidationCache


class ValidationTriggerMode(StrEnum):
    """How validation is triggered."""

    MANUAL = "manual"  # User explicitly requests
    QUERY_FLAG = "query_flag"  # Query with validate=true flag
    SCHEDULED = "scheduled"  # Periodic validation (optional)


@dataclass
class ValidationRequest:
    """User request for validation."""

    document_path: str
    mode: Literal["CODE_VS_DOC", "DOC_VS_CODE"]
    scope: Literal["all", "sections", "claims"] = "all"
    max_cost_usd: float = 0.50  # User's cost budget
    dry_run: bool = False  # Estimate cost without running


@dataclass
class CostEstimate:
    """Cost estimation before validation runs."""

    estimated_llm_calls: int
    estimated_tokens: int
    estimated_cost_usd: float
    exceeds_budget: bool
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "estimated_llm_calls": self.estimated_llm_calls,
            "estimated_tokens": self.estimated_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "exceeds_budget": self.exceeds_budget,
            "breakdown": self.breakdown,
        }


@dataclass
class ValidationTriggerResult:
    """Result of validation trigger request."""

    accepted: bool
    cost_estimate: CostEstimate | None
    validation_id: str | None  # Unique ID for tracking
    message: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "accepted": self.accepted,
            "cost_estimate": self.cost_estimate.to_dict() if self.cost_estimate else None,
            "validation_id": self.validation_id,
            "message": self.message,
        }


class ValidationTriggerAPI:
    """
    API for triggering on-demand validation.

    Flow:
    1. User requests validation with budget
    2. System estimates cost (graph queries only, no LLM)
    3. If within budget, user confirms → validation executes
    4. Results cached keyed by: hash(query + graph_state)

    IMPORTANT: Never run validation without cost estimate first.
    """

    COST_PER_LLM_CALL_USD = 0.01  # Approximate cost per extraction call
    TOKENS_PER_CLAIM = 500  # Average tokens per claim extraction

    # Provider-specific token pricing (USD per 1K tokens)
    PRICING_TABLE: dict[str, dict[str, float]] = {
        "openai": {"input": 0.005, "output": 0.015},
        "google": {"input": 0.00025, "output": 0.0005},
        "ollama": {"input": 0.0, "output": 0.0},
        "local": {"input": 0.0, "output": 0.0},
    }

    def __init__(
        self,
        llm_provider: str = "google",
        cache: ValidationCache | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._cache = cache
        self._pricing = self._get_provider_pricing(llm_provider)

    def _get_provider_pricing(self, provider: str) -> dict[str, float]:
        """Get provider-specific token pricing."""
        return self.PRICING_TABLE.get(provider, self.PRICING_TABLE["google"])

    async def request_validation(
        self, request: ValidationRequest
    ) -> ValidationTriggerResult:
        """
        Request validation with cost estimation.

        IMPORTANT: Never run validation without cost estimate first.
        """
        # Step 1: Estimate cost without running LLM
        estimate = await self._estimate_cost(request)

        # Step 2: Check budget
        if estimate.exceeds_budget:
            return ValidationTriggerResult(
                accepted=False,
                cost_estimate=estimate,
                validation_id=None,
                message=f"Estimated cost ${estimate.estimated_cost_usd:.2f} exceeds budget ${request.max_cost_usd:.2f}",
            )

        # Step 3: If dry_run, return estimate only
        if request.dry_run:
            return ValidationTriggerResult(
                accepted=False,
                cost_estimate=estimate,
                validation_id=None,
                message=f"Cost estimate: ${estimate.estimated_cost_usd:.2f} ({estimate.estimated_llm_calls} LLM calls)",
            )

        # Step 4: Generate validation ID and queue execution
        validation_id = self._generate_id(request)

        return ValidationTriggerResult(
            accepted=True,
            cost_estimate=estimate,
            validation_id=validation_id,
            message=f"Validation queued. Estimated cost: ${estimate.estimated_cost_usd:.2f}",
        )

    async def _estimate_cost(self, request: ValidationRequest) -> CostEstimate:
        """
        Estimate validation cost using graph queries only.

        NO LLM calls during estimation - use graph statistics.
        """
        # Get document statistics from graph
        doc_stats = await self._get_document_stats(request.document_path)

        # Estimate based on scope
        if request.scope == "all":
            estimated_claims = doc_stats.get("estimated_claims", 10)
        elif request.scope == "sections":
            estimated_claims = doc_stats.get("total_section_count", 5) * 2
        else:
            estimated_claims = 3  # Single claim validation

        llm_calls = estimated_claims * 2  # Extract + verify per claim
        tokens = estimated_claims * self.TOKENS_PER_CLAIM * 2

        # Calculate cost based on provider pricing
        input_cost = (tokens * 0.5) * self._pricing["input"] / 1000
        output_cost = (tokens * 0.5) * self._pricing["output"] / 1000
        call_cost = llm_calls * self.COST_PER_LLM_CALL_USD
        cost_usd = input_cost + output_cost + call_cost

        return CostEstimate(
            estimated_llm_calls=llm_calls,
            estimated_tokens=tokens,
            estimated_cost_usd=cost_usd,
            exceeds_budget=cost_usd > request.max_cost_usd,
            breakdown={
                "claim_extraction": estimated_claims * self.COST_PER_LLM_CALL_USD,
                "verification": estimated_claims * self.COST_PER_LLM_CALL_USD,
                "token_costs": input_cost + output_cost,
            },
        )

    async def _get_document_stats(self, document_path: str) -> dict:
        """Get document statistics from graph (no LLM)."""
        from ...services.graph_service import MemgraphIngestor
        from ...config import settings

        try:
            with MemgraphIngestor(
                host=settings.DOC_MEMGRAPH_HOST,
                port=settings.DOC_MEMGRAPH_PORT,
            ) as doc_graph:
                # Query for document statistics
                stats_query = """
                MATCH (d:Document)
                WHERE d.path = $path OR d.path CONTAINS $path
                OPTIONAL MATCH (d)-[:CONTAINS_SECTION]->(s:Section)
                OPTIONAL MATCH (d)-[:CONTAINS_CHUNK]->(c:Chunk)
                RETURN d.path as path,
                       count(DISTINCT s) as total_section_count,
                       count(DISTINCT c) as total_chunk_count,
                       sum(c.word_count) as word_count
                """
                results = doc_graph.fetch_all(stats_query, {"path": document_path})

                if results and results[0]:
                    result = results[0]
                    section_count = result.get("total_section_count", 0) or 0
                    chunk_count = result.get("total_chunk_count", 0) or 0
                    word_count = result.get("word_count", 0) or 0

                    # Estimate claims based on content size
                    estimated_claims = max(10, chunk_count // 3)

                    return {
                        "total_section_count": section_count,
                        "total_chunk_count": chunk_count,
                        "word_count": word_count,
                        "estimated_claims": estimated_claims,
                        "document_path": result.get("path", document_path),
                    }

        except Exception as e:
            from loguru import logger
            logger.warning(f"Could not get document stats: {e}")

        # Fallback to default estimates
        return {
            "total_section_count": 5,
            "estimated_claims": 10,
            "word_count": 1000,
        }

    def _generate_id(self, request: ValidationRequest) -> str:
        """Generate unique validation ID."""
        import hashlib

        content = f"{request.document_path}:{request.mode}:{request.scope}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


__all__ = [
    "ValidationTriggerMode",
    "ValidationRequest",
    "CostEstimate",
    "ValidationTriggerResult",
    "ValidationTriggerAPI",
]