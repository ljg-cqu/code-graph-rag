from __future__ import annotations

from pydantic import BaseModel, Field

from loguru import logger

from codebase_rag.config import settings


class SufficiencyAssessment(BaseModel):
    """LLM assessment of whether current results answer the user's question."""

    sufficient: bool = Field(
        ...,
        description="Whether the results fully answer the question",
    )
    reasoning: str = Field(
        ...,
        description="Explanation of the assessment",
    )
    missing_information: list[str] = Field(
        default_factory=list,
        description="What information is still needed",
    )
    suggested_tools: list[str] = Field(
        default_factory=list,
        description="Tools that might fill gaps",
    )
    requires_cross_validation: bool = Field(
        default=False,
        description="Whether code-doc cross-validation is needed",
    )


class LLMSufficiencyAnalyzer:
    """Uses LLM to determine if query results are sufficient."""

    __slots__ = ("agent",)

    SYSTEM_PROMPT = """You are a sufficiency analyzer. Given a user's question and the current results from various tools, determine if the answer is complete.

Respond with JSON:
{
  "sufficient": boolean,
  "reasoning": "...",
  "missing_information": ["what else is needed"],
  "suggested_tools": ["tool_name", ...],
  "requires_cross_validation": boolean
}

Rules:
- If results directly answer the question: sufficient=true
- If results are partial or ambiguous: sufficient=false with missing_information
- If the question requires verifying code against docs: requires_cross_validation=true
- suggested_tools should list tools that might fill gaps (e.g., "read_file", "semantic_search", "get_call_hierarchy")"""

    def __init__(self) -> None:
        self.agent = None

    def _initialize_agent(self) -> None:
        """Lazy initialization."""
        if self.agent is None:
            from pydantic_ai import Agent
            from codebase_rag.providers import _create_provider_model

            config = settings.active_orchestrator_config
            llm = _create_provider_model(config)

            self.agent = Agent(
                model=llm,
                system_prompt=self.SYSTEM_PROMPT,
                output_type=SufficiencyAssessment,
                retries=getattr(settings, "AGENT_RETRIES", 1),
            )

    async def assess(
        self,
        question: str,
        results: list[dict[str, object]],
    ) -> SufficiencyAssessment:
        """Assess whether results are sufficient to answer the question."""
        self._initialize_agent()

        context = f"Question: {question}\n\nResults: {results}"
        try:
            result = await self.agent.run(context)
            return result.output
        except Exception as e:
            logger.warning(f"Sufficiency assessment failed: {e}")
            return SufficiencyAssessment(
                sufficient=False,
                reasoning=f"Assessment failed: {e}",
                missing_information=["Unknown - assessment failed"],
                suggested_tools=[],
            )
