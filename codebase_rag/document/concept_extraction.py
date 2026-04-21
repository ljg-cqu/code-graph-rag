from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class ExtractedConcept(BaseModel):
    """A concept extracted from document content."""

    name: str = Field(..., description="Concept name")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names for this concept",
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


class LLMConceptExtractor:
    """LLM-based concept extraction implementation."""

    __slots__ = ("agent",)

    SYSTEM_PROMPT = """You are a concept extractor for technical documentation.
Given a document chunk, identify:
1. Key concepts mentioned (with definitions if available)
2. Relationships between concepts (RELATED_TO, IS_A, PART_OF, CAUSES)

Respond with JSON matching this structure:
{
  "concepts": [
    {
      "name": "concept name",
      "aliases": ["alternative name"],
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
- Confidence should reflect how clearly the concept is presented
- Relationship types must be one of: RELATED_TO, IS_A, PART_OF, CAUSES
- Strength reflects how explicitly the relationship is stated"""

    def __init__(self) -> None:
        self.agent = None

    def _initialize_agent(self) -> None:
        """Lazy initialization."""
        if self.agent is None:
            from pydantic_ai import Agent

            from codebase_rag.config import settings
            from codebase_rag.providers import _create_provider_model

            config = settings.active_orchestrator_config
            llm = _create_provider_model(config)

            self.agent = Agent(
                model=llm,
                system_prompt=self.SYSTEM_PROMPT,
                output_type=ExtractionResult,
                retries=1,
            )

    async def extract(self, chunk_content: str, chunk_qn: str) -> ExtractionResult:
        """Extract concepts from chunk content."""
        self._initialize_agent()

        try:
            result = await self.agent.run(chunk_content)
            # Add source attribution
            for concept in result.output.concepts:
                concept.source_chunk_qn = chunk_qn
            return result.output
        except Exception as e:
            from loguru import logger

            logger.warning(f"Concept extraction failed for {chunk_qn}: {e}")
            return ExtractionResult()
