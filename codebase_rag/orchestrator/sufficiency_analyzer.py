"""Sufficiency analyzer module.

NOTE: This module is deprecated. Use LLMSufficiencyAnalyzer from
codebase_rag.orchestrator.llm_sufficiency_analyzer for LLM-driven
sufficiency analysis.

This module remains for backward compatibility only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    """Question type classification."""

    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    DIAGNOSTIC = "diagnostic"


@dataclass
class InvestigationRequirements:
    """Investigation requirements for a question."""

    question_type: QuestionType
    requires_vector: bool
    requires_graph: bool
    requires_file_read: bool
    min_rounds: int
    requires_cross_validation: bool = False


def analyze_requirements(question: str) -> InvestigationRequirements:
    """Determine investigation requirements for a question.

    DEPRECATED: Use LLMSufficiencyAnalyzer.assess() for LLM-driven analysis.

    This function now returns conservative defaults suitable for most queries.
    It no longer performs keyword matching which produced false positives.

    Args:
        question: User's natural language question

    Returns:
        InvestigationRequirements with conservative defaults
    """
    return InvestigationRequirements(
        QuestionType.FUNCTIONAL,
        requires_vector=True,
        requires_graph=True,
        requires_file_read=True,
        min_rounds=2,
        requires_cross_validation=False,
    )
