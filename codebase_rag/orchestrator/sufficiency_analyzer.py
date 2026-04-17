from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QuestionType(Enum):
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    DIAGNOSTIC = "diagnostic"


@dataclass
class InvestigationRequirements:
    question_type: QuestionType
    requires_vector: bool
    requires_graph: bool
    requires_file_read: bool
    min_rounds: int
    requires_cross_validation: bool = False


_DIAGNOSTIC_KEYWORDS = frozenset({
    "why is", "why does", "what is wrong", "debug", "failing",
    "error", "issue", "problem", "not working", "broken",
    "stack trace", "exception", "crash",
})

_FUNCTIONAL_KEYWORDS = frozenset({
    "how does", "how do", "implementation", "logic", "algorithm",
    "work", "flow", "behavior", "what happens when", "step by step",
    "describe", "explain", "process", "mechanism",
})

_STRUCTURAL_KEYWORDS = frozenset({
    "what classes", "what functions", "list all", "show me",
    "find all", "how many", "count", "directory", "structure",
    "hierarchy", "dependencies", "relationships",
})


def analyze_requirements(question: str) -> InvestigationRequirements:
    q_lower = question.lower()

    if any(kw in q_lower for kw in _DIAGNOSTIC_KEYWORDS):
        return InvestigationRequirements(
            QuestionType.DIAGNOSTIC,
            requires_vector=True,
            requires_graph=True,
            requires_file_read=True,
            min_rounds=3,
            requires_cross_validation=True,
        )
    if any(kw in q_lower for kw in _FUNCTIONAL_KEYWORDS):
        return InvestigationRequirements(
            QuestionType.FUNCTIONAL,
            requires_vector=True,
            requires_graph=True,
            requires_file_read=True,
            min_rounds=3,
        )
    return InvestigationRequirements(
        QuestionType.STRUCTURAL,
        requires_vector=True,
        requires_graph=True,
        requires_file_read=False,
        min_rounds=2,
    )
