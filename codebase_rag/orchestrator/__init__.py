"""
Orchestrator module for parallel sub-agent execution.
"""

from .concurrency_eligibility_classifier import ConcurrencyEligibilityClassifier
from .dynamic_concurrency_controller import DynamicConcurrencyController
from .investigation_tracker import InvestigationState
from .result_aggregator import ResultAggregator
from .subagent_orchestrator import SubAgentOrchestrator
from .sufficiency_analyzer import InvestigationRequirements, QuestionType, analyze_requirements
from .sufficiency_gatekeeper import (
    SubtaskResult,
    SufficiencyMetadata,
    evaluate_sufficiency,
    evaluate_parallel_worker_sufficiency,
)
from .task_splitter import TaskSplitter

__all__ = [
    "TaskSplitter",
    "SubAgentOrchestrator",
    "ResultAggregator",
    "DynamicConcurrencyController",
    "ConcurrencyEligibilityClassifier",
    "InvestigationState",
    "InvestigationRequirements",
    "QuestionType",
    "analyze_requirements",
    "SubtaskResult",
    "SufficiencyMetadata",
    "evaluate_sufficiency",
    "evaluate_parallel_worker_sufficiency",
]
