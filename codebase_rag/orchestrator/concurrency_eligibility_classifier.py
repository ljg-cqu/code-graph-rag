"""
Concurrency Eligibility Classifier module for automatic parallel execution detection.
Determines if a task can be safely parallelized without explicit user request.
"""

import re

from loguru import logger

from codebase_rag.config import settings


class ConcurrencyEligibilityClassifier:
    """
    Classifies user requests/tasks to determine if they are eligible for automatic parallel execution.
    Uses a combination of rule-based matching and semantic pattern detection.
    """

    # Eligible task type patterns (rule-based)
    ELIGIBLE_PATTERNS: list[tuple[str, str, float]] = [
        # Multi-file code search patterns (flexible matching)
        (r"find all (?:.*)?(functions|classes|methods|files|occurrences)", "multi_file_search", 0.9),
        (r"(search|find).* across (all|the entire|multiple) (repository|codebase|files|project|repo)", "multi_file_search", 0.95),
        (r"look for (.*) in (all|multiple) files", "multi_file_search", 0.85),
        (r"find where (.*) is used across the (codebase|repo|project)", "multi_file_search", 0.9),

        # Bulk validation/analysis patterns (flexible matching)
        (r"(check|scan|audit) (all|multiple) (?:.*)?(files|modules|functions) for (errors|bugs|vulnerabilities|issues|anti-patterns|security risks)", "bulk_validation", 0.95),
        (r"validate (all|the entire) (codebase|repo|project)", "bulk_validation", 0.9),
        (r"run (analysis|lint|audit|security scan) on (all|multiple) files", "bulk_validation", 0.9),
        (r"scan for (security|performance|quality) issues across the (project|codebase|repo)", "bulk_validation", 0.9),

        # Large repository ingestion/indexing
        (r"(index|ingest|update|scan) (all|the entire|large|big) (repository|codebase|project|files)", "large_ingestion", 0.95),
        (r"full (index|reindex|scan|ingest) of the (codebase|repo|project)", "large_ingestion", 0.9),

        # Impact analysis across multiple components
        (r"(impact|effect) of changing (.*) across (the codebase|multiple modules|all files)", "impact_analysis", 0.9),
        (r"what depends on (.*) across the (project|repo|codebase)", "impact_analysis", 0.85),
        (r"find all dependencies of (.*) across multiple modules", "impact_analysis", 0.85),

        # Batch documentation generation
        (r"generate (docs|documentation|readme) for (all|multiple) (functions|classes|modules|files)", "batch_docs", 0.95),
        (r"document (all|the entire) (codebase|project|modules)", "batch_docs", 0.9),

        # Multiple independent tool calls / multi-step tasks
        (r"get (.*) and (.*) and (.*) from the codebase", "multi_tool", 0.8),
        (r"first find (.*), then get (.*), then (.*)", "multi_tool", 0.8),
        (r"perform the following (.*) tasks", "multi_tool", 0.85),
    ]

    # Non-eligible patterns (tasks that should never be parallelized)
    NON_ELIGIBLE_PATTERNS: list[tuple[str, float]] = [
        (r"single file", 0.95),
        (r"one (file|function|class)", 0.9),
        (r"only (.*) file", 0.85),
        (r"sequential execution", 0.95),
        (r"no parallel", 0.95),
        (r"run one at a time", 0.9),
        (r"(write|modify|delete|update|create) (file|code|config)", 0.8), # Write operations are sequential only
    ]

    def __init__(self):
        self.enabled: bool = getattr(settings, "CGR_AUTO_PARALLEL_ENABLED", True)
        self.threshold: float = getattr(settings, "CGR_PARALLEL_ELIGIBILITY_THRESHOLD", 0.8)
        self.min_subtask_count: int = 2 # Minimum subtasks required to justify parallel overhead

    def is_eligible(self, prompt: str, subtask_count: int | None = None, has_write_operations: bool = False) -> tuple[bool, str, float]:
        """
        Determine if a task is eligible for automatic parallel execution.

        Args:
            prompt: User's natural language request / task description
            subtask_count: Optional number of detected subtasks for this request
            has_write_operations: Whether the task includes any write/modify operations

        Returns:
            Tuple of (eligible: bool, task_type: str, confidence: float)
        """
        if not self.enabled:
            return False, "concurrency_disabled", 0.0

        # Write operations are never eligible for parallel execution
        if has_write_operations:
            logger.debug("Task not eligible for parallel execution: contains write operations")
            return False, "write_operation", 0.0

        # If subtask count is provided and below minimum, not eligible
        if subtask_count is not None and subtask_count < self.min_subtask_count:
            logger.debug(f"Task not eligible for parallel execution: only {subtask_count} subtasks (min {self.min_subtask_count})")
            return False, "insufficient_subtasks", 0.0

        # Check for non-eligible patterns first (override eligible patterns)
        lower_prompt = prompt.lower()
        for pattern, confidence in self.NON_ELIGIBLE_PATTERNS:
            if re.search(pattern, lower_prompt, flags=re.IGNORECASE):
                logger.debug(f"Task matches non-eligible pattern '{pattern}' with confidence {confidence}")
                return False, "non_eligible_pattern", confidence

        # Check for eligible patterns
        max_confidence: float = 0.0
        matched_type: str = "unknown"

        for pattern, task_type, confidence in self.ELIGIBLE_PATTERNS:
            if re.search(pattern, lower_prompt, flags=re.IGNORECASE):
                if confidence > max_confidence:
                    max_confidence = confidence
                    matched_type = task_type

        # If confidence meets threshold, eligible
        if max_confidence >= self.threshold:
            logger.info(f"Task automatically eligible for parallel execution: type={matched_type}, confidence={max_confidence:.2f}")
            return True, matched_type, max_confidence

        # No eligible patterns matched
        logger.debug(f"Task not eligible for parallel execution: max confidence {max_confidence:.2f} below threshold {self.threshold}")
        return False, "no_matching_pattern", max_confidence
