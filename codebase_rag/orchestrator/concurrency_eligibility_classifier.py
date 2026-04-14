"""
Concurrency Eligibility Classifier module for automatic parallel execution detection.
Determines if a task can be safely parallelized without explicit user request.
"""

import re
import json
from typing import Optional, Tuple
from loguru import logger
from pydantic_ai import Agent
from codebase_rag.config import settings
from codebase_rag.providers import get_provider_from_config


class ConcurrencyEligibilityClassifier:
    """
    Classifies user requests/tasks to determine if they are eligible for automatic parallel execution.
    Uses explicit user overrides, minimal safety rules, and LLM intent analysis (no rule-based pattern matching for eligible tasks).
    """

    # Explicit user override patterns (highest priority)
    EXPLICIT_PARALLEL_PATTERNS = [
        r"run in parallel",
        r"use parallel execution",
        r"split into parallel subtasks",
        r"parallelize this",
        r"execute in parallel",
    ]

    EXPLICIT_SEQUENTIAL_PATTERNS = [
        r"run sequentially",
        r"no parallel",
        r"run one at a time",
        r"sequential execution",
        r"do not parallelize",
    ]

    # Safety-focused non-eligible patterns (only these rule-based checks remain)
    SAFETY_NON_ELIGIBLE_PATTERNS = [
        (r"single file", 0.95),
        (r"one (file|function|class)", 0.9),
        (r"only (.*) file", 0.85),
        (
            r"(write|modify|delete|update|create) (file|code|config|document)",
            0.9,
        ),  # Write operations are sequential only
    ]

    # LLM prompt template for eligibility analysis (lightweight, low token usage)
    LLM_ELIGIBILITY_PROMPT = """
    You are a parallel task eligibility classifier. Evaluate if the following user request can be split into independent, non-overlapping subtasks that can be executed in parallel to speed up results.

    User request: {prompt}

    Respond ONLY with a valid JSON object with two keys:
    1. "eligible": boolean (true if request can be parallelized, false otherwise)
    2. "confidence": float between 0.0 and 1.0 indicating how confident you are in this assessment
    """

    def __init__(self):
        self.enabled: bool = getattr(settings, "CGR_AUTO_PARALLEL_ENABLED", True)
        self.threshold: float = getattr(
            settings, "CGR_PARALLEL_ELIGIBILITY_THRESHOLD", 0.7
        )
        self.min_subtask_count: int = (
            2  # Minimum subtasks required to justify parallel overhead
        )
        self.agent: Optional[Agent] = None
        # Dynamic calibration state (tracks success rates per task type)
        self.success_rate_tracker: dict[str, list[bool]] = {}

    async def _get_llm_eligibility(self, prompt: str) -> Tuple[float, str]:
        """Run lightweight LLM analysis to determine parallel eligibility and confidence score."""
        if not self.agent:
            config = settings.active_orchestrator_config
            provider = get_provider_from_config(config)
            llm = provider.create_model(config.model_id)
            self.agent = Agent(
                model=llm,
                system_prompt=self.LLM_ELIGIBILITY_PROMPT,
                output_type=dict,
                retries=settings.AGENT_RETRIES,
            )

        try:
            result = await self.agent.run(prompt)
            result_data = result.output
            confidence = max(0.0, min(1.0, float(result_data.get("confidence", 0.0))))
            task_type = (
                result_data.get("task_type", "llm_analyzed")
                if result_data.get("eligible", False)
                else "llm_rejected"
            )

            return confidence, task_type
        except Exception as e:
            logger.warning(
                f"LLM eligibility check failed: {str(e)}, falling back to sequential execution"
            )
            return 0.0, "llm_check_failed"

    async def is_eligible(
        self,
        prompt: str,
        subtask_count: int | None = None,
        has_write_operations: bool = False,
    ) -> tuple[bool, str, float]:
        """
        Determine if a task is eligible for automatic parallel execution (priority order enforced).

        Args:
            prompt: User's natural language request / task description
            subtask_count: Optional number of detected subtasks for this request
            has_write_operations: Whether the task includes any write/modify operations

        Returns:
            Tuple of (eligible: bool, task_type: str, confidence: float)
        """
        if not self.enabled:
            return False, "concurrency_disabled", 0.0

        # 1. HIGHEST PRIORITY: Explicit write operation check
        if has_write_operations:
            logger.debug(
                "Task not eligible for parallel execution: contains write operations"
            )
            return False, "write_operation", 0.0

        # 2. Check for explicit user overrides
        lower_prompt = prompt.lower()

        for pattern in self.EXPLICIT_PARALLEL_PATTERNS:
            if re.search(pattern, lower_prompt, flags=re.IGNORECASE):
                logger.info(
                    "Task eligible for parallel execution: explicit user request"
                )
                return True, "user_requested_parallel", 1.0

        for pattern in self.EXPLICIT_SEQUENTIAL_PATTERNS:
            if re.search(pattern, lower_prompt, flags=re.IGNORECASE):
                logger.debug(
                    "Task not eligible for parallel execution: explicit user request for sequential"
                )
                return False, "user_requested_sequential", 0.0

        # 3. Safety rule-based non-eligible checks
        for pattern, confidence in self.SAFETY_NON_ELIGIBLE_PATTERNS:
            if re.search(pattern, lower_prompt, flags=re.IGNORECASE):
                logger.debug(f"Task matches safety non-eligible pattern '{pattern}'")
                return False, "safety_rule_blocked", confidence

        # 4. Check minimum subtask count if provided
        if subtask_count is not None and subtask_count < self.min_subtask_count:
            logger.debug(
                f"Task not eligible for parallel execution: only {subtask_count} subtasks (min {self.min_subtask_count})"
            )
            return False, "insufficient_subtasks", 0.0

        # 5. LLM intent analysis (primary eligibility detection)
        confidence, task_type = await self._get_llm_eligibility(prompt)

        # 6. Apply dynamic calibration adjustment (based on past success rates)
        if (
            task_type in self.success_rate_tracker
            and len(self.success_rate_tracker[task_type]) >= 10
        ):
            success_rate = sum(self.success_rate_tracker[task_type]) / len(
                self.success_rate_tracker[task_type]
            )
            if success_rate > 0.9:
                confidence = min(
                    1.0, confidence * 1.1
                )  # Lower effective threshold for high success task types
            elif success_rate < 0.6:
                confidence = max(
                    0.0, confidence * 0.9
                )  # Raise effective threshold for low success task types

        # Final eligibility decision
        if confidence >= self.threshold:
            logger.info(
                f"Task eligible for parallel execution: type={task_type}, confidence={confidence:.2f}"
            )
            return True, task_type, confidence

        logger.debug(
            f"Task not eligible for parallel execution: LLM confidence {confidence:.2f} below threshold {self.threshold}"
        )
        return False, "llm_rejected", confidence

    def record_execution_result(self, task_type: str, success: bool) -> None:
        """Record execution result for dynamic confidence calibration."""
        if task_type not in self.success_rate_tracker:
            self.success_rate_tracker[task_type] = []
        self.success_rate_tracker[task_type].append(success)
        # Keep only last 100 results per task type for calibration
        if len(self.success_rate_tracker[task_type]) > 100:
            self.success_rate_tracker[task_type] = self.success_rate_tracker[task_type][
                -100:
            ]
