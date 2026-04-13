"""
Dynamic Concurrency Controller module for parallel sub-agent execution.
Handles worker count validation, auto-scaling, and dynamic adjustments.
"""

import re

from loguru import logger

from codebase_rag.config import settings


class DynamicConcurrencyController:
    """
    Controls concurrency levels for parallel sub-agent execution.
    Enforces safety limits, handles auto-scaling, and dynamic adjustments.
    """

    def __init__(self):
        self.max_workers = settings.CGR_MAX_PARALLEL_WORKERS
        self.default_workers = settings.CGR_DEFAULT_PARALLEL_WORKERS
        self.allow_override = settings.CGR_ALLOW_DYNAMIC_MAX_OVERRIDE
        self.auto_scale = settings.CGR_AUTO_SCALE_WORKERS

    def extract_worker_count_from_prompt(self, prompt: str) -> int | None:
        """
        Extract requested worker count from natural language prompt.

        Args:
            prompt: User's request

        Returns:
            Extracted worker count if present, None otherwise
        """
        # Pattern matching for worker count requests
        patterns = [
            r"(\d+)\s*(?:parallel)?\s*(?:worker|subagent|thread|process)",
            r"use\s+(\d+)\s+workers",
            r"with\s+(\d+)\s+parallel",
            r"spawn\s+(\d+)\s+subagent",
        ]

        lower_prompt = prompt.lower()
        for pattern in patterns:
            match = re.search(pattern, lower_prompt)
            if match:
                try:
                    count = int(match.group(1))
                    if count > 0:
                        logger.info(f"Extracted worker count: {count} from prompt")
                        return count
                except ValueError:
                    pass

        return None

    def get_effective_worker_count(
        self, requested_count: int | None = None, subtask_count: int | None = None
    ) -> int:
        """
        Calculate the effective worker count considering limits and auto-scaling.

        Args:
            requested_count: User-requested worker count (if any)
            subtask_count: Number of subtasks to execute (for auto-scaling)

        Returns:
            Effective worker count to use
        """
        # Start with requested count or default
        effective = requested_count or self.default_workers

        # Enforce max limit unless override is allowed
        if effective > self.max_workers:
            if self.allow_override:
                logger.warning(
                    f"Requested worker count {effective} exceeds max limit {self.max_workers}. "
                    "Override allowed, using requested count."
                )
            else:
                logger.warning(
                    f"Requested worker count {effective} exceeds max limit {self.max_workers}. "
                    f"Limiting to {self.max_workers}."
                )
                effective = self.max_workers

        # Auto-scale to match subtask count if enabled
        if self.auto_scale and subtask_count is not None:
            if effective > subtask_count:
                logger.info(
                    f"Auto-scaling workers from {effective} to {subtask_count} "
                    f"to match number of subtasks"
                )
                effective = subtask_count

        # Ensure minimum 1 worker
        effective = max(1, effective)
        logger.info(f"Effective worker count: {effective}")

        return effective

    def adjust_worker_count(self, current_count: int, adjustment: int) -> int:
        """
        Adjust worker count dynamically during execution.

        Args:
            current_count: Current number of active workers
            adjustment: Number of workers to add (positive) or remove (negative)

        Returns:
            New worker count
        """
        new_count = max(1, current_count + adjustment)
        new_count = (
            min(new_count, self.max_workers) if not self.allow_override else new_count
        )

        logger.info(f"Adjusted worker count from {current_count} to {new_count}")
        return new_count
