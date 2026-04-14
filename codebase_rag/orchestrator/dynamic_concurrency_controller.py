"""
Dynamic Concurrency Controller module for parallel sub-agent execution.
Implements 10 permanent base workers, 10 burst workers, round-robin assignment, and auto-scaling.
"""

from __future__ import annotations
import re
import math
import psutil
from typing import List, Optional, Any, TYPE_CHECKING
from loguru import logger
from codebase_rag.config import settings

if TYPE_CHECKING:
    from codebase_rag.orchestrator.subagent_orchestrator import SubAgentWorker


class DynamicConcurrencyController:
    """
    Controls concurrency levels for parallel sub-agent execution.
    Implements round-robin assignment, permanent base workers + burst workers architecture,
    and dynamic auto-scaling per specification.
    """

    # Fixed architecture parameters per spec
    PERMANENT_BASE_WORKERS = 10
    MAX_BURST_WORKERS = 10
    MAX_TOTAL_WORKERS = PERMANENT_BASE_WORKERS + MAX_BURST_WORKERS

    def __init__(self):
        self.auto_scale = settings.CGR_AUTO_SCALE_WORKERS
        # Round-robin assignment state
        self._next_worker_index = 0
        # Worker pools
        self.permanent_workers: List[SubAgentWorker] = []
        self.burst_workers: List[SubAgentWorker] = []

    def _initialize_permanent_workers(self, agent_factory: callable) -> None:
        """Initialize 10 permanent base workers that run continuously for zero cold start overhead."""
        from codebase_rag.orchestrator.subagent_orchestrator import (
            SubAgentWorker,
        )  # Import here to avoid circular import

        logger.info(
            f"Initializing {self.PERMANENT_BASE_WORKERS} permanent base workers"
        )
        for i in range(self.PERMANENT_BASE_WORKERS):
            worker = SubAgentWorker(worker_id=f"base-{i}", agent=agent_factory())
            self.permanent_workers.append(worker)

    def _get_cpu_core_limit(self) -> int:
        """Get maximum allowed workers based on available physical CPU cores (never exceed cores - 1)."""
        physical_cores = psutil.cpu_count(logical=False) or 4
        return max(1, physical_cores - 1)

    def extract_worker_count_from_prompt(self, prompt: str) -> int | None:
        """
        Extract requested worker count from natural language prompt.

        Args:
            prompt: User's request

        Returns:
            Extracted worker count if present, None otherwise
        """
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
                        return min(count, self.MAX_TOTAL_WORKERS)
                except ValueError:
                    pass

        return None

    def get_effective_worker_count(
        self, requested_count: int | None = None, subtask_count: int | None = None
    ) -> int:
        """
        Calculate the effective worker count following the spec scaling formula:
        worker_count = min(max(ceil(subtask_count / 2), 1), max_workers)
        Also enforces CPU core limit safeguard.

        Args:
            requested_count: User-requested worker count (if any)
            subtask_count: Number of subtasks to execute (for auto-scaling)

        Returns:
            Effective worker count to use
        """
        # Apply CPU core limit first (safety safeguard)
        max_allowed_workers = min(self.MAX_TOTAL_WORKERS, self._get_cpu_core_limit())

        # 1. Use user requested count if provided
        if requested_count is not None:
            effective = min(requested_count, max_allowed_workers)
        # 2. Auto-scale based on subtask count if enabled
        elif self.auto_scale and subtask_count is not None:
            effective = min(max(math.ceil(subtask_count / 2), 1), max_allowed_workers)
        # 3. Fall back to permanent base worker count
        else:
            effective = min(self.PERMANENT_BASE_WORKERS, max_allowed_workers)

        # Ensure minimum 1 worker
        effective = max(1, effective)
        logger.info(
            f"Effective worker count: {effective} (base workers: {self.PERMANENT_BASE_WORKERS}, burst workers allowed: {max(0, effective - self.PERMANENT_BASE_WORKERS)})"
        )
        return effective

    def adjust_worker_count(self, current_count: int, adjustment: int) -> int:
        """
        Adjust worker count up or down within allowed limits.

        Args:
            current_count: Current number of workers
            adjustment: Number of workers to add (positive) or remove (negative)

        Returns:
            New adjusted worker count
        """
        max_allowed_workers = min(self.MAX_TOTAL_WORKERS, self._get_cpu_core_limit())
        new_count = max(1, min(current_count + adjustment, max_allowed_workers))
        logger.info(f"Adjusted worker count from {current_count} to {new_count}")
        return new_count

    def scale_workers(
        self, target_count: int, agent_factory: callable
    ) -> List[SubAgentWorker]:
        """
        Scale worker pool to target count: use permanent base workers first, then spin up burst workers as needed.

        Args:
            target_count: Target number of active workers
            agent_factory: Factory function to create new agent instances for workers

        Returns:
            List of active workers ready for assignment
        """
        from codebase_rag.orchestrator.subagent_orchestrator import (
            SubAgentWorker,
        )  # Import here to avoid circular import

        # Initialize permanent workers if not done yet
        if not self.permanent_workers:
            self._initialize_permanent_workers(agent_factory)

        active_workers = self.permanent_workers.copy()
        required_burst = max(0, target_count - self.PERMANENT_BASE_WORKERS)

        # Spin up additional burst workers if needed
        current_burst_count = len(self.burst_workers)
        if required_burst > current_burst_count:
            add_count = required_burst - current_burst_count
            logger.info(f"Spinning up {add_count} additional burst workers")
            for i in range(current_burst_count, current_burst_count + add_count):
                worker = SubAgentWorker(worker_id=f"burst-{i}", agent=agent_factory())
                self.burst_workers.append(worker)

        # Return exactly target count of workers
        active_workers.extend(self.burst_workers[:required_burst])
        return active_workers[:target_count]

    def get_next_worker_round_robin(
        self, active_workers: List[SubAgentWorker]
    ) -> SubAgentWorker:
        """
        Get next available worker in strict round-robin order for even load distribution.

        Args:
            active_workers: List of currently active workers

        Returns:
            Next worker to assign subtask to
        """
        if not active_workers:
            raise ValueError("No active workers available")

        # Get next worker index, wrap around as needed
        worker = active_workers[self._next_worker_index % len(active_workers)]
        self._next_worker_index += 1
        return worker

    def reassign_failed_subtask(
        self, active_workers: List[SubAgentWorker], failed_worker_id: str
    ) -> SubAgentWorker:
        """
        Reassign a failed subtask to the next available worker in the round-robin queue, skipping the failed worker.

        Args:
            active_workers: List of currently active workers
            failed_worker_id: ID of the worker that failed to execute the subtask

        Returns:
            Next worker to retry the subtask on
        """
        # Skip the failed worker by incrementing index once
        self._next_worker_index += 1
        next_worker = self.get_next_worker_round_robin(active_workers)

        # If we got the same failed worker, increment again to get a different one
        while next_worker.worker_id == failed_worker_id and len(active_workers) > 1:
            self._next_worker_index += 1
            next_worker = self.get_next_worker_round_robin(active_workers)

        logger.info(
            f"Reassigning failed subtask from worker {failed_worker_id} to worker {next_worker.worker_id}"
        )
        return next_worker

    def shutdown_burst_workers(self) -> None:
        """Shut down all burst workers to free resources when not needed."""
        if self.burst_workers:
            logger.info(f"Shutting down {len(self.burst_workers)} burst workers")
            for worker in self.burst_workers:
                worker.shutdown()
            self.burst_workers = []
        # Reset round-robin index
        self._next_worker_index = 0
