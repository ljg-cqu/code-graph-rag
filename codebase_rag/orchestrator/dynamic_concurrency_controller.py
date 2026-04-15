"""
Dynamic Concurrency Controller module for parallel sub-agent execution.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import psutil
from loguru import logger

from codebase_rag.config import settings

if TYPE_CHECKING:
    from codebase_rag.orchestrator.subagent_orchestrator import SubAgentWorker


class DynamicConcurrencyController:
    """
    Controls concurrency levels for parallel sub-agent execution.
    Derives worker counts from runtime settings, CPU guardrails, and subtask volume.
    """

    def __init__(self):
        self.auto_scale = settings.CGR_AUTO_SCALE_WORKERS
        self.default_workers = max(1, settings.CGR_DEFAULT_PARALLEL_WORKERS)
        self.max_workers = max(self.default_workers, settings.CGR_MAX_PARALLEL_WORKERS)
        self.allow_dynamic_max_override = settings.CGR_ALLOW_DYNAMIC_MAX_OVERRIDE
        self._next_worker_index = 0
        self.permanent_workers: list[SubAgentWorker] = []
        self.burst_workers: list[SubAgentWorker] = []

    def _initialize_permanent_workers(self, agent_factory: callable) -> None:
        from codebase_rag.orchestrator.subagent_orchestrator import SubAgentWorker

        logger.info(f"Initializing {self.default_workers} baseline workers")
        for i in range(self.default_workers):
            worker = SubAgentWorker(worker_id=f"base-{i}", agent=agent_factory())
            self.permanent_workers.append(worker)

    def _get_cpu_core_limit(self) -> int:
        physical_cores = psutil.cpu_count(logical=False) or 4
        return max(1, physical_cores - 1)

    def _get_safe_limit(self, requested_count: int | None = None) -> int:
        cpu_limit = self._get_cpu_core_limit()
        if requested_count is not None and self.allow_dynamic_max_override:
            return max(1, min(max(requested_count, self.max_workers), cpu_limit))
        return max(1, min(self.max_workers, cpu_limit))

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
                except ValueError:
                    continue
                if count > 0:
                    logger.info(f"Extracted worker count: {count} from prompt")
                    return count

        return None

    def get_effective_worker_count(
        self, requested_count: int | None = None, subtask_count: int | None = None
    ) -> int:
        """
        Calculate the effective worker count from runtime inputs and safety limits.

        Args:
            requested_count: User-requested worker count
            subtask_count: Number of subtasks to execute

        Returns:
            Effective worker count to use
        """
        normalized_request = max(1, requested_count) if requested_count else None
        effective = normalized_request or self.default_workers

        if self.auto_scale and subtask_count is not None:
            effective = min(effective, max(1, subtask_count))

        safe_limit = self._get_safe_limit(normalized_request)
        effective = max(1, min(effective, safe_limit))
        logger.info(
            f"Effective worker count: {effective} (requested={normalized_request}, subtasks={subtask_count}, safe_limit={safe_limit})"
        )
        return effective

    def adjust_worker_count(self, current_count: int, adjustment: int) -> int:
        """
        Adjust worker count up or down within allowed limits.

        Args:
            current_count: Current number of workers
            adjustment: Number of workers to add or remove

        Returns:
            New adjusted worker count
        """
        requested = max(1, current_count + adjustment)
        safe_limit = self._get_safe_limit(requested if adjustment > 0 else None)
        new_count = max(1, min(requested, safe_limit))
        logger.info(f"Adjusted worker count from {current_count} to {new_count}")
        return new_count

    def scale_workers(
        self, target_count: int, agent_factory: callable
    ) -> list[SubAgentWorker]:
        """
        Scale worker pool to the requested count using baseline workers first and
        additional burst workers only when necessary.

        Args:
            target_count: Target number of active workers
            agent_factory: Factory function to create new agent instances

        Returns:
            List of active workers ready for assignment
        """
        from codebase_rag.orchestrator.subagent_orchestrator import SubAgentWorker

        target_count = self.get_effective_worker_count(requested_count=target_count)

        if not self.permanent_workers:
            self._initialize_permanent_workers(agent_factory)

        active_workers = self.permanent_workers.copy()
        required_burst = max(0, target_count - self.default_workers)
        current_burst_count = len(self.burst_workers)

        if required_burst > current_burst_count:
            add_count = required_burst - current_burst_count
            logger.info(f"Spinning up {add_count} additional burst workers")
            for i in range(current_burst_count, current_burst_count + add_count):
                worker = SubAgentWorker(worker_id=f"burst-{i}", agent=agent_factory())
                self.burst_workers.append(worker)

        active_workers.extend(self.burst_workers[:required_burst])
        return active_workers[:target_count]

    def get_next_worker_round_robin(
        self, active_workers: list[SubAgentWorker]
    ) -> SubAgentWorker:
        """
        Get next available worker in round-robin order.

        Args:
            active_workers: List of currently active workers

        Returns:
            Next worker to assign subtask to
        """
        if not active_workers:
            raise ValueError("No active workers available")

        worker = active_workers[self._next_worker_index % len(active_workers)]
        self._next_worker_index += 1
        return worker

    def reassign_failed_subtask(
        self, active_workers: list[SubAgentWorker], failed_worker_id: str
    ) -> SubAgentWorker:
        """
        Reassign a failed subtask to a different worker.

        Args:
            active_workers: List of currently active workers
            failed_worker_id: ID of the worker that failed

        Returns:
            Worker to retry the subtask on
        """
        self._next_worker_index += 1
        next_worker = self.get_next_worker_round_robin(active_workers)

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
        self._next_worker_index = 0
