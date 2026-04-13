"""
Sub-Agent Orchestrator module for parallel execution.
Manages sub-agent pool, task distribution, lifecycle, and execution guarantees.
"""

import signal
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from loguru import logger

from codebase_rag.config import settings

from .dynamic_concurrency_controller import DynamicConcurrencyController
from .result_aggregator import ResultAggregator


class SubAgentOrchestrator:
    """
    Orchestrates parallel execution of sub-agent tasks.
    Manages full lifecycle of sub-agents, execution guarantees, and cleanup.
    """

    def __init__(
        self,
        worker_count: int | None = None,
        agent_factory: Callable | None = None,
        scheduling_strategy: str = "fifo",  # Options: "fifo", "round-robin"
    ):
        self.dynamic_controller = DynamicConcurrencyController()
        self.worker_count = self.dynamic_controller.get_effective_worker_count(
            worker_count
        )
        self.agent_factory = agent_factory or self._default_agent_factory
        self.agent_pool: list[Any] = []
        self.running = False
        self._shutdown_called = False

        # Validate scheduling strategy
        valid_strategies = {"fifo", "round-robin"}
        scheduling_strategy = scheduling_strategy.lower()
        if scheduling_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid scheduling strategy: {scheduling_strategy}. Valid options: {', '.join(valid_strategies)}"
            )
        self.scheduling_strategy = scheduling_strategy
        self._round_robin_index = 0  # Counter for round-robin task assignment

        # Per-worker rate limiting to prevent LLM API throttling
        try:
            from codebase_rag.embeddings.rate_limiter import RateLimiter

            self.rate_limiter = RateLimiter(max_requests_per_minute=60)
        except ImportError:
            # Fallback to no rate limiting if rate limiter module not available
            self.rate_limiter = None

        # Safety check for worker count exceeding max limit
        if (
            self.worker_count > settings.CGR_MAX_PARALLEL_WORKERS
            and not settings.CGR_ALLOW_DYNAMIC_MAX_OVERRIDE
        ):
            logger.warning(
                f"Requested worker count {self.worker_count} exceeds max limit {settings.CGR_MAX_PARALLEL_WORKERS}. "
                f"Limiting to {settings.CGR_MAX_PARALLEL_WORKERS} workers. Set CGR_ALLOW_DYNAMIC_MAX_OVERRIDE=True to override."
            )
            self.worker_count = settings.CGR_MAX_PARALLEL_WORKERS

        # Register signal handlers for graceful shutdown on all termination signals
        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT]:
            try:
                signal.signal(sig, self._handle_shutdown)
            except ValueError:
                # Ignore signals that aren't supported on the current platform
                pass

    def _default_agent_factory(self) -> Any:
        """
        Default factory for creating sub-agent instances.
        TODO: Replace with actual RAG agent initialization once integrated.

        Returns:
            New sub-agent instance
        """

        # For MVP, this is a placeholder that returns a simple callable
        # In real implementation, this would create an instance of the core RAG agent
        # with isolated state, inherited config, and read-only permissions by default
        class SimpleSubAgent:
            def __init__(self):
                self.config = settings
                self.allow_write = settings.CGR_SUBAGENT_ALLOW_WRITE

            def execute(self, prompt: str) -> str:
                # Placeholder execution logic
                time.sleep(0.1)  # Simulate work
                return f"Processed prompt: {prompt[:50]}..."

        return SimpleSubAgent()

    def initialize_agents(self):
        """Initialize the pool of sub-agents."""
        logger.info(f"Initializing {self.worker_count} sub-agents")
        self.agent_pool = [self.agent_factory() for _ in range(self.worker_count)]
        logger.info("Sub-agent pool initialized successfully")

    def execute_tasks(
        self,
        subtasks: list[dict[str, Any]],
        result_aggregator: ResultAggregator | None = None,
        retry_attempts: int | None = None,
        dry_run: bool = False,
    ) -> ResultAggregator:
        """
        Execute a list of subtasks in parallel using the sub-agent pool.

        Args:
            subtasks: List of subtasks to execute
            result_aggregator: Optional aggregator to use for results
            retry_attempts: Number of retries for failed tasks (defaults to config value)
            dry_run: If True, simulate execution without making actual LLM calls (for cost estimation)
        """
        """
        Execute a list of subtasks in parallel using the sub-agent pool.

        Args:
            subtasks: List of subtasks to execute
            result_aggregator: Optional aggregator to use for results
            retry_attempts: Number of retries for failed tasks (defaults to config value)

        Returns:
            ResultAggregator with all results and errors
        """
        retry_attempts = retry_attempts or settings.CGR_SUBAGENT_RETRY_ATTEMPTS
        result_aggregator = result_aggregator or ResultAggregator()
        result_aggregator.set_total_subtasks(len(subtasks))

        # Auto-scale worker count to match subtask count
        self.worker_count = self.dynamic_controller.get_effective_worker_count(
            self.worker_count, len(subtasks)
        )

        # Initialize agents if not already done
        if not self.agent_pool:
            self.initialize_agents()

        # Handle dry run mode
        if dry_run:
            start_time = time.time()
            logger.info(
                f"Dry run: Would execute {len(subtasks)} subtasks with {self.worker_count} workers (scheduling: {self.scheduling_strategy})"
            )
            # Simulate execution for dry run without actual LLM calls
            for subtask in subtasks:
                result_aggregator.add_result(
                    subtask,
                    f"Dry run: Would process {subtask.get('relative_path', 'unknown file')} with round-robin worker assignment",
                    execution_time=0.0,
                )
            result_aggregator.set_total_execution_time(time.time() - start_time)
            logger.info(
                f"Dry run completed in {result_aggregator.metadata['total_execution_time']:.2f}s"
            )
            return result_aggregator

        self.running = True
        start_time = time.time()
        logger.info(
            f"Starting parallel execution of {len(subtasks)} subtasks with {self.worker_count} workers (scheduling: {self.scheduling_strategy})"
        )

        # Track remaining tasks and retries
        remaining_tasks = subtasks.copy()
        retry_counts = {st["id"]: 0 for st in subtasks}

        try:
            while remaining_tasks and self.running and not self._shutdown_called:
                with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
                    # Map subtasks to future objects
                    future_to_subtask = {}

                    # Assign tasks based on scheduling strategy (only assign as many as available agents)
                    num_tasks_to_assign = min(
                        len(remaining_tasks), len(self.agent_pool)
                    )
                    tasks_to_assign = remaining_tasks[:num_tasks_to_assign]

                    if self.scheduling_strategy == "round-robin":
                        # Round-robin assignment: cycle through available agents
                        for subtask in tasks_to_assign:
                            # Get next agent in round-robin order
                            agent = self.agent_pool[
                                self._round_robin_index % len(self.agent_pool)
                            ]
                            self._round_robin_index += 1
                            # Remove agent from pool temporarily during execution
                            self.agent_pool.remove(agent)
                            future = executor.submit(
                                self._execute_subtask, agent, subtask
                            )
                            future_to_subtask[future] = (subtask, agent)
                    else:
                        # Default FIFO assignment
                        for subtask in tasks_to_assign:
                            agent = self.agent_pool.pop()
                            future = executor.submit(
                                self._execute_subtask, agent, subtask
                            )
                            future_to_subtask[future] = (subtask, agent)

                    # Process completed futures
                    for future in as_completed(future_to_subtask):
                        subtask, agent = future_to_subtask[future]

                        # Return agent to pool
                        self.agent_pool.append(agent)

                        try:
                            result, execution_time = future.result()
                            result_aggregator.add_result(
                                subtask, result, execution_time
                            )
                            # Remove from remaining tasks
                            remaining_tasks = [
                                st
                                for st in remaining_tasks
                                if st["id"] != subtask["id"]
                            ]

                        except Exception as e:
                            error_msg = str(e)
                            retry_count = retry_counts.get(subtask["id"], 0)

                            if retry_count < retry_attempts:
                                # Retry the task
                                retry_counts[subtask["id"]] = retry_count + 1
                                logger.warning(
                                    f"Subtask {subtask['id']} failed (attempt {retry_count + 1}/{retry_attempts + 1}): {error_msg}. Retrying..."
                                )
                            else:
                                # Max retries reached, mark as failed
                                result_aggregator.add_error(subtask, error_msg)
                                remaining_tasks = [
                                    st
                                    for st in remaining_tasks
                                    if st["id"] != subtask["id"]
                                ]
                                logger.error(
                                    f"Subtask {subtask['id']} failed permanently after {retry_attempts + 1} attempts: {error_msg}"
                                )

        finally:
            self.running = False
            total_time = time.time() - start_time
            result_aggregator.set_total_execution_time(total_time)
            logger.info(f"Parallel execution completed in {total_time:.2f}s")

        return result_aggregator

    def _execute_subtask(
        self, agent: Any, subtask: dict[str, Any]
    ) -> tuple[Any, float]:
        """
        Execute a single subtask with a given agent.

        Args:
            agent: Sub-agent instance to use
            subtask: Subtask to execute

        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.time()
        timeout = settings.CGR_SUBAGENT_TIMEOUT

        try:
            # Execute the task with timeout
            result = agent.execute(subtask["prompt"])
            execution_time = time.time() - start_time

            if execution_time > timeout:
                raise TimeoutError(f"Subtask exceeded timeout of {timeout}s")

            return result, execution_time

        except Exception as e:
            execution_time = time.time() - start_time
            raise e

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals to gracefully terminate all workers."""
        logger.warning(f"Received signal {signum}, initiating graceful shutdown")
        self._shutdown_called = True
        self.running = False

    def shutdown(self):
        """Shutdown the orchestrator and cleanup all resources."""
        logger.info("Shutting down sub-agent orchestrator")
        self._shutdown_called = True
        self.running = False

        # Clear agent pool
        self.agent_pool.clear()
        logger.info("Sub-agent orchestrator shutdown complete")

    def get_current_progress(self) -> dict[str, Any]:
        """Get current execution progress."""
        # TODO: Implement real-time progress tracking
        return {}

    def adjust_worker_count(self, adjustment: int):
        """
        Adjust the number of workers dynamically during execution.

        Args:
            adjustment: Number of workers to add (positive) or remove (negative)
        """
        new_count = self.dynamic_controller.adjust_worker_count(
            self.worker_count, adjustment
        )

        if new_count > self.worker_count:
            # Add new workers
            new_workers = [
                self.agent_factory() for _ in range(new_count - self.worker_count)
            ]
            self.agent_pool.extend(new_workers)
        elif new_count < self.worker_count:
            # Remove excess workers
            remove_count = self.worker_count - new_count
            for _ in range(remove_count):
                if self.agent_pool:
                    self.agent_pool.pop()

        self.worker_count = new_count
