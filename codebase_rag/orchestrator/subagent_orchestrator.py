"""
Sub-Agent Orchestrator module for parallel execution.
Manages sub-agent pool, task distribution, lifecycle, and execution guarantees.
"""

import queue
import signal
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

from loguru import logger

from codebase_rag.config import ModelConfig, settings

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
        scheduling_strategy: str = "round-robin",  # Options: "fifo", "round-robin" (default: round-robin as per requirement)
    ):
        self.dynamic_controller = DynamicConcurrencyController()
        self.worker_count = self.dynamic_controller.get_effective_worker_count(
            worker_count
        )
        self.agent_factory = agent_factory or self._default_agent_factory
        self.agent_pool: queue.Queue[Any] = queue.Queue()
        self.running = False
        self._shutdown_called = False
        self._llm_assignment_index = 0  # Counter for round-robin LLM assignment

        # Validate scheduling strategy
        valid_strategies = {"fifo", "round-robin"}
        scheduling_strategy = scheduling_strategy.lower()
        if scheduling_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid scheduling strategy: {scheduling_strategy}. Valid options: {', '.join(valid_strategies)}"
            )
        self.scheduling_strategy = scheduling_strategy

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

    def _default_agent_factory(self, llm_config: ModelConfig | None = None) -> Any:
        """
        Default factory for creating sub-agent instances.

        Args:
            llm_config: Optional custom LLM config to use for this sub-agent instead of default orchestrator LLM

        Returns:
            New sub-agent instance with specified LLM configuration
        """
        worker_llm_config = llm_config or settings.active_orchestrator_config

        # For MVP, this is a placeholder that returns a simple callable
        # In real implementation, this would create an instance of the core RAG agent
        # with isolated state, inherited config, and read-only permissions by default
        class SimpleSubAgent:
            def __init__(self, llm_config: ModelConfig):
                self.config = settings
                self.llm_config = llm_config
                self.allow_write = settings.CGR_SUBAGENT_ALLOW_WRITE
                # Cypher LLM remains shared global config, unchanged
                self.cypher_llm_config = settings.active_cypher_config

            def execute(self, subtask: dict) -> str:
                # Placeholder execution logic
                prompt = subtask.get("prompt", "")
                time.sleep(0.1)  # Simulate work
                return f"Processed prompt with {self.llm_config.provider}:{self.llm_config.model_id}: {prompt[:50]}..."

            def reset(self):
                # Reset any mutable state between tasks
                pass

        return SimpleSubAgent(worker_llm_config)

    def initialize_agents(self):
        """Initialize the pool of sub-agents with round-robin LLM assignment."""
        logger.info(f"Initializing {self.worker_count} sub-agents")
        worker_llms = settings.active_worker_llms
        num_worker_llms = len(worker_llms)

        # Clear existing queue first
        while not self.agent_pool.empty():
            try:
                self.agent_pool.get_nowait()
            except queue.Empty:
                break

        # Reset LLM assignment index for new pool initialization
        self._llm_assignment_index = 0

        # Add new agents to queue with round-robin LLM assignment
        for _ in range(self.worker_count):
            if num_worker_llms > 0:
                # Get next LLM in round-robin sequence
                llm_config = worker_llms[self._llm_assignment_index % num_worker_llms]
                self._llm_assignment_index += 1
                agent = self.agent_factory(llm_config=llm_config)
            else:
                # No worker LLMs configured, use default orchestrator LLM
                agent = self.agent_factory()
            self.agent_pool.put(agent)

        if num_worker_llms > 0:
            logger.info(
                f"Sub-agent pool initialized with {num_worker_llms} worker LLMs (round-robin assignment)"
            )
        else:
            logger.info(
                "Sub-agent pool initialized successfully using orchestrator LLM as default"
            )

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
        retry_attempts = retry_attempts or settings.CGR_SUBAGENT_RETRY_ATTEMPTS
        result_aggregator = result_aggregator or ResultAggregator()
        result_aggregator.set_total_subtasks(len(subtasks))

        # Auto-scale worker count to match subtask count
        self.worker_count = self.dynamic_controller.get_effective_worker_count(
            self.worker_count, len(subtasks)
        )

        # Initialize agents if not already done
        if self.agent_pool.empty():
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
        logger.info(
            f"ACTIVE PARALLEL WORKERS: {self.worker_count} - running concurrently"
        )

        # Track remaining tasks and retries
        remaining_tasks = subtasks.copy()
        retry_counts = {st["id"]: 0 for st in subtasks}
        active_futures = {}

        try:
            # Single long-lived executor for all tasks (no per-batch recreation)
            with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
                while (
                    (remaining_tasks or active_futures)
                    and self.running
                    and not self._shutdown_called
                ):
                    # Submit new tasks as agents become available (continuous parallelism)
                    while remaining_tasks and not self.agent_pool.empty():
                        # Round-robin: take next task from front of queue
                        subtask = remaining_tasks.pop(0)
                        agent = self.agent_pool.get()
                        future = executor.submit(self._execute_subtask, agent, subtask)
                        active_futures[future] = (subtask, agent)

                    # Process completed tasks as they finish
                    if active_futures:
                        done, _ = wait(
                            active_futures.keys(), return_when="FIRST_COMPLETED"
                        )

                        for future in done:
                            subtask, agent = active_futures.pop(future)

                            # Reset agent state before returning to pool
                            if hasattr(agent, "reset") and callable(agent.reset):
                                agent.reset()
                            # Return agent to pool immediately for new tasks
                            self.agent_pool.put(agent)

                            try:
                                result, execution_time = future.result()
                                result_aggregator.add_result(
                                    subtask, result, execution_time
                                )
                            except Exception as e:
                                error_msg = str(e)
                                retry_count = retry_counts.get(subtask["id"], 0)

                                if retry_count < retry_attempts:
                                    # Retry the task, add back to front for round-robin
                                    retry_counts[subtask["id"]] = retry_count + 1
                                    logger.warning(
                                        f"Subtask {subtask['id']} failed (attempt {retry_count + 1}/{retry_attempts + 1}): {error_msg}. Retrying..."
                                    )
                                    remaining_tasks.insert(0, subtask)
                                else:
                                    # Max retries reached, mark as failed
                                    result_aggregator.add_error(subtask, error_msg)
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
            # Execute the task with PREEMPTIVE timeout using single-threaded executor
            with ThreadPoolExecutor(max_workers=1) as task_executor:
                future = task_executor.submit(agent.execute, subtask)
                result = future.result(timeout=timeout)

            execution_time = time.time() - start_time
            return result, execution_time

        except TimeoutError:
            execution_time = time.time() - start_time
            raise TimeoutError(f"Subtask exceeded timeout of {timeout}s")
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
        while not self.agent_pool.empty():
            try:
                self.agent_pool.get_nowait()
            except queue.Empty:
                break
        logger.info("Sub-agent orchestrator shutdown complete")

    def get_current_progress(self) -> dict[str, Any]:
        """Get current execution progress."""
        # TODO: Implement real-time progress tracking
        return {}

    def adjust_worker_count(self, adjustment: int):
        """
        Adjust the number of workers dynamically during execution.
        New workers are assigned LLMs continuing the round-robin sequence.

        Args:
            adjustment: Number of workers to add (positive) or remove (negative)
        """
        new_count = self.dynamic_controller.adjust_worker_count(
            self.worker_count, adjustment
        )

        if new_count > self.worker_count:
            # Add new workers with continued round-robin LLM assignment
            worker_llms = settings.active_worker_llms
            num_worker_llms = len(worker_llms)
            add_count = new_count - self.worker_count

            for _ in range(add_count):
                if num_worker_llms > 0:
                    llm_config = worker_llms[
                        self._llm_assignment_index % num_worker_llms
                    ]
                    self._llm_assignment_index += 1
                    agent = self.agent_factory(llm_config=llm_config)
                else:
                    agent = self.agent_factory()
                self.agent_pool.put(agent)

            logger.info(f"Added {add_count} new sub-agents to pool")
        elif new_count < self.worker_count:
            # Remove excess workers
            remove_count = self.worker_count - new_count
            for _ in range(remove_count):
                if not self.agent_pool.empty():
                    try:
                        self.agent_pool.get_nowait()
                    except queue.Empty:
                        pass
            logger.info(f"Removed {remove_count} sub-agents from pool")

        self.worker_count = new_count
