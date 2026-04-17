"""
Sub-Agent Orchestrator module for parallel execution.
Manages sub-agent pool, task distribution, lifecycle, and execution guarantees.
"""

from __future__ import annotations

import asyncio
import io
import signal
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from loguru import logger
from pydantic_ai import Agent, Tool
from rich.console import Console

from codebase_rag.config import ModelConfig, settings
from codebase_rag.services.graph_service import MemgraphIngestor
from codebase_rag.services.llm import (
    CypherGenerator,
    create_rag_orchestrator_with_config,
)
from codebase_rag.shared.query_router import QueryMode, QueryRouter
from codebase_rag.tools.code_retrieval import CodeRetriever, create_code_retrieval_tool
from codebase_rag.tools.codebase_query import create_query_tool
from codebase_rag.tools.directory_lister import (
    DirectoryLister,
    create_directory_lister_tool,
)
from codebase_rag.tools.document_analyzer import (
    DocumentAnalyzer,
    create_document_analyzer_tool,
)
from codebase_rag.tools.document_query import (
    create_query_both_graphs_tool,
    create_query_document_graph_tool,
)
from codebase_rag.tools.document_validation import (
    create_validate_code_against_spec_tool,
    create_validate_doc_against_code_tool,
)
from codebase_rag.tools.file_reader import FileReader, create_file_reader_tool
from codebase_rag.tools.graph_query import create_graph_query_tool
from codebase_rag.tools.semantic_search import (
    create_get_function_source_tool,
    create_semantic_search_tool,
)

from .dynamic_concurrency_controller import DynamicConcurrencyController
from .investigation_tracker import InvestigationState
from .result_aggregator import ResultAggregator
from .sufficiency_analyzer import analyze_requirements
from .sufficiency_gatekeeper import (
    SubtaskResult,
    SufficiencyMetadata,
    evaluate_parallel_worker_sufficiency,
)

READ_ONLY_SUBAGENT_PROMPT = """
You are a read-only parallel analysis worker for a codebase RAG system.

Rules:
- Investigate only the scope described in the user request and any literal file path attached to it.
- Use only the tools you were given.
- Do not claim to have modified files, executed shell commands, or applied fixes.
- Return evidence-based findings only. If a tool fails or no evidence is available, say so plainly.
- Summaries must stay scoped to the assigned subtask rather than the whole repository.
""".strip()


class ReadOnlySubAgent:
    def __init__(
        self,
        repo_path: str,
        llm_config: ModelConfig,
        enable_document_graph: bool = False,
        query_mode: QueryMode = QueryMode.CODE_ONLY,
        doc_workspace: str = "default",
        worker_index: int = 0,
    ):
        self.repo_path = repo_path
        self.llm_config = llm_config
        self.allow_write = False
        self.enable_document_graph = enable_document_graph
        self.query_mode = query_mode
        self.doc_workspace = doc_workspace
        self._worker_index = worker_index
        self.agent: Agent | None = None
        self.code_graph: MemgraphIngestor | None = None
        self.doc_graph: MemgraphIngestor | None = None
        self.query_router: QueryRouter | None = None
        self.cypher_generator: CypherGenerator | None = None
        self.console = Console(file=io.StringIO(), width=120, force_terminal=False)

    def _initialize(self) -> None:
        if self.agent is not None:
            return

        try:
            self.code_graph = MemgraphIngestor(
                host=settings.MEMGRAPH_HOST,
                port=settings.MEMGRAPH_PORT,
                batch_size=1,
                username=settings.MEMGRAPH_USERNAME,
                password=settings.MEMGRAPH_PASSWORD,
            )
            self.code_graph.__enter__()

            if self.enable_document_graph:
                self.doc_graph = MemgraphIngestor(
                    host=settings.DOC_MEMGRAPH_HOST,
                    port=settings.DOC_MEMGRAPH_PORT,
                    batch_size=1,
                    username=settings.DOC_MEMGRAPH_USERNAME,
                    password=settings.DOC_MEMGRAPH_PASSWORD,
                )
                self.doc_graph.__enter__()
                self.query_router = QueryRouter(
                    code_graph=self.code_graph,
                    doc_graph=self.doc_graph,
                )
                self.query_router.current_mode = self.query_mode

            self.cypher_generator = CypherGenerator()
            tools = self._build_tools()
            system_prompt = (
                READ_ONLY_SUBAGENT_PROMPT
                + "\n\nUse the available graph, file, document, and search tools to gather evidence before answering."
            )
            self.agent = create_rag_orchestrator_with_config(
                self.llm_config,
                tools,
                system_prompt=system_prompt,
                output_type=str,
            )
        except Exception:
            self.shutdown()
            raise

    def _build_tools(self) -> list[Tool]:
        if self.code_graph is None or self.cypher_generator is None:
            raise RuntimeError("Parallel sub-agent dependencies are not initialized")

        code_retriever = CodeRetriever(
            project_root=self.repo_path, ingestor=self.code_graph
        )
        file_reader = FileReader(project_root=self.repo_path)
        directory_lister = DirectoryLister(project_root=self.repo_path)
        document_analyzer = DocumentAnalyzer(
            project_root=self.repo_path,
            doc_graph=self.doc_graph,
        )

        tools: list[Tool] = [
            create_query_tool(self.code_graph, self.cypher_generator, self.console),
            create_code_retrieval_tool(code_retriever),
            create_file_reader_tool(file_reader),
            create_directory_lister_tool(directory_lister),
            *create_document_analyzer_tool(
                document_analyzer,
                enable_graph_queries=self.doc_graph is not None,
                workspace=self.doc_workspace,
            ),
            create_semantic_search_tool(),
            create_get_function_source_tool(),
        ]

        if self.query_router is not None:
            tools.extend(
                [
                    create_query_document_graph_tool(self.query_router),
                    create_query_both_graphs_tool(self.query_router),
                    create_validate_code_against_spec_tool(self.query_router),
                    create_validate_doc_against_code_tool(self.query_router),
                    create_graph_query_tool(self.query_router),
                ]
            )

        return tools

    def execute(self, subtask: dict[str, Any]) -> SubtaskResult:
        self._initialize()
        if self.agent is None:
            raise RuntimeError("Parallel sub-agent was not initialized")

        subtask_reqs = analyze_requirements(subtask.get("prompt", ""))
        state = InvestigationState.from_parallel_worker(worker_id=self._worker_index)

        response = asyncio.run(
            self.agent.run(subtask.get("prompt", ""), message_history=[])
        )

        if hasattr(response, "new_messages"):
            for msg in response.new_messages():
                for part in getattr(msg, "parts", []):
                    tool_name = getattr(part, "tool_name", None)
                    if tool_name is not None:
                        state.record_tool(tool_name, "")

        is_sufficient, warning = evaluate_parallel_worker_sufficiency(
            state, subtask_reqs, worker_id=self._worker_index
        )

        output = response.output if isinstance(response.output, str) else str(response.output)

        return SubtaskResult(
            content=output,
            sufficiency=SufficiencyMetadata(
                passed=is_sufficient,
                warning=warning,
                tools_used=list(state.tools_used),
                files_read=state.files_read,
            ),
        )

    def reset(self) -> None:
        if self.query_router is not None:
            self.query_router.current_mode = self.query_mode

    def shutdown(self) -> None:
        if self.doc_graph is not None:
            self.doc_graph.__exit__(None, None, None)
            self.doc_graph = None
        if self.code_graph is not None:
            self.code_graph.__exit__(None, None, None)
            self.code_graph = None
        self.query_router = None
        self.cypher_generator = None
        self.agent = None


class SubAgentWorker:
    """
    Worker wrapper for sub-agent instances.
    Manages worker lifecycle, task execution, and state.
    """

    def __init__(self, worker_id: str, agent: Any = None):
        self.worker_id = worker_id
        self.agent = agent
        self.busy = False
        self.last_task_time = 0.0

    def execute(self, subtask: dict[str, Any]) -> tuple[Any, float]:
        """Execute a subtask using this worker's agent."""
        self.busy = True
        try:
            agent = self.agent
            if agent is None or not hasattr(agent, "execute") or not callable(agent.execute):
                raise RuntimeError("Sub-agent worker has no executable agent")
            start_time = time.time()
            result = agent.execute(subtask)
            execution_time = time.time() - start_time
            self.last_task_time = time.time()
            return result, execution_time
        finally:
            self.busy = False

    def reset_agent(self, new_agent: Any) -> None:
        """Reset the agent instance for this worker."""
        self.agent = new_agent

    def shutdown(self) -> None:
        """Shutdown the worker and clean up resources."""
        self.busy = False
        if hasattr(self.agent, "shutdown") and callable(self.agent.shutdown):
            self.agent.shutdown()
        self.agent = None


class SubAgentOrchestrator:
    """
    Orchestrates parallel execution of sub-agent tasks.
    Manages full lifecycle of sub-agents, execution guarantees, and cleanup.
    """

    def __init__(
        self,
        worker_count: int | None = None,
        agent_factory: Callable | None = None,
        scheduling_strategy: str = "round-robin",
        repo_path: str | None = None,
        enable_document_graph: bool = False,
        query_mode: QueryMode = QueryMode.CODE_ONLY,
        doc_workspace: str = "default",
    ):
        self.dynamic_controller = DynamicConcurrencyController()
        self.requested_worker_count = worker_count
        self.worker_count = self.dynamic_controller.get_effective_worker_count(
            worker_count
        )
        self.repo_path = repo_path or settings.TARGET_REPO_PATH
        self.enable_document_graph = enable_document_graph
        self.query_mode = query_mode
        self.doc_workspace = doc_workspace
        self.agent_factory = agent_factory or self._default_agent_factory
        self.workers: list[SubAgentWorker] = []
        self.running = False
        self._shutdown_called = False
        self._llm_assignment_index = 0

        valid_strategies = {"fifo", "round-robin"}
        scheduling_strategy = scheduling_strategy.lower()
        if scheduling_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid scheduling strategy: {scheduling_strategy}. Valid options: {', '.join(valid_strategies)}"
            )
        self.scheduling_strategy = scheduling_strategy

        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT]:
            try:
                signal.signal(sig, self._handle_shutdown)
            except ValueError:
                pass

    def _default_agent_factory(
        self, llm_config: ModelConfig | None = None, worker_index: int = 0
    ) -> Any:
        worker_llm_config = llm_config or settings.active_orchestrator_config
        return ReadOnlySubAgent(
            repo_path=self.repo_path,
            llm_config=worker_llm_config,
            enable_document_graph=self.enable_document_graph,
            query_mode=self.query_mode,
            doc_workspace=self.doc_workspace,
            worker_index=worker_index,
        )

    def initialize_agents(self) -> None:
        logger.info(f"Initializing {self.worker_count} sub-agents")
        worker_llms = settings.active_worker_llms
        num_worker_llms = len(worker_llms)

        for worker in self.workers[self.worker_count :]:
            worker.shutdown()
        self.workers = self.workers[: self.worker_count]

        for index in range(len(self.workers), self.worker_count):
            if num_worker_llms > 0:
                llm_config = worker_llms[self._llm_assignment_index % num_worker_llms]
                self._llm_assignment_index += 1
                agent = self.agent_factory(llm_config=llm_config, worker_index=index)
            else:
                agent = self.agent_factory(worker_index=index)
            self.workers.append(
                SubAgentWorker(worker_id=self._build_worker_id(index), agent=agent)
            )

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
            retry_attempts: Number of retries for failed tasks
            dry_run: If True, emit execution plan metadata without running sub-agents
        """
        if len(subtasks) > settings.CGR_PARALLEL_MAX_QUEUE_SIZE:
            raise ValueError(
                f"Subtask queue size {len(subtasks)} exceeds configured maximum of {settings.CGR_PARALLEL_MAX_QUEUE_SIZE}"
            )

        retry_attempts = retry_attempts or settings.CGR_SUBAGENT_RETRY_ATTEMPTS
        result_aggregator = result_aggregator or ResultAggregator()
        result_aggregator.set_total_subtasks(len(subtasks))
        result_aggregator.metadata["dry_run"] = dry_run
        result_aggregator.metadata["scheduling_strategy"] = self.scheduling_strategy

        self.worker_count = self.dynamic_controller.get_effective_worker_count(
            self.requested_worker_count, len(subtasks)
        )
        result_aggregator.metadata["worker_count"] = self.worker_count

        if dry_run:
            start_time = time.time()
            logger.info(
                f"Dry run: Would execute {len(subtasks)} subtasks with {self.worker_count} workers (scheduling: {self.scheduling_strategy})"
            )
            for index, subtask in enumerate(subtasks):
                worker_metadata = self._build_planned_worker_metadata(
                    index % max(self.worker_count, 1)
                )
                result_aggregator.add_result(
                    subtask,
                    {
                        "status": "planned",
                        "target": subtask.get("relative_path")
                        or subtask.get("target_entity")
                        or subtask.get("id"),
                    },
                    execution_time=0.0,
                    status="planned",
                    worker_metadata={**worker_metadata, "status": "planned"},
                )
            result_aggregator.set_total_execution_time(time.time() - start_time)
            logger.info(
                f"Dry run completed in {result_aggregator.metadata['total_execution_time']:.2f}s"
            )
            return result_aggregator

        self.initialize_agents()

        self.running = True
        start_time = time.time()
        logger.info(
            f"Starting parallel execution of {len(subtasks)} subtasks with {self.worker_count} workers (scheduling: {self.scheduling_strategy})"
        )

        remaining_tasks = subtasks.copy()
        retry_counts = {subtask["id"]: 0 for subtask in subtasks}
        active_futures: dict[Any, tuple[dict[str, Any], SubAgentWorker, float]] = {}

        try:
            with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
                while (
                    (remaining_tasks or active_futures)
                    and self.running
                    and not self._shutdown_called
                ):
                    while remaining_tasks:
                        worker = self._get_available_worker()
                        if worker is None:
                            break
                        subtask = remaining_tasks.pop(0)
                        worker.busy = True
                        future = executor.submit(self._execute_subtask, worker, subtask)
                        active_futures[future] = (subtask, worker, time.time())

                    if not active_futures:
                        continue

                    done, _ = wait(active_futures, return_when=FIRST_COMPLETED)

                    for future in done:
                        subtask, worker, started_at = active_futures.pop(future)
                        execution_time = time.time() - started_at
                        worker_metadata = self._build_worker_metadata(
                            worker,
                            retry_counts.get(subtask["id"], 0),
                        )

                        if hasattr(worker.agent, "reset") and callable(
                            worker.agent.reset
                        ):
                            worker.agent.reset()

                        try:
                            result, execution_time = future.result()
                            result_aggregator.add_result(
                                subtask,
                                result,
                                execution_time=execution_time,
                                worker_metadata={
                                    **worker_metadata,
                                    "execution_time": execution_time,
                                    "status": "completed",
                                },
                            )
                        except Exception as e:
                            error_msg = str(e)
                            retry_count = retry_counts.get(subtask["id"], 0)

                            if retry_count < retry_attempts:
                                retry_counts[subtask["id"]] = retry_count + 1
                                logger.warning(
                                    f"Subtask {subtask['id']} failed (attempt {retry_count + 1}/{retry_attempts + 1}): {error_msg}. Retrying..."
                                )
                                remaining_tasks.insert(0, subtask)
                            else:
                                result_aggregator.add_error(
                                    subtask,
                                    error_msg,
                                    execution_time=execution_time,
                                    worker_metadata={
                                        **worker_metadata,
                                        "execution_time": execution_time,
                                        "status": "failed",
                                    },
                                )
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
        self, worker: SubAgentWorker, subtask: dict[str, Any]
    ) -> tuple[Any, float]:
        """
        Execute a single subtask with a given worker.

        Args:
            worker: Sub-agent worker wrapper to use
            subtask: Subtask to execute

        Returns:
            Tuple of (result, execution_time_seconds)
        """
        start_time = time.time()
        timeout = settings.CGR_SUBAGENT_TIMEOUT
        task_executor = ThreadPoolExecutor(max_workers=1)
        future = task_executor.submit(worker.execute, subtask)

        try:
            result, _ = future.result(timeout=timeout)
            execution_time = time.time() - start_time
            return result, execution_time

        except TimeoutError as e:
            future.cancel()
            raise TimeoutError(f"Subtask exceeded timeout of {timeout}s") from e
        finally:
            task_executor.shutdown(wait=False, cancel_futures=True)

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

        for worker in self.workers:
            worker.shutdown()
        self.workers = []
        logger.info("Sub-agent orchestrator shutdown complete")

    def get_current_progress(self) -> dict[str, Any]:
        """Get current execution progress."""
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
        self.requested_worker_count = new_count
        self.worker_count = new_count

    def _build_worker_id(self, index: int) -> str:
        if index < self.dynamic_controller.default_workers:
            return f"base-{index}"
        return f"burst-{index - self.dynamic_controller.default_workers}"

    def _get_available_worker(self) -> SubAgentWorker | None:
        available_workers = [worker for worker in self.workers if not worker.busy]
        if not available_workers:
            return None
        if self.scheduling_strategy == "fifo":
            return available_workers[0]
        return self.dynamic_controller.get_next_worker_round_robin(available_workers)

    def _build_worker_metadata(
        self, worker: SubAgentWorker, retry_count: int
    ) -> dict[str, Any]:
        llm_config = getattr(worker.agent, "llm_config", None)
        if llm_config is None:
            llm_config = settings.active_orchestrator_config
        return {
            "worker_id": worker.worker_id,
            "provider": getattr(llm_config, "provider", None),
            "model_id": getattr(llm_config, "model_id", None),
            "retry_count": retry_count,
        }

    def _build_planned_worker_metadata(self, worker_index: int) -> dict[str, Any]:
        worker_llms = settings.active_worker_llms
        llm_config = (
            worker_llms[worker_index % len(worker_llms)]
            if worker_llms
            else settings.active_orchestrator_config
        )
        return {
            "worker_id": self._build_worker_id(worker_index),
            "provider": getattr(llm_config, "provider", None),
            "model_id": getattr(llm_config, "model_id", None),
            "retry_count": 0,
        }
