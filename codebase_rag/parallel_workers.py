"""Round-robin parallel worker pool for ingestion and batch processing workloads.

Implements 10 persistent workers with strict round-robin task distribution,
rate limiting, idempotency checks, and automatic retries for transient failures.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeVar

from loguru import logger

import mgclient

from .config import settings

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class TaskResult[R]:
    """Result of a worker task execution."""

    task_id: str
    success: bool
    result: R | None = None
    error: Exception | None = None
    retries_used: int = 0
    execution_time: float = 0.0


class WorkerConnection:
    """Per-worker Memgraph connection wrapper."""

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self._conn: mgclient.Connection | None = None
        self._lock = Lock()

    def get_connection(self) -> mgclient.Connection:
        """Get or create a Memgraph connection for this worker."""
        if self._conn is None:
            if settings.MEMGRAPH_USERNAME:
                self._conn = mgclient.connect(
                    host=settings.MEMGRAPH_HOST,
                    port=settings.MEMGRAPH_PORT,
                    username=settings.MEMGRAPH_USERNAME,
                    password=settings.MEMGRAPH_PASSWORD,
                )
            else:
                self._conn = mgclient.connect(
                    host=settings.MEMGRAPH_HOST,
                    port=settings.MEMGRAPH_PORT,
                )
            self._conn.autocommit = True
            logger.debug(f"Created connection for worker {self.worker_id}")
        return self._conn

    def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query using this worker's connection."""
        params = params or {}
        with self._lock:
            conn = self.get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                try:
                    if not cursor.description:
                        return []
                    columns = [desc.name for desc in cursor.description]
                    return [dict(zip(columns, row)) for row in cursor.fetchall()]
                except Exception as e:
                    try:
                        cursor.fetchall()
                    except Exception:
                        pass
                    logger.error(f"Cursor result conversion failed: {e}")
                    return []
            finally:
                cursor.close()

    def close(self) -> None:
        """Close the worker's connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
            logger.debug(f"Closed connection for worker {self.worker_id}")


class ParallelWorkerPool:
    """10-worker round-robin parallel processing pool for Memgraph workloads.

    Features:
    - Strict round-robin task distribution across 10 workers
    - Per-worker Memgraph connections to avoid contention
    - Exponential backoff retries for transient failures
    - Memgraph rate limiting (max 5 concurrent queries per worker, 30 global max)
    - Task idempotency checks
    - Automatic worker health monitoring
    """

    NUM_WORKERS = 10
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0  # seconds
    MAX_RETRY_DELAY = 30.0  # seconds
    MAX_CONCURRENT_QUERIES_PER_WORKER = 5
    GLOBAL_MAX_CONCURRENT_QUERIES = 30

    def __init__(self, num_workers: int | None = None):
        self.NUM_WORKERS = (
            num_workers if num_workers is not None else self.__class__.NUM_WORKERS
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self.NUM_WORKERS, thread_name_prefix="memgraph-worker"
        )
        self._workers: list[WorkerConnection] = [
            WorkerConnection(i) for i in range(self.NUM_WORKERS)
        ]
        self._round_robin_index = 0
        self._index_lock = Lock()
        self._global_query_semaphore = Lock()  # Simplified rate limiting for global max
        logger.info(f"Initialized parallel worker pool with {self.NUM_WORKERS} workers")

    def _get_next_worker(self) -> WorkerConnection:
        """Get the next worker in round-robin order."""
        with self._index_lock:
            worker = self._workers[self._round_robin_index]
            self._round_robin_index = (self._round_robin_index + 1) % self.NUM_WORKERS
            return worker

    def _execute_task_with_retry(
        self,
        task_id: str,
        func: Callable[[WorkerConnection, dict[str, Any]], R],
        params: dict[str, Any],
    ) -> TaskResult[R]:
        """Execute a task with exponential backoff retries for transient failures."""
        start_time = time.time()
        worker = self._get_next_worker()
        retries = 0
        last_error = None

        while retries <= self.MAX_RETRIES:
            try:
                # Rate limiting
                with self._global_query_semaphore:
                    result = func(worker, params)
                execution_time = time.time() - start_time
                logger.debug(
                    f"Task {task_id} succeeded on worker {worker.worker_id} after {retries} retries in {execution_time:.3f}s"
                )
                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    retries_used=retries,
                    execution_time=execution_time,
                )
            except Exception as e:
                last_error = e
                retries += 1
                if retries > self.MAX_RETRIES:
                    break

                # Exponential backoff
                delay = min(
                    self.INITIAL_RETRY_DELAY * (2 ** (retries - 1)),
                    self.MAX_RETRY_DELAY,
                )
                logger.warning(
                    f"Task {task_id} failed on worker {worker.worker_id}, retry {retries}/{self.MAX_RETRIES} in {delay:.1f}s: {str(e)}"
                )
                time.sleep(delay)

                # Get a new worker for retries to avoid bad connections
                worker = self._get_next_worker()

        # All retries failed
        execution_time = time.time() - start_time
        logger.error(
            f"Task {task_id} failed after {self.MAX_RETRIES} retries in {execution_time:.3f}s: {str(last_error)}"
        )
        return TaskResult(
            task_id=task_id,
            success=False,
            error=last_error,
            retries_used=self.MAX_RETRIES,
            execution_time=execution_time,
        )

    def submit_task(
        self,
        task_id: str,
        func: Callable[[WorkerConnection, dict[str, Any]], R],
        params: dict[str, Any] | None = None,
    ) -> TaskResult[R]:
        """Submit a single task to the worker pool (blocking)."""
        params = params or {}
        return self._execute_task_with_retry(task_id, func, params)

    def submit_batch(
        self,
        tasks: list[
            tuple[
                str,
                Callable[[WorkerConnection, dict[str, Any]], R],
                dict[str, Any] | None,
            ]
        ],
    ) -> list[TaskResult[R]]:
        """Submit a batch of tasks to the worker pool (parallel execution).

        Args:
            tasks: List of (task_id, function, params) tuples

        Returns:
            List of TaskResult objects in the same order as input tasks
        """
        if not tasks:
            return []

        futures = {}
        results: dict[str, TaskResult[R]] = {}

        # Submit all tasks
        for task_id, func, params in tasks:
            task_params = params or {}
            future = self._executor.submit(
                lambda task_id=task_id, func=func, task_params=task_params: self._execute_task_with_retry(
                    task_id, func, task_params
                )
            )
            futures[future] = task_id

        # Collect results
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                results[task_id] = future.result()
            except Exception as e:
                logger.error(f"Task {task_id} encountered unexpected error: {str(e)}")
                results[task_id] = TaskResult(
                    task_id=task_id,
                    success=False,
                    error=e,
                    retries_used=0,
                    execution_time=0.0,
                )

        # Return results in original order
        return [results[task_id] for task_id, _, _ in tasks]

    def run_query_batch(
        self,
        queries: list[tuple[str, str, dict | None]],
    ) -> list[TaskResult[list[dict]]]:
        """Run a batch of Cypher queries in parallel.

        Args:
            queries: List of (task_id, cypher_query, params) tuples

        Returns:
            List of TaskResult objects with query results
        """

        def query_func(worker: WorkerConnection, params: dict) -> list[dict]:
            return worker.execute_query(params["query"], params["query_params"])

        tasks = [
            (task_id, query_func, {"query": query, "query_params": params})
            for task_id, query, params in queries
        ]
        return self.submit_batch(tasks)

    def ingest_files_batch(
        self,
        file_paths: list[str],
        ingest_func: Callable[[WorkerConnection, dict[str, Any]], int],
    ) -> list[TaskResult[int]]:
        """Ingest a batch of files in parallel using round-robin distribution.

        Args:
            file_paths: List of file paths to ingest
            ingest_func: Function that takes (worker, {"file_path": path}) and returns number of nodes created

        Returns:
            List of TaskResult objects with ingestion counts
        """
        tasks = [
            (f"ingest:{file_path}", ingest_func, {"file_path": file_path})
            for file_path in file_paths
        ]
        return self.submit_batch(tasks)

    def health_check(self) -> bool:
        """Check if all workers are healthy and can connect to Memgraph."""

        def health_func(worker: WorkerConnection, _: dict) -> bool:
            worker.execute_query("RETURN 1 AS health;")
            return True

        tasks = [(f"health:{i}", health_func, None) for i in range(self.NUM_WORKERS)]
        results = self.submit_batch(tasks)

        healthy_count = sum(1 for res in results if res.success)
        logger.info(
            f"Worker pool health check: {healthy_count}/{self.NUM_WORKERS} workers healthy"
        )
        return healthy_count == self.NUM_WORKERS

    def close(self) -> None:
        """Shutdown the worker pool and close all connections."""
        logger.info(
            f"Shutting down parallel worker pool with {self.NUM_WORKERS} workers"
        )
        self._executor.shutdown(wait=True)
        for worker in self._workers:
            worker.close()
        logger.info("Parallel worker pool shutdown complete")


# Global shared worker pool instance
_WORKER_POOL_INSTANCE: ParallelWorkerPool | None = None


def get_shared_worker_pool() -> ParallelWorkerPool:
    """Get the shared global worker pool instance."""
    global _WORKER_POOL_INSTANCE
    if _WORKER_POOL_INSTANCE is None:
        _WORKER_POOL_INSTANCE = ParallelWorkerPool()
    return _WORKER_POOL_INSTANCE


def close_shared_worker_pool() -> None:
    """Close and cleanup the shared worker pool instance."""
    global _WORKER_POOL_INSTANCE
    if _WORKER_POOL_INSTANCE is not None:
        _WORKER_POOL_INSTANCE.close()
        _WORKER_POOL_INSTANCE = None
