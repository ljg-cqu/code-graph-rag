from __future__ import annotations

import socket
import sys
import threading
import time
import types
from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime

from loguru import logger

import mgclient
from codebase_rag.config import settings
from codebase_rag.types_defs import CursorProtocol, ResultValue

from .. import exceptions as ex
from .. import logs as ls
from ..constants import (
    ERR_SUBSTR_ALREADY_EXISTS,
    ERR_SUBSTR_CONSTRAINT,
    KEY_CREATED,
    KEY_FROM_VAL,
    KEY_NAME,
    KEY_PROJECT_NAME,
    KEY_PROPS,
    KEY_QUALIFIED_NAME,
    KEY_TO_VAL,
    NODE_UNIQUE_CONSTRAINTS,
    REL_TYPE_CALLS,
    REL_TYPE_IMPORTS,
    NodeLabel,
)
from ..cypher_queries import (
    CYPHER_DELETE_ALL,
    CYPHER_DELETE_PROJECT,
    CYPHER_EXPORT_NODES,
    CYPHER_EXPORT_RELATIONSHIPS,
    CYPHER_LIST_PROJECTS,
    build_constraint_query,
    build_create_node_query,
    build_create_relationship_query,
    build_index_query,
    build_merge_node_query,
    build_merge_relationship_query,
    wrap_with_unwind,
)
from ..types_defs import (
    BatchParams,
    BatchWrapper,
    GraphData,
    GraphMetadata,
    NodeBatchRow,
    PropertyDict,
    PropertyValue,
    RelBatchRow,
    ResultRow,
)
from ..utils.shutdown_manager import shutdown_manager
from .error_guidance import (
    ErrorContext,
    LLMErrorGuidance,
    UserExpertiseLevel,
)
from .failure_classifier import (
    _KNOWN_THIRD_PARTY,
    FailureType,
    classify_memgraph_failure,
    is_stdlib_module,
)


@dataclass
class ConnectionRetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 10.0
    exponential_base: float = 2.0
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
    )


class MemgraphIngestor:
    _TRANSIENT_ERROR_MARKERS = (
        "broken pipe",
        "bad session",
        "connection reset",
        "connection aborted",
        "connection refused",
        "connection closed",
        "server closed the connection",
        "network is unreachable",
        "temporarily unavailable",
        "socket",
        "transport",
        "failed to send chunk data",
        "failed to send message end marker",
    )

    _REL_BUFFER_WARNING_THRESHOLD = 10000

    __slots__ = (
        "_conn_lock",
        "_executor",
        "_host",
        "_port",
        "_username",
        "_password",
        "_use_merge",
        "_rel_count",
        "_rel_groups",
        "_dynamic_algorithms_supported",
        "_connection_timeout",
        "_last_health_check",
        "_health_check_interval",
        "batch_size",
        "conn",
        "node_buffer",
    )

    def __init__(
        self,
        host: str,
        port: int,
        batch_size: int = 1000,
        username: str | None = None,
        password: str | None = None,
        use_merge: bool = True,
        connection_timeout: int | None = None,
    ):
        self._host = host
        self._port = port
        self._username = username.strip() if username and username.strip() else None
        self._password = password.strip() if password and password.strip() else None
        if (self._username is None) != (self._password is None):
            raise ValueError(ex.AUTH_INCOMPLETE)

        # Security check: Block default credentials for non-localhost connections
        if self._host not in ("localhost", "127.0.0.1", "::1"):
            if (
                self._username == "admin"
                and self._password == "REPLACE_WITH_SECURE_PASSWORD_IN_PRODUCTION"
            ):
                raise PermissionError(
                    "SECURITY VIOLATION: Default Memgraph credentials cannot be used "
                    "with non-localhost connections. Please set a secure password "
                    "in your environment configuration."
                )
        if batch_size < 1:
            raise ValueError(ex.BATCH_SIZE)
        self.batch_size = batch_size
        self._use_merge = use_merge
        self._conn_lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self.conn: mgclient.Connection | None = None
        self.node_buffer: list[tuple[str, dict[str, PropertyValue]]] = []
        self._rel_count = 0
        self._rel_groups: defaultdict[
            tuple[str, str, str, str, str], list[RelBatchRow]
        ] = defaultdict(list)
        self._dynamic_algorithms_supported: bool | None = None
        self._connection_timeout = connection_timeout
        self._last_health_check = time.time()
        self._health_check_interval = 30.0  # Check connection health every 30 seconds

    @property
    def dynamic_algorithms_enabled(self) -> bool:
        """
        Get if dynamic algorithms should be used.
        Auto-enables if running on Memgraph Enterprise, unless explicitly disabled via config.
        """
        # If explicitly disabled in config, never use
        if settings.MEMGRAPH_USE_DYNAMIC_ALGORITHMS is False:
            return False
        # If explicitly enabled in config, always use
        if settings.MEMGRAPH_USE_DYNAMIC_ALGORITHMS is True:
            return True
        # Auto-enable if supported (Enterprise detected)
        return bool(self._dynamic_algorithms_supported)

    def _detect_dynamic_algorithm_support(self) -> bool:
        """
        Detect if Memgraph supports dynamic incremental algorithms (Enterprise feature).
        Returns True if Enterprise edition, False otherwise.
        """
        if not self.conn:
            return False

        # Silent execution without error logging for expected failure
        try:
            # Try to use a dynamic algorithm function to check support
            # This will fail on Community edition with "Function not found" error
            with self._get_cursor() as cursor:
                cursor.execute(
                    "RETURN dynamic_graph_update_is_supported() AS supported LIMIT 1"
                )
                return True
        except Exception:
            # Fallback: check version string for enterprise
            try:
                with self._get_cursor() as cursor:
                    cursor.execute("SHOW VERSION AS version")
                    results = self._cursor_to_results(cursor)
                    if (
                        results
                        and "enterprise" in str(results[0].get("version", "")).lower()
                    ):
                        return True
            except Exception:
                pass
            return False

    def __enter__(self) -> MemgraphIngestor:
        logger.info(ls.MG_CONNECTING.format(host=self._host, port=self._port))
        self.conn = self._create_connection_with_timeout()  # <-- CHANGED
        self._executor = ThreadPoolExecutor(max_workers=settings.FLUSH_THREAD_POOL_SIZE)

        # Auto-detect Enterprise edition and dynamic algorithm support
        self._dynamic_algorithms_supported = self._detect_dynamic_algorithm_support()
        if self._dynamic_algorithms_supported:
            if settings.MEMGRAPH_USE_DYNAMIC_ALGORITHMS is False:
                logger.info(
                    "Memgraph Enterprise detected, but dynamic algorithms are explicitly disabled in config"
                )
            else:
                logger.info(
                    "Memgraph Enterprise detected: dynamic incremental algorithms automatically enabled"
                )
        else:
            logger.debug(
                "Memgraph Community detected: using full algorithm recalculations after updates"
            )

        logger.info(ls.MG_CONNECTED)
        # Register cleanup handler for graceful shutdown
        shutdown_manager.register_handler(self._cleanup_on_shutdown, priority=20)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        connection_healthy = self.conn is not None
        if connection_healthy:
            try:
                self.conn.cursor().execute("RETURN 1")
            except Exception:
                connection_healthy = False

        try:
            if exc_type and connection_healthy:
                logger.exception(ls.MG_FLUSH_ON_EXCEPTION)
                try:
                    self.flush_all()
                except Exception as flush_err:
                    logger.error(ls.MG_FLUSH_ON_EXCEPTION_FAILED.format(error=flush_err))
            elif connection_healthy:
                self.flush_all()
        finally:
            if self._executor:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
                self.conn = None
                logger.info(ls.MG_DISCONNECTED)
            shutdown_manager.unregister_handler(self._cleanup_on_shutdown)

    def _cleanup_on_shutdown(self) -> None:
        """Cleanup resources on shutdown."""
        if not self.conn:
            return
        try:
            self.flush_all()
        except Exception as e:
            logger.error(f"Failed to flush during shutdown: {e}")
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None
            logger.info(ls.MG_DISCONNECTED)

    async def __aenter__(self) -> MemgraphIngestor:
        import asyncio

        return await asyncio.to_thread(self.__enter__)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        import asyncio

        await asyncio.to_thread(self.__exit__, exc_type, exc_val, exc_tb)

    @contextmanager
    def _get_cursor(self) -> Generator[CursorProtocol, None, None]:
        self._ensure_connection()
        if not self.conn:
            raise ConnectionError(ex.CONN)
        with self._conn_lock:
            cursor: CursorProtocol | None = None
            try:
                cursor = self.conn.cursor()
                yield cursor
            finally:
                if cursor:
                    cursor.close()

    def _cursor_to_results(self, cursor: CursorProtocol) -> list[ResultRow]:
        try:
            if not cursor.description:
                return []
            column_names = [desc.name for desc in cursor.description]
            return [
                dict[str, ResultValue](zip(column_names, row)) for row in cursor.fetchall()
            ]
        except Exception as e:
            # Consume any pending results to clear exception state
            try:
                cursor.fetchall()
            except Exception:
                pass
            logger.error(f"Cursor result conversion failed: {e}")
            return []

    def _check_connection_health(self) -> bool:
        """Check if the current connection is still healthy.

        Returns:
            True if connection is healthy, False otherwise.
        """
        if not self.conn:
            return False

        # Skip health check if we've checked recently (within interval)
        current_time = time.time()
        if current_time - self._last_health_check < self._health_check_interval:
            return True

        try:
            # Simple health check query — use cursor directly to avoid
            # recursion through _get_cursor() → _ensure_connection().
            cursor = self.conn.cursor()
            try:
                cursor.execute("RETURN 1")
                cursor.fetchall()
                self._last_health_check = current_time
                return True
            finally:
                cursor.close()
        except Exception:
            # Connection is unhealthy
            self._last_health_check = current_time
            return False

    def _ensure_connection(self) -> None:
        """Ensure we have a healthy connection, reconnecting if necessary."""
        if not self.conn:
            return
        if not self._check_connection_health():
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = self._create_connection_with_timeout()

    @classmethod
    def _is_retryable_memgraph_error(cls, error: Exception) -> bool:
        """Determine if a Memgraph error is retryable using failure classifier."""
        classification = classify_memgraph_failure(error)
        return classification.should_retry

    def _retry_delay_seconds(self, attempt: int) -> float:
        return settings.MEMGRAPH_RETRY_BASE_DELAY * attempt

    def _reset_shared_connection(self) -> None:
        current_conn = self.conn
        self.conn = None
        if current_conn is not None:
            try:
                current_conn.close()
            except Exception:
                pass
        self.conn = self._create_connection_with_timeout()

    def _should_retry_shared_connection_error(
        self,
        error: Exception,
        attempt: int,
        max_attempts: int,
    ) -> bool:
        from .failure_classifier import classify_memgraph_failure

        classification = classify_memgraph_failure(error)
        if not classification.should_retry:
            return False
        # Only retry known transient error types, not generic unknown errors
        if classification.failure_type not in (
            FailureType.TRANSIENT_NETWORK,
            FailureType.TRANSIENT_TIMEOUT,
            FailureType.RESOURCE_EXHAUSTION,
            FailureType.TRANSACTION_CONFLICT,
            FailureType.MGCLIENT_STATE_CORRUPTION,
        ):
            return False
        return attempt < min(max_attempts, classification.max_retries + 1)

    def _execute_query(
        self,
        query: str,
        params: dict[str, PropertyValue] | None = None,
    ) -> list[ResultRow]:
        from .failure_classifier import classify_memgraph_failure

        params = params or {}
        max_attempts = settings.MEMGRAPH_QUERY_MAX_RETRIES + 1
        for attempt in range(1, max_attempts + 1):
            try:
                if attempt > 1 and self.conn is None:
                    try:
                        self._reset_shared_connection()
                    except Exception as reconnect_error:
                        if self._should_retry_shared_connection_error(
                            reconnect_error, attempt, max_attempts
                        ):
                            logger.warning(
                                f"Transient Memgraph reconnect failure (attempt {attempt}/{max_attempts}), retrying: {reconnect_error}"
                            )
                            time.sleep(self._retry_delay_seconds(attempt))
                            continue
                        raise
                with self._get_cursor() as cursor:
                    if "embedding" in params:
                        embedding = params["embedding"]
                        expected_dim = settings.get_effective_vector_dim(
                            self._vector_graph_type()
                        )
                        if (
                            isinstance(embedding, list)
                            and len(embedding) != expected_dim
                        ):
                            raise ex.DimensionMismatchError(
                                existing_dim=len(embedding),
                                configured_dim=expected_dim,
                                message=f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}. "
                                f"Check your embedding model configuration or run `{self._vector_recreate_command()}` to fix.",
                            )
                    cursor.execute(query, params)
                    return self._cursor_to_results(cursor)
            except Exception as e:
                classification = classify_memgraph_failure(e)

                if classification.failure_type == FailureType.SYNTAX_ERROR:
                    logger.warning(f"Cypher syntax error, not retrying: {e}")
                    if (
                        ERR_SUBSTR_ALREADY_EXISTS not in str(e).lower()
                        and ERR_SUBSTR_CONSTRAINT not in str(e).lower()
                    ):
                        logger.error(ls.MG_CYPHER_ERROR.format(error=e))
                        logger.error(ls.MG_CYPHER_QUERY.format(query=query))
                        logger.error(ls.MG_CYPHER_PARAMS.format(params=params))
                    raise

                if classification.failure_type == FailureType.MISSING_PROCEDURE:
                    logger.warning(f"Procedure not found, not retrying: {e}")
                    raise

                if classification.failure_type in (
                    FailureType.AUTHENTICATION_FAILURE,
                    FailureType.PERMISSION_DENIED,
                ):
                    logger.error(f"Authentication/permission failure: {e}")
                    raise

                if self._should_retry_shared_connection_error(
                    e, attempt, max_attempts
                ):
                    logger.warning(
                        f"Retryable error (attempt {attempt}/{max_attempts}): {e}"
                    )
                    try:
                        self._reset_shared_connection()
                    except Exception as reconnect_error:
                        if self._should_retry_shared_connection_error(
                            reconnect_error, attempt, max_attempts
                        ):
                            logger.warning(
                                f"Transient Memgraph reconnect failure (attempt {attempt}/{max_attempts}), retrying: {reconnect_error}"
                            )
                            time.sleep(self._retry_delay_seconds(attempt))
                            continue
                        raise
                    time.sleep(self._retry_delay_seconds(attempt))
                    continue

                if (
                    ERR_SUBSTR_ALREADY_EXISTS not in str(e).lower()
                    and ERR_SUBSTR_CONSTRAINT not in str(e).lower()
                ):
                    logger.error(ls.MG_CYPHER_ERROR.format(error=e))
                    logger.error(ls.MG_CYPHER_QUERY.format(query=query))
                    logger.error(ls.MG_CYPHER_PARAMS.format(params=params))
                raise
        return []

    async def _execute_query_with_guidance(
        self,
        query: str,
        params: dict[str, PropertyValue] | None = None,
        operation_type: str = "cypher_query",
        llm_guidance: LLMErrorGuidance | None = None,
    ) -> list[ResultRow]:
        """Execute a Cypher query with LLM-guided error handling.

        Per LLM-First design:
        - Classification is deterministic (via classify_memgraph_failure)
        - User-facing guidance is LLM-generated (contextual, user-friendly)

        Args:
            query: Cypher query to execute
            params: Query parameters
            operation_type: Context for error messages (e.g., "cypher_query", "document_indexing")
            llm_guidance: Optional LLM guidance generator. If None, uses static fallback.

        Returns:
            Query results as list of ResultRow

        Raises:
            GraphQueryError: With LLM-generated user guidance on failure
        """
        params = params or {}
        try:
            # Use sync _execute_query for the actual query execution
            return self._execute_query(query, params)
        except Exception as e:
            # 1. Deterministic classification (fast, reliable)
            classification = classify_memgraph_failure(e)

            # 2. Build error context for LLM guidance
            context = ErrorContext(
                operation_type=operation_type,
                error_category=classification.failure_type,
                graph_type="code" if self._port == settings.MEMGRAPH_PORT else "document",
                embedding_provider=settings.EMBEDDING_PROVIDER,
                user_expertise=UserExpertiseLevel.INTERMEDIATE,
            )

            # 3. Generate LLM-based guidance (or fallback to static)
            guidance_generator = llm_guidance or LLMErrorGuidance(model_call=None)
            guidance = await guidance_generator.generate_guidance(e, context, classification)

            # 4. Raise structured exception with LLM guidance
            raise GraphQueryError(
                original_error=e,
                classification=classification,
                user_guidance=guidance,
                query=query,
            ) from e

    def _get_connection_timeout(self) -> int:
        """Resolve the effective connection timeout based on port/config."""
        if self._connection_timeout is not None:
            return self._connection_timeout
        if self._port == settings.DOC_MEMGRAPH_PORT:
            return settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT
        elif self._port == settings.JSON_MEMGRAPH_PORT:
            return settings.JSON_MEMGRAPH_CONNECTION_TIMEOUT
        return settings.MEMGRAPH_CONNECTION_TIMEOUT

    def _create_connection(self) -> mgclient.Connection:
        """Create a new Memgraph connection with timeout configuration.

        Sets up TCP keepalive and socket timeout to prevent connection drops
        and indefinite blocking during long-running operations.
        """
        timeout = self._get_connection_timeout()  # <-- CHANGED: use helper

        # Create connection (mgclient.connect() does not accept a timeout parameter;
        # timeout is enforced via socket.settimeout() below and threading wrapper
        # in _create_connection_with_timeout() for callers that need it)
        if self._username is not None:
            conn = mgclient.connect(
                host=self._host,
                port=self._port,
                username=self._username,
                password=self._password,
            )
        else:
            conn = mgclient.connect(host=self._host, port=self._port)
        conn.autocommit = True

        # Set socket timeout to prevent indefinite blocking on I/O operations
        # (see Fix 2 for details)
        try:
            if hasattr(conn, "socket") or hasattr(conn, "_socket"):
                sock = getattr(conn, "socket", getattr(conn, "_socket", None))
                if sock:
                    sock.settimeout(timeout)
        except (OSError, AttributeError) as e:
            logger.warning(
                f"Could not set socket timeout for {self._host}:{self._port}: {e}"
            )

        # Configure TCP keepalive for long-running connections
        try:
            # Get the underlying socket from the connection
            # This depends on mgclient implementation details
            if hasattr(conn, "socket") or hasattr(conn, "_socket"):
                sock = getattr(conn, "socket", getattr(conn, "_socket", None))
                if sock:
                    # Enable TCP keepalive
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    # Set keepalive parameters (platform-specific)
                    try:
                        # TCP_KEEPIDLE (seconds before first keepalive probe)
                        # TCP_KEEPINTVL (seconds between probes)
                        # TCP_KEEPCNT (number of failed probes before dropping)
                        sock.setsockopt(
                            socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60
                        )  # Start keepalive after 60 seconds
                        sock.setsockopt(
                            socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10
                        )  # Probe every 10 seconds
                        sock.setsockopt(
                            socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6
                        )  # Drop after 6 failed probes (60s total idle timeout)
                    except (OSError, AttributeError):
                        # Some platforms may not support these options
                        # Use platform-appropriate alternatives
                        if hasattr(socket, "TCP_KEEPIDLE"):
                            try:
                                sock.setsockopt(
                                    socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60
                                )
                            except OSError:
                                pass
                        if hasattr(socket, "TCP_KEEPINTVL"):
                            try:
                                sock.setsockopt(
                                    socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10
                                )
                            except OSError:
                                pass
                        if hasattr(socket, "TCP_KEEPCNT"):
                            try:
                                sock.setsockopt(
                                    socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6
                                )
                            except OSError:
                                pass
        except (OSError, AttributeError) as e:
            # Non-fatal: connection will still work, just without keepalive optimization
            logger.debug(
                f"Could not configure TCP keepalive for {self._host}:{self._port}: {e}"
            )

        return conn

    def _create_connection_with_timeout(self) -> mgclient.Connection:
        """Create connection with timeout enforcement.

        Worker threads inside ThreadPoolExecutor must not spawn child threads,
        as nested thread creation causes futex deadlocks on Linux. When called
        from the main thread (or a process worker), the original threaded
        timeout is preserved. When called from a worker thread, a socket
        pre-check is used instead.
        """
        timeout = self._get_connection_timeout()

        if threading.current_thread() is not threading.main_thread():
            # Worker thread: use socket pre-check to avoid nested threads
            try:
                with socket.create_connection(
                    (self._host, self._port),
                    timeout=min(timeout, 30.0),
                ):
                    pass
            except TimeoutError as e:
                raise TimeoutError(
                    f"Connection to Memgraph at {self._host}:{self._port} "
                    f"timed out after {timeout}s. Check if Memgraph is running "
                    f"and accessible."
                ) from e
            except OSError as e:
                raise ConnectionError(
                    f"Cannot connect to Memgraph at {self._host}:{self._port}: {e}"
                ) from e
            return self._create_connection()

        # Main thread: safe to use threaded timeout
        result: list[mgclient.Connection | None] = [None]
        error: list[Exception | None] = [None]

        def connect_worker():
            try:
                result[0] = self._create_connection()
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=connect_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Connection attempt timed out. The daemon thread may still complete
            # the connection in the background — attempt to close any connection
            # that was created to prevent a connection leak. There is an inherent
            # race condition between this check and the thread writing result[0],
            # so this is best-effort cleanup rather than guaranteed.
            if result[0] is not None:
                try:
                    result[0].close()
                except Exception:
                    pass
            raise TimeoutError(
                f"Connection to Memgraph at {self._host}:{self._port} "
                f"timed out after {timeout}s. Check if Memgraph is running "
                f"and accessible."
            )

        if error[0] is not None:
            raise error[0]

        if result[0] is None:
            raise ConnectionError("Failed to create Memgraph connection")

        return result[0]

    def _create_connection_with_retry(
        self,
        policy: ConnectionRetryPolicy | None = None,
    ) -> mgclient.Connection:
        """Create connection with exponential backoff on transient failures."""
        policy = policy or ConnectionRetryPolicy()
        delay = policy.base_delay_seconds
        last_exception: Exception | None = None

        for attempt in range(1, policy.max_attempts + 1):
            try:
                return self._create_connection_with_timeout()
            except policy.retryable_exceptions as e:
                last_exception = e
                if attempt == policy.max_attempts:
                    break
                logger.warning(
                    ls.MG_CONNECTION_RETRY_ATTEMPT.format(
                        attempt=attempt,
                        max_attempts=policy.max_attempts,
                        error=e,
                        delay=delay,
                    )
                )
                time.sleep(delay)
                delay = min(delay * policy.exponential_base, policy.max_delay_seconds)

        raise ConnectionError(
            ls.MG_CONNECTION_RETRY_EXHAUSTED.format(
                host=self._host,
                port=self._port,
                max_attempts=policy.max_attempts,
                error=last_exception,
            )
        ) from last_exception

    @contextmanager
    def _socket_timeout_context(
        self,
        conn: mgclient.Connection,
        timeout: int,
    ) -> Generator[None, None, None]:
        """Temporarily set socket timeout on a connection, restoring it on exit.
        
        Uses the same socket access pattern as existing keepalive/timeout code
        in _create_connection() for consistency.
        """
        original_timeout = None
        try:
            if hasattr(conn, "socket") or hasattr(conn, "_socket"):
                sock = getattr(conn, "socket", getattr(conn, "_socket", None))
                if sock:
                    original_timeout = sock.gettimeout()
                    sock.settimeout(timeout)
        except (OSError, AttributeError):
            pass
        try:
            yield
        finally:
            if original_timeout is not None:
                try:
                    if hasattr(conn, "socket") or hasattr(conn, "_socket"):
                        sock = getattr(conn, "socket", getattr(conn, "_socket", None))
                        if sock:
                            sock.settimeout(original_timeout)
                except (OSError, AttributeError):
                    pass

    def _execute_batch_with_timeout(
        self,
        conn: mgclient.Connection,
        query: str,
        params_list: Sequence[BatchParams],
        timeout: int | None = None,
    ) -> None:
        """Execute batch query with timeout enforcement (no return value)."""
        if timeout is None:
            timeout = settings.MEMGRAPH_QUERY_TIMEOUT
        with self._socket_timeout_context(conn, timeout):
            self._execute_batch_on(conn, query, params_list)

    def _execute_batch_with_return_with_timeout(
        self,
        conn: mgclient.Connection,
        query: str,
        params_list: Sequence[BatchParams],
        timeout: int | None = None,
    ) -> list[ResultRow]:
        """Execute batch query with timeout enforcement (returns results).
        
        This is the variant used by relationship flushing via
        _flush_rel_pattern_group(), which is the primary hang scenario.
        """
        if timeout is None:
            timeout = settings.MEMGRAPH_QUERY_TIMEOUT
        with self._socket_timeout_context(conn, timeout):
            return self._execute_batch_with_return_on(conn, query, params_list)

    def _execute_batch_on(
        self,
        conn: mgclient.Connection,
        query: str,
        params_list: Sequence[BatchParams],
    ) -> None:
        if not params_list:
            return
        use_shared_connection = conn is self.conn
        max_attempts = (
            settings.MEMGRAPH_QUERY_MAX_RETRIES + 1 if use_shared_connection else 1
        )
        for attempt in range(1, max_attempts + 1):
            cursor = None
            try:
                if use_shared_connection:
                    if attempt > 1 and self.conn is None:
                        try:
                            self._reset_shared_connection()
                        except Exception as reconnect_error:
                            if self._should_retry_shared_connection_error(
                                reconnect_error, attempt, max_attempts
                            ):
                                logger.warning(
                                    f"Transient Memgraph reconnect failure (attempt {attempt}/{max_attempts}), retrying: {reconnect_error}"
                                )
                                time.sleep(self._retry_delay_seconds(attempt))
                                continue
                            raise
                    with self._get_cursor() as shared_cursor:
                        shared_cursor.execute(
                            wrap_with_unwind(query), BatchWrapper(batch=params_list)
                        )
                    return
                cursor = conn.cursor()
                cursor.execute(wrap_with_unwind(query), BatchWrapper(batch=params_list))
                return
            except Exception as e:
                if use_shared_connection and self._should_retry_shared_connection_error(
                    e, attempt, max_attempts
                ):
                    logger.warning(
                        f"Transient Memgraph batch failure (attempt {attempt}/{max_attempts}), reconnecting: {e}"
                    )
                    try:
                        self._reset_shared_connection()
                    except Exception as reconnect_error:
                        if self._should_retry_shared_connection_error(
                            reconnect_error, attempt, max_attempts
                        ):
                            logger.warning(
                                f"Transient Memgraph reconnect failure (attempt {attempt}/{max_attempts}), retrying: {reconnect_error}"
                            )
                            time.sleep(self._retry_delay_seconds(attempt))
                            continue
                        raise
                    time.sleep(self._retry_delay_seconds(attempt))
                    continue
                if ERR_SUBSTR_ALREADY_EXISTS not in str(e).lower():
                    logger.error(ls.MG_BATCH_ERROR.format(error=e))
                    logger.error(ls.MG_CYPHER_QUERY.format(query=query))
                    if len(params_list) > 10:
                        logger.error(
                            ls.MG_BATCH_PARAMS_TRUNCATED.format(
                                count=len(params_list), params=params_list[:10]
                            )
                        )
                    else:
                        logger.error(ls.MG_CYPHER_PARAMS.format(params=params_list))
                raise
            finally:
                if cursor:
                    cursor.close()

    def _execute_batch_with_return_on(
        self,
        conn: mgclient.Connection,
        query: str,
        params_list: Sequence[BatchParams],
    ) -> list[ResultRow]:
        if not params_list:
            return []
        use_shared_connection = conn is self.conn
        max_attempts = (
            settings.MEMGRAPH_QUERY_MAX_RETRIES + 1 if use_shared_connection else 1
        )
        for attempt in range(1, max_attempts + 1):
            cursor = None
            try:
                if use_shared_connection:
                    if attempt > 1 and self.conn is None:
                        try:
                            self._reset_shared_connection()
                        except Exception as reconnect_error:
                            if self._should_retry_shared_connection_error(
                                reconnect_error, attempt, max_attempts
                            ):
                                logger.warning(
                                    f"Transient Memgraph reconnect failure (attempt {attempt}/{max_attempts}), retrying: {reconnect_error}"
                                )
                                time.sleep(self._retry_delay_seconds(attempt))
                                continue
                            raise
                    with self._get_cursor() as shared_cursor:
                        shared_cursor.execute(
                            wrap_with_unwind(query), BatchWrapper(batch=params_list)
                        )
                        return self._cursor_to_results(shared_cursor)
                cursor = conn.cursor()
                cursor.execute(wrap_with_unwind(query), BatchWrapper(batch=params_list))
                return self._cursor_to_results(cursor)
            except Exception as e:
                if use_shared_connection and self._should_retry_shared_connection_error(
                    e, attempt, max_attempts
                ):
                    logger.warning(
                        f"Transient Memgraph batch-return failure (attempt {attempt}/{max_attempts}), reconnecting: {e}"
                    )
                    try:
                        self._reset_shared_connection()
                    except Exception as reconnect_error:
                        if self._should_retry_shared_connection_error(
                            reconnect_error, attempt, max_attempts
                        ):
                            logger.warning(
                                f"Transient Memgraph reconnect failure (attempt {attempt}/{max_attempts}), retrying: {reconnect_error}"
                            )
                            time.sleep(self._retry_delay_seconds(attempt))
                            continue
                        raise
                    time.sleep(self._retry_delay_seconds(attempt))
                    continue
                logger.error(ls.MG_BATCH_ERROR.format(error=e))
                logger.error(ls.MG_CYPHER_QUERY.format(query=query))
                raise
            finally:
                if cursor:
                    cursor.close()
        return []

    def clean_database(self) -> None:
        logger.info(ls.MG_CLEANING_DB)
        self._execute_query(CYPHER_DELETE_ALL)
        logger.info(ls.MG_DB_CLEANED)

    def _vector_graph_type(self) -> str:
        if self._port == settings.DOC_MEMGRAPH_PORT:
            return "document"
        if self._port == settings.JSON_MEMGRAPH_PORT:
            return "json"
        return "code"

    def _vector_recreate_command(self) -> str:
        graph_type = self._vector_graph_type()
        option = {
            "code": "--code",
            "document": "--docs",
            "json": "--json",
        }[graph_type]
        return f"cgr vector recreate-indexes {option}"

    def list_projects(self) -> list[str]:
        result = self.fetch_all(CYPHER_LIST_PROJECTS)
        return [str(r[KEY_NAME]) for r in result]

    def delete_project(self, project_name: str) -> None:
        logger.info(ls.MG_DELETING_PROJECT.format(project_name=project_name))
        self._execute_query(CYPHER_DELETE_PROJECT, {KEY_PROJECT_NAME: project_name})
        logger.info(ls.MG_PROJECT_DELETED.format(project_name=project_name))

    def ensure_constraints(self, labels: tuple[str, ...] | None = None) -> None:
        logger.info(ls.MG_ENSURING_CONSTRAINTS)
        targets = (
            {k: v for k, v in NODE_UNIQUE_CONSTRAINTS.items() if k in labels}
            if labels
            else NODE_UNIQUE_CONSTRAINTS
        )
        for label, prop in targets.items():
            try:
                self._execute_query(build_constraint_query(label, prop))
            except Exception:
                pass
        logger.info(ls.MG_CONSTRAINTS_DONE)
        self._ensure_indexes(labels)

    def _ensure_indexes(self, labels: tuple[str, ...] | None = None) -> None:
        logger.info(ls.MG_ENSURING_INDEXES)
        targets = (
            {k: v for k, v in NODE_UNIQUE_CONSTRAINTS.items() if k in labels}
            if labels
            else NODE_UNIQUE_CONSTRAINTS
        )
        for label, prop in targets.items():
            try:
                self._execute_query(build_index_query(label, prop))
            except Exception:
                pass
        logger.info(ls.MG_INDEXES_DONE)

    def ensure_node_batch(
        self, label: str, properties: dict[str, PropertyValue]
    ) -> None:
        self.node_buffer.append((label, properties))
        if len(self.node_buffer) >= self.batch_size:
            logger.debug(ls.MG_NODE_BUFFER_FLUSH, size=self.batch_size)
            self.flush_nodes()

    def ensure_node(self, label: str, properties: dict[str, PropertyValue]) -> None:
        self.ensure_node_batch(label, properties)

    def ensure_edge(
        self,
        rel_type: str,
        from_identifier: str,
        to_identifier: str,
        properties: PropertyDict | None = None,
    ) -> None:
        # Simplified interface for graph_updater.py
        # Determine node label based on relationship type and identifier pattern
        def _get_label(identifier: str) -> str:
            # Simple heuristic: if identifier contains a dot and the part after
            # the last dot starts with lowercase, it might be a method
            # This is language-dependent but works for many cases
            if "." in identifier:
                # Check if it looks like a method (e.g., ClassName.methodName)
                parts = identifier.split(".")
                if len(parts) > 1 and parts[-1][0].islower():
                    return NodeLabel.METHOD
            return NodeLabel.FUNCTION

        if rel_type == REL_TYPE_CALLS:
            # Skip relationships to builtin functions (not part of indexed codebase)
            if to_identifier.startswith("builtin."):
                return
            # For CALLS relationships, try to determine if nodes are methods or functions
            from_label = _get_label(from_identifier)
            to_label = _get_label(to_identifier)
        else:
            from_label = NodeLabel.FUNCTION
            to_label = NodeLabel.FUNCTION

        self.ensure_relationship_batch(
            (from_label, KEY_QUALIFIED_NAME, from_identifier),
            rel_type,
            (to_label, KEY_QUALIFIED_NAME, to_identifier),
            properties,
        )

    def ensure_relationship_batch(
        self,
        from_spec: tuple[str, str, PropertyValue],
        rel_type: str,
        to_spec: tuple[str, str, PropertyValue],
        properties: dict[str, PropertyValue] | None = None,
    ) -> None:
        from_label, from_key, from_val = from_spec
        to_label, to_key, to_val = to_spec
        pattern = (from_label, from_key, rel_type, to_label, to_key)
        self._rel_groups[pattern].append(
            RelBatchRow(from_val=from_val, to_val=to_val, props=properties or {})
        )
        self._rel_count += 1

        # Add memory warning (not automatic flush)
        if self._rel_count > 0 and self._rel_count % self._REL_BUFFER_WARNING_THRESHOLD == 0:
            buffer_size_mb = sys.getsizeof(self._rel_groups) / (1024 * 1024)
            logger.debug(
                f"Relationship buffer size: {self._rel_count} relationships, "
                f"~{buffer_size_mb:.1f}MB in memory"
            )

        if self._rel_count >= self.batch_size:
            logger.debug(ls.MG_REL_BUFFER_FLUSH, size=self.batch_size)
            self.flush_nodes()
            # Note: Relationships are flushed at the end of ingestion via flush_all()
            # to ensure all target nodes exist before relationships are created.
            # Do NOT flush relationships here - it causes failures when target
            # nodes haven't been created yet.

    def _flush_node_label_group(
        self,
        label: str,
        props_list: list[dict[str, PropertyValue]],
        conn: mgclient.Connection | None = None,
    ) -> tuple[int, int]:
        if not props_list:
            return 0, 0

        id_key = NODE_UNIQUE_CONSTRAINTS.get(label)
        if not id_key:
            logger.warning(ls.MG_NO_CONSTRAINT.format(label=label))
            return 0, len(props_list)

        batch_rows: list[NodeBatchRow] = []
        skipped = 0
        for props in props_list:
            if id_key not in props:
                logger.warning(
                    ls.MG_MISSING_PROP.format(
                        label=label, key=id_key, prop_keys=list(props.keys())
                    )
                )
                skipped += 1
                continue
            row_props: PropertyDict = {k: v for k, v in props.items() if k != id_key}
            batch_rows.append(NodeBatchRow(id=props[id_key], props=row_props))

        if not batch_rows:
            return 0, skipped

        build_query = (
            build_merge_node_query if self._use_merge else build_create_node_query
        )
        query = build_query(label, id_key)
        target_conn = conn or self.conn
        if not target_conn:
            logger.warning(ls.MG_NO_CONN_NODES.format(label=label))
            return 0, skipped + len(batch_rows)
        lock = self._conn_lock if conn is None else nullcontext()
        with lock:
            self._execute_batch_with_timeout(target_conn, query, batch_rows)
        return len(batch_rows), skipped

    def _flush_node_group_with_own_conn(
        self,
        label: str,
        props_list: list[dict[str, PropertyValue]],
    ) -> tuple[int, int]:
        conn = self._create_connection_with_retry()
        try:
            return self._flush_node_label_group(label, props_list, conn=conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass  # Best-effort close — consistent with _flush_rel_group_with_own_conn

    def _flush_with_retry(
        self,
        flush_fn: Callable[[], tuple[int, int]],
        pattern: tuple[str, str, str, str, str],
        max_retries: int = 3,
    ) -> tuple[int, int]:
        """Execute flush operation with retry for transaction conflicts."""

        for attempt in range(max_retries + 1):
            try:
                return flush_fn()
            except Exception as e:
                classification = classify_memgraph_failure(e)

                if not classification.should_retry or attempt >= max_retries:
                    raise

                if classification.failure_type == FailureType.TRANSACTION_CONFLICT:
                    backoff = (
                        2 ** attempt * settings.MEMGRAPH_CONFLICT_RETRY_BASE_DELAY
                    )
                    logger.warning(
                        ls.MG_TRANSACTION_CONFLICT_RETRY.format(
                            pattern=pattern,
                            backoff=backoff,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                        )
                    )
                    time.sleep(backoff)
                else:
                    raise

    def _flush_rel_group_with_own_conn(
        self,
        pattern: tuple[str, str, str, str, str],
        params_list: list[RelBatchRow],
    ) -> tuple[int, int]:
        conn = self._create_connection_with_timeout()
        try:
            return self._flush_rel_pattern_group(pattern, params_list, conn=conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass  # Best-effort close — connection may be degraded after timeout

    def flush_nodes_with_stats(self) -> dict[str, int]:
        if not self.node_buffer:
            return {"attempted": 0, "flushed": 0, "failed": 0}

        buffer_size = len(self.node_buffer)
        nodes_by_label: defaultdict[str, list[dict[str, PropertyValue]]] = defaultdict(
            list
        )
        for label, props in self.node_buffer:
            nodes_by_label[label].append(props)

        flushed_total = 0
        skipped_total = 0
        total_failed = 0
        first_error: Exception | None = None

        if self._executor and len(nodes_by_label) > 1:
            logger.info(
                ls.MG_PARALLEL_FLUSH_NODES.format(
                    count=len(nodes_by_label),
                    workers=settings.FLUSH_THREAD_POOL_SIZE,
                )
            )
            futures = {
                self._executor.submit(
                    self._flush_node_group_with_own_conn, label, props_list
                ): label
                for label, props_list in nodes_by_label.items()
            }
            pending = set(futures.keys())
            start_time = time.monotonic()

            while pending:
                elapsed = time.monotonic() - start_time
                remaining = settings.FLUSH_OPERATION_TIMEOUT - elapsed
                if remaining <= 0:
                    for future in pending:
                        future.cancel()
                    raise ex.FlushTimeoutError(
                        ls.MG_FLUSH_TIMEOUT.format(
                            timeout=settings.FLUSH_OPERATION_TIMEOUT,
                            pending=len(pending),
                            total=len(futures),
                        )
                    )

                done, pending = wait(
                    pending,
                    timeout=min(
                        remaining, settings.FLUSH_PROGRESS_LOG_INTERVAL
                    ),
                    return_when=FIRST_COMPLETED,
                )

                if not done:
                    logger.warning(
                        ls.MG_FLUSH_NO_PROGRESS.format(
                            interval=settings.FLUSH_PROGRESS_LOG_INTERVAL,
                            pending=len(pending),
                        )
                    )
                    continue

                for future in done:
                    label = futures[future]
                    try:
                        flushed, skipped = future.result()
                        flushed_total += flushed
                        skipped_total += skipped
                    except Exception as e:
                        group_size = len(nodes_by_label[label])
                        total_failed += group_size
                        if first_error is None:
                            first_error = e
                        logger.error(
                            ls.MG_LABEL_FLUSH_ERROR.format(label=label, error=e)
                        )
        else:
            for label, props_list in nodes_by_label.items():
                try:
                    flushed, skipped = self._flush_node_label_group(label, props_list)
                    flushed_total += flushed
                    skipped_total += skipped
                except Exception as e:
                    group_size = len(props_list)
                    total_failed += group_size
                    if first_error is None:
                        first_error = e
                    logger.error(ls.MG_LABEL_FLUSH_ERROR.format(label=label, error=e))

        logger.info(
            ls.MG_NODES_FLUSHED.format(flushed=flushed_total, total=buffer_size)
        )
        if skipped_total:
            logger.info(ls.MG_NODES_SKIPPED.format(count=skipped_total))
        self.node_buffer.clear()

        if first_error is not None and flushed_total == 0:
            raise first_error

        return {
            "attempted": buffer_size,
            "flushed": flushed_total,
            "failed": total_failed,
        }

    def flush_nodes(self) -> None:
        """Backward-compatible wrapper that raises on any failure."""
        stats = self.flush_nodes_with_stats()
        if stats["failed"] > 0:
            raise ex.FlushError(
                ls.MG_FLUSH_PARTIAL_FAILURE.format(
                    attempted=stats["attempted"],
                    flushed=stats["flushed"],
                    failed=stats["failed"],
                )
            )

    def _classify_import_failure(self, to_val: str) -> str:
        """Classify import failure type.

        Args:
            to_val: Target module qualified name.

        Returns:
            "STDLIB", "THIRD_PARTY", or "INTERNAL"
        """
        base_module = to_val.split('.')[0]

        # Fast path: deterministic stdlib check
        if is_stdlib_module(base_module):
            return "STDLIB"

        # Known third-party packages (deterministic)
        if base_module in _KNOWN_THIRD_PARTY:
            return "THIRD_PARTY"

        # Unknown - could be internal or third-party
        return "INTERNAL"

    def _flush_rel_pattern_group_impl(
        self,
        pattern: tuple[str, str, str, str, str],
        params_list: list[RelBatchRow],
        conn: mgclient.Connection | None = None,
    ) -> tuple[int, int]:
        from_label, from_key, rel_type, to_label, to_key = pattern
        build_rel_query = (
            build_merge_relationship_query
            if self._use_merge
            else build_create_relationship_query
        )
        has_props = any(p[KEY_PROPS] for p in params_list)
        query = build_rel_query(
            from_label, from_key, rel_type, to_label, to_key, has_props
        )

        target_conn = conn or self.conn
        if not target_conn:
            logger.warning(ls.MG_NO_CONN_RELS.format(pattern=pattern))
            return len(params_list), 0
        lock = self._conn_lock if conn is None else nullcontext()
        with lock:
            results = self._execute_batch_with_return_with_timeout(
                target_conn, query, params_list
            )
        batch_successful = 0
        failed_indices: list[int] = []
        for i, r in enumerate(results):
            created = r.get(KEY_CREATED, 0)
            if isinstance(created, int):
                batch_successful += created
            else:
                failed_indices.append(i)

        # Log failures for all relationship types (not just CALLS)
        failed = len(params_list) - batch_successful
        if failed > 0:
            failure_rate = failed / len(params_list)

            # IMPORTS-specific handling: classify failures
            if rel_type == REL_TYPE_IMPORTS:
                stdlib_count = 0
                third_party_count = 0
                internal_count = 0

                for failed_idx in failed_indices:
                    sample = params_list[failed_idx]
                    classification = self._classify_import_failure(sample[KEY_TO_VAL])
                    if classification == "STDLIB":
                        stdlib_count += 1
                    elif classification == "THIRD_PARTY":
                        third_party_count += 1
                    else:
                        internal_count += 1

                # External imports (stdlib, third-party) are expected - log at DEBUG if enabled
                external_count = stdlib_count + third_party_count
                if external_count > 0 and settings.LOG_EXTERNAL_IMPORT_FAILURES:
                    logger.debug(
                        f"IMPORTS failures: {external_count} external modules "
                        f"(stdlib={stdlib_count}, third-party={third_party_count}) - expected"
                    )

                # Internal imports should be investigated - log at WARNING if enabled
                if internal_count > 0 and settings.LOG_INTERNAL_IMPORT_FAILURES:
                    logger.warning(
                        f"IMPORTS failures: {internal_count} internal modules - investigate"
                    )
                    for idx, failed_idx in enumerate(failed_indices[:3]):
                        sample = params_list[failed_idx]
                        classification = self._classify_import_failure(sample[KEY_TO_VAL])
                        if classification == "INTERNAL":
                            logger.warning(
                                f"  Internal import failed: {sample[KEY_FROM_VAL]} -> {sample[KEY_TO_VAL]}"
                            )
            else:
                # Non-IMPORTS: existing behavior - WARNING if high failure rate, DEBUG otherwise
                log_fn = logger.warning if failure_rate > 0.1 else logger.debug
                log_fn(
                    ls.MG_REL_FLUSH_FAILURES.format(
                        rel_type=rel_type,
                        failed=failed,
                        total=len(params_list),
                        from_label=from_label,
                        to_label=to_label,
                    )
                )
                # Log sample failures for diagnosis (up to 3)
                for idx, failed_idx in enumerate(failed_indices[:3]):
                    sample = params_list[failed_idx]
                    log_fn(
                        ls.MG_REL_FLUSH_FAILURE_SAMPLE.format(
                            index=idx + 1,
                            from_label=from_label,
                            from_val=sample[KEY_FROM_VAL],
                            to_label=to_label,
                            to_val=sample[KEY_TO_VAL],
                            props=sample.get(KEY_PROPS, {}),
                        )
                    )

        # Keep existing CALLS-specific debug logging for backward compatibility
        if rel_type == REL_TYPE_CALLS and failed > 0:
            logger.debug(ls.MG_CALLS_FAILED.format(count=failed))
            for i, sample in enumerate(params_list[:3]):
                logger.debug(
                    ls.MG_CALLS_SAMPLE.format(
                        index=i + 1,
                        from_label=from_label,
                        from_val=sample[KEY_FROM_VAL],
                        to_label=to_label,
                        to_val=sample[KEY_TO_VAL],
                    )
                )

        return len(params_list), batch_successful

    def _flush_rel_pattern_group(
        self,
        pattern: tuple[str, str, str, str, str],
        params_list: list[RelBatchRow],
        conn: mgclient.Connection | None = None,
    ) -> tuple[int, int]:
        """Flush relationship group with automatic retry for transaction conflicts."""
        if not settings.MEMGRAPH_CONFLICT_RETRY_ENABLED:
            return self._flush_rel_pattern_group_impl(pattern, params_list, conn)

        return self._flush_with_retry(
            lambda: self._flush_rel_pattern_group_impl(pattern, params_list, conn),
            pattern,
            max_retries=settings.MEMGRAPH_CONFLICT_RETRY_MAX_ATTEMPTS,
        )

    def flush_relationships_with_stats(self) -> dict[str, int]:
        if not self._rel_count:
            logger.debug(ls.MG_NO_RELS_TO_FLUSH)
            return {"attempted": 0, "flushed": 0, "failed": 0}

        total_attempted = 0
        total_successful = 0
        total_failed = 0
        first_error: Exception | None = None

        logger.info(
            ls.MG_PARALLEL_FLUSH_RELS.format(
                count=len(self._rel_groups),
                workers=settings.FLUSH_THREAD_POOL_SIZE,
            )
        )
        if self._executor and len(self._rel_groups) > 1:
            futures = {
                self._executor.submit(
                    self._flush_rel_group_with_own_conn, pattern, params_list
                ): pattern
                for pattern, params_list in self._rel_groups.items()
            }
            pending = set(futures.keys())
            start_time = time.monotonic()

            while pending:
                elapsed = time.monotonic() - start_time
                remaining = settings.FLUSH_OPERATION_TIMEOUT - elapsed
                if remaining <= 0:
                    for future in pending:
                        future.cancel()
                    raise ex.FlushTimeoutError(
                        ls.MG_FLUSH_TIMEOUT.format(
                            timeout=settings.FLUSH_OPERATION_TIMEOUT,
                            pending=len(pending),
                            total=len(futures),
                        )
                    )

                done, pending = wait(
                    pending,
                    timeout=min(
                        remaining, settings.FLUSH_PROGRESS_LOG_INTERVAL
                    ),
                    return_when=FIRST_COMPLETED,
                )

                if not done:
                    logger.warning(
                        ls.MG_FLUSH_NO_PROGRESS.format(
                            interval=settings.FLUSH_PROGRESS_LOG_INTERVAL,
                            pending=len(pending),
                        )
                    )
                    continue

                for future in done:
                    pattern = futures[future]
                    try:
                        attempted, successful = future.result()
                        total_attempted += attempted
                        total_successful += successful
                    except Exception as e:
                        group_size = len(self._rel_groups[pattern])
                        total_failed += group_size
                        if first_error is None:
                            first_error = e
                        logger.error(
                            ls.MG_REL_FLUSH_ERROR.format(pattern=pattern, error=e)
                        )
        else:
            for pattern, params_list in self._rel_groups.items():
                try:
                    attempted, successful = self._flush_rel_pattern_group(
                        pattern, params_list
                    )
                    total_attempted += attempted
                    total_successful += successful
                except Exception as e:
                    group_size = len(params_list)
                    total_failed += group_size
                    if first_error is None:
                        first_error = e
                    logger.error(ls.MG_REL_FLUSH_ERROR.format(pattern=pattern, error=e))

        logger.info(
            ls.MG_RELS_FLUSHED.format(
                total=self._rel_count,
                success=total_successful,
                failed=total_attempted - total_successful,
            )
        )
        self._rel_count = 0
        self._rel_groups.clear()

        if first_error is not None and total_successful == 0:
            raise first_error

        return {
            "attempted": total_attempted,
            "flushed": total_successful,
            "failed": total_failed,
        }

    def flush_relationships(self) -> None:
        """Backward-compatible wrapper that raises on any failure."""
        stats = self.flush_relationships_with_stats()
        if stats["failed"] > 0:
            raise ex.FlushError(
                ls.MG_FLUSH_PARTIAL_FAILURE.format(
                    attempted=stats["attempted"],
                    flushed=stats["flushed"],
                    failed=stats["failed"],
                )
            )

    def flush_all_with_stats(self) -> dict[str, int]:
        """Flush all pending writes, returning partial-success statistics."""
        stats: dict[str, int] = {
            "nodes_attempted": 0,
            "nodes_flushed": 0,
            "nodes_failed": 0,
            "relationships_attempted": 0,
            "relationships_flushed": 0,
            "relationships_failed": 0,
        }

        try:
            node_stats = self.flush_nodes_with_stats()
            stats["nodes_attempted"] = node_stats["attempted"]
            stats["nodes_flushed"] = node_stats["flushed"]
            stats["nodes_failed"] = node_stats["failed"]
        except Exception as e:
            logger.error(ls.MG_FLUSH_NODES_FAILED.format(error=e))
            stats["nodes_failed"] = stats["nodes_attempted"]

        try:
            rel_stats = self.flush_relationships_with_stats()
            stats["relationships_attempted"] = rel_stats["attempted"]
            stats["relationships_flushed"] = rel_stats["flushed"]
            stats["relationships_failed"] = rel_stats["failed"]
        except Exception as e:
            logger.error(ls.MG_FLUSH_RELS_FAILED.format(error=e))
            stats["relationships_failed"] = stats["relationships_attempted"]

        return stats

    def flush_all(self) -> None:
        logger.info(ls.MG_FLUSH_START)
        self.flush_nodes()
        self.flush_relationships()
        logger.info(ls.MG_FLUSH_COMPLETE)

    def fetch_all(
        self, query: str, params: dict[str, PropertyValue] | None = None
    ) -> list[ResultRow]:
        logger.debug(ls.MG_FETCH_QUERY, query=query, params=params)
        return self._execute_query(query, params)

    async def fetch_all_async(
        self, query: str, params: dict[str, PropertyValue] | None = None
    ) -> list[ResultRow]:
        """Async wrapper for fetch_all using asyncio.to_thread.

        Required for async-native execution in the query orchestrator,
        which runs inside async contexts (MCP server, pydantic-ai agent loop).
        """
        import asyncio

        return await asyncio.to_thread(self.fetch_all, query, params)

    def execute_write(
        self, query: str, params: dict[str, PropertyValue] | None = None
    ) -> None:
        logger.debug(ls.MG_WRITE_QUERY, query=query, params=params)
        self._execute_query(query, params)

    def export_graph_to_dict(self) -> GraphData:
        logger.info(ls.MG_EXPORTING)

        nodes_data = self.fetch_all(CYPHER_EXPORT_NODES)
        relationships_data = self.fetch_all(CYPHER_EXPORT_RELATIONSHIPS)

        metadata = GraphMetadata(
            total_nodes=len(nodes_data),
            total_relationships=len(relationships_data),
            exported_at=self._get_current_timestamp(),
        )

        logger.info(
            ls.MG_EXPORTED.format(nodes=len(nodes_data), rels=len(relationships_data))
        )
        return GraphData(
            nodes=nodes_data,
            relationships=relationships_data,
            metadata=metadata,
        )

    def _get_current_timestamp(self) -> str:
        return datetime.now(UTC).isoformat()
