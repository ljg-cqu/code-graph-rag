from __future__ import annotations

import socket
import threading
import time
import types
from collections import defaultdict
from collections.abc import Generator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
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
        self._conn_lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self.conn: mgclient.Connection | None = None
        self.node_buffer: list[tuple[str, dict[str, PropertyValue]]] = []
        self._rel_count = 0
        self._rel_groups: defaultdict[
            tuple[str, str, str, str, str], list[RelBatchRow]
        ] = defaultdict(list)
        self._dynamic_algorithms_supported: bool | None = None
        self._connection_timeout = connection_timeout
        self._last_health_check = 0.0
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

        try:
            # Try to use a dynamic algorithm function to check support
            # This will fail on Community edition with "Function not found" error
            self._execute_query(
                """
                RETURN dynamic_graph_update_is_supported() AS supported
                LIMIT 1
                """
            )
            return True
        except Exception:
            # Fallback: check version string for enterprise
            try:
                results = self._execute_query("SHOW VERSION AS version")
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
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        try:
            if exc_type:
                logger.exception(ls.MG_EXCEPTION.format(error=exc_val))
                # (H) Best-effort flush: attempt to persist buffered nodes/relationships
                # (H) even when an exception occurred. Catching broad Exception so a
                # (H) secondary flush failure never masks the original exception.
                try:
                    self.flush_all()
                except Exception as flush_err:
                    logger.error(ls.MG_FLUSH_ERROR.format(error=flush_err))
            else:
                self.flush_all()
        finally:
            if self._executor:
                # Shutdown without waiting to allow forced exit.
                # cancel_futures=True (Python ≥3.9) prevents waiting for stuck workers.
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
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
        if not cursor.description:
            return []
        column_names = [desc.name for desc in cursor.description]
        return [
            dict[str, ResultValue](zip(column_names, row)) for row in cursor.fetchall()
        ]

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
            # Simple health check query
            with self._get_cursor() as cursor:
                cursor.execute("RETURN 1")
                results = cursor.fetchall()
                self._last_health_check = current_time
                return bool(results and results[0][0] == 1)
        except Exception:
            # Connection is unhealthy
            self._last_health_check = current_time
            return False

    def _ensure_connection(self) -> None:
        """Ensure we have a healthy connection, reconnecting if necessary."""
        if not self.conn or not self._check_connection_health():
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass
            self.conn = self._create_connection_with_timeout()  # <-- CHANGED

    @classmethod
    def _is_retryable_memgraph_error(cls, error: Exception) -> bool:
        message = str(error).lower()
        return any(marker in message for marker in cls._TRANSIENT_ERROR_MARKERS)

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
        self.conn = self._create_connection_with_timeout()  # <-- CHANGED

    def _should_retry_shared_connection_error(
        self,
        error: Exception,
        attempt: int,
        max_attempts: int,
    ) -> bool:
        return attempt < max_attempts and self._is_retryable_memgraph_error(error)

    def _execute_query(
        self,
        query: str,
        params: dict[str, PropertyValue] | None = None,
    ) -> list[ResultRow]:
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
                    # Validate embedding dimension if present in parameters
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
                if self._should_retry_shared_connection_error(e, attempt, max_attempts):
                    logger.warning(
                        f"Transient Memgraph query failure (attempt {attempt}/{max_attempts}), reconnecting: {e}"
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

        Uses threading to enforce connection timeout since mgclient.connect()
        doesn't support a timeout parameter natively. If the connection attempt
        times out, any connection object that may have been created by the
        background thread is closed to prevent connection leaks.
        """
        timeout = self._get_connection_timeout()

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

    def ensure_constraints(self) -> None:
        logger.info(ls.MG_ENSURING_CONSTRAINTS)
        for label, prop in NODE_UNIQUE_CONSTRAINTS.items():
            try:
                self._execute_query(build_constraint_query(label, prop))
            except Exception:
                pass
        logger.info(ls.MG_CONSTRAINTS_DONE)
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        logger.info(ls.MG_ENSURING_INDEXES)
        for label, prop in NODE_UNIQUE_CONSTRAINTS.items():
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
        if self._rel_count >= self.batch_size:
            logger.debug(ls.MG_REL_BUFFER_FLUSH, size=self.batch_size)
            self.flush_nodes()
            self.flush_relationships()

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
        conn = self._create_connection_with_timeout()  # <-- CHANGED
        try:
            return self._flush_node_label_group(label, props_list, conn=conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass  # Best-effort close — consistent with _flush_rel_group_with_own_conn

    def _flush_rel_group_with_own_conn(
        self,
        pattern: tuple[str, str, str, str, str],
        params_list: list[RelBatchRow],
    ) -> tuple[int, int]:
        conn = self._create_connection_with_timeout()  # <-- CHANGED
        try:
            return self._flush_rel_pattern_group(pattern, params_list, conn=conn)
        finally:
            try:
                conn.close()
            except Exception:
                pass  # Best-effort close — connection may be degraded after timeout

    def flush_nodes(self) -> None:
        if not self.node_buffer:
            return

        buffer_size = len(self.node_buffer)
        nodes_by_label: defaultdict[str, list[dict[str, PropertyValue]]] = defaultdict(
            list
        )
        for label, props in self.node_buffer:
            nodes_by_label[label].append(props)

        flushed_total = 0
        skipped_total = 0

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
            for future in as_completed(futures):
                label = futures[future]
                try:
                    flushed, skipped = future.result()
                    flushed_total += flushed
                    skipped_total += skipped
                except Exception as e:
                    logger.error(ls.MG_LABEL_FLUSH_ERROR.format(label=label, error=e))
                    if first_error is None:
                        first_error = e
        else:
            for label, props_list in nodes_by_label.items():
                try:
                    flushed, skipped = self._flush_node_label_group(label, props_list)
                    flushed_total += flushed
                    skipped_total += skipped
                except Exception as e:
                    logger.error(ls.MG_LABEL_FLUSH_ERROR.format(label=label, error=e))
                    if first_error is None:
                        first_error = e

        logger.info(
            ls.MG_NODES_FLUSHED.format(flushed=flushed_total, total=buffer_size)
        )
        if skipped_total:
            logger.info(ls.MG_NODES_SKIPPED.format(count=skipped_total))
        self.node_buffer.clear()

        if first_error is not None:
            raise first_error

    def _flush_rel_pattern_group(
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
        for r in results:
            created = r.get(KEY_CREATED, 0)
            if isinstance(created, int):
                batch_successful += created

        if rel_type == REL_TYPE_CALLS:
            failed = len(params_list) - batch_successful
            if failed > 0:
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

    def flush_relationships(self) -> None:
        if not self._rel_count:
            logger.debug("No relationships to flush, skipping")
            return

        total_attempted = 0
        total_successful = 0
        first_error: Exception | None = None

        # Always log relationship flush start (previously conditional on executor + >1 groups)
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
            for future in as_completed(futures):
                pattern = futures[future]
                try:
                    attempted, successful = future.result()
                    total_attempted += attempted
                    total_successful += successful
                except Exception as e:
                    logger.error(ls.MG_REL_FLUSH_ERROR.format(pattern=pattern, error=e))
                    if first_error is None:
                        first_error = e
        else:
            for pattern, params_list in self._rel_groups.items():
                try:
                    attempted, successful = self._flush_rel_pattern_group(
                        pattern, params_list
                    )
                    total_attempted += attempted
                    total_successful += successful
                except Exception as e:
                    logger.error(ls.MG_REL_FLUSH_ERROR.format(pattern=pattern, error=e))
                    if first_error is None:
                        first_error = e

        logger.info(
            ls.MG_RELS_FLUSHED.format(
                total=self._rel_count,
                success=total_successful,
                failed=total_attempted - total_successful,
            )
        )
        self._rel_count = 0
        self._rel_groups.clear()

        if first_error is not None:
            raise first_error

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
