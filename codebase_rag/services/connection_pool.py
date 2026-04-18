"""Connection pool for parallel Memgraph access.

Provides shared connection pooling for SubAgentOrchestrator workers
instead of each worker creating its own MemgraphIngestor instance.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from queue import Queue
from typing import Any

import mgclient
from codebase_rag.types_defs import PropertyDict, ResultRow

from . import QueryProtocol
from ..utils.shutdown_manager import shutdown_manager

_SHUTDOWN_HANDLER_REGISTERED = False


def _register_shutdown_handler_if_needed() -> None:
    global _SHUTDOWN_HANDLER_REGISTERED
    if not _SHUTDOWN_HANDLER_REGISTERED:
        shutdown_manager.register_handler(close_all_pools, priority=15)
        _SHUTDOWN_HANDLER_REGISTERED = True


class MemgraphConnectionPool:
    """Thread-safe connection pool for Memgraph connections."""

    _pools: dict[tuple[str, int, str], MemgraphConnectionPool] = {}
    _pool_lock = threading.Lock()

    @classmethod
    def get_pool(
        cls,
        host: str,
        port: int,
        username: str | None = None,
        password: str | None = None,
        max_connections: int = 20,
        connection_timeout: int = 60,
    ) -> MemgraphConnectionPool:
        key = (host, port, username or "")
        with cls._pool_lock:
            if key not in cls._pools:
                cls._pools[key] = cls(
                    host, port, username, password, max_connections, connection_timeout
                )
            return cls._pools[key]

    @classmethod
    def close_all_pools(cls) -> None:
        with cls._pool_lock:
            for pool in cls._pools.values():
                pool.close()
            cls._pools.clear()

    def __init__(
        self,
        host: str,
        port: int,
        username: str | None = None,
        password: str | None = None,
        max_connections: int = 20,
        connection_timeout: int = 60,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self._connections: Queue[mgclient.Connection] = Queue(maxsize=max_connections)
        self._active_count = 0
        self._lock = threading.Lock()
        self._closed = False

    def get_connection(self, timeout: float = 30.0) -> mgclient.Connection:
        with self._lock:
            if self._closed:
                raise RuntimeError("Connection pool is closed")

        try:
            return self._connections.get_nowait()
        except Exception:
            pass

        with self._lock:
            if self._closed:
                raise RuntimeError("Connection pool is closed")
            if self._active_count < self.max_connections:
                try:
                    conn = self._create_connection()
                    self._active_count += 1
                    return conn
                except Exception:
                    # Don't increment count if connection creation failed
                    raise

        return self._connections.get(timeout=timeout)

    def return_connection(self, conn: mgclient.Connection) -> None:
        with self._lock:
            if self._closed:
                try:
                    conn.close()
                except Exception:
                    pass
                self._active_count -= 1
                return

        try:
            self._connections.put_nowait(conn)
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._active_count -= 1

    def _create_connection(self) -> mgclient.Connection:
        """Create connection using MemgraphIngestor's proven logic."""
        from codebase_rag.services.graph_service import MemgraphIngestor

        ingestor = object.__new__(MemgraphIngestor)
        ingestor._host = self.host
        ingestor._port = self.port
        ingestor._username = self.username
        ingestor._password = self.password
        ingestor._connection_timeout = self.connection_timeout
        return ingestor._create_connection_with_timeout()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True

            while True:
                try:
                    conn = self._connections.get_nowait()
                    try:
                        conn.close()
                    except Exception:
                        pass
                except Exception:
                    break

            self._active_count = 0


def get_connection_pool(
    host: str,
    port: int,
    username: str | None = None,
    password: str | None = None,
    max_connections: int = 20,
    connection_timeout: int = 60,
) -> MemgraphConnectionPool:
    _register_shutdown_handler_if_needed()
    return MemgraphConnectionPool.get_pool(
        host, port, username, password, max_connections, connection_timeout
    )


def close_all_pools() -> None:
    MemgraphConnectionPool.close_all_pools()


class PooledMemgraphProxy:
    """Lightweight proxy that borrows connections from a pool.

    Provides the same query interface as MemgraphIngestor via
    QueryProtocol but uses pooled connections instead of creating
    new ones per instance.
    """

    def __init__(self, pool: MemgraphConnectionPool) -> None:
        self._pool = pool

    @contextmanager
    def _get_cursor(self) -> Generator[Any, None, None]:
        conn = self._pool.get_connection()
        try:
            cursor = conn.cursor()
            yield cursor
        finally:
            try:
                cursor.close()
            except Exception:
                pass
            self._pool.return_connection(conn)

    def fetch_all(
        self, query: str, params: PropertyDict | None = None
    ) -> list[ResultRow]:
        with self._get_cursor() as cursor:
            cursor.execute(query, params or {})
            if not cursor.description:
                return []
            column_names = [desc.name for desc in cursor.description]
            return [dict[str, Any](zip(column_names, row)) for row in cursor.fetchall()]

    async def fetch_all_async(
        self, query: str, params: PropertyDict | None = None
    ) -> list[ResultRow]:
        """Async wrapper for fetch_all using asyncio.to_thread.

        Required for async-native execution in the query orchestrator,
        which runs inside async contexts (MCP server, pydantic-ai agent loop).
        """
        import asyncio

        return await asyncio.to_thread(self.fetch_all, query, params)

    def execute_write(
        self, query: str, params: PropertyDict | None = None
    ) -> None:
        with self._get_cursor() as cursor:
            cursor.execute(query, params or {})


def _check_protocol_compliance() -> None:
    _proxy: QueryProtocol = PooledMemgraphProxy.__new__(PooledMemgraphProxy)
