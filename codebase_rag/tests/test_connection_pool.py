"""Tests for connection pool and pooled proxy."""

import threading
from unittest.mock import MagicMock, patch

from codebase_rag.services.connection_pool import (
    MemgraphConnectionPool,
    PooledMemgraphProxy,
    close_all_pools,
    get_connection_pool,
)


class TestMemgraphConnectionPool:
    def setup_method(self) -> None:
        close_all_pools()

    def teardown_method(self) -> None:
        close_all_pools()

    @patch.object(MemgraphConnectionPool, "_create_connection")
    def test_get_and_return_connection(self, mock_create: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_create.return_value = mock_conn

        pool = MemgraphConnectionPool("localhost", 7687, max_connections=5)
        conn = pool.get_connection()
        assert conn is mock_conn

        pool.return_connection(conn)
        conn2 = pool.get_connection()
        assert conn2 is mock_conn

    @patch.object(MemgraphConnectionPool, "_create_connection")
    def test_pool_creates_new_connections(self, mock_create: MagicMock) -> None:
        mock_conns = [MagicMock(name=f"conn_{i}") for i in range(3)]
        mock_create.side_effect = mock_conns

        pool = MemgraphConnectionPool("localhost", 7687, max_connections=5)
        c1 = pool.get_connection()
        c2 = pool.get_connection()
        c3 = pool.get_connection()

        assert c1 is mock_conns[0]
        assert c2 is mock_conns[1]
        assert c3 is mock_conns[2]
        assert mock_create.call_count == 3

    @patch.object(MemgraphConnectionPool, "_create_connection")
    def test_close_closes_all_connections(self, mock_create: MagicMock) -> None:
        mock_conns = [MagicMock() for _ in range(2)]
        mock_create.side_effect = mock_conns

        pool = MemgraphConnectionPool("localhost", 7687, max_connections=5)
        c1 = pool.get_connection()
        c2 = pool.get_connection()
        pool.return_connection(c1)
        pool.return_connection(c2)

        pool.close()

        for conn in mock_conns:
            conn.close.assert_called()

        try:
            pool.get_connection()
            assert False, "Should raise"
        except RuntimeError:
            pass

    def test_get_pool_returns_singleton(self) -> None:
        p1 = get_connection_pool("localhost", 7687, max_connections=5)
        p2 = get_connection_pool("localhost", 7687, max_connections=5)
        assert p1 is p2

    def test_get_pool_different_hosts(self) -> None:
        p1 = get_connection_pool("host1", 7687, max_connections=5)
        p2 = get_connection_pool("host2", 7687, max_connections=5)
        assert p1 is not p2

    @patch.object(MemgraphConnectionPool, "_create_connection")
    def test_return_connection_when_closed(self, mock_create: MagicMock) -> None:
        mock_conn = MagicMock()
        mock_create.return_value = mock_conn

        pool = MemgraphConnectionPool("localhost", 7687, max_connections=5)
        conn = pool.get_connection()
        pool.close()

        pool.return_connection(conn)
        mock_conn.close.assert_called()


class TestPooledMemgraphProxy:
    def test_fetch_all(self) -> None:
        mock_pool = MagicMock(spec=MemgraphConnectionPool)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.get_connection.return_value = mock_conn

        mock_cursor.description = [
            MagicMock(name="col1"),
            MagicMock(name="col2"),
        ]
        mock_cursor.description[0].name = "name"
        mock_cursor.description[1].name = "age"
        mock_cursor.fetchall.return_value = [("Alice", 30), ("Bob", 25)]

        proxy = PooledMemgraphProxy(mock_pool)
        results = proxy.fetch_all("MATCH (n) RETURN n.name, n.age")

        assert len(results) == 2
        assert results[0] == {"name": "Alice", "age": 30}
        assert results[1] == {"name": "Bob", "age": 25}
        mock_pool.get_connection.assert_called_once()
        mock_pool.return_connection.assert_called_once_with(mock_conn)

    def test_fetch_all_no_description(self) -> None:
        mock_pool = MagicMock(spec=MemgraphConnectionPool)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.get_connection.return_value = mock_conn
        mock_cursor.description = None

        proxy = PooledMemgraphProxy(mock_pool)
        results = proxy.fetch_all("CREATE (n:Test)")

        assert results == []

    def test_execute_write(self) -> None:
        mock_pool = MagicMock(spec=MemgraphConnectionPool)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.get_connection.return_value = mock_conn

        proxy = PooledMemgraphProxy(mock_pool)
        proxy.execute_write("CREATE (n:Test {name: 'test'})")

        mock_cursor.execute.assert_called_once_with(
            "CREATE (n:Test {name: 'test'})", {}
        )
        mock_pool.return_connection.assert_called_once_with(mock_conn)

    def test_cursor_closed_on_error(self) -> None:
        mock_pool = MagicMock(spec=MemgraphConnectionPool)
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_pool.get_connection.return_value = mock_conn
        mock_cursor.execute.side_effect = RuntimeError("query failed")

        proxy = PooledMemgraphProxy(mock_pool)
        try:
            proxy.fetch_all("BAD QUERY")
        except RuntimeError:
            pass

        mock_cursor.close.assert_called_once()
        mock_pool.return_connection.assert_called_once_with(mock_conn)


class TestConnectionPoolConcurrency:
    @patch.object(MemgraphConnectionPool, "_create_connection")
    def test_concurrent_access(self, mock_create: MagicMock) -> None:
        mock_conns = [MagicMock(name=f"conn_{i}") for i in range(20)]
        mock_create.side_effect = mock_conns

        pool = MemgraphConnectionPool("localhost", 7687, max_connections=5)
        acquired: list = []
        lock = threading.Lock()

        def worker() -> None:
            conn = pool.get_connection(timeout=5)
            with lock:
                acquired.append(conn)
            pool.return_connection(conn)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(acquired) == 10
