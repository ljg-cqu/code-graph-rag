from __future__ import annotations

import socket
import time
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

from codebase_rag.services.graph_service import MemgraphIngestor

if TYPE_CHECKING:
    import mgclient


@pytest.fixture(scope="session")
def memgraph_container() -> Generator[dict[str, str | int], None, None]:
    pytest.importorskip("testcontainers")
    import time

    from testcontainers.core.container import DockerContainer
    from testcontainers.core.wait_strategies import LogMessageWaitStrategy

    container = DockerContainer("memgraph/memgraph:latest")
    container.with_exposed_ports(7687)
    container.waiting_for(LogMessageWaitStrategy("You are running Memgraph"))

    container.start()

    host = container.get_container_host_ip()
    port = int(container.get_exposed_port(7687))

    max_retries = 30
    for attempt in range(max_retries):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((host, port))
            sock.close()
            break
        except (TimeoutError, ConnectionRefusedError, OSError):
            if attempt == max_retries - 1:
                container.stop()
                pytest.fail(
                    f"Memgraph port {port} not ready after {max_retries} attempts"
                )
            time.sleep(0.5)

    yield {"host": host, "port": port}

    container.stop()


@pytest.fixture(scope="function")
def memgraph_connection(
    memgraph_container: dict[str, str | int],
) -> Generator[mgclient.Connection, None, None]:
    import mgclient  # ty: ignore[unresolved-import]

    host = str(memgraph_container["host"])
    port = int(memgraph_container["port"])

    max_retries = 10
    conn: mgclient.Connection | None = None

    for attempt in range(max_retries):
        try:
            conn = mgclient.connect(host=host, port=port)
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute("MATCH (n) DETACH DELETE n")
            cursor.close()
            break
        except Exception as e:
            if attempt == max_retries - 1:
                pytest.fail(
                    f"Failed to connect to Memgraph after {max_retries} attempts: {e}"
                )
            time.sleep(0.5)

    if conn is None:
        pytest.fail("Failed to establish Memgraph connection")

    yield conn

    assert conn is not None
    cursor = conn.cursor()
    cursor.execute("MATCH (n) DETACH DELETE n")
    cursor.close()
    conn.close()


@pytest.fixture(scope="function")
def memgraph_ingestor(
    memgraph_container: dict[str, str | int],
) -> Generator[MemgraphIngestor, None, None]:
    host = str(memgraph_container["host"])
    port = int(memgraph_container["port"])

    max_retries = 10

    for attempt in range(max_retries):
        try:
            ingestor = MemgraphIngestor(host=host, port=port)
            ingestor.__enter__()
            ingestor._execute_query("MATCH (n) DETACH DELETE n")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                pytest.fail(
                    f"Failed to connect to Memgraph after {max_retries} attempts: {e}"
                )
            time.sleep(0.5)

    yield ingestor

    ingestor._execute_query("MATCH (n) DETACH DELETE n")
    ingestor.__exit__(None, None, None)


@pytest.fixture(scope="function")
def mock_memgraph_backend():
    """Mock vector backend for hybrid retrieval tests."""
    from unittest.mock import MagicMock

    backend = MagicMock()
    # search returns tuples of (node_id, similarity_score)
    backend.search.return_value = [
        (1, 0.9),
        (2, 0.8),
        (3, 0.7),
    ]
    backend.health_check.return_value = True
    backend.LABELS_TO_INDEX = ["Function", "Method", "Class"]
    return backend


@pytest.fixture(scope="function")
def mock_graph_ingestor():
    """Mock graph ingestor for hybrid retrieval tests."""
    from unittest.mock import MagicMock

    ingestor = MagicMock()
    ingestor.get_pagerank_scores.return_value = {"n1": 0.5, "n2": 0.4, "n3": 0.3}
    ingestor.get_community_scores.return_value = {"n1": 0.6, "n2": 0.5, "n3": 0.4}
    return ingestor


@pytest.fixture(scope="function")
def mock_embedding_provider():
    """Mock embedding provider for hybrid retrieval tests."""
    from unittest.mock import MagicMock

    provider = MagicMock()
    provider.embed.return_value = [0.1] * 768
    return provider
