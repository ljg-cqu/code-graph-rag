from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self
from unittest.mock import MagicMock, call

import pytest
from loguru import logger

from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers

if TYPE_CHECKING:
    pass  # ty: ignore[unresolved-import]

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class NodeProtocol(Protocol):
    @property
    def type(self) -> str: ...
    @property
    def children(self) -> list[Self]: ...
    @property
    def parent(self) -> Self | None: ...
    @property
    def text(self) -> bytes: ...
    def child_by_field_name(self, name: str) -> Self | None: ...


@dataclass
class MockNode:
    node_type: str
    node_children: list[MockNode] = field(default_factory=list)
    node_parent: MockNode | None = None
    node_fields: dict[str, MockNode | None] = field(default_factory=dict)
    node_text: bytes = b""

    @property
    def type(self) -> str:
        return self.node_type

    @property
    def children(self) -> list[MockNode]:
        return self.node_children

    @property
    def parent(self) -> MockNode | None:
        return self.node_parent

    @parent.setter
    def parent(self, value: MockNode | None) -> None:
        self.node_parent = value

    @property
    def text(self) -> bytes:
        return self.node_text

    def child_by_field_name(self, name: str) -> MockNode | None:
        return self.node_fields.get(name)


def create_mock_node(
    node_type: str,
    text: str = "",
    fields: dict[str, MockNode | None] | None = None,
    children: list[MockNode] | None = None,
    parent: MockNode | None = None,
) -> MockNode:
    node = MockNode(
        node_type=node_type,
        node_children=children or [],
        node_parent=parent,
        node_fields=fields or {},
        node_text=text.encode(),
    )
    for child in node.node_children:
        child.node_parent = node
    return node


logger.remove()


@pytest.fixture
def temp_repo() -> Generator[Path, None, None]:
    """Creates a temporary repository path for a test and cleans up afterward."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


from codebase_rag.services import QueryProtocol


class _MockIngestor:
    _TRACKED = (
        "fetch_all",
        "execute_write",
        "ensure_node_batch",
        "ensure_relationship_batch",
        "flush_all",
    )

    def __init__(self) -> None:
        self.fetch_all = MagicMock(return_value=[])
        self.execute_write = MagicMock()
        self.ensure_node_batch = MagicMock()
        self.ensure_relationship_batch = MagicMock()
        self.flush_all = MagicMock()
        self._fallback = MagicMock()

    def reset_mock(self) -> None:
        for name in (*self._TRACKED, "_fallback"):
            getattr(self, name).reset_mock()

    @property
    def method_calls(self) -> list:
        result = []
        for name in self._TRACKED:
            mock_attr = self.__dict__[name]
            for c in mock_attr.call_args_list:
                result.append(getattr(call, name)(*c.args, **c.kwargs))
        result.extend(self._fallback.method_calls)
        return result

    def __getattr__(self, name: str) -> MagicMock:
        return getattr(self._fallback, name)


QueryProtocol.register(_MockIngestor)


@pytest.fixture
def mock_ingestor() -> _MockIngestor:
    return _MockIngestor()


def run_updater(
    repo_path: Path, mock_ingestor: MagicMock, skip_if_missing: str | None = None
) -> None:
    create_and_run_updater(repo_path, mock_ingestor, skip_if_missing)


def create_and_run_updater(
    repo_path: Path, mock_ingestor: MagicMock, skip_if_missing: str | None = None
) -> GraphUpdater:
    parsers, queries = load_parsers()
    if skip_if_missing and skip_if_missing not in parsers:
        pytest.skip(f"{skip_if_missing} parser not available")
    updater = GraphUpdater(
        ingestor=mock_ingestor,
        repo_path=repo_path,
        parsers=parsers,
        queries=queries,
    )
    updater.run()
    return updater


def get_relationships(mock_ingestor: MagicMock, rel_type: str) -> list:
    """Extract relationships of a specific type from mock_ingestor calls."""
    return [
        c
        for c in mock_ingestor.ensure_relationship_batch.call_args_list
        if c.args[1] == rel_type
    ]


def get_nodes(mock_ingestor: MagicMock, node_type: str) -> list:
    """Extract nodes of a specific type from mock_ingestor calls."""
    return [
        call
        for call in mock_ingestor.ensure_node_batch.call_args_list
        if call[0][0] == node_type
    ]


def get_qualified_names(calls: list) -> set[str]:
    """Extract qualified names from a list of node calls."""
    return {call[0][1]["qualified_name"] for call in calls}


def get_node_names(mock_ingestor: MagicMock, node_type: str) -> set[str]:
    """Get qualified names of all nodes of a specific type."""
    return get_qualified_names(get_nodes(mock_ingestor, node_type))


@pytest.fixture
def mock_updater(temp_repo: Path, mock_ingestor: MagicMock) -> MagicMock:
    """Provides a mocked GraphUpdater instance with necessary dependencies."""
    parsers, queries = load_parsers()
    mock = MagicMock(spec=GraphUpdater)
    mock.repo_path = temp_repo
    mock.ingestor = mock_ingestor
    mock.parsers = parsers
    mock.queries = queries

    mock.factory = MagicMock()
    mock.factory.definition_processor = MagicMock()
    mock.factory.structure_processor = MagicMock()
    mock.factory.structure_processor.structural_elements = {}

    mock_root_node = MagicMock()
    mock.factory.definition_processor.process_file.return_value = (
        mock_root_node,
        "python",
    )

    mock.ast_cache = {}

    return mock


# =============================================================================
# Dependency Check Functions
# =============================================================================

def _check_pydantic_ai() -> bool:
    """Check if pydantic_ai is available."""
    try:
        from codebase_rag.compat.pydantic_ai import HAS_PYDANTIC_AI
        return HAS_PYDANTIC_AI
    except ImportError:
        return False


def _check_memgraph() -> bool:
    """Check if Memgraph is running and accessible."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", 7687))
        sock.close()
        return result == 0
    except Exception:
        return False


def _check_embeddings() -> bool:
    """Check if embedding model is available."""
    try:
        # Check if semantic dependencies are installed
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


# =============================================================================
# Skipif Helpers (for use as pytest.mark.skipif)
# =============================================================================

requires_pydantic_ai = pytest.mark.skipif(
    not _check_pydantic_ai(),
    reason="pydantic_ai not installed (install with: uv sync --extra ai)"
)

requires_memgraph = pytest.mark.skipif(
    not _check_memgraph(),
    reason="Memgraph not running (start with: docker run -p 7687:7687 memgraph/memgraph)"
)

requires_embeddings = pytest.mark.skipif(
    not _check_embeddings(),
    reason="Embedding model not available (install with: uv sync --extra semantic)"
)


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_pydantic_ai():
    """Fixture providing mock pydantic_ai classes.

    Returns:
        Dict with MockAgent, MockTool, and MockRunResult classes
    """
    from codebase_rag.tests.mocks.pydantic_ai_mock import MockAgent, MockTool, MockRunResult
    return {
        "Agent": MockAgent,
        "Tool": MockTool,
        "RunResult": MockRunResult,
    }


@pytest.fixture
def mock_memgraph():
    """Fixture providing a mock Memgraph connection.

    Yields:
        MockMemgraphConnection instance
    """
    from codebase_rag.tests.mocks.memgraph_mock import MockMemgraphConnection
    from unittest.mock import patch

    conn = MockMemgraphConnection()
    with patch("codebase_rag.services.graph_service.connect", return_value=conn):
        yield conn


@pytest.fixture
def mock_embedding():
    """Fixture providing a mock embedding provider.

    Returns:
        MockEmbeddingProvider instance
    """
    from codebase_rag.tests.mocks.embedding_mock import MockEmbeddingProvider
    return MockEmbeddingProvider()


@pytest.fixture
def sample_code_file(temp_repo: Path) -> Path:
    """Create a sample Python file for testing.

    Args:
        temp_repo: Temporary repository path

    Returns:
        Path to the created file
    """
    file_path = temp_repo / "sample.py"
    file_path.write_text('''
def hello():
    """Return a greeting."""
    return "world"


class MyClass:
    """A sample class."""

    def method(self):
        """A sample method."""
        pass


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
''')
    return file_path


@pytest.fixture
def sample_document(temp_repo: Path) -> Path:
    """Create a sample markdown document for testing.

    Args:
        temp_repo: Temporary repository path

    Returns:
        Path to the created file
    """
    file_path = temp_repo / "sample.md"
    file_path.write_text('''
# Sample Document

This is a sample document for testing document indexing.

## Features

- Feature 1: Does something
- Feature 2: Does something else

## Code Example

```python
def example():
    return "example"
```
''')
    return file_path


# =============================================================================
# Mock Classification and Guidance Fixtures
# =============================================================================

@pytest.fixture
def mock_classification():
    """Provide a mock FailureClassification."""
    from codebase_rag.services.failure_classifier import (
        FailureClassification,
        FailureType,
    )
    return FailureClassification(
        failure_type=FailureType.TRANSIENT_NETWORK,
        message="Mock error",
        should_retry=True,
        max_retries=3,
    )


@pytest.fixture
def mock_guidance():
    """Provide a mock ErrorGuidance."""
    from codebase_rag.services.error_guidance import ErrorGuidance
    return ErrorGuidance(
        summary="Mock Summary",
        explanation="Mock explanation.",
        suggested_fix="Mock fix.",
    )
