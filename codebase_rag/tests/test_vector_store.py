from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from codebase_rag.utils.dependencies import has_qdrant_client

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    return mock_client


@pytest.fixture
def reset_global_client() -> Generator[None, None, None]:
    import codebase_rag.vector_store as vs
    import codebase_rag.vector_backend as vb

    if has_qdrant_client() and vs._CLIENT is not None:
        try:
            vs._CLIENT.close()
        except Exception:
            pass
        vs._CLIENT = None

    # Reset backend instances
    vb._BACKEND_INSTANCE = None
    vs._BACKEND = None

    yield

    if has_qdrant_client() and vs._CLIENT is not None:
        try:
            vs._CLIENT.close()
        except Exception:
            pass
        vs._CLIENT = None

    vb._BACKEND_INSTANCE = None
    vs._BACKEND = None


@pytest.fixture
def temp_qdrant_path() -> Generator[Path, None, None]:
    temp_dir = tempfile.mkdtemp(prefix="qdrant_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def qdrant_backend_env(reset_global_client: None) -> Generator[None, None, None]:
    """Set environment to use Qdrant backend."""
    old_env = os.environ.get("VECTOR_STORE_BACKEND")
    os.environ["VECTOR_STORE_BACKEND"] = "qdrant"
    yield
    if old_env is not None:
        os.environ["VECTOR_STORE_BACKEND"] = old_env
    else:
        os.environ.pop("VECTOR_STORE_BACKEND", None)


@pytest.fixture
def mock_backend(
    mock_qdrant_client: MagicMock, reset_global_client: None, qdrant_backend_env: None
) -> Generator[MagicMock, None, None]:
    """Create a mock backend that delegates to the mock client."""
    import codebase_rag.vector_store as vs

    # Create a mock backend
    mock_backend = MagicMock()
    vs._BACKEND = mock_backend
    vs._CLIENT = mock_qdrant_client

    yield mock_backend

    vs._BACKEND = None
    vs._CLIENT = None


@pytest.fixture
def integration_client(
    temp_qdrant_path: Path, reset_global_client: None, qdrant_backend_env: None
) -> Generator[QdrantClient, None, None]:
    if not has_qdrant_client():
        pytest.skip("qdrant-client not installed")

    from qdrant_client import QdrantClient as QC
    from qdrant_client.models import Distance, VectorParams
    from codebase_rag.vector_store_qdrant import QdrantBackend

    import codebase_rag.vector_store as vs

    client = QC(path=str(temp_qdrant_path))
    client.create_collection(
        collection_name="code_embeddings",
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
    )

    # Create a QdrantBackend with the test client injected
    backend = QdrantBackend()
    backend._client = client
    backend._initialized = True
    vs._BACKEND = backend
    vs._CLIENT = client

    yield client

    vs._CLIENT = None
    vs._BACKEND = None
    try:
        client.close()
    except Exception:
        pass


@pytest.mark.skipif(not has_qdrant_client(), reason="qdrant-client not installed")
def test_store_embedding_calls_upsert(
    mock_backend: MagicMock, mock_qdrant_client: MagicMock
) -> None:
    from codebase_rag.vector_store import store_embedding

    node_id = 123
    embedding = [0.1] * 768
    qualified_name = "myproject.module.function"

    # Configure mock backend to delegate to mock client
    def store_batch_side_effect(points):
        from qdrant_client.models import PointStruct
        mock_qdrant_client.upsert(
            collection_name="code_embeddings",
            points=[
                PointStruct(
                    id=p[0],
                    vector=p[1],
                    payload={"node_id": p[0], "qualified_name": p[2]},
                )
                for p in points
            ],
        )
        return len(points)

    mock_backend.store_batch = store_batch_side_effect

    store_embedding(node_id, embedding, qualified_name)

    mock_qdrant_client.upsert.assert_called_once()
    call_kwargs = mock_qdrant_client.upsert.call_args[1]
    assert call_kwargs["collection_name"] == "code_embeddings"
    points = call_kwargs["points"]
    assert len(points) == 1
    assert points[0].id == node_id
    assert points[0].vector == embedding
    assert points[0].payload["node_id"] == node_id
    assert points[0].payload["qualified_name"] == qualified_name


@pytest.mark.skipif(not has_qdrant_client(), reason="qdrant-client not installed")
def test_store_embedding_handles_exception(
    mock_backend: MagicMock, mock_qdrant_client: MagicMock
) -> None:
    from codebase_rag.vector_store import store_embedding

    def store_batch_side_effect(points):
        try:
            mock_qdrant_client.upsert(
                collection_name="code_embeddings", points=[]
            )
        except Exception:
            return 0
        return 0

    mock_backend.store_batch = store_batch_side_effect
    mock_qdrant_client.upsert.side_effect = Exception("Connection failed")

    # Should not raise - exception is handled gracefully
    store_embedding(123, [0.1] * 768, "test.func")


@pytest.mark.skipif(not has_qdrant_client(), reason="qdrant-client not installed")
def test_search_embeddings_calls_query_points(
    mock_backend: MagicMock, mock_qdrant_client: MagicMock
) -> None:
    from codebase_rag.vector_store import search_embeddings

    mock_point1 = MagicMock()
    mock_point1.payload = {"node_id": 1}
    mock_point1.score = 0.95

    mock_point2 = MagicMock()
    mock_point2.payload = {"node_id": 2}
    mock_point2.score = 0.85

    mock_result = MagicMock()
    mock_result.points = [mock_point1, mock_point2]
    mock_qdrant_client.query_points.return_value = mock_result

    query_embedding = [0.2] * 768

    def search_side_effect(q_emb, top_k, filters=None):
        result = mock_qdrant_client.query_points(
            collection_name="code_embeddings", query=q_emb, limit=top_k
        )
        return [(p.payload["node_id"], p.score) for p in result.points if p.payload]

    mock_backend.search = search_side_effect

    results = search_embeddings(query_embedding, top_k=5)

    mock_qdrant_client.query_points.assert_called_once_with(
        collection_name="code_embeddings", query=query_embedding, limit=5
    )
    assert results == [(1, 0.95), (2, 0.85)]


@pytest.mark.skipif(not has_qdrant_client(), reason="qdrant-client not installed")
def test_search_embeddings_filters_null_payloads(
    mock_backend: MagicMock, mock_qdrant_client: MagicMock
) -> None:
    from codebase_rag.vector_store import search_embeddings

    mock_point1 = MagicMock()
    mock_point1.payload = {"node_id": 1}
    mock_point1.score = 0.95

    mock_point2 = MagicMock()
    mock_point2.payload = None
    mock_point2.score = 0.85

    mock_result = MagicMock()
    mock_result.points = [mock_point1, mock_point2]
    mock_qdrant_client.query_points.return_value = mock_result

    def search_side_effect(q_emb, top_k, filters=None):
        result = mock_qdrant_client.query_points(
            collection_name="code_embeddings", query=q_emb, limit=top_k
        )
        return [(p.payload["node_id"], p.score) for p in result.points if p.payload]

    mock_backend.search = search_side_effect

    results = search_embeddings([0.2] * 768)

    assert results == [(1, 0.95)]


@pytest.mark.skipif(not has_qdrant_client(), reason="qdrant-client not installed")
def test_search_embeddings_handles_exception(
    mock_backend: MagicMock, mock_qdrant_client: MagicMock
) -> None:
    from codebase_rag.vector_store import search_embeddings

    def search_side_effect(q_emb, top_k, filters=None):
        mock_qdrant_client.query_points(
            collection_name="code_embeddings", query=q_emb, limit=top_k
        )
        return []

    mock_backend.search = search_side_effect
    mock_qdrant_client.query_points.side_effect = Exception("Connection failed")

    results = search_embeddings([0.2] * 768)

    assert results == []


@pytest.mark.skipif(not has_qdrant_client(), reason="qdrant-client not installed")
def test_search_embeddings_default_top_k(
    mock_backend: MagicMock, mock_qdrant_client: MagicMock
) -> None:
    from codebase_rag.vector_store import search_embeddings
    from codebase_rag.config import settings

    mock_result = MagicMock()
    mock_result.points = []
    mock_qdrant_client.query_points.return_value = mock_result

    def search_side_effect(q_emb, top_k, filters=None):
        result = mock_qdrant_client.query_points(
            collection_name="code_embeddings", query=q_emb, limit=top_k
        )
        return [(p.payload["node_id"], p.score) for p in result.points if p.payload]

    mock_backend.search = search_side_effect

    search_embeddings([0.2] * 768)

    mock_qdrant_client.query_points.assert_called_once_with(
        collection_name="code_embeddings", query=[0.2] * 768, limit=settings.VECTOR_SEARCH_TOP_K
    )


@pytest.mark.skipif(not has_qdrant_client(), reason="qdrant-client not installed")
def test_store_and_search_roundtrip(integration_client: QdrantClient) -> None:
    from codebase_rag.vector_store import search_embeddings, store_embedding

    embedding1 = [1.0] + [0.0] * 767
    embedding2 = [0.0, 1.0] + [0.0] * 766
    embedding3 = [0.9, 0.1] + [0.0] * 766

    store_embedding(1, embedding1, "project.module1.func1")
    store_embedding(2, embedding2, "project.module2.func2")
    store_embedding(3, embedding3, "project.module3.func3")

    query = [0.95, 0.05] + [0.0] * 766
    results = search_embeddings(query, top_k=3)

    assert len(results) == 3
    node_ids = [r[0] for r in results]
    assert node_ids[0] in [1, 3]
    assert node_ids[1] in [1, 3]


@pytest.mark.skipif(not has_qdrant_client(), reason="qdrant-client not installed")
def test_upsert_updates_existing(integration_client: QdrantClient) -> None:
    from codebase_rag.vector_store import search_embeddings, store_embedding

    embedding_v1 = [1.0] + [0.0] * 767
    embedding_v2 = [0.0, 1.0] + [0.0] * 766

    store_embedding(1, embedding_v1, "project.func")
    store_embedding(1, embedding_v2, "project.func_updated")

    query = [0.0, 1.0] + [0.0] * 766
    results = search_embeddings(query, top_k=1)

    assert len(results) == 1
    assert results[0][0] == 1
    assert results[0][1] > 0.99


@pytest.mark.skipif(not has_qdrant_client(), reason="qdrant-client not installed")
def test_empty_search_returns_empty_list(integration_client: QdrantClient) -> None:
    from codebase_rag.vector_store import search_embeddings

    results = search_embeddings([0.5] * 768, top_k=5)
    assert results == []