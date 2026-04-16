from __future__ import annotations

from unittest.mock import MagicMock, patch

from codebase_rag.config import settings


def test_store_embedding_wraps_single_point() -> None:
    from codebase_rag.vector_store import store_embedding

    with patch(
        "codebase_rag.vector_store.store_embedding_batch",
        return_value=1,
    ) as mock_store_batch:
        store_embedding(123, [0.1, 0.2], "pkg.module.function")

    mock_store_batch.assert_called_once_with(
        [(123, [0.1, 0.2], "pkg.module.function")]
    )


def test_store_embedding_batch_delegates_to_backend() -> None:
    from codebase_rag.vector_store import store_embedding_batch

    backend = MagicMock()
    backend.store_batch.return_value = 2
    points = [
        (1, [0.1, 0.2], "mod.func1"),
        (2, [0.3, 0.4], "mod.func2"),
    ]

    with patch("codebase_rag.vector_store._get_backend", return_value=backend):
        assert store_embedding_batch(points) == 2

    backend.store_batch.assert_called_once_with(points)


def test_store_embedding_batch_returns_zero_for_empty_input() -> None:
    from codebase_rag.vector_store import store_embedding_batch

    with patch("codebase_rag.vector_store._get_backend") as mock_get_backend:
        assert store_embedding_batch([]) == 0

    mock_get_backend.assert_not_called()


def test_delete_project_embeddings_skips_empty_ids() -> None:
    from codebase_rag.vector_store import delete_project_embeddings

    with patch("codebase_rag.vector_store._get_backend") as mock_get_backend:
        delete_project_embeddings("project", [])

    mock_get_backend.assert_not_called()


def test_verify_stored_ids_delegates_to_backend() -> None:
    from codebase_rag.vector_store import verify_stored_ids

    backend = MagicMock()
    backend.verify_ids.return_value = {1, 3}

    with patch("codebase_rag.vector_store._get_backend", return_value=backend):
        assert verify_stored_ids({1, 2, 3}) == {1, 3}

    backend.verify_ids.assert_called_once_with({1, 2, 3})


def test_search_embeddings_uses_default_top_k() -> None:
    from codebase_rag.vector_store import search_embeddings

    backend = MagicMock()
    backend.search.return_value = [(7, 0.91)]
    query_embedding = [0.2, 0.4]

    with patch("codebase_rag.vector_store._get_backend", return_value=backend):
        assert search_embeddings(query_embedding) == [(7, 0.91)]

    backend.search.assert_called_once_with(
        query_embedding,
        settings.VECTOR_SEARCH_TOP_K,
    )


def test_search_embeddings_returns_empty_on_backend_error() -> None:
    from codebase_rag.vector_store import search_embeddings

    backend = MagicMock()
    backend.search.side_effect = RuntimeError("boom")

    with patch("codebase_rag.vector_store._get_backend", return_value=backend):
        assert search_embeddings([0.2, 0.4], top_k=3) == []

    backend.search.assert_called_once_with([0.2, 0.4], 3)


def test_close_vector_backend_resets_cached_backend() -> None:
    import codebase_rag.vector_store as vs

    vs._BACKEND = MagicMock()

    with patch("codebase_rag.vector_store.close_shared_backend") as mock_close:
        vs.close_vector_backend()

    assert vs._BACKEND is None
    mock_close.assert_called_once_with()
