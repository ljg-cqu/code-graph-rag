from unittest.mock import patch

from codebase_rag.vector_store_memgraph import MemgraphBackend


def test_store_batch_falls_back_to_individual_writes_on_batch_failure() -> None:
    backend = MemgraphBackend()
    points = [
        (1, [0.1, 0.2], "mod.func1"),
        (2, [0.3, 0.4], "mod.func2"),
    ]

    with (
        patch.object(
            backend,
            "_execute_query",
            side_effect=[Exception("batch failure"), [{"stored": 1}], [{"stored": 1}]],
        ) as mock_execute,
        patch.object(backend, "verify_ids", return_value={1, 2}) as mock_verify,
    ):
        assert backend.store_batch(points) == 2

    assert mock_execute.call_count == 3
    assert mock_verify.call_count == 1


def test_store_batch_retries_missing_points_after_partial_write() -> None:
    backend = MemgraphBackend()
    points = [
        (1, [0.1, 0.2], "mod.func1"),
        (2, [0.3, 0.4], "mod.func2"),
    ]

    with (
        patch.object(
            backend,
            "_execute_query",
            side_effect=[[{"stored": 1}], [{"stored": 1}]],
        ) as mock_execute,
        patch.object(backend, "verify_ids", side_effect=[{1}, {1, 2}]) as mock_verify,
    ):
        assert backend.store_batch(points) == 2

    assert mock_execute.call_count == 2
    assert mock_verify.call_count == 2
