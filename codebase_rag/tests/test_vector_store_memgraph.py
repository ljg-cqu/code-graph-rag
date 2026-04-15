from unittest.mock import MagicMock, patch

from codebase_rag.config import settings
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


def test_initialize_skips_recreate_when_existing_dimension_matches_as_string() -> None:
    backend = MemgraphBackend()
    backend.LABELS_TO_INDEX = ("Function",)
    effective_dim = settings.get_effective_vector_dim()
    query_generator = MagicMock()
    query_generator.capabilities.supports_vector_index = True
    backend._query_generator = query_generator

    with patch.object(
        backend,
        "_execute_query",
        return_value=[
            {"index_name": "function_embedding_index", "dimension": str(effective_dim)}
        ],
    ) as mock_execute:
        backend.initialize()

    queries = [call.args[0] for call in mock_execute.call_args_list]
    assert "SHOW VECTOR INDEX INFO;" in queries
    assert not any(query.startswith("DROP VECTOR INDEX") for query in queries)
    assert not any("SET n.embedding = NULL" in query for query in queries)
    query_generator.generate_vector_index_creation_query.assert_not_called()


def test_store_single_reports_specific_recreate_command() -> None:
    backend = MemgraphBackend()

    with (
        patch.object(
            backend,
            "_execute_query",
            side_effect=Exception(
                "vector index property must have the same number of dimensions"
            ),
        ),
        patch("codebase_rag.vector_store_memgraph.logger.warning") as mock_warning,
    ):
        stored = backend._store_single(1, [0.1, 0.2], "mod.func")

    assert stored == 0
    warning_message = mock_warning.call_args.args[0]
    assert "cgr vector recreate-indexes --code" in warning_message
