from unittest.mock import patch

from codebase_rag.graph_algorithms import GraphAlgorithms


def test_run_community_detection_falls_back_to_louvain() -> None:
    algo = GraphAlgorithms()

    with patch.object(
        algo,
        "_execute_query",
        side_effect=[Exception("Leiden failed"), [{"updated_count": 4}]],
    ) as mock_execute:
        assert algo.run_community_detection() == 4

    assert mock_execute.call_count == 2
    assert "CALL leiden_community_detection.get()" in mock_execute.call_args_list[0].args[0]
    assert "CALL community_detection.get()" in mock_execute.call_args_list[1].args[0]


def test_run_community_detection_skips_missing_procedures() -> None:
    algo = GraphAlgorithms()

    with patch.object(
        algo,
        "_execute_query",
        side_effect=[
            Exception("There is no procedure named 'leiden_community_detection.get'."),
            Exception("There is no procedure named 'community_detection.get'."),
        ],
    ) as mock_execute:
        assert algo.run_community_detection() == 0

    assert mock_execute.call_count == 2


def test_run_community_detection_uses_memgraph_call_syntax() -> None:
    algo = GraphAlgorithms()

    with patch.object(
        algo,
        "_execute_query",
        return_value=[{"updated_count": 3}],
    ) as mock_execute:
        assert algo.run_community_detection() == 3

    assert mock_execute.call_count == 1
    cypher = mock_execute.call_args.args[0]
    assert "CALL leiden_community_detection.get()" in cypher
    assert "weight_property:" not in cypher


def test_run_community_detection_downgrades_no_communities_to_info() -> None:
    algo = GraphAlgorithms()

    with (
        patch.object(
            algo,
            "_execute_query",
            side_effect=[
                Exception(
                    "leiden_community_detection.get: No communities detected."
                ),
                [{"updated_count": 4}],
            ],
        ) as mock_execute,
        patch("codebase_rag.graph_algorithms.logger") as mock_logger,
    ):
        assert algo.run_community_detection() == 4

    assert mock_execute.call_count == 2
    mock_logger.warning.assert_not_called()


def test_run_community_detection_returns_zero_when_no_algorithms_find_communities() -> None:
    algo = GraphAlgorithms()

    with (
        patch.object(
            algo,
            "_execute_query",
            side_effect=[
                Exception(
                    "leiden_community_detection.get: No communities detected."
                ),
                Exception("community_detection.get: No communities detected."),
            ],
        ) as mock_execute,
        patch("codebase_rag.graph_algorithms.logger") as mock_logger,
    ):
        assert algo.run_community_detection() == 0

    assert mock_execute.call_count == 2
    mock_logger.warning.assert_not_called()
    mock_logger.error.assert_not_called()
