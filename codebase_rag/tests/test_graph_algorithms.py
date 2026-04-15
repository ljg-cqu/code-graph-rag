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


def test_run_community_detection_skips_missing_procedures() -> None:
    algo = GraphAlgorithms()

    with patch.object(
        algo,
        "_execute_query",
        side_effect=[
            Exception("There is no procedure named 'graph_algorithms.leiden'."),
            Exception("There is no procedure named 'graph_algorithms.louvain'."),
        ],
    ) as mock_execute:
        assert algo.run_community_detection() == 0

    assert mock_execute.call_count == 2
