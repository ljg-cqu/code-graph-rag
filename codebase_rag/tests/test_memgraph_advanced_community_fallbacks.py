from __future__ import annotations

from unittest.mock import MagicMock, patch

from codebase_rag.memgraph_advanced.dynamic_algorithms import DynamicGraphAlgorithms
from codebase_rag.memgraph_advanced.qfs import CommunityQFS


def _make_mock_ingestor(side_effect: object) -> MagicMock:
    mock_ingestor = MagicMock()
    mock_ingestor.fetch_all.side_effect = side_effect
    mock_ingestor.__enter__ = MagicMock(return_value=mock_ingestor)
    mock_ingestor.__exit__ = MagicMock(return_value=False)
    return mock_ingestor


def test_dynamic_community_recalculation_skips_missing_procedure() -> None:
    mock_ingestor = _make_mock_ingestor(
        Exception("There is no procedure named 'graph_algorithms.leiden'.")
    )

    with patch(
        "codebase_rag.memgraph_advanced.dynamic_algorithms.MemgraphIngestor",
        return_value=mock_ingestor,
    ):
        algo = DynamicGraphAlgorithms(use_dynamic=False)

        assert algo.update_communities_dynamic(None) == {
            "updated_communities": 0,
            "total_nodes_updated": 0,
            "method": "unavailable",
        }


def test_qfs_returns_empty_summaries_when_community_detection_is_unavailable() -> None:
    provider = MagicMock()
    provider.create_model.return_value = MagicMock()
    mock_ingestor = _make_mock_ingestor(
        Exception("There is no procedure named 'graph_algorithms.leiden'.")
    )

    with (
        patch(
            "codebase_rag.memgraph_advanced.qfs.get_provider_from_config",
            return_value=provider,
        ),
        patch(
            "codebase_rag.memgraph_advanced.qfs.MemgraphIngestor",
            return_value=mock_ingestor,
        ),
    ):
        qfs = CommunityQFS()

        assert qfs.build_community_summaries() == []
        assert (
            qfs.query_focused_summary("What does this code do?")
            == "No code communities are available for summarization in the current graph."
        )