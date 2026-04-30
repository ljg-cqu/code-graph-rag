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
        Exception("There is no procedure named 'leiden_community_detection.get'.")
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


def test_dynamic_community_recalculation_returns_no_communities_when_detected() -> None:
    mock_ingestor = _make_mock_ingestor(
        Exception("leiden_community_detection.get: No communities detected.")
    )

    with patch(
        "codebase_rag.memgraph_advanced.dynamic_algorithms.MemgraphIngestor",
        return_value=mock_ingestor,
    ):
        algo = DynamicGraphAlgorithms(use_dynamic=False)

        assert algo.update_communities_dynamic(None) == {
            "updated_communities": 0,
            "total_nodes_updated": 0,
            "method": "no_communities",
        }


def test_qfs_returns_empty_summaries_when_community_detection_is_unavailable() -> None:
    provider = MagicMock()
    provider.create_model.return_value = MagicMock()
    mock_ingestor = _make_mock_ingestor(
        Exception("There is no procedure named 'leiden_community_detection.get'.")
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


def test_qfs_returns_empty_summaries_when_no_communities_detected() -> None:
    provider = MagicMock()
    provider.create_model.return_value = MagicMock()
    mock_ingestor = _make_mock_ingestor(
        Exception("leiden_community_detection.get: No communities detected.")
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


def test_dynamic_community_recalculation_uses_memgraph_call_syntax() -> None:
    mock_ingestor = _make_mock_ingestor([[{"community_id": 1, "size": 2}]])

    with patch(
        "codebase_rag.memgraph_advanced.dynamic_algorithms.MemgraphIngestor",
        return_value=mock_ingestor,
    ):
        algo = DynamicGraphAlgorithms(use_dynamic=False)

        assert algo.update_communities_dynamic(None) == {
            "updated_communities": 1,
            "total_nodes_updated": 2,
            "method": "full",
        }

    cypher = mock_ingestor.fetch_all.call_args.args[0]
    assert "CALL leiden_community_detection.get()" in cypher
    assert "weight_property:" not in cypher


def test_qfs_uses_memgraph_call_syntax() -> None:
    provider = MagicMock()
    provider.create_model.return_value = MagicMock()
    mock_ingestor = _make_mock_ingestor([[]])

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

    cypher = mock_ingestor.fetch_all.call_args.args[0]
    assert "CALL leiden_community_detection.get()" in cypher
    assert "weight_property:" not in cypher
