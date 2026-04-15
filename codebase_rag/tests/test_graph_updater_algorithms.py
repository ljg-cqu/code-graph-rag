from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers


@pytest.fixture
def updater(temp_repo: Path, mock_ingestor: MagicMock) -> GraphUpdater:
    parsers, queries = load_parsers()
    return GraphUpdater(
        ingestor=mock_ingestor,
        repo_path=temp_repo,
        parsers=parsers,
        queries=queries,
    )


def test_post_ingestion_algorithms_respect_disable_flags(updater: GraphUpdater) -> None:
    mock_algo = MagicMock()
    mock_algo.count_nodes.return_value = 42
    mock_algo.run_pagerank.return_value = 0
    mock_algo.run_community_detection.return_value = 0

    with (
        patch(
            "codebase_rag.graph_algorithms.get_shared_algorithms",
            return_value=mock_algo,
        ),
        patch("codebase_rag.graph_updater.settings.ALGORITHM_ENABLE_PAGERANK", False),
        patch(
            "codebase_rag.graph_updater.settings.ALGORITHM_ENABLE_COMMUNITY_DETECTION",
            False,
        ),
    ):
        updater._run_post_ingestion_algorithms()

    mock_algo.analyze_graph.assert_called_once()
    mock_algo.run_pagerank.assert_not_called()
    mock_algo.count_nodes.assert_not_called()
    mock_algo.run_community_detection.assert_not_called()
    mock_algo.close.assert_called_once()


def test_post_ingestion_master_flag_skips_optional_algorithms(
    updater: GraphUpdater,
) -> None:
    mock_algo = MagicMock()
    mock_algo.count_nodes.return_value = 42
    mock_algo.run_pagerank.return_value = 9
    mock_algo.run_community_detection.return_value = 6

    with (
        patch(
            "codebase_rag.graph_algorithms.get_shared_algorithms",
            return_value=mock_algo,
        ),
        patch(
            "codebase_rag.graph_updater.settings.ALGORITHM_RUN_POST_INGESTION",
            False,
        ),
        patch("codebase_rag.graph_updater.settings.ALGORITHM_ENABLE_PAGERANK", True),
        patch(
            "codebase_rag.graph_updater.settings.ALGORITHM_ENABLE_COMMUNITY_DETECTION",
            True,
        ),
    ):
        updater._run_post_ingestion_algorithms()

    mock_algo.analyze_graph.assert_called_once()
    mock_algo.run_pagerank.assert_not_called()
    mock_algo.count_nodes.assert_not_called()
    mock_algo.run_community_detection.assert_not_called()
    mock_algo.close.assert_called_once()


def test_post_ingestion_algorithms_use_configured_louvain(
    updater: GraphUpdater,
) -> None:
    mock_algo = MagicMock()
    mock_algo.count_nodes.return_value = 25
    mock_algo.run_pagerank.return_value = 0
    mock_algo.run_community_detection.return_value = 7

    with (
        patch(
            "codebase_rag.graph_algorithms.get_shared_algorithms",
            return_value=mock_algo,
        ),
        patch("codebase_rag.graph_updater.settings.ALGORITHM_ENABLE_PAGERANK", False),
        patch(
            "codebase_rag.graph_updater.settings.ALGORITHM_ENABLE_COMMUNITY_DETECTION",
            True,
        ),
        patch(
            "codebase_rag.graph_updater.settings.ALGORITHM_COMMUNITY_ALGORITHM",
            "louvain",
        ),
    ):
        updater._run_post_ingestion_algorithms()

    mock_algo.analyze_graph.assert_called_once()
    mock_algo.run_pagerank.assert_not_called()
    mock_algo.count_nodes.assert_called_once()
    mock_algo.run_community_detection.assert_called_once_with(use_leiden=False)
    mock_algo.close.assert_called_once()


def test_post_ingestion_algorithms_default_invalid_algorithm_to_leiden(
    updater: GraphUpdater,
) -> None:
    mock_algo = MagicMock()
    mock_algo.count_nodes.return_value = 25
    mock_algo.run_pagerank.return_value = 11
    mock_algo.run_community_detection.return_value = 6

    with (
        patch(
            "codebase_rag.graph_algorithms.get_shared_algorithms",
            return_value=mock_algo,
        ),
        patch("codebase_rag.graph_updater.settings.ALGORITHM_ENABLE_PAGERANK", True),
        patch(
            "codebase_rag.graph_updater.settings.ALGORITHM_ENABLE_COMMUNITY_DETECTION",
            True,
        ),
        patch(
            "codebase_rag.graph_updater.settings.ALGORITHM_COMMUNITY_ALGORITHM",
            "unknown",
        ),
    ):
        updater._run_post_ingestion_algorithms()

    mock_algo.run_pagerank.assert_called_once()
    mock_algo.count_nodes.assert_called_once()
    mock_algo.run_community_detection.assert_called_once_with(use_leiden=True)
    mock_algo.close.assert_called_once()
