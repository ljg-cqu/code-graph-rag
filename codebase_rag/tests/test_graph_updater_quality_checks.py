from pathlib import Path
from unittest.mock import MagicMock, patch

from codebase_rag import constants as cs
from codebase_rag.graph_updater import GraphUpdater
from codebase_rag.parser_loader import load_parsers
from codebase_rag.services.graph_service import MemgraphIngestor


def test_run_uses_all_embeddable_labels_for_quality_validation(
    temp_repo: Path,
) -> None:
    query_ingestor = MagicMock(spec=MemgraphIngestor)
    parsers, queries = load_parsers()
    updater = GraphUpdater(
        ingestor=query_ingestor,
        repo_path=temp_repo,
        parsers=parsers,
        queries=queries,
    )
    checker_instance = MagicMock()
    checker_instance.validate_ingestion_quality.return_value = []

    with (
        patch.object(updater, "_process_files"),
        patch.object(updater, "_process_function_calls"),
        patch.object(query_ingestor, "flush_all"),
        patch.object(updater, "_repair_legacy_function_parent_relationships"),
        patch.object(updater, "_prune_orphan_nodes"),
        patch.object(updater, "_generate_semantic_embeddings"),
        patch.object(updater, "_run_post_ingestion_algorithms"),
        patch("codebase_rag.graph_updater.HealthChecker", return_value=checker_instance),
    ):
        updater.run()

    checker_instance.validate_ingestion_quality.assert_called_once_with(
        embedded_node_label="|".join(cs.EMBEDDABLE_CODE_NODE_LABELS)
    )
