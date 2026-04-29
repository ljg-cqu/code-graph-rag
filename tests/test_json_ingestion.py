"""Tests for JSON ingestion relationship counting and edge cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.json_ingestion import (
    DatasetReferences,
    OperationSummary,
    _get_actual_relationship_count,
    ingest_json_data,
    ingest_relationships,
)


class TestIngestRelationshipsCounting:
    """Tests for relationship ingestion counting, including symmetric edges."""

    @pytest.fixture
    def mock_graph(self):
        """Return a mocked graph connection."""
        conn = MagicMock()
        conn.fetch_all.return_value = [{"relationship_id": 42}]
        return conn

    @pytest.fixture
    def dataset_refs(self):
        """Return dataset references with resolved entities."""
        refs = DatasetReferences()
        refs.ids = {"entity_a": "ds::entity_a", "entity_b": "ds::entity_b"}
        return refs

    def test_non_symmetric_relationship_ingested(self, mock_graph, dataset_refs):
        """A non-symmetric relationship counts as 1 ingested."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "relates_to",
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        assert summary.ingested == 1
        assert summary.updated == 0
        assert summary.failed == 0
        assert mock_graph.fetch_all.call_count == 1

    def test_relationship_properties_include_source_and_target_ids(self, mock_graph, dataset_refs):
        """source_id and target_id are included in rel_props passed to MERGE."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "relates_to",
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        assert summary.ingested == 1
        args, _kwargs = mock_graph.fetch_all.call_args
        params = args[1]
        assert params["properties"].get("source_id") == "ds::entity_a"
        assert params["properties"].get("target_id") == "ds::entity_b"

    def test_symmetric_relationship_properties_include_source_and_target_ids(self, mock_graph, dataset_refs):
        """Symmetric relationships include source_id and target_id in both MERGE calls."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "similar_to",
                "symmetric": True,
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        assert summary.ingested == 2
        assert mock_graph.fetch_all.call_count == 2
        for call in mock_graph.fetch_all.call_args_list:
            args, _kwargs = call
            params = args[1]
            assert params["properties"].get("source_id") == "ds::entity_a"
            assert params["properties"].get("target_id") == "ds::entity_b"

    def test_symmetric_relationship_both_edges_ingested(self, mock_graph, dataset_refs):
        """A symmetric relationship with no existing edges counts as 2 ingested."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "similar_to",
                "symmetric": True,
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        assert summary.ingested == 2
        assert summary.updated == 0
        assert summary.failed == 0
        assert mock_graph.fetch_all.call_count == 2

    def test_symmetric_forward_exists_reverse_new(self, mock_graph, dataset_refs):
        """Symmetric relationship where forward exists but reverse is new.

        Should count as 1 updated (forward) + 1 ingested (reverse).
        """
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "similar_to",
                "symmetric": True,
            }
        ]
        existing = {
            ("ds::entity_a", "similar_to", "ds::entity_b"): None,
        }

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value=existing,
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        assert summary.ingested == 1
        assert summary.updated == 1
        assert summary.failed == 0
        assert mock_graph.fetch_all.call_count == 2

    def test_symmetric_both_edges_exist(self, mock_graph, dataset_refs):
        """Symmetric relationship where both forward and reverse exist.

        Should count as 2 updated.
        """
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "similar_to",
                "symmetric": True,
            }
        ]
        existing = {
            ("ds::entity_a", "similar_to", "ds::entity_b"): None,
            ("ds::entity_b", "similar_to", "ds::entity_a"): None,
        }

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value=existing,
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        assert summary.ingested == 0
        assert summary.updated == 2
        assert summary.failed == 0
        assert mock_graph.fetch_all.call_count == 2

    def test_dry_run_symmetric_counts_both_edges(self, dataset_refs):
        """Dry run with symmetric relationship counts both edges."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "similar_to",
                "symmetric": True,
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                dry_run=True,
            )

        assert summary.ingested == 2
        assert summary.updated == 0
        assert summary.skipped == 0
        assert summary.failed == 0

    def test_dry_run_symmetric_reverse_exists(self, mock_graph, dataset_refs):
        """Dry run with symmetric relationship where reverse edge exists.

        skip_existing=True is used so that existing_relationships is fetched
        for the dry run, allowing accurate reverse-edge detection.
        """
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "similar_to",
                "symmetric": True,
            }
        ]
        existing = {
            ("ds::entity_b", "similar_to", "ds::entity_a"): None,
        }

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value=existing,
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                dry_run=True,
                skip_existing=True,
                graph_connection=mock_graph,
            )

        assert summary.ingested == 1
        assert summary.updated == 1
        assert summary.skipped == 0
        assert summary.failed == 0

    def test_skip_existing_symmetric_skips_whole_relationship(self, mock_graph, dataset_refs):
        """skip_existing=True skips the entire symmetric relationship if forward exists."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "similar_to",
                "symmetric": True,
            }
        ]
        existing = {
            ("ds::entity_a", "similar_to", "ds::entity_b"): None,
        }

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value=existing,
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                skip_existing=True,
                graph_connection=mock_graph,
            )

        assert summary.skipped == 1
        assert summary.ingested == 0
        assert summary.updated == 0
        assert mock_graph.fetch_all.call_count == 0

    def test_multiple_mixed_relationships(self, mock_graph, dataset_refs):
        """Mixed batch of symmetric and non-symmetric relationships."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "relates_to",
            },
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "similar_to",
                "symmetric": True,
            },
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        assert summary.ingested == 3
        assert summary.updated == 0
        assert summary.failed == 0
        assert mock_graph.fetch_all.call_count == 3

    def test_failed_lookup_does_not_count(self, dataset_refs):
        """Unresolved references count as failed, not ingested."""
        relationships = [
            {
                "source": "unknown_entity",
                "target": "entity_b",
                "relationship": "relates_to",
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            summary = ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
            )

        assert summary.failed == 1
        assert summary.ingested == 0
        assert summary.updated == 0


class TestGetActualRelationshipCount:
    """Tests for post-ingestion relationship count verification."""

    def test_returns_count_for_single_dataset_id(self) -> None:
        """Should query count for a single dataset ID."""
        mock_conn = MagicMock()
        mock_conn.fetch_all.return_value = [{"actual_count": 41}]

        with patch(
            "codebase_rag.json_ingestion._create_json_ingestor",
            return_value=MagicMock(__enter__=MagicMock(return_value=mock_conn), __exit__=MagicMock(return_value=False)),
        ):
            result = _get_actual_relationship_count(["ds-1"])

        assert result == 41
        mock_conn.fetch_all.assert_called_once()
        call_args = mock_conn.fetch_all.call_args
        assert "$dataset_id" in str(call_args)

    def test_returns_count_for_multiple_dataset_ids(self) -> None:
        """Should use IN clause for multiple dataset IDs."""
        mock_conn = MagicMock()
        mock_conn.fetch_all.return_value = [{"actual_count": 100}]

        with patch(
            "codebase_rag.json_ingestion._create_json_ingestor",
            return_value=MagicMock(__enter__=MagicMock(return_value=mock_conn), __exit__=MagicMock(return_value=False)),
        ):
            result = _get_actual_relationship_count(["ds-1", "ds-2"])

        assert result == 100
        call_args = mock_conn.fetch_all.call_args
        assert "$dataset_ids" in str(call_args)

    def test_returns_zero_for_empty_dataset_ids(self) -> None:
        """Should return 0 immediately when no dataset IDs provided."""
        result = _get_actual_relationship_count([])
        assert result == 0

    def test_returns_negative_one_on_error(self) -> None:
        """Should return -1 when database query fails."""
        mock_conn = MagicMock()
        mock_conn.fetch_all.side_effect = Exception("connection refused")

        with patch(
            "codebase_rag.json_ingestion._create_json_ingestor",
            return_value=MagicMock(__enter__=MagicMock(return_value=mock_conn), __exit__=MagicMock(return_value=False)),
        ):
            result = _get_actual_relationship_count(["ds-1"])

        assert result == -1

    def test_returns_zero_when_no_rows(self) -> None:
        """Should return 0 when query returns no rows."""
        mock_conn = MagicMock()
        mock_conn.fetch_all.return_value = []

        with patch(
            "codebase_rag.json_ingestion._create_json_ingestor",
            return_value=MagicMock(__enter__=MagicMock(return_value=mock_conn), __exit__=MagicMock(return_value=False)),
        ):
            result = _get_actual_relationship_count(["ds-1"])

        assert result == 0


class TestIngestJsonDataVerification:
    """Tests for post-ingestion relationship count verification in ingest_json_data."""

    @pytest.fixture
    def sample_json_data(self) -> dict[str, object]:
        return {
            "metadata": {"dataset_id": "test-ds"},
            "entities": [
                {"id": "a", "name": "Entity A"},
                {"id": "b", "name": "Entity B"},
            ],
            "relationships": [
                {"source": "Entity A", "target": "Entity B", "relationship": "relates_to"}
            ],
        }

    def test_verification_warns_on_count_mismatch(self, sample_json_data) -> None:
        """Should log warning when logged count differs from stored count."""
        with (
            patch("codebase_rag.json_ingestion._ingest_entity_file", return_value=OperationSummary()),
            patch(
                "codebase_rag.json_ingestion._ingest_relationship_file",
                return_value=OperationSummary(ingested=2),
            ),
            patch("codebase_rag.json_ingestion._get_actual_relationship_count", return_value=1),
            patch("codebase_rag.json_ingestion._create_json_graph_indexes"),
            patch("codebase_rag.json_ingestion._compute_json_graph_pagerank"),
            patch("codebase_rag.json_ingestion.logger") as mock_logger,
        ):
            result = ingest_json_data(
                pre_loaded_data=[(Path("test.json"), sample_json_data)],
                dry_run=False,
            )

        assert result.relationships_ingested == 2
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert any("Relationship count mismatch: logged 2, actually stored 1" in c for c in warning_calls)
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("2 (1 stored) [1 symmetric duplicate merged] relationships ingested" in c for c in info_calls)

    def test_verification_skipped_in_dry_run(self, sample_json_data) -> None:
        """Should skip verification when dry_run=True."""
        with (
            patch("codebase_rag.json_ingestion._ingest_entity_file", return_value=OperationSummary()),
            patch(
                "codebase_rag.json_ingestion._ingest_relationship_file",
                return_value=OperationSummary(ingested=2),
            ),
            patch("codebase_rag.json_ingestion._get_actual_relationship_count") as mock_verify,
            patch("codebase_rag.json_ingestion.logger") as mock_logger,
        ):
            result = ingest_json_data(
                pre_loaded_data=[(Path("test.json"), sample_json_data)],
                dry_run=True,
            )

        assert result.relationships_ingested == 2
        mock_verify.assert_not_called()
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert not any("Relationship count mismatch" in c for c in warning_calls)

    def test_verification_normal_log_when_counts_match(self, sample_json_data) -> None:
        """Should use normal log format when counts match."""
        with (
            patch("codebase_rag.json_ingestion._ingest_entity_file", return_value=OperationSummary()),
            patch(
                "codebase_rag.json_ingestion._ingest_relationship_file",
                return_value=OperationSummary(ingested=2),
            ),
            patch("codebase_rag.json_ingestion._get_actual_relationship_count", return_value=2),
            patch("codebase_rag.json_ingestion._create_json_graph_indexes"),
            patch("codebase_rag.json_ingestion._compute_json_graph_pagerank"),
            patch("codebase_rag.json_ingestion.logger") as mock_logger,
        ):
            result = ingest_json_data(
                pre_loaded_data=[(Path("test.json"), sample_json_data)],
                dry_run=False,
            )

        assert result.relationships_ingested == 2
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert not any("Relationship count mismatch" in c for c in warning_calls)
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("2 relationships ingested" in c for c in info_calls)
        assert not any("stored" in c for c in info_calls)

    def test_verification_handles_negative_one_gracefully(self, sample_json_data) -> None:
        """Should not warn when verification returns -1 (error/unavailable)."""
        with (
            patch("codebase_rag.json_ingestion._ingest_entity_file", return_value=OperationSummary()),
            patch(
                "codebase_rag.json_ingestion._ingest_relationship_file",
                return_value=OperationSummary(ingested=2),
            ),
            patch("codebase_rag.json_ingestion._get_actual_relationship_count", return_value=-1),
            patch("codebase_rag.json_ingestion._create_json_graph_indexes"),
            patch("codebase_rag.json_ingestion._compute_json_graph_pagerank"),
            patch("codebase_rag.json_ingestion.logger") as mock_logger,
        ):
            result = ingest_json_data(
                pre_loaded_data=[(Path("test.json"), sample_json_data)],
                dry_run=False,
            )

        assert result.relationships_ingested == 2
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert not any("Relationship count mismatch" in c for c in warning_calls)
        info_calls = [str(call) for call in mock_logger.info.call_args_list]
        assert any("2 relationships ingested" in c for c in info_calls)
