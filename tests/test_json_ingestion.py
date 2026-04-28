"""Tests for JSON ingestion relationship counting and edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.json_ingestion import (
    DatasetReferences,
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
