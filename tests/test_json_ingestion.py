"""Tests for JSON ingestion relationship counting and edge cases."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.json_ingestion import (
    DatasetReferences,
    OperationSummary,
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

    def test_relationship_properties_include_source_and_target_ids(
        self, mock_graph, dataset_refs
    ):
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

    def test_symmetric_relationship_properties_include_source_and_target_ids(
        self, mock_graph, dataset_refs
    ):
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
            ("ds::entity_a", "COMPARATIVE", "ds::entity_b"): None,
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
            ("ds::entity_a", "COMPARATIVE", "ds::entity_b"): None,
            ("ds::entity_b", "COMPARATIVE", "ds::entity_a"): None,
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
            ("ds::entity_b", "COMPARATIVE", "ds::entity_a"): None,
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

    def test_skip_existing_symmetric_skips_whole_relationship(
        self, mock_graph, dataset_refs
    ):
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
            ("ds::entity_a", "COMPARATIVE", "ds::entity_b"): None,
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
                "relationship": "causes",
            },
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "contains",
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


class TestCreatedMatchedCounters:
    """Tests for the new created/matched counter semantics."""

    @pytest.fixture
    def mock_graph(self):
        conn = MagicMock()
        conn.fetch_all.return_value = [{"relationship_id": 42}]
        return conn

    @pytest.fixture
    def dataset_refs(self):
        refs = DatasetReferences()
        refs.ids = {"entity_a": "ds::entity_a", "entity_b": "ds::entity_b"}
        return refs

    def test_operation_summary_backward_compat(self) -> None:
        """Legacy .ingested and .updated properties map to created/matched."""
        summary = OperationSummary(created=5, matched=3)

        assert summary.created == 5
        assert summary.matched == 3
        assert summary.ingested == 5
        assert summary.updated == 3

        summary.ingested = 10
        summary.updated = 7
        assert summary.created == 10
        assert summary.matched == 7

    def test_self_referential_symmetric_counts_one(self, mock_graph, dataset_refs):
        """Self-referential symmetric relationship stores 1 edge, counts 1 created."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_a",
                "relationship": "self_relates",
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

        assert summary.created == 1
        assert summary.matched == 0
        assert summary.failed == 0
        assert mock_graph.fetch_all.call_count == 1

    def test_intra_batch_duplicate_counts_as_matched(self, mock_graph, dataset_refs):
        """Duplicate relationship in same batch counts second as matched."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "relates_to",
            },
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "relates_to",
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

        assert summary.created == 1
        assert summary.matched == 1
        assert summary.failed == 0
        assert mock_graph.fetch_all.call_count == 2


class TestReingestionNoFalseWarnings:
    """Tests that re-ingestion does not produce false mismatch warnings."""

    @pytest.fixture
    def sample_json_data(self):
        return {
            "metadata": {"dataset_id": "test-ds"},
            "entities": [
                {"id": "a", "name": "Entity A"},
                {"id": "b", "name": "Entity B"},
            ],
            "relationships": [
                {
                    "source": "Entity A",
                    "target": "Entity B",
                    "relationship": "relates_to",
                }
            ],
        }

    def test_reingestion_same_dataset_no_warning(self, sample_json_data):
        """Re-ingesting same dataset must NOT produce mismatch warning."""
        with (
            patch(
                "codebase_rag.json_ingestion._ingest_entity_file",
                return_value=OperationSummary(),
            ),
            patch(
                "codebase_rag.json_ingestion._ingest_relationship_file",
                return_value=OperationSummary(created=2),
            ),
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
        assert any("2 created, 0 matched" in c for c in info_calls)

    def test_reingestion_matched_relationships_no_warning(self, sample_json_data):
        """Re-ingestion where all relationships match must NOT produce mismatch warning."""
        with (
            patch(
                "codebase_rag.json_ingestion._ingest_entity_file",
                return_value=OperationSummary(),
            ),
            patch(
                "codebase_rag.json_ingestion._ingest_relationship_file",
                return_value=OperationSummary(matched=2),
            ),
            patch("codebase_rag.json_ingestion._create_json_graph_indexes"),
            patch("codebase_rag.json_ingestion._compute_json_graph_pagerank"),
            patch("codebase_rag.json_ingestion.logger") as mock_logger,
        ):
            result = ingest_json_data(
                pre_loaded_data=[(Path("test.json"), sample_json_data)],
                dry_run=False,
            )

        assert result.relationships_matched == 2
        warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
        assert not any("Relationship count mismatch" in c for c in warning_calls)


class TestIngestionResultBackwardCompat:
    """Tests for IngestionResult backward compatibility."""

    def test_ingestion_result_backward_compat(self) -> None:
        """Verify IngestionResult backward compatibility properties."""
        from codebase_rag.schemas.models import IngestionResult

        result = IngestionResult(
            relationships_created=10,
            relationships_matched=5,
        )

        assert result.relationships_created == 10
        assert result.relationships_matched == 5
        assert result.ingested == 10
        assert result.updated == 5

    def test_ingestion_result_legacy_field_aliases(self) -> None:
        """Verify legacy fields relationships_ingested/relationships_updated still work.

        These are kept for backward compatibility with existing code
        that accesses result.relationships_ingested directly.
        """
        from codebase_rag.schemas.models import IngestionResult

        result = IngestionResult(relationships_ingested=7, relationships_updated=2)

        assert result.relationships_ingested == 7
        assert result.relationships_updated == 2
        assert result.relationships_created == 7
        assert result.relationships_matched == 2


class TestRelationshipEdgeLabels:
    """Tests that relationship edge labels use canonical MECE categories."""

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

    def test_category_field_used_for_edge_label(self, mock_graph, dataset_refs):
        """When category is present, edge label uses canonical MECE type."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "analogous_to",
                "category": "ANALOGICAL",
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        cypher = mock_graph.fetch_all.call_args[0][0]
        assert "-[r:`ANALOGICAL` {" in cypher
        assert "analogous_to" not in cypher.split("MERGE")[1]

    def test_verb_fallback_for_edge_label(self, mock_graph, dataset_refs):
        """When category is missing, edge label falls back to normalized verb."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "causes",
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        cypher = mock_graph.fetch_all.call_args[0][0]
        assert "-[r:`CAUSAL` {" in cypher

    def test_unknown_verb_maps_to_related_to(self, mock_graph, dataset_refs):
        """Unknown verbs without category map to RELATED_TO edge label."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "some_random_verb",
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        cypher = mock_graph.fetch_all.call_args[0][0]
        assert "-[r:`RELATED_TO` {" in cypher

    def test_comparable_to_maps_to_comparative(self, mock_graph, dataset_refs):
        """COMPARABLE_TO verb maps to COMPARATIVE edge label."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "comparable_to",
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        cypher = mock_graph.fetch_all.call_args[0][0]
        assert "-[r:`COMPARATIVE` {" in cypher

    def test_verb_preserved_in_properties(self, mock_graph, dataset_refs):
        """Original verb is preserved in relationship properties."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "analogous_to",
                "category": "ANALOGICAL",
            }
        ]

        with patch(
            "codebase_rag.json_ingestion._fetch_existing_relationships",
            return_value={},
        ):
            ingest_relationships(
                dataset_id="ds",
                relationships=relationships,
                dataset_references=dataset_refs,
                metadata={},
                graph_connection=mock_graph,
            )

        params = mock_graph.fetch_all.call_args[0][1]
        assert params["properties"]["verb"] == "analogous_to"
        assert params["properties"]["category"] == "ANALOGICAL"

    def test_symmetric_relationship_uses_canonical_types(self, mock_graph, dataset_refs):
        """Symmetric relationships use canonical edge labels for both directions."""
        relationships = [
            {
                "source": "entity_a",
                "target": "entity_b",
                "relationship": "analogous_to",
                "category": "ANALOGICAL",
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
            cypher = call[0][0]
            assert "-[r:`ANALOGICAL` {" in cypher
