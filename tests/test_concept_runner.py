"""Unit tests for standalone concept extraction runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codebase_rag.document.concept_extraction import (
    ConceptRelationship,
    ExtractedConcept,
    ExtractionResult,
)
from codebase_rag.document.concept_runner import (
    AdaptiveRetryResult,
    ConceptExtractionRunner,
    StandaloneExtractionStats,
)
from codebase_rag.document.error_handling import ErrorType


class TestStandaloneExtractionStats:
    """Test StandaloneExtractionStats dataclass."""

    def test_default_values(self):
        """Test default values are all zero/empty."""
        stats = StandaloneExtractionStats()
        assert stats.total_chunks == 0
        assert stats.processed_chunks == 0
        assert stats.successful_extractions == 0
        assert stats.failed_extractions == 0
        assert stats.concepts_created == 0
        assert stats.relationships_created == 0
        assert stats.skipped_existing == 0
        assert stats.errors_by_type == {}

    def test_custom_values(self):
        """Test custom values are stored correctly."""
        stats = StandaloneExtractionStats(
            total_chunks=100,
            processed_chunks=50,
            successful_extractions=45,
            failed_extractions=5,
            concepts_created=120,
            relationships_created=30,
            skipped_existing=10,
            errors_by_type={"TimeoutError": 3, "RateLimitError": 2},
        )
        assert stats.total_chunks == 100
        assert stats.processed_chunks == 50
        assert stats.successful_extractions == 45
        assert stats.failed_extractions == 5
        assert stats.concepts_created == 120
        assert stats.relationships_created == 30
        assert stats.skipped_existing == 10
        assert stats.errors_by_type == {"TimeoutError": 3, "RateLimitError": 2}


class TestConceptExtractionRunnerInit:
    """Test ConceptExtractionRunner initialization."""

    def test_init_defaults(self, tmp_path: Path):
        """Test initialization with default values."""
        with patch("codebase_rag.document.concept_runner.settings") as mock_settings:
            mock_settings.CONCEPT_MEMGRAPH_BATCH_SIZE = 1000
            mock_settings.DOC_CONCEPT_EXTRACTION_CONCURRENCY = 10

            runner = ConceptExtractionRunner(repo_path=tmp_path)

            assert runner.repo_path == tmp_path
            assert runner.workspace == "default"
            assert runner.batch_size == 1000
            assert runner.concurrency == 10

    def test_init_custom_values(self, tmp_path: Path):
        """Test initialization with custom values."""
        runner = ConceptExtractionRunner(
            repo_path=tmp_path,
            workspace="my-project",
            batch_size=500,
            concurrency=25,
        )

        assert runner.repo_path == tmp_path
        assert runner.workspace == "my-project"
        assert runner.batch_size == 500
        assert runner.concurrency == 25

    def test_workspace_validation_valid(self, tmp_path: Path):
        """Test valid workspace identifiers."""
        valid_workspaces = ["default", "my-project", "project_123", "test-workspace"]
        for ws in valid_workspaces:
            runner = ConceptExtractionRunner(repo_path=tmp_path, workspace=ws)
            assert runner.workspace == ws

    def test_workspace_validation_invalid(self, tmp_path: Path):
        """Test invalid workspace identifiers raise ValueError."""
        invalid_workspaces = ["my project", "project@123", "project.name", ""]
        for ws in invalid_workspaces:
            with pytest.raises(ValueError, match="Invalid workspace identifier"):
                ConceptExtractionRunner(repo_path=tmp_path, workspace=ws)

    def test_concurrency_cap(self, tmp_path: Path):
        """Test concurrency is capped at 50."""
        runner = ConceptExtractionRunner(repo_path=tmp_path, concurrency=100)
        assert runner.concurrency == 50


class TestConnectGraphs:
    """Test dual-graph connection context manager."""

    def test_connect_graphs_success(self, tmp_path: Path):
        """Test successful dual-graph connection."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_doc_ingestor = MagicMock()
        mock_concept_ingestor = MagicMock()

        with patch("codebase_rag.document.concept_runner.connect_doc_memgraph") as mock_connect_doc, \
             patch("codebase_rag.document.concept_runner.connect_concept_memgraph") as mock_connect_concept, \
             patch("codebase_rag.document.concept_runner.settings") as mock_settings:

            mock_settings.CONCEPT_MEMGRAPH_ENABLED = True
            mock_connect_doc.return_value = mock_doc_ingestor
            mock_connect_concept.return_value = mock_concept_ingestor
            mock_doc_ingestor.__enter__.return_value = mock_doc_ingestor
            mock_concept_ingestor.__enter__.return_value = mock_concept_ingestor

            with runner._connect_graphs() as connections:
                assert connections == (mock_doc_ingestor, mock_concept_ingestor)

    def test_connect_graphs_doc_graph_fails(self, tmp_path: Path):
        """Test when document graph connection fails."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        with patch("codebase_rag.document.concept_runner.connect_doc_memgraph") as mock_connect, \
             patch("codebase_rag.document.concept_runner.logger"):

            mock_connect.side_effect = Exception("Connection refused")

            with runner._connect_graphs() as connections:
                assert connections is None

    def test_connect_graphs_concept_disabled(self, tmp_path: Path):
        """Test when concept graph is disabled."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_doc_ingestor = MagicMock()

        with patch("codebase_rag.document.concept_runner.connect_doc_memgraph") as mock_connect_doc, \
             patch("codebase_rag.document.concept_runner.settings") as mock_settings, \
             patch("codebase_rag.document.concept_runner.logger"):

            mock_settings.CONCEPT_MEMGRAPH_ENABLED = False
            mock_connect_doc.return_value = mock_doc_ingestor
            mock_doc_ingestor.__enter__.return_value = mock_doc_ingestor

            with runner._connect_graphs() as connections:
                assert connections is None

    def test_connect_graphs_concept_fails(self, tmp_path: Path):
        """Test when concept graph connection fails."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_doc_ingestor = MagicMock()

        with patch("codebase_rag.document.concept_runner.connect_doc_memgraph") as mock_connect_doc, \
             patch("codebase_rag.document.concept_runner.connect_concept_memgraph") as mock_connect_concept, \
             patch("codebase_rag.document.concept_runner.settings") as mock_settings, \
             patch("codebase_rag.document.concept_runner.logger"):

            mock_settings.CONCEPT_MEMGRAPH_ENABLED = True
            mock_connect_doc.return_value = mock_doc_ingestor
            mock_doc_ingestor.__enter__.return_value = mock_doc_ingestor
            mock_connect_concept.side_effect = Exception("Connection refused")

            with runner._connect_graphs() as connections:
                assert connections is None


class TestInitializeExtractor:
    """Test LLM concept extractor initialization."""

    def test_initialize_disabled(self, tmp_path: Path):
        """Test when concept extraction is disabled."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        with patch("codebase_rag.document.concept_runner.settings") as mock_settings, \
             patch("codebase_rag.document.concept_runner.logger"):

            mock_settings.DOC_CONCEPT_EXTRACTION_ENABLED = False

            result = runner._initialize_extractor()
            assert result is False

    def test_initialize_success(self, tmp_path: Path):
        """Test successful extractor initialization."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        with patch("codebase_rag.document.concept_runner.settings") as mock_settings, \
             patch("codebase_rag.document.concept_runner.LLMConceptExtractor") as mock_extractor, \
             patch("codebase_rag.document.concept_runner.DeadLetterQueue"):

            mock_settings.DOC_CONCEPT_EXTRACTION_ENABLED = True
            mock_extractor.return_value = MagicMock()

            result = runner._initialize_extractor()
            assert result is True
            assert runner.extractor is not None

    def test_initialize_creates_cgr_dir(self, tmp_path: Path):
        """Test that .cgr directory is created."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        with patch("codebase_rag.document.concept_runner.settings") as mock_settings, \
             patch("codebase_rag.document.concept_runner.LLMConceptExtractor") as mock_extractor, \
             patch("codebase_rag.document.concept_runner.DeadLetterQueue"):

            mock_settings.DOC_CONCEPT_EXTRACTION_ENABLED = True
            mock_extractor.return_value = MagicMock()

            runner._initialize_extractor()
            assert (tmp_path / ".cgr").exists()


class TestGetAllChunks:
    """Test chunk query from document graph."""

    def test_get_all_chunks(self, tmp_path: Path):
        """Test querying all chunks from document graph."""
        runner = ConceptExtractionRunner(repo_path=tmp_path, workspace="test")

        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = [
            {"qualified_name": "doc1:chunk1", "content": "content1"},
            {"qualified_name": "doc1:chunk2", "content": "content2"},
        ]

        chunks = runner._get_all_chunks(mock_ingestor)

        assert len(chunks) == 2
        assert chunks[0]["qualified_name"] == "doc1:chunk1"
        mock_ingestor.fetch_all.assert_called_once()

    def test_get_all_chunks_empty(self, tmp_path: Path):
        """Test when no chunks exist."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = []

        chunks = runner._get_all_chunks(mock_ingestor)
        assert chunks == []


class TestGetExtractedChunkQns:
    """Test querying extracted chunks from concept graph."""

    def test_get_extracted_qns(self, tmp_path: Path):
        """Test getting set of extracted chunk QNs."""
        runner = ConceptExtractionRunner(repo_path=tmp_path, workspace="test")

        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = [
            {"qn": "doc1:chunk1"},
            {"qn": "doc1:chunk2"},
        ]

        qns = runner._get_extracted_chunk_qns(mock_ingestor)

        assert qns == {"doc1:chunk1", "doc1:chunk2"}

    def test_get_extracted_qns_empty(self, tmp_path: Path):
        """Test when no chunks have concepts."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = []

        qns = runner._get_extracted_chunk_qns(mock_ingestor)
        assert qns == set()

    def test_get_extracted_qns_filters_none(self, tmp_path: Path):
        """Test that None values are filtered out."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = [
            {"qn": "doc1:chunk1"},
            {"qn": None},
            {"qn": "doc1:chunk2"},
        ]

        qns = runner._get_extracted_chunk_qns(mock_ingestor)
        assert qns == {"doc1:chunk1", "doc1:chunk2"}


class TestGetChunksNeedingExtraction:
    """Test chunk filtering logic."""

    def test_force_returns_all(self, tmp_path: Path):
        """Test that force=True returns all chunks."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_doc = MagicMock()
        mock_doc.fetch_all.return_value = [
            {"qualified_name": "doc1:chunk1", "content": "content1"},
            {"qualified_name": "doc1:chunk2", "content": "content2"},
        ]
        mock_concept = MagicMock()

        chunks, skipped = runner._get_chunks_needing_extraction(
            mock_doc, mock_concept, force=True
        )

        assert len(chunks) == 2
        assert skipped == 0

    def test_filters_extracted_chunks(self, tmp_path: Path):
        """Test that already-extracted chunks are filtered."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_doc = MagicMock()
        mock_doc.fetch_all.return_value = [
            {"qualified_name": "doc1:chunk1", "content": "content1"},
            {"qualified_name": "doc1:chunk2", "content": "content2"},
            {"qualified_name": "doc1:chunk3", "content": "content3"},
        ]

        mock_concept = MagicMock()
        mock_concept.fetch_all.return_value = [
            {"qn": "doc1:chunk1"},
            {"qn": "doc1:chunk3"},
        ]

        chunks, skipped = runner._get_chunks_needing_extraction(
            mock_doc, mock_concept, force=False
        )

        assert len(chunks) == 1
        assert chunks[0]["qualified_name"] == "doc1:chunk2"
        assert skipped == 2


class TestCleanupExistingConcepts:
    """Test cleanup of existing concepts."""

    def test_cleanup_empty_list(self, tmp_path: Path):
        """Test that empty list does nothing."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        runner._cleanup_existing_concepts([], mock_ingestor)

        mock_ingestor.fetch_all.assert_not_called()

    def test_cleanup_executes_cypher(self, tmp_path: Path):
        """Test that cleanup executes Cypher query."""
        runner = ConceptExtractionRunner(repo_path=tmp_path, workspace="test")

        mock_ingestor = MagicMock()
        runner._cleanup_existing_concepts(["chunk1", "chunk2"], mock_ingestor)

        mock_ingestor.fetch_all.assert_called_once()
        call_args = mock_ingestor.fetch_all.call_args
        assert "UNWIND" in call_args[0][0]


class TestRunDryRun:
    """Test dry-run mode."""

    def test_dry_run_no_chunks(self, tmp_path: Path):
        """Test dry-run when no chunks exist."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_doc = MagicMock()
        mock_doc.fetch_all.return_value = []
        mock_concept = MagicMock()

        with patch.object(runner, "_connect_graphs") as mock_connect:
            mock_connect.return_value.__enter__ = lambda self: (mock_doc, mock_concept)
            mock_connect.return_value.__exit__ = lambda self, *args: None

            stats = runner.run(dry_run=True)

            assert stats.total_chunks == 0
            assert stats.processed_chunks == 0

    def test_dry_run_with_chunks(self, tmp_path: Path):
        """Test dry-run shows what would be processed."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_doc = MagicMock()
        mock_doc.fetch_all.return_value = [
            {"qualified_name": "doc1:chunk1", "content": "content1"},
        ]
        mock_concept = MagicMock()
        mock_concept.fetch_all.return_value = []  # No existing concepts

        with patch.object(runner, "_connect_graphs") as mock_connect, \
             patch.object(runner, "_check_indexes") as mock_check:

            mock_connect.return_value.__enter__ = lambda self: (mock_doc, mock_concept)
            mock_connect.return_value.__exit__ = lambda self, *args: None
            mock_check.return_value = {"Concept_qualified_name": True}

            stats = runner.run(dry_run=True)

            assert stats.total_chunks == 1
            assert stats.processed_chunks == 1

    def test_dry_run_skips_existing(self, tmp_path: Path):
        """Test dry-run shows skipped chunks."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_doc = MagicMock()
        mock_doc.fetch_all.return_value = [
            {"qualified_name": "doc1:chunk1", "content": "content1"},
        ]
        mock_concept = MagicMock()
        mock_concept.fetch_all.return_value = [{"qn": "doc1:chunk1"}]

        with patch.object(runner, "_connect_graphs") as mock_connect:
            mock_connect.return_value.__enter__ = lambda self: (mock_doc, mock_concept)
            mock_connect.return_value.__exit__ = lambda self, *args: None

            stats = runner.run(dry_run=True)

            assert stats.total_chunks == 0  # None needing extraction
            assert stats.skipped_existing == 1


class TestRunLimit:
    """Test limit parameter."""

    def test_limit_restricts_chunks(self, tmp_path: Path):
        """Test that limit restricts number of chunks processed."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_doc = MagicMock()
        mock_doc.fetch_all.return_value = [
            {"qualified_name": f"doc1:chunk{i}", "content": f"content{i}"}
            for i in range(10)
        ]
        mock_concept = MagicMock()
        mock_concept.fetch_all.return_value = []

        with patch.object(runner, "_connect_graphs") as mock_connect, \
             patch.object(runner, "_initialize_extractor") as mock_init, \
             patch.object(runner, "_ensure_indexes"):

            mock_connect.return_value.__enter__ = lambda self: (mock_doc, mock_concept)
            mock_connect.return_value.__exit__ = lambda self, *args: None
            mock_init.return_value = True

            # Use asyncio.run mock to avoid actual LLM calls
            with patch.object(runner, "_process_chunks_async") as mock_process:
                mock_process.return_value = StandaloneExtractionStats(
                    total_chunks=10,
                    processed_chunks=3,
                )

                runner.run(limit=3, dry_run=False)

                # Check that only 3 chunks were passed
                call_args = mock_process.call_args[0]
                assert len(call_args[0]) == 3


class TestEnsureIndexes:
    """Test index creation."""

    def test_ensure_indexes_success(self, tmp_path: Path):
        """Test successful index creation."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        runner._ensure_indexes(mock_ingestor)

        # Should have called fetch_all for each index
        assert mock_ingestor.fetch_all.call_count >= 6  # At least node indexes

    def test_ensure_indexes_handles_existing(self, tmp_path: Path):
        """Test that existing indexes are handled gracefully."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.side_effect = Exception("already exists")

        with patch("codebase_rag.document.concept_runner.logger"):
            # Should not raise
            runner._ensure_indexes(mock_ingestor)


class TestCheckIndexes:
    """Test index existence checking."""

    def test_check_indexes_all_exist(self, tmp_path: Path):
        """Test when all indexes exist."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = [
            {"index label": "Concept", "index property": "qualified_name"},
            {"index label": "Concept", "index property": "workspace"},
            {"index label": "ChunkRef", "index property": "qualified_name"},
            {"index label": "ChunkRef", "index property": "workspace"},
        ]

        status = runner._check_indexes(mock_ingestor)

        assert status["Concept_qualified_name"] is True
        assert status["Concept_workspace"] is True
        assert status["ChunkRef_qualified_name"] is True
        assert status["ChunkRef_workspace"] is True

    def test_check_indexes_missing(self, tmp_path: Path):
        """Test when indexes are missing."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.return_value = []

        status = runner._check_indexes(mock_ingestor)

        assert all(v is False for v in status.values())

    def test_check_indexes_query_fails(self, tmp_path: Path):
        """Test when query fails."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        mock_ingestor.fetch_all.side_effect = Exception("Query failed")

        status = runner._check_indexes(mock_ingestor)

        assert all(v is False for v in status.values())


class TestStoreExtractionResults:
    """Test storage of extraction results."""

    def test_store_creates_chunkref(self, tmp_path: Path):
        """Test that ChunkRef is created."""
        runner = ConceptExtractionRunner(repo_path=tmp_path, workspace="test")

        mock_ingestor = MagicMock()
        chunk = {"qualified_name": "doc1:chunk1", "content": "test content"}

        result = ExtractionResult(
            concepts=[
                ExtractedConcept(
                    name="TestConcept",
                    definition="A test concept",
                    confidence=0.9,
                )
            ],
            relationships=[],
        )

        runner._store_extraction_results(chunk, result, mock_ingestor)

        # Should have at least called for ChunkRef and Concept
        assert mock_ingestor.fetch_all.call_count >= 2

    def test_store_creates_mentions(self, tmp_path: Path):
        """Test that MENTIONS relationships are created."""
        runner = ConceptExtractionRunner(repo_path=tmp_path, workspace="test")

        mock_ingestor = MagicMock()
        chunk = {"qualified_name": "doc1:chunk1", "content": "test TestConcept content"}

        result = ExtractionResult(
            concepts=[
                ExtractedConcept(
                    name="TestConcept",
                    definition="A test concept",
                    confidence=0.9,
                )
            ],
            relationships=[],
        )

        runner._store_extraction_results(chunk, result, mock_ingestor)

        # Check that MENTIONS Cypher was executed
        calls = [str(call) for call in mock_ingestor.fetch_all.call_args_list]
        mentions_found = any("MENTIONS" in str(call) for call in calls)
        assert mentions_found

    def test_store_creates_relationships(self, tmp_path: Path):
        """Test that concept relationships are created."""
        runner = ConceptExtractionRunner(repo_path=tmp_path, workspace="test")

        mock_ingestor = MagicMock()
        chunk = {"qualified_name": "doc1:chunk1", "content": "test content"}

        result = ExtractionResult(
            concepts=[
                ExtractedConcept(
                    name="ConceptA",
                    definition="First concept",
                    confidence=0.9,
                ),
                ExtractedConcept(
                    name="ConceptB",
                    definition="Second concept",
                    confidence=0.9,
                ),
            ],
            relationships=[
                ConceptRelationship(
                    from_concept="ConceptA",
                    to_concept="ConceptB",
                    verb="causes",
                    category="CAUSAL",
                    strength=0.8,
                )
            ],
        )

        runner._store_extraction_results(chunk, result, mock_ingestor)

        # Check that relationship Cypher was executed
        calls = [str(call) for call in mock_ingestor.fetch_all.call_args_list]
        rel_found = any("HIERARCHICAL" in str(call) or "CAUSAL" in str(call) for call in calls)
        assert rel_found


class TestProcessChunksAsync:
    """Test async chunk processing."""

    @pytest.mark.asyncio
    async def test_process_chunks_success(self, tmp_path: Path):
        """Test successful chunk processing."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)
        runner.extractor = MagicMock()
        runner.extractor.extract_with_retry = AsyncMock(return_value=ExtractionResult(
            concepts=[
                ExtractedConcept(name="Test", definition="test", confidence=0.9)
            ],
            relationships=[],
        ))

        mock_ingestor = MagicMock()
        chunks = [
            {"qualified_name": "doc1:chunk1", "content": "content1"},
        ]
        stats = StandaloneExtractionStats()

        result = await runner._process_chunks_async(chunks, stats, mock_ingestor)

        assert result.successful_extractions == 1
        assert result.concepts_created == 1

    @pytest.mark.asyncio
    async def test_process_chunks_handles_exceptions(self, tmp_path: Path):
        """Test that exceptions are tracked in stats."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)
        runner.extractor = MagicMock()
        runner.extractor.extract_with_retry = AsyncMock(side_effect=Exception("LLM error"))

        mock_ingestor = MagicMock()
        chunks = [
            {"qualified_name": "doc1:chunk1", "content": "content1"},
        ]
        stats = StandaloneExtractionStats()

        result = await runner._process_chunks_async(chunks, stats, mock_ingestor)

        assert result.failed_extractions == 1
        assert "Exception" in result.errors_by_type

    @pytest.mark.asyncio
    async def test_process_chunks_tracks_skewed_count(self, tmp_path: Path):
        """Test that was_rebalanced increments skewed_chunks_count."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)
        runner.extractor = MagicMock()
        runner.extractor.extract_with_retry = AsyncMock(return_value=ExtractionResult(
            concepts=[
                ExtractedConcept(name="Test", definition="test", confidence=0.9)
            ],
            relationships=[],
            was_rebalanced=True,
        ))

        mock_ingestor = MagicMock()
        chunks = [
            {"qualified_name": "doc1:chunk1", "content": "content1"},
        ]
        stats = StandaloneExtractionStats()

        result = await runner._process_chunks_async(chunks, stats, mock_ingestor)

        assert result.skewed_chunks_count == 1
        assert result.successful_extractions == 1


class TestBatchStorageMethods:
    """Test batch storage methods."""

    def test_merge_concept_nodes_batch(self, tmp_path: Path):
        """Test batch merge of concept nodes."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        concept_nodes = [
            {
                "qualified_name": "test:Concept1",
                "workspace": "test",
                "name": "Concept1",
                "aliases": [],
                "type": "ABSTRACT_CONCEPT",
                "definition": "Test",
                "confidence": 0.9,
                "entity_category": "ABSTRACT_CONCEPT",
                "entity_subtype": None,
                "entity_emoji": "💡",
            }
        ]

        runner._merge_concept_nodes_batch(mock_ingestor, concept_nodes)

        mock_ingestor.fetch_all.assert_called_once()
        assert "MERGE" in mock_ingestor.fetch_all.call_args[0][0]

    def test_create_mentions_batch(self, tmp_path: Path):
        """Test batch creation of MENTIONS relationships."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        mention_rels = [
            {
                "chunk_qn": "doc1:chunk1",
                "concept_qn": "test:Concept1",
                "frequency": 2,
                "context": "test context",
            }
        ]

        runner._create_mentions_batch(mock_ingestor, mention_rels, "test")

        mock_ingestor.fetch_all.assert_called_once()
        assert "MENTIONS" in mock_ingestor.fetch_all.call_args[0][0]

    def test_store_concept_relationships_batch(self, tmp_path: Path):
        """Test batch storage of concept relationships."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_ingestor = MagicMock()
        relationships = [
            ("test:ConceptA", "test:ConceptB", "causes", "⚡", "CAUSAL", 0.8),
        ]

        runner._store_concept_relationships_batch(mock_ingestor, relationships, "test")

        mock_ingestor.fetch_all.assert_called_once()
        # Check that FOREACH CAUSAL branch is in the query
        assert "CAUSAL" in mock_ingestor.fetch_all.call_args[0][0]


class TestRetryDlq:
    """Test DLQ retry functionality."""

    def test_retry_dlq_force_cleans_existing(self, tmp_path: Path):
        """When force=True, retry_dlq cleans up existing concepts first."""
        runner = ConceptExtractionRunner(repo_path=tmp_path)

        mock_error = MagicMock()
        mock_error.error_type = ErrorType.CONCEPT_CONTEXT_OVERFLOW
        mock_error.chunk_qn = "doc:chunk1"
        mock_error.chunk_content = "x" * 200
        mock_error.path = "doc:chunk1"

        mock_dlq = MagicMock()
        mock_dlq.get_pending.return_value = [mock_error]
        runner.dead_letter_queue = mock_dlq
        runner.extractor = MagicMock()

        mock_doc = MagicMock()
        mock_concept = MagicMock()

        with patch.object(runner, "_connect_graphs") as mock_connect, \
             patch.object(runner, "_initialize_extractor") as mock_init, \
             patch.object(runner, "_cleanup_existing_concepts") as mock_cleanup, \
             patch.object(runner, "_store_extraction_results"), \
             patch("codebase_rag.document.concept_runner.asyncio.run") as mock_asyncio_run:

            mock_connect.return_value.__enter__ = lambda self: (mock_doc, mock_concept)
            mock_connect.return_value.__exit__ = lambda self, *args: None
            mock_init.return_value = True
            mock_asyncio_run.return_value = AdaptiveRetryResult(
                success=True,
                concepts=1,
                relationships=0,
                result=ExtractionResult(
                    concepts=[ExtractedConcept(name="C", definition="d", confidence=0.9)],
                    relationships=[],
                ),
            )

            stats = runner.retry_dlq(force=True)

            mock_cleanup.assert_called_once_with(["doc:chunk1"], mock_concept)
            assert stats.successful_extractions == 1
            mock_dlq.remove.assert_called_once_with(mock_error)
