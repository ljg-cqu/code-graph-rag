"""Tests for 4th Memgraph instance concept storage.

Covers: connect_concept_memgraph, ChunkRef MERGE pattern,
cross-instance chunk QN query, cross-instance cleanup coordination.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from codebase_rag.config import settings
from codebase_rag.document.document_updater import DocumentGraphUpdater


class TestConnectConceptMemgraph:
    """Tests for connect_concept_memgraph connection function."""

    def test_connect_concept_memgraph_uses_correct_settings(self) -> None:
        from codebase_rag.main import connect_concept_memgraph

        with patch("codebase_rag.main.MemgraphIngestor") as mock_ingestor:
            connect_concept_memgraph(batch_size=500)
            mock_ingestor.assert_called_once_with(
                host=settings.CONCEPT_MEMGRAPH_HOST,
                port=settings.CONCEPT_MEMGRAPH_PORT,
                batch_size=500,
                username=settings.CONCEPT_MEMGRAPH_USERNAME,
                password=settings.CONCEPT_MEMGRAPH_PASSWORD,
            )

    def test_connect_concept_memgraph_default_batch_size(self) -> None:
        from codebase_rag.main import connect_concept_memgraph

        with patch("codebase_rag.main.MemgraphIngestor") as mock_ingestor:
            connect_concept_memgraph()
            mock_ingestor.assert_called_once_with(
                host=settings.CONCEPT_MEMGRAPH_HOST,
                port=settings.CONCEPT_MEMGRAPH_PORT,
                batch_size=1000,
                username=settings.CONCEPT_MEMGRAPH_USERNAME,
                password=settings.CONCEPT_MEMGRAPH_PASSWORD,
            )


class TestConnectAllGraphs:
    """Tests for connect_all_graphs context manager."""

    def test_yields_all_three_graphs_when_concept_enabled(self) -> None:
        from codebase_rag.main import connect_all_graphs

        with patch("codebase_rag.main.MemgraphIngestor") as mock_cls:
            mock_cls.side_effect = [
                MagicMock(),
                MagicMock(),
                MagicMock(),
            ]
            with patch.object(settings, "CONCEPT_MEMGRAPH_ENABLED", True):
                with connect_all_graphs(batch_size=100) as (
                    code_graph,
                    doc_graph,
                    concept_graph,
                ):
                    assert code_graph is not None
                    assert doc_graph is not None
                    assert concept_graph is not None
                    assert mock_cls.call_count == 3

    def test_concept_graph_is_none_when_disabled(self) -> None:
        from codebase_rag.main import connect_all_graphs

        with patch("codebase_rag.main.MemgraphIngestor") as mock_cls:
            mock_cls.side_effect = [
                MagicMock(),
                MagicMock(),
            ]
            with patch.object(settings, "CONCEPT_MEMGRAPH_ENABLED", False):
                with connect_all_graphs(batch_size=100) as (
                    code_graph,
                    doc_graph,
                    concept_graph,
                ):
                    assert code_graph is not None
                    assert doc_graph is not None
                    assert concept_graph is None
                    assert mock_cls.call_count == 2

    def test_concept_graph_is_none_on_connection_failure(self) -> None:
        from codebase_rag.main import connect_all_graphs

        with patch("codebase_rag.main.MemgraphIngestor") as mock_cls:
            mock_cls.side_effect = [
                MagicMock(),
                MagicMock(),
                Exception("connection refused"),
            ]
            with patch.object(settings, "CONCEPT_MEMGRAPH_ENABLED", True):
                with connect_all_graphs(batch_size=100) as (
                    code_graph,
                    doc_graph,
                    concept_graph,
                ):
                    assert code_graph is not None
                    assert doc_graph is not None
                    assert concept_graph is None

    def test_enters_and_exits_all_graphs(self) -> None:
        from codebase_rag.main import connect_all_graphs

        code_mock = MagicMock()
        doc_mock = MagicMock()
        concept_mock = MagicMock()

        with patch("codebase_rag.main.MemgraphIngestor") as mock_cls:
            mock_cls.side_effect = [code_mock, doc_mock, concept_mock]
            with patch.object(settings, "CONCEPT_MEMGRAPH_ENABLED", True):
                with connect_all_graphs(batch_size=100) as (
                    code_graph,
                    doc_graph,
                    concept_graph,
                ):
                    code_mock.__enter__.assert_called_once()
                    doc_mock.__enter__.assert_called_once()
                    concept_mock.__enter__.assert_called_once()

                code_mock.__exit__.assert_called_once()
                doc_mock.__exit__.assert_called_once()
                concept_mock.__exit__.assert_called_once()

    def test_cleanup_on_concept_enter_failure(self) -> None:
        from codebase_rag.main import connect_all_graphs

        code_mock = MagicMock()
        doc_mock = MagicMock()
        concept_mock = MagicMock()
        concept_mock.__enter__.side_effect = Exception("boom")

        with patch("codebase_rag.main.MemgraphIngestor") as mock_cls:
            mock_cls.side_effect = [code_mock, doc_mock, concept_mock]
            with patch.object(settings, "CONCEPT_MEMGRAPH_ENABLED", True):
                with pytest.raises(Exception, match="boom"):
                    with connect_all_graphs(batch_size=100):
                        pass

                doc_mock.__exit__.assert_called_once()
                code_mock.__exit__.assert_called_once()


class TestInitializeServicesConceptGraph:
    """Tests for _initialize_services_and_agent passing concept_graph."""

    _PATCH_TARGETS = [
        "codebase_rag.main.create_rag_orchestrator",
        "codebase_rag.main._validate_provider_config",
        "codebase_rag.main._determine_default_query_mode",
        "codebase_rag.main.CypherGenerator",
        "codebase_rag.main.CodeRetriever",
        "codebase_rag.main.DocumentAnalyzer",
        "codebase_rag.main.create_query_tool",
        "codebase_rag.main.create_document_analyzer_tool",
        "codebase_rag.main.create_semantic_search_tool",
        "codebase_rag.main.create_get_function_source_tool",
        "codebase_rag.main.create_file_reader_tool",
        "codebase_rag.main.create_file_writer_tool",
        "codebase_rag.main.create_file_editor_tool",
        "codebase_rag.main.create_shell_command_tool",
        "codebase_rag.main.create_directory_lister_tool",
        "codebase_rag.main.create_inspect_python_object_tool",
        "codebase_rag.main.create_find_references_tool",
        "codebase_rag.main.create_get_call_hierarchy_tool",
        "codebase_rag.main.create_find_implementations_tool",
        "codebase_rag.main.create_get_project_structure_tool",
        "codebase_rag.main.create_get_import_dependencies_tool",
        "codebase_rag.main.create_code_retrieval_tool",
        "codebase_rag.main.create_graph_query_tool",
        "codebase_rag.main.create_query_document_graph_tool",
        "codebase_rag.main.create_query_both_graphs_tool",
        "codebase_rag.main.create_validate_code_against_spec_tool",
        "codebase_rag.main.create_validate_doc_against_code_tool",
        "codebase_rag.main.create_index_documents_tool",
    ]

    def _run_under_patches(self, func: callable) -> None:
        from contextlib import ExitStack

        with ExitStack() as stack:
            for target in self._PATCH_TARGETS:
                kwargs = {"return_value": []} if "create_document_analyzer_tool" in target else {}
                stack.enter_context(patch(target, **kwargs))
            func()

    def test_passes_concept_ingestor_to_query_router(self) -> None:
        from codebase_rag.main import _initialize_services_and_agent

        code_mock = MagicMock()
        doc_mock = MagicMock()
        concept_mock = MagicMock()
        router_cls = None

        def _inner() -> None:
            nonlocal router_cls
            with patch("codebase_rag.main.QueryRouter") as mock_router_cls:
                router_cls = mock_router_cls
                _initialize_services_and_agent(
                    "/tmp/test",
                    code_mock,
                    doc_ingestor=doc_mock,
                    concept_ingestor=concept_mock,
                    query_mode=None,
                )

        self._run_under_patches(_inner)

        router_cls.assert_called_once()
        call_kwargs = router_cls.call_args.kwargs
        assert call_kwargs.get("code_graph") is code_mock
        assert call_kwargs.get("doc_graph") is doc_mock
        assert call_kwargs.get("concept_graph") is concept_mock

    def test_concept_graph_defaults_to_none(self) -> None:
        from codebase_rag.main import _initialize_services_and_agent

        code_mock = MagicMock()
        doc_mock = MagicMock()
        router_cls = None

        def _inner() -> None:
            nonlocal router_cls
            with patch("codebase_rag.main.QueryRouter") as mock_router_cls:
                router_cls = mock_router_cls
                _initialize_services_and_agent(
                    "/tmp/test",
                    code_mock,
                    doc_ingestor=doc_mock,
                )

        self._run_under_patches(_inner)

        router_cls.assert_called_once()
        call_kwargs = router_cls.call_args.kwargs
        assert call_kwargs.get("concept_graph") is None


class TestChunkRefMentionsBatch:
    """Tests for _create_mentions_batch using ChunkRef MERGE pattern."""

    def test_mentions_batch_uses_chunkref_merge(self) -> None:
        updater = _make_dummy_updater()
        ingestor = MagicMock()
        updater._create_mentions_batch(
            ingestor,
            [{"chunk_qn": "doc.md#0", "concept_qn": "c:Concept", "frequency": 1, "context": "ctx"}],
            "default",
        )
        cypher = ingestor.fetch_all.call_args[0][0]
        assert "MERGE (c:ChunkRef" in cypher
        assert "MATCH (c:Chunk" not in cypher

    def test_mentions_batch_still_merges_concept(self) -> None:
        updater = _make_dummy_updater()
        ingestor = MagicMock()
        updater._create_mentions_batch(
            ingestor,
            [{"chunk_qn": "doc.md#0", "concept_qn": "c:Concept", "frequency": 1, "context": "ctx"}],
            "default",
        )
        cypher = ingestor.fetch_all.call_args[0][0]
        assert "MERGE (concept:Concept" in cypher

    def test_mentions_batch_preserves_frequency_and_context(self) -> None:
        updater = _make_dummy_updater()
        ingestor = MagicMock()
        updater._create_mentions_batch(
            ingestor,
            [{"chunk_qn": "doc.md#0", "concept_qn": "c:Concept", "frequency": 2, "context": "some context"}],
            "default",
        )
        cypher = ingestor.fetch_all.call_args[0][0]
        assert "m.frequency = rel.frequency" in cypher
        assert "m.context = rel.context" in cypher


class TestConceptDeduplication:
    """Tests for _deduplicate_concept_nodes in DocumentUpdater."""

    def test_deduplicate_keeps_highest_confidence(self) -> None:
        updater = _make_dummy_updater()
        nodes = [
            {"qualified_name": "ws:A", "name": "A", "confidence": 0.7, "aliases": ["a1"], "definition": "def1"},
            {"qualified_name": "ws:A", "name": "A", "confidence": 0.9, "aliases": ["a2"], "definition": "def2"},
            {"qualified_name": "ws:B", "name": "B", "confidence": 0.8, "aliases": [], "definition": "def3"},
        ]
        result = updater._deduplicate_concept_nodes(nodes)
        assert len(result) == 2
        a_node = next(n for n in result if n["qualified_name"] == "ws:A")
        assert a_node["confidence"] == 0.9  # highest wins
        assert sorted(a_node["aliases"]) == ["a1", "a2"]  # merged

    def test_deduplicate_sorts_by_qualified_name(self) -> None:
        updater = _make_dummy_updater()
        nodes = [
            {"qualified_name": "ws:C", "name": "C", "confidence": 0.5},
            {"qualified_name": "ws:A", "name": "A", "confidence": 0.5},
            {"qualified_name": "ws:B", "name": "B", "confidence": 0.5},
        ]
        result = updater._deduplicate_concept_nodes(nodes)
        assert [n["qualified_name"] for n in result] == ["ws:A", "ws:B", "ws:C"]

    def test_deduplicate_empty_list(self) -> None:
        updater = _make_dummy_updater()
        assert updater._deduplicate_concept_nodes([]) == []


class TestGetChunkQnsForDocument:
    """Tests for _get_chunk_qns_for_document cross-instance query."""

    def test_returns_chunk_qns_from_doc_instance(self) -> None:
        updater = _make_dummy_updater()
        doc_ingestor = MagicMock()
        doc_ingestor.fetch_all.return_value = [
            {"chunk_qn": "doc.md#0"},
            {"chunk_qn": "doc.md#1"},
        ]
        result = updater._get_chunk_qns_for_document("doc.md", doc_ingestor)
        assert result == ["doc.md#0", "doc.md#1"]

    def test_returns_empty_list_when_no_chunks(self) -> None:
        updater = _make_dummy_updater()
        doc_ingestor = MagicMock()
        doc_ingestor.fetch_all.return_value = []
        result = updater._get_chunk_qns_for_document("doc.md", doc_ingestor)
        assert result == []

    def test_queries_with_correct_params(self) -> None:
        updater = _make_dummy_updater()
        doc_ingestor = MagicMock()
        doc_ingestor.fetch_all.return_value = [{"chunk_qn": "doc.md#0"}]
        updater._get_chunk_qns_for_document("doc.md", doc_ingestor)
        params = doc_ingestor.fetch_all.call_args[0][1]
        assert params["doc_path"] == "doc.md"
        assert params["workspace"] == "default"


class TestCleanupConceptsCrossInstance:
    """Tests for _cleanup_concepts_for_document cross-instance coordination."""

    def test_early_return_when_concept_ingestor_none(self) -> None:
        updater = _make_dummy_updater()
        doc_ingestor = MagicMock()
        updater._cleanup_concepts_for_document("doc.md", doc_ingestor, None)
        doc_ingestor.fetch_all.assert_not_called()

    def test_early_return_when_concept_disabled(self) -> None:
        updater = _make_dummy_updater()
        doc_ingestor = MagicMock()
        concept_ingestor = MagicMock()
        with patch.object(settings, "CONCEPT_MEMGRAPH_ENABLED", False):
            updater._cleanup_concepts_for_document("doc.md", doc_ingestor, concept_ingestor)
        doc_ingestor.fetch_all.assert_not_called()
        concept_ingestor.fetch_all.assert_not_called()

    def test_queries_doc_instance_for_chunk_qns(self) -> None:
        updater = _make_dummy_updater()
        doc_ingestor = MagicMock()
        doc_ingestor.fetch_all.return_value = [{"chunk_qn": "doc.md#0"}]
        concept_ingestor = MagicMock()
        concept_ingestor.fetch_all.return_value = [{"removed_count": 1}]

        updater._cleanup_concepts_for_document("doc.md", doc_ingestor, concept_ingestor)

        doc_call = doc_ingestor.fetch_all.call_args
        assert "Document" in doc_call[0][0]
        assert "CONTAINS_CHUNK" in doc_call[0][0]

    def test_deletes_from_concept_instance(self) -> None:
        updater = _make_dummy_updater()
        doc_ingestor = MagicMock()
        doc_ingestor.fetch_all.return_value = [{"chunk_qn": "doc.md#0"}]
        concept_ingestor = MagicMock()
        concept_ingestor.fetch_all.return_value = [{"removed_count": 1}]

        updater._cleanup_concepts_for_document("doc.md", doc_ingestor, concept_ingestor)

        concept_call = concept_ingestor.fetch_all.call_args
        cypher = concept_call[0][0]
        assert "ChunkRef" in cypher
        assert "MENTIONS" in cypher
        assert "DETACH DELETE concept" in cypher


class TestEnsureConceptIndexes:
    """Tests for _ensure_concept_indexes with ChunkRef indexes."""

    def test_chunkref_indexes_included(self) -> None:
        ingestor = MagicMock()
        updater = _make_dummy_updater()
        updater._concept_indexes_ensured = False

        updater._ensure_concept_indexes(ingestor)

        all_calls = [call[0][0] for call in ingestor.fetch_all.call_args_list]
        all_indexes = " ".join(all_calls)
        assert ":ChunkRef(qualified_name)" in all_indexes
        assert ":ChunkRef(workspace)" in all_indexes


class TestMigrationListStaleIndexes:
    """Tests for cleanup_doc_concepts _list_stale_indexes deduplication."""

    def test_returns_unique_matches_only(self) -> None:
        from codebase_rag.migrations.cleanup_doc_concepts import _list_stale_indexes

        ingestor = MagicMock()
        ingestor.fetch_all.return_value = [
            {"index label": "Concept", "index property": "qualified_name"},
            {"index label": "Concept", "index property": "workspace"},
            {"index label": "Chunk", "index property": "qualified_name"},
            {"index label": "HIERARCHICAL", "index property": "verb"},
            {"index label": "RELATED_TO", "index property": "verb"},
        ]
        result = _list_stale_indexes(ingestor)
        assert result == [
            "Concept(qualified_name)",
            "Concept(workspace)",
            "HIERARCHICAL(verb)",
            "RELATED_TO(verb)",
        ]

    def test_returns_empty_list_when_no_matches(self) -> None:
        from codebase_rag.migrations.cleanup_doc_concepts import _list_stale_indexes

        ingestor = MagicMock()
        ingestor.fetch_all.return_value = [
            {"index label": "Chunk", "index property": "qualified_name"},
            {"index label": "Document", "index property": "path"},
        ]
        result = _list_stale_indexes(ingestor)
        assert result == []

    def test_gracefully_handles_empty_result(self) -> None:
        from codebase_rag.migrations.cleanup_doc_concepts import _list_stale_indexes

        ingestor = MagicMock()
        ingestor.fetch_all.return_value = []
        result = _list_stale_indexes(ingestor)
        assert result == []

    def test_gracefully_handles_show_index_failure(self) -> None:
        from codebase_rag.migrations.cleanup_doc_concepts import _list_stale_indexes

        ingestor = MagicMock()
        ingestor.fetch_all.side_effect = Exception("unavailable")
        result = _list_stale_indexes(ingestor)
        assert result == []


def _make_dummy_updater() -> DocumentGraphUpdater:
    """Create a minimal DocumentGraphUpdater for testing."""
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    with patch.object(DocumentGraphUpdater, "__init__", lambda self: None):
        updater = DocumentGraphUpdater.__new__(DocumentGraphUpdater)
        updater.host = "localhost"
        updater.port = 7688
        updater.repo_path = Path("/tmp/test")
        updater.workspace = "default"
        updater.batch_size = 100
        updater.username = None
        updater.password = None
        updater.concept_host = "localhost"
        updater.concept_port = 7690
        updater.concept_username = None
        updater.concept_password = None
        updater.concept_extractor = MagicMock()
        updater._concept_indexes_ensured = False
    return updater
