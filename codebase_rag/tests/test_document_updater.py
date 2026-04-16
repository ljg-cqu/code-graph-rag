from pathlib import Path
from unittest.mock import MagicMock, patch

from codebase_rag.config import settings
from codebase_rag.document.chunking import DocumentChunk
from codebase_rag.document.document_updater import (
    DocumentGraphUpdater,
    ensure_document_vector_index,
)
from codebase_rag.document.extractors.base import ExtractedDocument


def test_collect_documents_skips_internal_artifacts(tmp_path: Path) -> None:
    included = tmp_path / "guide.md"
    included.write_text("# Guide\n", encoding="utf-8")

    egg_info_dir = tmp_path / "demo.egg-info"
    egg_info_dir.mkdir()
    (egg_info_dir / "artifact.md").write_text("# Artifact\n", encoding="utf-8")

    cgr_dir = tmp_path / ".cgr"
    cgr_dir.mkdir()
    (cgr_dir / "state.md").write_text("# Internal\n", encoding="utf-8")

    provider = MagicMock()

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    documents = updater._collect_documents()

    assert included in documents
    assert egg_info_dir / "artifact.md" not in documents
    assert cgr_dir / "state.md" not in documents


def test_collect_documents_respects_cgrignore_patterns(tmp_path: Path) -> None:
    included = tmp_path / "guide.md"
    included.write_text("# Guide\n", encoding="utf-8")

    excluded_by_glob = tmp_path / "notes.txt"
    excluded_by_glob.write_text("notes\n", encoding="utf-8")

    excluded_by_exact = tmp_path / "docs" / "tree-sitter.pdf"
    excluded_by_exact.parent.mkdir(parents=True)
    excluded_by_exact.write_text("pdf placeholder\n", encoding="utf-8")

    (tmp_path / ".cgrignore").write_text(
        "*.txt\n/docs/tree-sitter.pdf\n",
        encoding="utf-8",
    )

    provider = MagicMock()

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    documents = updater._collect_documents()

    assert included in documents
    assert excluded_by_glob not in documents
    assert excluded_by_exact not in documents


def test_delete_stale_documents_removes_ignored_paths(tmp_path: Path) -> None:
    included = tmp_path / "guide.md"
    included.write_text("# Guide\n", encoding="utf-8")

    ignored = tmp_path / "notes.txt"
    ignored.write_text("notes\n", encoding="utf-8")

    (tmp_path / ".cgrignore").write_text("*.txt\n", encoding="utf-8")

    provider = MagicMock()

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    ingestor = MagicMock()
    ingestor.fetch_all.return_value = [
        {"path": str(included)},
        {"path": str(ignored)},
        {"path": "/outside/workspace/other.md"},
    ]

    with (
        patch.object(updater, "_delete_document_nodes") as delete_document_nodes,
        patch.object(updater.version_cache, "remove") as remove_version_cache,
    ):
        removed = updater._delete_stale_documents([included], ingestor)

    assert removed == 1
    delete_document_nodes.assert_called_once_with(str(ignored), ingestor)
    remove_version_cache.assert_called_once_with(str(ignored))


def test_refresh_code_reference_index_builds_lookup(tmp_path: Path) -> None:
    provider = MagicMock()

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    with patch("codebase_rag.document.document_updater.MemgraphIngestor") as mock_cls:
        mock_ingestor = mock_cls.return_value.__enter__.return_value
        mock_ingestor.fetch_all.return_value = [
            {
                "qualified_name": "proj.auth.authenticate_user",
                "name": "authenticate_user",
            },
            {"qualified_name": "proj.models.User", "name": "User"},
        ]

        updater._refresh_code_reference_index()

    assert "proj.auth.authenticate_user" in updater._code_reference_qns
    assert updater._code_reference_simple_lookup["authenticate_user"] == (
        "proj.auth.authenticate_user",
    )
    assert updater._code_reference_simple_lookup["User"] == ("proj.models.User",)


def test_resolve_code_reference_names_skips_ambiguous_matches(tmp_path: Path) -> None:
    provider = MagicMock()

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    updater._code_reference_qns = {
        "proj.auth.authenticate_user",
        "proj.models.User",
    }
    updater._code_reference_simple_lookup = {
        "authenticate_user": ("proj.auth.authenticate_user",),
        "User": ("proj.models.User",),
        "common": ("proj.first.common", "proj.second.common"),
    }

    resolved = updater._resolve_code_reference_names(
        [
            "proj.auth.authenticate_user",
            "User",
            "common",
            "missing",
        ]
    )

    assert resolved == ["proj.auth.authenticate_user", "proj.models.User"]


def test_store_chunks_with_embeddings_persists_resolved_references(
    tmp_path: Path,
) -> None:
    provider = MagicMock()

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    updater._code_reference_qns = {"proj.auth.authenticate_user"}
    updater._code_reference_simple_lookup = {
        "authenticate_user": ("proj.auth.authenticate_user",)
    }

    chunk = DocumentChunk(
        content="Use `authenticate_user()` before creating the session.",
        section_title="Authentication",
        start_line=10,
        end_line=10,
        token_count=12,
        document_path="docs/auth.md",
        chunk_index=0,
    )
    doc = ExtractedDocument(
        path="docs/auth.md",
        file_type=".md",
        content=chunk.content,
        sections=[],
        code_blocks=[],
        code_references=[],
        word_count=8,
        modified_date="2026-04-15T00:00:00+00:00",
    )
    ingestor = MagicMock()

    stored_count = updater._store_chunks_with_embeddings(
        doc=doc,
        embeddings_data=([chunk], [[0.1, 0.2, 0.3]]),
        section_info=[
            {
                "qualified_name": "docs/auth.md#L10:Authentication",
                "title": "Authentication",
                "start_line": 10,
                "end_line": 12,
                "level": 1,
            }
        ],
        ingestor=ingestor,
        indexed_at="2026-04-15T00:00:00+00:00",
    )

    assert stored_count == 1

    chunk_call = ingestor.ensure_node_batch.call_args_list[0]
    assert chunk_call[0][0] == "Chunk"
    assert chunk_call[0][1]["code_references"] == ["authenticate_user"]
    assert chunk_call[0][1]["resolved_code_references"] == [
        "proj.auth.authenticate_user"
    ]


def test_delete_document_nodes_uses_separate_linear_deletes(tmp_path: Path) -> None:
    provider = MagicMock()

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    ingestor = MagicMock()

    updater._delete_document_nodes("docs/guide.md", ingestor)

    assert ingestor.execute_write.call_count == 4

    queries = [call.args[0] for call in ingestor.execute_write.call_args_list]

    assert "CONTAINS_SECTION" in queries[0]
    assert "CONTAINS_CHUNK" not in queries[0]
    assert "CONTAINS_CHUNK" in queries[1]
    assert "CONTAINS_SECTION" not in queries[1]
    assert "qualified_name STARTS WITH $path_prefix" in queries[2]
    assert "qualified_name STARTS WITH $path_prefix" in queries[3]


def test_prepare_embeddings_batches_large_documents(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.dimension = 1
    provider.embed_batch.side_effect = lambda texts, batch_size=32: [
        [float(len(text))] for text in texts
    ]

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    doc = ExtractedDocument(
        path="docs/guide.md",
        file_type=".md",
        content="content",
        sections=[],
        code_blocks=[],
        code_references=[],
        word_count=5,
        modified_date="2026-04-15T00:00:00+00:00",
    )
    chunks = [
        DocumentChunk(
            content=f"chunk {index} content",
            section_title="Guide",
            start_line=index,
            end_line=index,
            token_count=20,
            document_path="docs/guide.md",
            chunk_index=index,
        )
        for index in range(5)
    ]

    with patch.object(settings, "VECTOR_EMBEDDING_BATCH_SIZE", 2):
        chunks_list, embeddings = updater._prepare_embeddings(doc, chunks)

    assert len(chunks_list) == 5
    assert len(embeddings) == 5
    assert provider.embed_batch.call_count == 3
    assert [len(call.args[0]) for call in provider.embed_batch.call_args_list] == [2, 2, 1]
    assert [call.kwargs["batch_size"] for call in provider.embed_batch.call_args_list] == [2, 2, 1]


def test_prepare_embeddings_logs_large_documents_at_info(tmp_path: Path) -> None:
    provider = MagicMock()
    provider.dimension = 1
    provider.embed_batch.side_effect = lambda texts, batch_size=32: [
        [float(len(text))] for text in texts
    ]

    with patch(
        "codebase_rag.document.document_updater.get_embedding_provider",
        return_value=provider,
    ):
        updater = DocumentGraphUpdater("localhost", 7688, tmp_path)

    doc = ExtractedDocument(
        path="docs/large-guide.md",
        file_type=".md",
        content="content",
        sections=[],
        code_blocks=[],
        code_references=[],
        word_count=5,
        modified_date="2026-04-15T00:00:00+00:00",
    )
    chunks = [
        DocumentChunk(
            content=f"chunk {index} content",
            section_title="Guide",
            start_line=index,
            end_line=index,
            token_count=20,
            document_path="docs/large-guide.md",
            chunk_index=index,
        )
        for index in range(100)
    ]

    with (
        patch.object(settings, "VECTOR_EMBEDDING_BATCH_SIZE", 50),
        patch("codebase_rag.document.document_updater.logger") as mock_logger,
    ):
        chunks_list, embeddings = updater._prepare_embeddings(doc, chunks)

    assert len(chunks_list) == 100
    assert len(embeddings) == 100
    warning_messages = [call.args[0] for call in mock_logger.warning.call_args_list]
    info_messages = [call.args[0] for call in mock_logger.info.call_args_list]
    assert not any("generated 100 embedding chunks" in message for message in warning_messages)
    assert any("generated 100 embedding chunks" in message for message in info_messages)


def test_ensure_vector_index_recreates_mismatched_dimension() -> None:
    ingestor = MagicMock()
    ingestor.fetch_all.return_value = [
        {"index_name": settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME, "dimension": 768}
    ]

    ensure_document_vector_index(ingestor, dimension=1024)

    queries = [call.args[0] for call in ingestor.execute_write.call_args_list]
    assert any(
        "MATCH (n:Chunk)" in query and "SET n.embedding = NULL" in query
        for query in queries
    )
    assert any(
        f"DROP VECTOR INDEX {settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME};" in query
        for query in queries
    )
    assert any(
        f"CREATE VECTOR INDEX {settings.DOC_MEMGRAPH_VECTOR_INDEX_NAME}" in query
        for query in queries
    )
