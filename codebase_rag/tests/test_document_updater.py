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
