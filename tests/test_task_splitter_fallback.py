"""Unit tests for TaskSplitter fallback and mode suggestion."""
from __future__ import annotations

from pathlib import Path

from codebase_rag.orchestrator.task_splitter import SplitInfo, TaskSplitter
from codebase_rag.shared.query_router import QueryMode


class TestSuggestQueryMode:
    """Test _suggest_query_mode based on graph content."""

    def test_suggests_document_only_when_code_empty_and_docs_present(self):
        splitter = TaskSplitter(
            query_mode=QueryMode.CODE_ONLY,
            code_count=0,
            doc_count=5,
        )
        assert splitter._suggest_query_mode() == QueryMode.DOCUMENT_ONLY

    def test_suggests_code_only_when_doc_empty_and_code_present(self):
        splitter = TaskSplitter(
            query_mode=QueryMode.DOCUMENT_ONLY,
            code_count=10,
            doc_count=0,
        )
        assert splitter._suggest_query_mode() == QueryMode.CODE_ONLY

    def test_suggests_both_merged_when_both_present(self):
        splitter = TaskSplitter(
            query_mode=QueryMode.CODE_ONLY,
            code_count=10,
            doc_count=5,
        )
        assert splitter._suggest_query_mode() == QueryMode.BOTH_MERGED

    def test_suggests_both_merged_from_document_only_when_both_present(self):
        splitter = TaskSplitter(
            query_mode=QueryMode.DOCUMENT_ONLY,
            code_count=10,
            doc_count=5,
        )
        assert splitter._suggest_query_mode() == QueryMode.BOTH_MERGED

    def test_no_suggestion_when_content_matches_mode(self):
        splitter = TaskSplitter(
            query_mode=QueryMode.CODE_ONLY,
            code_count=10,
            doc_count=0,
        )
        assert splitter._suggest_query_mode() is None

    def test_no_suggestion_when_both_empty(self):
        splitter = TaskSplitter(
            query_mode=QueryMode.CODE_ONLY,
            code_count=0,
            doc_count=0,
        )
        assert splitter._suggest_query_mode() is None


class TestLastSplitInfo:
    """Test last_split_info population during split_task."""

    def test_last_split_info_populated_on_empty_result(self, tmp_path: Path):
        import asyncio

        splitter = TaskSplitter(
            repo_path=str(tmp_path),
            query_mode=QueryMode.CODE_ONLY,
            code_count=0,
            doc_count=5,
        )
        subtasks = asyncio.run(splitter.split_task("test query", strategy="file"))
        assert subtasks == []
        assert splitter.last_split_info is not None
        assert splitter.last_split_info.suggested_mode == QueryMode.DOCUMENT_ONLY
        assert splitter.last_split_info.code_count == 0
        assert splitter.last_split_info.doc_count == 5

    def test_last_split_info_reset_on_successful_split(self, tmp_path: Path):
        import asyncio

        (tmp_path / "test.py").write_text("def foo(): pass\n")
        splitter = TaskSplitter(
            repo_path=str(tmp_path),
            query_mode=QueryMode.CODE_ONLY,
            code_count=10,
            doc_count=0,
        )
        subtasks = asyncio.run(splitter.split_task("test query", strategy="file"))
        assert len(subtasks) > 0
        assert splitter.last_split_info is not None
        assert splitter.last_split_info.subtask_count == len(subtasks)
        assert splitter.last_split_info.suggested_mode is None

    def test_last_split_info_on_scope_paths(self, tmp_path: Path):
        import asyncio

        (tmp_path / "main.py").write_text("def foo(): pass\n")
        splitter = TaskSplitter(
            repo_path=str(tmp_path),
            query_mode=QueryMode.CODE_ONLY,
            code_count=10,
            doc_count=0,
        )
        subtasks = asyncio.run(splitter.split_task(
            f"review {tmp_path / 'main.py'}", strategy="file"
        ))
        assert len(subtasks) > 0
        assert splitter.last_split_info is not None
        assert splitter.last_split_info.subtask_count == len(subtasks)

    def test_split_info_dataclass_defaults(self):
        info = SplitInfo()
        assert info.subtask_count == 0
        assert info.warning is None
        assert info.suggested_mode is None
        assert info.code_count == 0
        assert info.doc_count == 0
