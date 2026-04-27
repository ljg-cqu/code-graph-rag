"""Tests for document chunking and tiny chunk merging."""

from __future__ import annotations

from codebase_rag.document.chunking import DocumentChunk, merge_tiny_chunks


class TestMergeTinyChunks:
    """Tests for merge_tiny_chunks function."""

    def _chunk(
        self,
        content: str,
        token_count: int,
        start_line: int = 0,
        end_line: int = 0,
        chunk_index: int = 0,
    ) -> DocumentChunk:
        return DocumentChunk(
            content=content,
            section_title="",
            start_line=start_line,
            end_line=end_line,
            token_count=token_count,
            document_path="test.md",
            chunk_index=chunk_index,
        )

    def test_no_chunks_returns_empty(self):
        """Empty list returns empty list."""
        assert merge_tiny_chunks([], min_tokens=10) == []

    def test_all_large_chunks_unchanged(self):
        """Chunks above threshold are not modified."""
        chunks = [
            self._chunk("First paragraph with enough tokens", 15, chunk_index=0),
            self._chunk("Second paragraph with enough tokens", 20, chunk_index=1),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        assert len(result) == 2
        assert result[0].content == chunks[0].content
        assert result[1].content == chunks[1].content

    def test_tiny_chunk_in_middle_merges_with_previous(self):
        """Tiny chunk between two large chunks merges with previous."""
        chunks = [
            self._chunk("First paragraph with enough tokens", 15, start_line=0, end_line=0, chunk_index=0),
            self._chunk("---", 3, start_line=1, end_line=1, chunk_index=1),
            self._chunk("Third paragraph with enough tokens", 18, start_line=2, end_line=2, chunk_index=2),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        assert len(result) == 2
        assert result[0].chunk_index == 0
        assert result[1].chunk_index == 1
        assert "---" in result[0].content
        assert result[0].start_line == 0
        assert result[0].end_line == 1

    def test_tiny_chunk_at_start_merges_with_next(self):
        """Tiny chunk at start merges with next chunk."""
        chunks = [
            self._chunk("---", 3, start_line=0, end_line=0, chunk_index=0),
            self._chunk("Second paragraph with enough tokens", 18, start_line=1, end_line=1, chunk_index=1),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        assert len(result) == 1
        assert result[0].chunk_index == 0
        assert "---" in result[0].content
        assert "Second paragraph" in result[0].content
        assert result[0].start_line == 0
        assert result[0].end_line == 1

    def test_tiny_chunk_at_end_merges_with_previous(self):
        """Tiny chunk at end merges with previous chunk."""
        chunks = [
            self._chunk("First paragraph with enough tokens", 18, start_line=0, end_line=0, chunk_index=0),
            self._chunk("---", 3, start_line=1, end_line=1, chunk_index=1),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        assert len(result) == 1
        assert result[0].chunk_index == 0
        assert "---" in result[0].content
        assert result[0].start_line == 0
        assert result[0].end_line == 1

    def test_renumbers_indices_after_merge(self):
        """chunk_index is renumbered sequentially after merge."""
        chunks = [
            self._chunk("A", 3, chunk_index=5),
            self._chunk("B", 3, chunk_index=7),
            self._chunk("C", 3, chunk_index=9),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        for idx, chunk in enumerate(result):
            assert chunk.chunk_index == idx

    def test_multiple_tiny_chunks_merge_correctly(self):
        """Multiple tiny chunks merge into adjacent large chunks."""
        chunks = [
            self._chunk("Large first", 20, start_line=0, end_line=2, chunk_index=0),
            self._chunk("---", 3, start_line=3, end_line=3, chunk_index=1),
            self._chunk("***", 3, start_line=4, end_line=4, chunk_index=2),
            self._chunk("Large last", 20, start_line=5, end_line=7, chunk_index=3),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        assert len(result) == 2
        assert result[0].chunk_index == 0
        assert result[1].chunk_index == 1
        assert "---" in result[0].content
        assert "***" in result[0].content

    def test_section_title_preserved_from_merge_target(self):
        """Section title is preserved from the chunk being merged into."""
        chunks = [
            DocumentChunk(
                content="Large section content here",
                section_title="Section A",
                start_line=0,
                end_line=0,
                token_count=20,
                document_path="test.md",
                chunk_index=0,
            ),
            DocumentChunk(
                content="---",
                section_title="Section B",
                start_line=1,
                end_line=1,
                token_count=3,
                document_path="test.md",
                chunk_index=1,
            ),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        assert result[0].section_title == "Section A"

    def test_document_path_preserved(self):
        """Document path is preserved through merge."""
        chunks = [
            self._chunk("First", 15, chunk_index=0),
            self._chunk("---", 3, chunk_index=1),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        assert all(c.document_path == "test.md" for c in result)

    def test_cross_section_chunks_not_merged(self):
        """Tiny chunks from different sections remain separate."""
        chunks = [
            DocumentChunk(
                content="Section A content",
                section_title="Section A",
                start_line=0,
                end_line=0,
                token_count=5,
                document_path="test.md",
                chunk_index=0,
            ),
            DocumentChunk(
                content="Section B content",
                section_title="Section B",
                start_line=1,
                end_line=1,
                token_count=5,
                document_path="test.md",
                chunk_index=1,
            ),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        assert len(result) == 2
        assert result[0].section_title == "Section A"
        assert result[1].section_title == "Section B"

    def test_same_section_chunks_merge(self):
        """Tiny chunks within the same section are merged."""
        chunks = [
            DocumentChunk(
                content="Part one",
                section_title="Section A",
                start_line=0,
                end_line=0,
                token_count=5,
                document_path="test.md",
                chunk_index=0,
            ),
            DocumentChunk(
                content="Part two",
                section_title="Section A",
                start_line=1,
                end_line=1,
                token_count=5,
                document_path="test.md",
                chunk_index=1,
            ),
        ]
        result = merge_tiny_chunks(chunks, min_tokens=10)
        assert len(result) == 1
        assert result[0].section_title == "Section A"
        assert "Part one" in result[0].content
        assert "Part two" in result[0].content
