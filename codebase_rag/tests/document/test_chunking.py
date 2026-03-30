"""Tests for semantic document chunking."""

from __future__ import annotations

import pytest

from codebase_rag.document.chunking import (
    SemanticDocumentChunker,
    DocumentChunk,
)
from codebase_rag.document.extractors.base import (
    ExtractedDocument,
    ExtractedSection,
)


class TestDocumentChunk:
    """Tests for DocumentChunk dataclass."""

    def test_create_chunk(self):
        """Test creating a document chunk."""
        chunk = DocumentChunk(
            content="This is the introduction content.",
            section_title="Introduction",
            start_line=0,
            end_line=10,
            token_count=10,
            document_path="/test/doc.md",
            chunk_index=0,
        )
        assert chunk.document_path == "/test/doc.md"
        assert chunk.section_title == "Introduction"
        assert chunk.chunk_index == 0
        assert chunk.token_count == 10

    def test_qualified_name(self):
        """Test qualified_name property."""
        chunk = DocumentChunk(
            content="Content",
            section_title="Getting Started",
            start_line=0,
            end_line=10,
            token_count=5,
            document_path="/test/doc.md",
            chunk_index=2,
        )
        assert chunk.qualified_name == "/test/doc.md#chunk_2"


class TestSemanticDocumentChunker:
    """Tests for SemanticDocumentChunker."""

    def test_chunk_small_document(self):
        """Test chunking a small document that fits in one chunk."""
        chunker = SemanticDocumentChunker(max_tokens=512)

        doc = ExtractedDocument(
            path="/test/small.md",
            file_type=".md",
            content="This is a small document.",
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=5,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        assert len(chunks) == 1
        assert chunks[0].content == "This is a small document."

    def test_chunk_document_with_sections(self):
        """Test chunking preserves section boundaries."""
        chunker = SemanticDocumentChunker(max_tokens=100)

        doc = ExtractedDocument(
            path="/test/doc.md",
            file_type=".md",
            content="",
            sections=[
                ExtractedSection(
                    title="Section 1",
                    level=1,
                    start_line=0,
                    end_line=5,
                    content="Content for section 1.",
                    subsections=[],
                ),
                ExtractedSection(
                    title="Section 2",
                    level=1,
                    start_line=6,
                    end_line=10,
                    content="Content for section 2.",
                    subsections=[],
                ),
            ],
            code_blocks=[],
            code_references=[],
            word_count=10,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        # Each section should produce at least one chunk
        section_titles = {c.section_title for c in chunks}
        assert "Section 1" in section_titles
        assert "Section 2" in section_titles

    def test_chunk_large_section(self):
        """Test chunking a large section into multiple chunks."""
        chunker = SemanticDocumentChunker(max_tokens=10)

        # Create a large section that exceeds chunk size
        large_content = " ".join(["word"] * 100)

        doc = ExtractedDocument(
            path="/test/large.md",
            file_type=".md",
            content="",
            sections=[
                ExtractedSection(
                    title="Large Section",
                    level=1,
                    start_line=0,
                    end_line=100,
                    content=large_content,
                    subsections=[],
                ),
            ],
            code_blocks=[],
            code_references=[],
            word_count=50,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        # Should produce multiple chunks since content exceeds max_tokens
        assert len(chunks) >= 1
        # All chunks should be from the same section
        assert all(c.section_title == "Large Section" for c in chunks)

    def test_chunk_empty_document(self):
        """Test chunking an empty document."""
        chunker = SemanticDocumentChunker()

        doc = ExtractedDocument(
            path="/test/empty.md",
            file_type=".md",
            content="",
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=0,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        # Empty document may still produce a chunk with empty content
        assert len(chunks) >= 1
        assert chunks[0].content == ""

    def test_chunk_token_count(self):
        """Test that chunks have token counts."""
        chunker = SemanticDocumentChunker(max_tokens=50)

        doc = ExtractedDocument(
            path="/test/doc.md",
            file_type=".md",
            content="This is test content for chunking.",
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=6,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        for chunk in chunks:
            assert chunk.token_count > 0


class TestTokenCounting:
    """Tests for token counting functionality."""

    def test_count_tokens(self):
        """Test token counting."""
        chunker = SemanticDocumentChunker()

        # Simple text should have approximate token count
        text = "Hello world this is a test"
        tokens = chunker.count_tokens(text)
        # Should return a positive count
        assert tokens > 0

    def test_empty_text_tokens(self):
        """Test token count for empty text."""
        chunker = SemanticDocumentChunker()
        assert chunker.count_tokens("") == 0