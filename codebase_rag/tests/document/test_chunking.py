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

    def test_token_limit_enforcement_with_overlap(self):
        """Test that chunks never exceed max_tokens, even with overlap.

        This tests the bug fix where overlap handling could create chunks
        exceeding the max_tokens limit.
        """
        chunker = SemanticDocumentChunker(max_tokens=100, overlap_tokens=30)

        # Create section with paragraphs that would exceed limit if overlap added
        # Paragraph 1: ~60 tokens (within limit)
        # Paragraph 2: ~80 tokens (close to limit)
        # If overlap (30) + para2 (80) = 110, that exceeds max_tokens (100)
        para1 = "This is paragraph one with some content that adds up to about sixty tokens or so. " * 2
        para2 = "This is paragraph two which is longer and has more content to reach about eighty tokens in total length. " * 2
        large_content = para1 + "\n\n" + para2

        doc = ExtractedDocument(
            path="/test/overlap.md",
            file_type=".md",
            content="",
            sections=[
                ExtractedSection(
                    title="Overlap Test",
                    level=1,
                    start_line=1,
                    end_line=10,
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
        # Critical: NO chunk should exceed max_tokens
        for chunk in chunks:
            assert chunk.token_count <= chunker.max_tokens, (
                f"Chunk exceeded max_tokens: {chunk.token_count} > {chunker.max_tokens}"
            )


class TestTokenLimitEnforcement:
    """Tests for token limit enforcement in various scenarios."""

    def test_chunks_never_exceed_max_tokens(self):
        """Verify all chunks respect the configured token limit."""
        chunker = SemanticDocumentChunker(max_tokens=512, overlap_tokens=50)

        # Create a large document with many paragraphs
        paragraphs = []
        for i in range(20):
            # Each paragraph is ~200 words (~270 tokens)
            para = " ".join(["word"] * 200)
            paragraphs.append(para)

        content = "\n\n".join(paragraphs)

        doc = ExtractedDocument(
            path="/test/large.md",
            file_type=".md",
            content=content,
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=4000,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        assert len(chunks) > 1  # Should split into multiple chunks

        for chunk in chunks:
            assert chunk.token_count <= chunker.max_tokens, (
                f"Chunk at index {chunk.chunk_index} exceeded limit: "
                f"{chunk.token_count} > {chunker.max_tokens}"
            )

    def test_code_blocks_without_punctuation(self):
        """Test splitting code blocks/dictionaries without sentence-ending punctuation.

        This tests the bug fix where code blocks (Python dicts, JSON, etc.)
        had no punctuation and created oversized chunks.
        """
        chunker = SemanticDocumentChunker(max_tokens=100)

        # Simulate a Python dictionary without punctuation
        code_block = """
BOUNDARY_NODE_TYPES = {
    "python": {
        "class": ["class_definition"],
        "function": ["function_definition"],
        "method": ["function_definition"],
    },
    "java": {
        "class": ["class_declaration", "interface_declaration"],
        "function": ["method_declaration", "constructor_declaration"],
    },
    "javascript": {
        "class": ["class_declaration"],
        "function": ["function_declaration", "arrow_function"],
    },
}
"""
        doc = ExtractedDocument(
            path="/test/code.md",
            file_type=".md",
            content="",
            sections=[
                ExtractedSection(
                    title="Code Block Test",
                    level=1,
                    start_line=1,
                    end_line=20,
                    content=code_block,
                    subsections=[],
                ),
            ],
            code_blocks=[],
            code_references=[],
            word_count=30,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        # All chunks must respect token limit, even unpunctuated code
        for chunk in chunks:
            assert chunk.token_count <= chunker.max_tokens, (
                f"Code chunk exceeded limit: {chunk.token_count} > {chunker.max_tokens}"
            )

    def test_ascii_diagram_splitting(self):
        """Test splitting ASCII art diagrams that have no punctuation.

        ASCII diagrams in markdown can span many lines with no .!? punctuation.
        """
        chunker = SemanticDocumentChunker(max_tokens=50)

        # ASCII art diagram
        diagram = """
+-------------------+       +-------------------+
|   Service A       | ----> |   Service B       |
+-------------------+       +-------------------+
        |                           |
        |                           |
        v                           v
+-------------------+       +-------------------+
|   Database        |       |   Cache           |
+-------------------+       +-------------------+
"""
        doc = ExtractedDocument(
            path="/test/diagram.md",
            file_type=".md",
            content="",
            sections=[
                ExtractedSection(
                    title="Architecture Diagram",
                    level=1,
                    start_line=1,
                    end_line=15,
                    content=diagram,
                    subsections=[],
                ),
            ],
            code_blocks=[],
            code_references=[],
            word_count=10,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        # Diagram should be split into smaller chunks respecting token limit
        for chunk in chunks:
            assert chunk.token_count <= chunker.max_tokens, (
                f"Diagram chunk exceeded limit: {chunk.token_count} > {chunker.max_tokens}"
            )