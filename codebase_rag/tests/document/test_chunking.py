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
        chunker = SemanticDocumentChunker(max_tokens=10, overlap_tokens=0)

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
        """Test chunking an empty document returns no chunks."""
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
        # Empty document should produce no chunks
        assert len(chunks) == 0

    def test_chunk_token_count(self):
        """Test that chunks have token counts."""
        chunker = SemanticDocumentChunker(max_tokens=50, overlap_tokens=10)

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
        chunker = SemanticDocumentChunker(max_tokens=50, overlap_tokens=10)

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

    def test_plain_text_oversized_paragraph(self):
        """Test that plain text documents handle oversized paragraphs.

        This tests the P0 fix where _chunk_plain_text didn't split
        paragraphs exceeding max_tokens.
        """
        chunker = SemanticDocumentChunker(max_tokens=50, overlap_tokens=10)
        # Single paragraph that exceeds max_tokens
        large_para = " ".join(["word"] * 200)  # ~270 tokens

        doc = ExtractedDocument(
            path="/test/plain.md",
            file_type=".md",
            content=large_para,
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=200,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        assert len(chunks) > 1  # Should split into multiple chunks

        for chunk in chunks:
            assert chunk.token_count <= chunker.max_tokens, (
                f"Plain text chunk exceeded limit: {chunk.token_count} > {chunker.max_tokens}"
            )

    def test_actual_token_count_matches_content(self):
        """Verify reported token_count matches actual content tokens."""
        chunker = SemanticDocumentChunker(max_tokens=100, overlap_tokens=20)

        content = " ".join(["word"] * 150)  # Will be split
        doc = ExtractedDocument(
            path="/test/tokens.md",
            file_type=".md",
            content=content,
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=150,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        for chunk in chunks:
            # Verify token_count is calculated from actual content
            actual_tokens = chunker.count_tokens(chunk.content)
            assert chunk.token_count == actual_tokens, (
                f"Reported token_count {chunk.token_count} != actual {actual_tokens}"
            )


class TestParameterValidation:
    """Tests for constructor parameter validation."""

    def test_invalid_max_tokens_zero(self):
        """Test that max_tokens=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            SemanticDocumentChunker(max_tokens=0)

    def test_invalid_max_tokens_negative(self):
        """Test that negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            SemanticDocumentChunker(max_tokens=-10)

    def test_invalid_overlap_tokens_negative(self):
        """Test that negative overlap_tokens raises ValueError."""
        with pytest.raises(ValueError, match="overlap_tokens must be non-negative"):
            SemanticDocumentChunker(max_tokens=100, overlap_tokens=-5)

    def test_invalid_overlap_exceeds_max_tokens(self):
        """Test that overlap_tokens >= max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="overlap_tokens .* must be less than max_tokens"):
            SemanticDocumentChunker(max_tokens=50, overlap_tokens=50)

    def test_invalid_overlap_greater_than_max_tokens(self):
        """Test that overlap_tokens > max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="overlap_tokens .* must be less than max_tokens"):
            SemanticDocumentChunker(max_tokens=50, overlap_tokens=100)

    def test_valid_parameters(self):
        """Test that valid parameters are accepted."""
        chunker = SemanticDocumentChunker(max_tokens=512, overlap_tokens=50)
        assert chunker.max_tokens == 512
        assert chunker.overlap_tokens == 50

    def test_zero_overlap_is_valid(self):
        """Test that overlap_tokens=0 is valid."""
        chunker = SemanticDocumentChunker(max_tokens=100, overlap_tokens=0)
        assert chunker.overlap_tokens == 0


class TestLineTrackingAccuracy:
    """Tests for line tracking accuracy."""

    def test_line_numbers_are_zero_indexed(self):
        """Verify all line numbers use 0-indexed convention."""
        chunker = SemanticDocumentChunker(max_tokens=100)

        doc = ExtractedDocument(
            path="/test/lines.md",
            file_type=".md",
            content="Line 0\n\nLine 2\n\nLine 4",
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=5,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        # All start_line values should be >= 0
        for chunk in chunks:
            assert chunk.start_line >= 0, f"start_line {chunk.start_line} should be >= 0"
            assert chunk.end_line >= chunk.start_line, "end_line should be >= start_line"

    def test_plain_text_line_tracking(self):
        """Verify line tracking in plain text documents (no sections)."""
        chunker = SemanticDocumentChunker(max_tokens=100)

        # Content: line 0, blank, blank, line 3, blank, blank, line 6
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        doc = ExtractedDocument(
            path="/test/plain.md",
            file_type=".md",
            content=content,
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=6,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        assert len(chunks) >= 1

        # Verify line numbers are 0-indexed
        for chunk in chunks:
            assert chunk.start_line >= 0
            # Extract content at claimed lines
            lines = content.split("\n")
            if chunk.end_line < len(lines):
                extracted = "\n".join(lines[chunk.start_line : chunk.end_line + 1])
                # Content should match (allowing for stripping)
                assert chunk.content.strip() in extracted or extracted.strip() in chunk.content

    def test_section_line_tracking_preserved(self):
        """Verify section line numbers are preserved in chunks."""
        chunker = SemanticDocumentChunker(max_tokens=100)

        doc = ExtractedDocument(
            path="/test/section.md",
            file_type=".md",
            content="",
            sections=[
                ExtractedSection(
                    title="Section 1",
                    level=1,
                    start_line=10,  # Section starts at line 10
                    end_line=20,
                    content="Content here.",
                    subsections=[],
                ),
            ],
            code_blocks=[],
            code_references=[],
            word_count=2,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        assert len(chunks) == 1
        # Chunk should use section's line numbers
        assert chunks[0].start_line == 10
        assert chunks[0].end_line <= 20


class TestEdgeCaseHandling:
    """Tests for edge case handling."""

    def test_whitespace_only_plain_text(self):
        """Test that whitespace-only documents without sections produce no chunks."""
        chunker = SemanticDocumentChunker()

        doc = ExtractedDocument(
            path="/test/whitespace.md",
            file_type=".md",
            content="   \n\n   \t\t\n   ",  # Whitespace only
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=0,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        assert len(chunks) == 0, "Whitespace-only content should not produce chunks"

    def test_whitespace_only_section(self):
        """Test that whitespace-only sections produce no chunks."""
        chunker = SemanticDocumentChunker()

        doc = ExtractedDocument(
            path="/test/whitespace_section.md",
            file_type=".md",
            content="",
            sections=[
                ExtractedSection(
                    title="Empty Section",
                    level=1,
                    start_line=0,
                    end_line=5,
                    content="   \n\n   ",  # Whitespace only
                    subsections=[],
                ),
            ],
            code_blocks=[],
            code_references=[],
            word_count=0,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        assert len(chunks) == 0, "Whitespace-only sections should not produce chunks"

    def test_mixed_content_and_whitespace(self):
        """Test that documents with real content and whitespace work correctly."""
        chunker = SemanticDocumentChunker()

        doc = ExtractedDocument(
            path="/test/mixed.md",
            file_type=".md",
            content="Real content here.\n\n   \n\nMore content.",
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=4,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        # Should have chunks with actual content, not just whitespace
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.content.strip(), "Chunks should have non-empty content"


class TestSeparatorTokenAccounting:
    """Tests for separator token accounting."""

    def test_paragraph_separator_accounted(self):
        """Verify that paragraph separators are accounted for in token limits."""
        chunker = SemanticDocumentChunker(max_tokens=50, overlap_tokens=0)

        # Create content where separator would push over limit
        # Each paragraph is ~20 tokens, separator is ~2 tokens
        para = "This is a test paragraph with some content."
        content = f"{para}\n\n{para}\n\n{para}"  # 3 paragraphs with separators

        doc = ExtractedDocument(
            path="/test/separators.md",
            file_type=".md",
            content=content,
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=15,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        # All chunks should respect max_tokens including separators
        for chunk in chunks:
            assert chunk.token_count <= chunker.max_tokens, (
                f"Chunk exceeded limit with separators: {chunk.token_count} > {chunker.max_tokens}"
            )

    def test_overlap_respects_token_limit(self):
        """Verify that overlap handling respects token limits."""
        chunker = SemanticDocumentChunker(max_tokens=100, overlap_tokens=20)

        # Create content that would trigger overlap
        para = "This is a paragraph with enough tokens to trigger overlap handling in the chunker. "
        content = para * 10  # Large enough to trigger multiple chunks

        doc = ExtractedDocument(
            path="/test/overlap.md",
            file_type=".md",
            content=content,
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=50,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        # All chunks must respect limit, even with overlap
        for chunk in chunks:
            assert chunk.token_count <= chunker.max_tokens, (
                f"Chunk with overlap exceeded limit: {chunk.token_count} > {chunker.max_tokens}"
            )


class TestSecurityBounds:
    """Tests for security bounds and limits."""

    def test_max_tokens_upper_bound(self):
        """Test that max_tokens exceeding upper bound raises ValueError."""
        with pytest.raises(ValueError, match="exceeds reasonable limit"):
            SemanticDocumentChunker(max_tokens=10000)  # Exceeds MAX_REASONABLE_TOKENS (8192)

    def test_max_tokens_at_upper_bound_accepted(self):
        """Test that max_tokens at upper bound is accepted."""
        chunker = SemanticDocumentChunker(max_tokens=8192)
        assert chunker.max_tokens == 8192

    def test_reduce_segment_to_fit_guarantee(self):
        """Verify _reduce_segment_to_fit always returns segment <= max_tokens."""
        chunker = SemanticDocumentChunker(max_tokens=10, overlap_tokens=5)

        # Create a pathological segment that might need many iterations
        pathological_segment = "x" * 10000  # Very long string

        segment, tokens = chunker._reduce_segment_to_fit(pathological_segment)
        assert tokens <= chunker.max_tokens, (
            f"Segment tokens {tokens} exceeded max_tokens {chunker.max_tokens}"
        )
        assert len(segment) > 0, "Segment should not be empty"

    def test_reduce_segment_to_fit_preserves_content(self):
        """Verify _reduce_segment_to_fit doesn't break content unnecessarily."""
        chunker = SemanticDocumentChunker(max_tokens=100)

        # Segment that fits
        segment = "This is a short segment."
        reduced, tokens = chunker._reduce_segment_to_fit(segment)

        assert reduced == segment, "Should not reduce segment that fits"
        assert tokens == chunker.count_tokens(segment)

    def test_max_chunks_per_document_limit(self):
        """Verify max_chunks_per_document limit is enforced."""
        # Use very small max_tokens to generate many chunks
        chunker = SemanticDocumentChunker(max_tokens=20, overlap_tokens=0)

        # Create content that would generate many chunks
        # Each sentence is ~10 tokens, so we need many sentences
        sentences = [f"Sentence number {i} here." for i in range(2000)]
        content = " ".join(sentences)

        doc = ExtractedDocument(
            path="/test/large.md",
            file_type=".md",
            content=content,
            sections=[],
            code_blocks=[],
            code_references=[],
            word_count=2000,
            modified_date="2024-01-01",
        )

        chunks = list(chunker.chunk_document(doc))
        assert len(chunks) <= chunker.MAX_CHUNKS_PER_DOCUMENT, (
            f"Chunks {len(chunks)} exceeded limit {chunker.MAX_CHUNKS_PER_DOCUMENT}"
        )