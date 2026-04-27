"""Tests for _split_chunk helper in concept_runner.

Covers Markdown heading, paragraph, and sentence splitting heuristics.
"""

from __future__ import annotations

from unittest.mock import patch

from codebase_rag.document.chunk_splitting import _split_chunk


class TestSplitChunk:
    """Test chunk splitting heuristics."""

    def test_markdown_heading_split(self):
        content = (
            "# Heading 1\n"
            + "Some text here that is long enough to exceed the minimum length requirement for a valid section. "
            + "This additional text ensures the section is well over one hundred characters in total length. "
            + "We keep adding more and more text to make absolutely sure this exceeds five hundred characters. "
            + "The quick brown fox jumps over the lazy dog repeatedly to inflate the character count sufficiently.\n"
            + "## Heading 2\n"
            + "More text here that is also long enough to not be dropped by the filtering mechanism. "
            + "We need extra words to make sure this section exceeds the minimum threshold comfortably. "
            + "Adding substantial content guarantees that all sections pass the length validation check easily. "
            + "Continuous prose increases the section size well beyond any reasonable minimum length boundary.\n"
            + "### Heading 3\n"
            + "Even more text here with sufficient length to survive the section length filter. "
            + "Adding more content guarantees that all three sections pass the length validation check. "
            + "We must ensure every single heading section has more than five hundred characters total. "
            + "This final paragraph provides the necessary length to satisfy the minimum size constraint."
        )
        with patch("codebase_rag.document.chunk_splitting.settings") as mock_settings:
            mock_settings.DOC_CHUNK_SIZE = 4000
            mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 100
            result = _split_chunk(content)
        assert len(result) == 3
        assert "Heading 1" in result[0]
        assert "Heading 2" in result[1]
        assert "Heading 3" in result[2]

    def test_markdown_heading_drops_short_fragments(self):
        content = "# Heading 1\nShort.\n## Heading 2\nAlso short."
        with patch("codebase_rag.document.chunk_splitting.settings") as mock_settings:
            mock_settings.DOC_CHUNK_SIZE = 4000
            mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 100
            result = _split_chunk(content)
        # Each section is < 100 chars → dropped
        assert len(result) == 0

    def test_paragraph_split(self):
        content = (
            "Paragraph one with enough text to exceed the minimum length requirement for splitting and to be considered a valid standalone batch.\n\n"
            "Paragraph two also has sufficient content to be considered a valid batch on its own without being dropped by the filtering mechanism.\n\n"
            "Paragraph three continues the pattern with adequate length to survive the filtering mechanism."
        )
        with patch("codebase_rag.document.chunk_splitting.settings") as mock_settings:
            mock_settings.DOC_CHUNK_SIZE = 4000
            mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 100
            result = _split_chunk(content)
        assert len(result) >= 1
        for r in result:
            assert len(r) > 100

    def test_sentence_split(self):
        content = (
            "This is sentence one with enough words to make it substantial. "
            "This is sentence two also with sufficient length. "
            "This is sentence three continuing the pattern. "
            "This is sentence four with extra words to make it longer and more meaningful."
        )
        with patch("codebase_rag.document.chunk_splitting.settings") as mock_settings:
            mock_settings.DOC_CHUNK_SIZE = 4000
            mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 100
            result = _split_chunk(content)
        # No headings or paragraphs → falls back to sentence split
        assert len(result) >= 1

    def test_single_chunk_no_split_needed(self):
        content = "Short content."
        with patch("codebase_rag.document.chunk_splitting.settings") as mock_settings:
            mock_settings.DOC_CHUNK_SIZE = 4000
            mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 100
            result = _split_chunk(content)
        assert len(result) == 1
        assert result[0] == "Short content."

    def test_max_chars_parameter(self):
        # Two paragraphs, each exceeding max_chars, should each be truncated
        content = (
            "Paragraph one with enough text to exceed the max_chars limit and be truncated accordingly. "
            * 3
            + "\n\n"
            + "Paragraph two with enough text to exceed the max_chars limit and be truncated accordingly. "
            * 3
        )
        result = _split_chunk(content, max_chars=80)
        # Should split into multiple truncated batches
        assert len(result) >= 2
        for r in result:
            assert len(r) <= 80

    def test_empty_content(self):
        result = _split_chunk("")
        assert result == [""]

    def test_fallback_to_content_slice(self):
        """If no sentences, paragraphs, or headings, return content[:max_chars]."""
        content = "word " * 1000  # No sentence endings
        result = _split_chunk(content, max_chars=100)
        assert len(result) == 1
        assert len(result[0]) <= 100

    def test_single_oversized_paragraph_truncated(self):
        """A single paragraph exceeding max_chars should be truncated."""
        content = "word " * 1000  # No paragraph breaks, no sentence endings
        with patch("codebase_rag.document.chunk_splitting.settings") as mock_settings:
            mock_settings.DOC_CHUNK_SIZE = 200
            mock_settings.DOC_CONCEPT_MIN_CHUNK_SIZE = 100
            result = _split_chunk(content)
        # Should truncate to max_chars (max(1000, 200//2) = 1000)
        # Wait, default max_chars = max(1000, DOC_CHUNK_SIZE // 2) = 1000
        # So with DOC_CHUNK_SIZE=200, max_chars = max(1000, 100) = 1000
        # Hmm, let me use explicit max_chars
        result = _split_chunk(content, max_chars=100)
        assert len(result) == 1
        assert len(result[0]) <= 100
