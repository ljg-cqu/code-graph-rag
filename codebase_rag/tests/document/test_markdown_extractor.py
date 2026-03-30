"""Tests for Markdown extractor."""

from __future__ import annotations

import pytest
from pathlib import Path

from codebase_rag.document.extractors.markdown_extractor import MarkdownExtractor
from codebase_rag.document.extractors.base import ExtractedDocument
from codebase_rag.document.error_handling import ErrorType, ExtractionException


class TestMarkdownExtractor:
    """Tests for MarkdownExtractor."""

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extractor = MarkdownExtractor()
        assert ".md" in extractor.supported_extensions
        assert ".rst" in extractor.supported_extensions
        assert ".txt" in extractor.supported_extensions

    def test_extract_simple_markdown(self, tmp_path):
        """Test extracting a simple markdown file with hierarchical sections."""
        extractor = MarkdownExtractor()

        # Create test file
        md_file = tmp_path / "test.md"
        md_file.write_text(
            """# Title

This is the introduction.

## Section 1

Content for section 1.

### Subsection 1.1

More content here.

## Section 2

Final section.
"""
        )

        doc = extractor.extract(md_file)

        assert isinstance(doc, ExtractedDocument)
        assert doc.file_type == ".md"
        assert "Title" in doc.content

        # Check hierarchical structure
        # Root: Title (H1)
        assert len(doc.sections) == 1
        assert doc.sections[0].title == "Title"
        assert doc.sections[0].level == 1

        # Title has 2 subsections: Section 1 and Section 2 (both H2)
        assert len(doc.sections[0].subsections) == 2
        assert doc.sections[0].subsections[0].title == "Section 1"
        assert doc.sections[0].subsections[0].level == 2
        assert doc.sections[0].subsections[1].title == "Section 2"
        assert doc.sections[0].subsections[1].level == 2

        # Section 1 has 1 subsection: Subsection 1.1 (H3)
        assert len(doc.sections[0].subsections[0].subsections) == 1
        assert doc.sections[0].subsections[0].subsections[0].title == "Subsection 1.1"
        assert doc.sections[0].subsections[0].subsections[0].level == 3

    def test_extract_code_blocks(self, tmp_path):
        """Test extracting code blocks from markdown."""
        extractor = MarkdownExtractor()

        md_file = tmp_path / "code.md"
        md_file.write_text(
            """# Code Example

```python
def hello():
    print("Hello, World!")
```

```javascript
function greet() {
    console.log("Hi!");
}
```
"""
        )

        doc = extractor.extract(md_file)

        assert len(doc.code_blocks) == 2
        assert "def hello():" in doc.code_blocks[0]
        assert "function greet()" in doc.code_blocks[1]

    def test_headers_in_code_blocks_ignored(self, tmp_path):
        """Test that headers inside fenced code blocks are not extracted as sections."""
        extractor = MarkdownExtractor()

        md_file = tmp_path / "code_headers.md"
        md_file.write_text(
            """# Real Header

Some content here.

```python
# This is a comment, not a header
def fake_header():
    pass

# Another comment
x = 1
```

## Real Subsection

More content.
"""
        )

        doc = extractor.extract(md_file)

        # Should only have 2 sections: Real Header and Real Subsection
        # The # comments in the code block should NOT be detected as headers
        assert len(doc.sections) == 1  # Just the root "Real Header"
        assert doc.sections[0].title == "Real Header"
        # Real Subsection should be a subsection of Real Header
        assert len(doc.sections[0].subsections) == 1
        assert doc.sections[0].subsections[0].title == "Real Subsection"

    def test_extract_code_references(self, tmp_path):
        """Test extracting code references from markdown."""
        extractor = MarkdownExtractor()

        md_file = tmp_path / "refs.md"
        md_file.write_text(
            """# API Reference

Use the `UserService.create_user()` method to create users.

The `User` class handles user data.

Call `authenticate()` to verify credentials.
"""
        )

        doc = extractor.extract(md_file)

        # Should extract function/class references
        assert len(doc.code_references) > 0

    def test_extract_rst_file(self, tmp_path):
        """Test extracting reStructuredText file."""
        extractor = MarkdownExtractor()

        rst_file = tmp_path / "test.rst"
        rst_file.write_text(
            """Title
=====

Introduction text.

Section 1
---------

Content for section 1.
"""
        )

        doc = extractor.extract(rst_file)

        assert doc.file_type == ".rst"
        assert "Title" in doc.content
        assert len(doc.sections) >= 1

    def test_extract_text_file(self, tmp_path):
        """Test extracting plain text file."""
        extractor = MarkdownExtractor()

        txt_file = tmp_path / "test.txt"
        txt_file.write_text(
            """Plain text content.
No headers or special formatting.
Just text.
"""
        )

        doc = extractor.extract(txt_file)

        assert doc.file_type == ".txt"
        assert "Plain text content" in doc.content

    def test_file_not_found(self, tmp_path):
        """Test extraction error for missing file."""
        extractor = MarkdownExtractor()

        with pytest.raises(ExtractionException) as exc_info:
            extractor.extract(tmp_path / "nonexistent.md")

        assert exc_info.value.error_type == ErrorType.FILE_NOT_FOUND

    def test_word_count(self, tmp_path):
        """Test word count calculation."""
        extractor = MarkdownExtractor()

        md_file = tmp_path / "words.md"
        md_file.write_text("One two three four five.")

        doc = extractor.extract(md_file)

        assert doc.word_count == 5

    def test_content_hash(self, tmp_path):
        """Test content hash is computed."""
        extractor = MarkdownExtractor()

        md_file = tmp_path / "hash.md"
        md_file.write_text("Content for hashing")

        doc = extractor.extract(md_file)

        assert doc.content_hash != ""
        assert len(doc.content_hash) == 64  # SHA-256 hex length

    def test_async_extraction(self, tmp_path):
        """Test async extraction method."""
        import asyncio

        extractor = MarkdownExtractor()

        md_file = tmp_path / "async.md"
        md_file.write_text("# Async Test\n\nContent here.")

        async def run_async():
            return await extractor.extract_async(md_file)

        doc = asyncio.run(run_async())

        assert doc.file_type == ".md"
        assert "Async Test" in doc.content

    def test_modified_date(self, tmp_path):
        """Test modified date is extracted."""
        extractor = MarkdownExtractor()

        md_file = tmp_path / "date.md"
        md_file.write_text("Content")

        doc = extractor.extract(md_file)

        assert doc.modified_date != ""
        assert "T" in doc.modified_date  # ISO format

    def test_section_levels(self, tmp_path):
        """Test section levels are correct in hierarchical structure."""
        extractor = MarkdownExtractor()

        md_file = tmp_path / "levels.md"
        md_file.write_text(
            """# H1

## H2

### H3

#### H4
"""
        )

        doc = extractor.extract(md_file)

        # Root level has H1
        assert len(doc.sections) == 1
        assert doc.sections[0].level == 1
        assert doc.sections[0].title == "H1"

        # H1 has H2 as subsection
        assert len(doc.sections[0].subsections) == 1
        assert doc.sections[0].subsections[0].level == 2
        assert doc.sections[0].subsections[0].title == "H2"

        # H2 has H3 as subsection
        assert len(doc.sections[0].subsections[0].subsections) == 1
        assert doc.sections[0].subsections[0].subsections[0].level == 3
        assert doc.sections[0].subsections[0].subsections[0].title == "H3"

        # H3 has H4 as subsection
        assert len(doc.sections[0].subsections[0].subsections[0].subsections) == 1
        assert doc.sections[0].subsections[0].subsections[0].subsections[0].level == 4
        assert doc.sections[0].subsections[0].subsections[0].subsections[0].title == "H4"