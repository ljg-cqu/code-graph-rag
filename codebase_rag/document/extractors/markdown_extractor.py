"""Markdown document extractor.

Supports .md, .rst, and .txt files with section extraction,
code block detection, and code reference extraction.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path

from .base import (
    BaseDocumentExtractor,
    ExtractedDocument,
    ExtractedSection,
)
from ..error_handling import ErrorType, ExtractionException


class MarkdownExtractor(BaseDocumentExtractor):
    """
    Extractor for Markdown files (.md, .rst, .txt).

    Features:
    - Section extraction via header detection
    - Code block extraction with language detection
    - Code reference extraction (function/class names)
    - Content hashing for incremental updates
    """

    # Regex patterns for section detection
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    RST_HEADER_PATTERN = re.compile(r"^([=\-~^\"']+)\n(.+)\n\1$", re.MULTILINE)

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".rst", ".txt"]

    def extract(self, file_path: Path) -> ExtractedDocument:
        """
        Extract content from Markdown file.

        Args:
            file_path: Path to the markdown file

        Returns:
            ExtractedDocument with sections, code blocks, and references

        Raises:
            ExtractionException: If file cannot be read or parsed
        """
        # Validate path
        validated_path = self._validate_path(file_path)

        # Check file size limit
        max_size_mb = self.get_config("max_file_size_mb", 50)
        file_size_mb = validated_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.FILE_TOO_LARGE,
                message=f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)",
            )

        # Read file content
        try:
            content = validated_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            # Try with latin-1 as fallback
            try:
                content = validated_path.read_text(encoding="latin-1")
            except Exception:
                raise ExtractionException(
                    path=str(file_path),
                    error_type=ErrorType.ENCODING_ERROR,
                    message=f"Cannot decode file: {e}",
                )
        except PermissionError as e:
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.PERMISSION_DENIED,
                message=str(e),
            )
        except OSError as e:
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.MALFORMED_FILE,
                message=str(e),
            )

        # Get modification date
        modified_date = datetime.fromtimestamp(
            validated_path.stat().st_mtime, UTC
        ).isoformat()

        # Extract components based on file type
        if file_path.suffix.lower() == ".rst":
            sections = self._extract_rst_sections(content)
        else:
            sections = self._extract_markdown_sections(content)

        code_blocks = self._extract_code_blocks(content)
        code_references = self._extract_code_references(content)
        word_count = len(content.split())

        return ExtractedDocument(
            path=str(file_path),
            file_type=file_path.suffix.lower(),
            content=content,
            sections=sections,
            code_blocks=code_blocks,
            code_references=code_references,
            word_count=word_count,
            modified_date=modified_date,
        )

    async def extract_async(self, file_path: Path) -> ExtractedDocument:
        """
        Async extraction for large files.

        Uses asyncio.to_thread to avoid blocking the event loop.
        """
        return await asyncio.to_thread(self.extract, file_path)

    def _extract_markdown_sections(self, content: str) -> list[ExtractedSection]:
        """Extract header hierarchy from Markdown."""
        sections: list[ExtractedSection] = []
        lines = content.split("\n")

        # Find all header positions
        header_positions: list[tuple[int, int, str, str]] = []  # (line, level, title, content_start)

        for i, line in enumerate(lines):
            match = self.HEADER_PATTERN.match(line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()
                header_positions.append((i, level, title, i + 1))

        # Build sections with content
        for idx, (line_num, level, title, content_start) in enumerate(header_positions):
            # Determine end line (next header or end of file)
            if idx + 1 < len(header_positions):
                end_line = header_positions[idx + 1][0] - 1
            else:
                end_line = len(lines) - 1

            # Extract section content
            section_content = "\n".join(lines[content_start : end_line + 1])

            sections.append(
                ExtractedSection(
                    title=title,
                    level=level,
                    start_line=line_num,
                    end_line=end_line,
                    content=section_content.strip(),
                    subsections=[],
                )
            )

        return sections

    def _extract_rst_sections(self, content: str) -> list[ExtractedSection]:
        """Extract sections from reStructuredText files."""
        sections: list[ExtractedSection] = []
        lines = content.split("\n")

        # RST uses underlines for headers
        # Different characters indicate different levels
        level_map: dict[str, int] = {}
        current_level = 0

        i = 0
        while i < len(lines) - 1:
            line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else ""

            # Check for underline-style header
            if next_line and all(c == next_line[0] for c in next_line) and len(next_line) >= len(line):
                if line.strip():  # Non-empty title
                    underline_char = next_line[0]

                    # Assign level based on character
                    if underline_char not in level_map:
                        current_level += 1
                        level_map[underline_char] = current_level

                    level = level_map[underline_char]
                    title = line.strip()

                    # Find end of section
                    end_line = len(lines) - 1
                    for j in range(i + 2, len(lines) - 1):
                        potential_underline = lines[j + 1] if j + 1 < len(lines) else ""
                        if potential_underline and all(
                            c == potential_underline[0] for c in potential_underline
                        ):
                            if lines[j].strip():
                                end_line = j - 1
                                break

                    section_content = "\n".join(lines[i + 2 : end_line + 1])

                    sections.append(
                        ExtractedSection(
                            title=title,
                            level=level,
                            start_line=i,
                            end_line=end_line,
                            content=section_content.strip(),
                            subsections=[],
                        )
                    )

                    i = end_line + 1
                    continue

            i += 1

        return sections

    def _extract_code_blocks(self, content: str) -> list[str]:
        """Extract code blocks from Markdown."""
        matches = self.CODE_BLOCK_PATTERN.findall(content)
        return [m[1].strip() for m in matches if m[1].strip()]


__all__ = ["MarkdownExtractor"]