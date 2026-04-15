"""PDF document extractor.

Optional extractor that requires PyPDF2 or pdfplumber.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from ..error_handling import ErrorType, ExtractionException
from .base import (
    BaseDocumentExtractor,
    ExtractedDocument,
    ExtractedSection,
)


class PDFExtractor(BaseDocumentExtractor):
    """
    Extractor for PDF files (.pdf).

    Requires PyPDF2 or pdfplumber for text extraction.
    Falls back gracefully if not installed.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def _extract(self, file_path: Path) -> ExtractedDocument:
        """Extract content from PDF file."""
        validated_path = file_path

        # Check file size
        max_size_mb = self.get_config("max_file_size_mb", 50)
        file_size_mb = validated_path.stat().st_size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ExtractionException(
                path=str(file_path),
                error_type=ErrorType.FILE_TOO_LARGE,
                message=f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)",
            )

        # Try to import PDF library
        try:
            import pdfplumber  # type: ignore # noqa: F401
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore # noqa: F401
            except ImportError:
                raise ExtractionException(
                    path=str(file_path),
                    error_type=ErrorType.MISSING_DEPENDENCY,
                    message="PDF extraction requires pdfplumber or PyPDF2. Install with: uv add pdfplumber",
                )
            else:
                return self._extract_with_pypdf2(validated_path, file_path)
        else:
            return self._extract_with_pdfplumber(validated_path, file_path)

    def _extract_with_pdfplumber(
        self, validated_path: Path, original_path: Path
    ) -> ExtractedDocument:
        """Extract using pdfplumber (preferred)."""
        import pdfplumber

        page_texts: list[str] = []

        with pdfplumber.open(validated_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    page_texts.append(text)

        return self._build_document_from_pages(
            page_texts=page_texts,
            validated_path=validated_path,
            original_path=original_path,
        )

    def _extract_with_pypdf2(
        self, validated_path: Path, original_path: Path
    ) -> ExtractedDocument:
        """Extract using PyPDF2 (fallback)."""
        from PyPDF2 import PdfReader

        reader = PdfReader(validated_path)
        page_texts: list[str] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                page_texts.append(text)

        return self._build_document_from_pages(
            page_texts=page_texts,
            validated_path=validated_path,
            original_path=original_path,
        )

    def _build_document_from_pages(
        self,
        page_texts: list[str],
        validated_path: Path,
        original_path: Path,
    ) -> ExtractedDocument:
        content_lines: list[str] = []
        sections: list[ExtractedSection] = []

        for index, text in enumerate(page_texts, start=1):
            page_title = f"Page {index}"
            page_lines = text.splitlines() or [text]
            start_line = len(content_lines)
            content_lines.append(page_title)
            content_lines.extend(page_lines)
            end_line = len(content_lines) - 1

            sections.append(
                ExtractedSection(
                    title=page_title,
                    level=1,
                    start_line=start_line,
                    end_line=end_line,
                    content=text,
                    subsections=[],
                )
            )

        content = "\n".join(content_lines)
        modified_date = datetime.fromtimestamp(
            validated_path.stat().st_mtime, UTC
        ).isoformat()

        return ExtractedDocument(
            path=str(original_path),
            file_type=".pdf",
            content=content,
            sections=sections,
            code_blocks=[],
            code_references=self._extract_code_references(content),
            word_count=len(content.split()),
            modified_date=modified_date,
        )

    async def _extract_async(self, file_path: Path) -> ExtractedDocument:
        """Async extraction for PDF files."""
        return await asyncio.to_thread(self._extract, file_path)


__all__ = ["PDFExtractor"]
