"""PDF document extractor.

Optional extractor that requires PyPDF2 or pdfplumber.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

from .base import (
    BaseDocumentExtractor,
    ExtractedDocument,
    ExtractedSection,
)
from ..error_handling import ErrorType, ExtractionException


class PDFExtractor(BaseDocumentExtractor):
    """
    Extractor for PDF files (.pdf).

    Requires PyPDF2 or pdfplumber for text extraction.
    Falls back gracefully if not installed.
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract content from PDF file."""
        # Validate path
        validated_path = self._validate_path(file_path)

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
            import pdfplumber  # type: ignore
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore
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

        content_parts: list[str] = []
        sections: list[ExtractedSection] = []

        with pdfplumber.open(validated_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    content_parts.append(text)

                    # Create a section per page
                    sections.append(
                        ExtractedSection(
                            title=f"Page {i + 1}",
                            level=1,
                            start_line=i,
                            end_line=i,
                            content=text,
                            subsections=[],
                        )
                    )

        content = "\n\n".join(content_parts)
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

    def _extract_with_pypdf2(
        self, validated_path: Path, original_path: Path
    ) -> ExtractedDocument:
        """Extract using PyPDF2 (fallback)."""
        from PyPDF2 import PdfReader

        reader = PdfReader(validated_path)
        content_parts: list[str] = []
        sections: list[ExtractedSection] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                content_parts.append(text)

                sections.append(
                    ExtractedSection(
                        title=f"Page {i + 1}",
                        level=1,
                        start_line=i,
                        end_line=i,
                        content=text,
                        subsections=[],
                    )
                )

        content = "\n\n".join(content_parts)
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

    async def extract_async(self, file_path: Path) -> ExtractedDocument:
        """Async extraction for PDF files."""
        return await asyncio.to_thread(self.extract, file_path)


__all__ = ["PDFExtractor"]