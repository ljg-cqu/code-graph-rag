"""Tests for BaseDocumentExtractor path validation and skip behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from codebase_rag.document.error_handling import ErrorType, ExtractionException
from codebase_rag.document.extractors.base import BaseDocumentExtractor


class MockExtractor(BaseDocumentExtractor):
    """Concrete extractor for testing base behavior."""

    def supported_extensions(self) -> list[str]:
        return [".txt"]

    def _extract(self, file_path: Path) -> MagicMock:
        return MagicMock()

    async def _extract_async(self, file_path: Path) -> MagicMock:
        return MagicMock()


class TestSkipValidation:
    """Tests for the skip_validation parameter on extract/extract_async."""

    def test_extract_with_skip_validation_bypasses_existence_check(
        self, tmp_path: Path
    ):
        """extract(skip_validation=True) should not check if file exists."""
        extractor = MockExtractor()
        nonexistent = tmp_path / "does_not_exist.txt"

        result = extractor.extract(nonexistent, skip_validation=True)

        assert result is not None

    def test_extract_without_skip_validation_checks_existence(self, tmp_path: Path):
        """extract(skip_validation=False) should raise for missing files."""
        extractor = MockExtractor()
        nonexistent = tmp_path / "does_not_exist.txt"

        with pytest.raises(ExtractionException) as exc_info:
            extractor.extract(nonexistent, skip_validation=False)

        assert exc_info.value.error_type == ErrorType.FILE_NOT_FOUND

    def test_extract_async_with_skip_validation_bypasses_existence_check(
        self, tmp_path: Path
    ):
        """extract_async(skip_validation=True) should not check if file exists."""
        extractor = MockExtractor()
        nonexistent = tmp_path / "does_not_exist.txt"

        import asyncio

        result = asyncio.run(extractor.extract_async(nonexistent, skip_validation=True))

        assert result is not None

    def test_extract_async_without_skip_validation_checks_existence(
        self, tmp_path: Path
    ):
        """extract_async(skip_validation=False) should raise for missing files."""
        extractor = MockExtractor()
        nonexistent = tmp_path / "does_not_exist.txt"

        import asyncio

        with pytest.raises(ExtractionException) as exc_info:
            asyncio.run(extractor.extract_async(nonexistent, skip_validation=False))

        assert exc_info.value.error_type == ErrorType.FILE_NOT_FOUND

    def test_extract_defaults_to_validation_enabled(self, tmp_path: Path):
        """extract() default behavior should validate file existence."""
        extractor = MockExtractor()
        nonexistent = tmp_path / "does_not_exist.txt"

        with pytest.raises(ExtractionException) as exc_info:
            extractor.extract(nonexistent)

        assert exc_info.value.error_type == ErrorType.FILE_NOT_FOUND

    def test_extract_async_defaults_to_validation_enabled(self, tmp_path: Path):
        """extract_async() default behavior should validate file existence."""
        extractor = MockExtractor()
        nonexistent = tmp_path / "does_not_exist.txt"

        import asyncio

        with pytest.raises(ExtractionException) as exc_info:
            asyncio.run(extractor.extract_async(nonexistent))

        assert exc_info.value.error_type == ErrorType.FILE_NOT_FOUND

    def test_extract_with_existing_file_succeeds_with_validation(self, tmp_path: Path):
        """extract() should succeed for existing files with default validation."""
        extractor = MockExtractor()
        existing = tmp_path / "exists.txt"
        existing.write_text("hello")

        result = extractor.extract(existing)

        assert result is not None

    def test_extract_async_with_existing_file_succeeds_with_validation(
        self, tmp_path: Path
    ):
        """extract_async() should succeed for existing files with default validation."""
        extractor = MockExtractor()
        existing = tmp_path / "exists.txt"
        existing.write_text("hello")

        import asyncio

        result = asyncio.run(extractor.extract_async(existing))

        assert result is not None
