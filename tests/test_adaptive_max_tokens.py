"""Tests for adaptive max_tokens calculation and content density estimation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from codebase_rag.document.concept_extraction import (
    DENSITY_CODE,
    DENSITY_LIST,
    DENSITY_PLAIN,
    DENSITY_TABLE,
    _estimate_content_density,
    calculate_adaptive_max_tokens,
)
from codebase_rag.document.concept_runner import (
    AdaptiveRetryResult,
    ConceptExtractionRunner,
)
from codebase_rag.document.error_handling import ErrorType, ExtractionError


class TestEstimateContentDensity:
    """Content density classification based on structural markers."""

    def test_empty_content_returns_plain(self) -> None:
        result = _estimate_content_density("")
        assert result.category == DENSITY_PLAIN
        assert result.multiplier == 1.0

    def test_plain_text(self) -> None:
        text = "Hello world. This is plain text with no special markers."
        result = _estimate_content_density(text)
        assert result.category == DENSITY_PLAIN
        assert result.multiplier == 1.0

    def test_table_dense_content(self) -> None:
        table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"
        result = _estimate_content_density(table)
        assert result.category == DENSITY_TABLE
        assert result.multiplier == 2.0

    def test_list_dense_content(self) -> None:
        lst = "- item 1\n- item 2\n- item 3\n- item 4\nplain line"
        result = _estimate_content_density(lst)
        assert result.category == DENSITY_LIST
        assert result.multiplier == 1.5

    def test_code_fence_content(self) -> None:
        code = "```python\nx = 1\n```\nSome text after."
        result = _estimate_content_density(code)
        assert result.category == DENSITY_CODE
        assert result.multiplier == 1.2

    def test_mixed_content_dominant_category_wins(self) -> None:
        mixed = "| A | B |\n|---|---|\n| 1 | 2 |\n- list item\n- list item"
        result = _estimate_content_density(mixed)
        assert result.category == DENSITY_TABLE


class TestCalculateAdaptiveMaxTokens:
    """Adaptive max_tokens calculation based on chunk characteristics."""

    def test_plain_text_1000_chars(self) -> None:
        text = "word " * 200
        tokens = calculate_adaptive_max_tokens(text, min_tokens=256)
        assert tokens == 500

    def test_table_dense_doubles_tokens(self) -> None:
        table = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        table_tokens = calculate_adaptive_max_tokens(table, min_tokens=10)
        assert table_tokens == 38

    def test_respects_min_tokens(self) -> None:
        short = "hi"
        tokens = calculate_adaptive_max_tokens(short, min_tokens=2048)
        assert tokens == 2048

    def test_respects_max_tokens(self) -> None:
        long_text = "x" * 100000
        tokens = calculate_adaptive_max_tokens(long_text, max_tokens=4096)
        assert tokens == 4096

    def test_custom_tokens_per_char(self) -> None:
        text = "word " * 200
        tokens = calculate_adaptive_max_tokens(text, tokens_per_char=1.0, min_tokens=256)
        assert tokens == 1000

    def test_custom_base_tokens(self) -> None:
        text = "word " * 200
        tokens = calculate_adaptive_max_tokens(text, base_tokens=8192, min_tokens=256)
        assert tokens == 500


class TestAdaptiveRetryResult:
    """AdaptiveRetryResult dataclass behavior."""

    def test_defaults(self) -> None:
        result = AdaptiveRetryResult(success=False)
        assert result.concepts == 0
        assert result.relationships == 0
        assert result.reason is None
        assert result.tokens_used is None
        assert result.result is None

    def test_with_result(self) -> None:
        from codebase_rag.document.concept_extraction import ExtractionResult

        result = AdaptiveRetryResult(
            success=True,
            concepts=3,
            relationships=2,
            tokens_used=8192,
            result=ExtractionResult(),
        )
        assert result.success is True
        assert result.concepts == 3
        assert result.relationships == 2
        assert result.tokens_used == 8192
        assert result.result is not None


class TestRetryWithAdaptiveTokens:
    """ConceptExtractionRunner._retry_with_adaptive_tokens behavior."""

    def test_missing_content_returns_failure(self, tmp_path) -> None:
        runner = ConceptExtractionRunner(repo_path=tmp_path)
        error = ExtractionError(
            path="test",
            error_type=ErrorType.CONCEPT_CONTEXT_OVERFLOW,
            message="overflow",
            chunk_content="",
        )
        result = pytest.importorskip("asyncio").run(
            runner._retry_with_adaptive_tokens(error)
        )
        assert result.success is False
        assert result.reason == "No chunk content in DLQ entry"

    def test_uses_dlq_multiplier(self, tmp_path) -> None:
        runner = ConceptExtractionRunner(repo_path=tmp_path)
        runner.extractor = MagicMock()
        runner.extractor.extract = AsyncMock(return_value=MagicMock(
            concepts=[MagicMock()],
            relationships=[],
        ))

        error = ExtractionError(
            path="test",
            error_type=ErrorType.CONCEPT_CONTEXT_OVERFLOW,
            message="overflow",
            chunk_content="word " * 1000,
        )

        import asyncio
        result = asyncio.run(runner._retry_with_adaptive_tokens(error))

        assert result.success is True
        assert result.tokens_used is not None
        assert result.tokens_used > 0
        runner.extractor.extract.assert_awaited_once()
        call_kwargs = runner.extractor.extract.call_args.kwargs
        assert "max_tokens" in call_kwargs
