"""Tests for smart truncation functionality."""

import pytest

from codebase_rag.models import TruncationResult
from codebase_rag.utils.token_utils import (
    _truncate_at_statement_boundary,
    _truncate_at_word_boundary,
    _truncate_balanced,
    _truncate_by_relevance,
    _truncate_fifo,
    count_tokens,
    truncate_results_smart,
    truncate_row_fields,
)


class TestTruncateResultsSmart:
    """Tests for truncate_results_smart function."""

    def test_empty_results(self) -> None:
        """Empty input returns empty result."""
        result = truncate_results_smart([], max_tokens=1000)
        assert result.results == []
        assert result.tokens_used == 0
        assert result.was_truncated is False
        assert result.dropped_count == 0

    def test_fifo_strategy(self) -> None:
        """FIFO strategy should keep first rows until limit."""
        results = [
            {"id": 1, "content": "a"},
            {"id": 2, "content": "b"},
            {"id": 3, "content": "c"},
        ]
        result = truncate_results_smart(results, max_tokens=50, strategy="fifo")
        assert len(result.results) >= 1
        assert result.results[0]["id"] == 1  # First should be kept

    def test_relevance_strategy_prioritizes_high_scores(self) -> None:
        """Relevance strategy should prioritize high scores."""
        results = [
            {"id": 1, "relevance_score": 0.3, "content": "a"},
            {"id": 2, "relevance_score": 0.9, "content": "b"},
            {"id": 3, "relevance_score": 0.5, "content": "c"},
        ]
        result = truncate_results_smart(
            results, max_tokens=1000, strategy="relevance"
        )
        # All should fit, but sorted by relevance
        ids = [r["id"] for r in result.results]
        assert 2 in ids  # Highest score should be included

    def test_relevance_strategy_preserves_order(self) -> None:
        """Relevance strategy should restore original order."""
        results = [
            {"id": 1, "relevance_score": 0.3},
            {"id": 2, "relevance_score": 0.9},
            {"id": 3, "relevance_score": 0.5},
        ]
        result = truncate_results_smart(
            results, max_tokens=1000, strategy="relevance"
        )
        # Order should be restored to original
        ids = [r["id"] for r in result.results]
        assert ids == sorted(ids)

    def test_balanced_strategy_prevents_budget_hogging(self) -> None:
        """Balanced strategy should cap individual row sizes."""
        results = [
            {"id": 1, "content": "x" * 10000},  # Large row
            {"id": 2, "content": "y" * 100},  # Small row
            {"id": 3, "content": "z" * 100},  # Small row
        ]
        result = truncate_results_smart(
            results, max_tokens=5000, strategy="balanced", max_row_tokens=2000
        )
        assert len(result.results) == 3  # All rows kept

    def test_balanced_strategy_min_rows_guarantee(self) -> None:
        """Balanced strategy should guarantee minimum rows."""
        results = [{"id": i, "content": "x" * 1000} for i in range(100)]
        result = truncate_results_smart(
            results, max_tokens=5000, strategy="balanced", min_rows=10
        )
        assert len(result.results) >= 10

    def test_balanced_with_single_massive_row(self) -> None:
        """One row exceeding entire budget should be capped."""
        results = [{"content": "x" * 100000}]  # 100k chars
        result = truncate_results_smart(
            results, max_tokens=5000, strategy="balanced"
        )
        assert len(result.results) == 1
        # Row should be capped
        result_tokens = count_tokens(str(result.results[0]))
        assert result_tokens < 100000

    def test_row_cap_applied_first(self) -> None:
        """Row cap should be applied before token counting."""
        results = [{"id": i, "content": "x"} for i in range(100)]
        result = truncate_results_smart(
            results, max_tokens=100000, row_cap=10
        )
        assert len(result.results) == 10

    def test_returns_truncation_result(self) -> None:
        """Should return TruncationResult dataclass."""
        results = [{"id": 1, "content": "test"}]
        result = truncate_results_smart(results, max_tokens=1000)
        assert isinstance(result, TruncationResult)
        assert hasattr(result, "results")
        assert hasattr(result, "tokens_used")
        assert hasattr(result, "was_truncated")
        assert hasattr(result, "dropped_count")


class TestTruncateFifo:
    """Tests for _truncate_fifo function."""

    def test_keeps_first_rows(self) -> None:
        """Should keep first rows until limit."""
        row_token_counts = [
            ({"id": 1}, 10),
            ({"id": 2}, 10),
            ({"id": 3}, 10),
        ]
        kept, tokens = _truncate_fifo(row_token_counts, max_tokens=25)
        assert len(kept) == 2
        assert kept[0]["id"] == 1
        assert kept[1]["id"] == 2

    def test_empty_input(self) -> None:
        """Empty input should return empty."""
        kept, tokens = _truncate_fifo([], max_tokens=100)
        assert kept == []
        assert tokens == 0


class TestTruncateByRelevance:
    """Tests for _truncate_by_relevance function."""

    def test_prioritizes_high_relevance(self) -> None:
        """Should prioritize high relevance scores."""
        row_token_counts = [
            ({"id": 1, "relevance_score": 0.1}, 10),
            ({"id": 2, "relevance_score": 0.9}, 10),
            ({"id": 3, "relevance_score": 0.5}, 10),
        ]
        kept, tokens = _truncate_by_relevance(row_token_counts, max_tokens=25)
        ids = [r["id"] for r in kept]
        assert 2 in ids  # Highest relevance should be kept

    def test_handles_missing_relevance_score(self) -> None:
        """Should handle missing relevance_score with default."""
        row_token_counts = [
            ({"id": 1}, 10),  # No relevance_score
            ({"id": 2}, 10),
        ]
        kept, tokens = _truncate_by_relevance(row_token_counts, max_tokens=100)
        assert len(kept) == 2

    def test_restores_original_order(self) -> None:
        """Should restore original order after sorting by relevance."""
        row_token_counts = [
            ({"id": 1, "relevance_score": 0.1}, 10),
            ({"id": 2, "relevance_score": 0.9}, 10),
            ({"id": 3, "relevance_score": 0.5}, 10),
        ]
        kept, tokens = _truncate_by_relevance(row_token_counts, max_tokens=100)
        ids = [r["id"] for r in kept]
        assert ids == [1, 2, 3]  # Original order


class TestTruncateBalanced:
    """Tests for _truncate_balanced function."""

    def test_empty_input(self) -> None:
        """Empty input should return empty."""
        kept, tokens = _truncate_balanced([], max_tokens=100)
        assert kept == []
        assert tokens == 0

    def test_few_rows_allows_full_content(self) -> None:
        """Few rows should allow full content."""
        row_token_counts = [
            ({"id": 1, "content": "x" * 1000}, 500),
            ({"id": 2, "content": "y" * 1000}, 500),
        ]
        kept, tokens = _truncate_balanced(
            row_token_counts, max_tokens=10000, min_rows=5
        )
        assert len(kept) == 2

    def test_respects_min_rows(self) -> None:
        """Should respect min_rows parameter."""
        row_token_counts = [
            ({"id": i, "content": "x" * 100}, 50) for i in range(100)
        ]
        kept, tokens = _truncate_balanced(
            row_token_counts, max_tokens=500, min_rows=10
        )
        assert len(kept) >= 10


class TestTruncateRowFields:
    """Tests for truncate_row_fields function."""

    def test_small_row_unchanged(self) -> None:
        """Small row should not be modified."""
        row = {"id": 1, "content": "small"}
        result = truncate_row_fields(row, max_tokens=100)
        assert result == row

    def test_large_code_field_truncated(self) -> None:
        """Large code field should be truncated at statement boundary."""
        row = {"id": 1, "content": "x = 1;\ny = 2;\nz = 3;"}
        result = truncate_row_fields(row, max_tokens=10)
        assert "..." in result["content"] or len(result["content"]) < len(row["content"])

    def test_prioritizes_largest_fields(self) -> None:
        """Should prioritize truncating largest fields."""
        row = {
            "id": 1,
            "small": "abc",
            "large": "x" * 10000,
        }
        result = truncate_row_fields(row, max_tokens=100)
        # Large field should be truncated
        assert len(result["large"]) < len(row["large"])


class TestTruncateAtStatementBoundary:
    """Tests for _truncate_at_statement_boundary function."""

    def test_short_text_unchanged(self) -> None:
        """Short text should not be modified."""
        text = "x = 1"
        result = _truncate_at_statement_boundary(text, max_chars=100)
        assert result == text

    def test_truncates_at_semicolon(self) -> None:
        """Should truncate at semicolon boundary."""
        text = "x = 1;\ny = 2;\nz = 3;"
        result = _truncate_at_statement_boundary(text, max_chars=15)
        assert "..." in result
        # Should have kept at least one statement
        assert ";" in result or "x" in result

    def test_fallback_to_word_boundary(self) -> None:
        """Should fallback to word boundary if no statement boundary found."""
        text = "no boundaries here just words"
        result = _truncate_at_statement_boundary(text, max_chars=15)
        assert "..." in result


class TestTruncateAtWordBoundary:
    """Tests for _truncate_at_word_boundary function."""

    def test_short_text_unchanged(self) -> None:
        """Short text should not be modified."""
        text = "short text"
        result = _truncate_at_word_boundary(text, max_chars=100)
        assert result == text

    def test_truncates_at_space(self) -> None:
        """Should truncate at word boundary."""
        text = "this is a long text with many words"
        result = _truncate_at_word_boundary(text, max_chars=20)
        assert "..." in result
        # Should not cut in middle of word
        words = result.replace("... [truncated]", "").split()
        assert all(w in text.split() for w in words)

    def test_no_space_fallback(self) -> None:
        """Should still truncate even if no space found."""
        text = "verylongwordwithoutspaces"
        result = _truncate_at_word_boundary(text, max_chars=10)
        assert "..." in result
        assert len(result) < len(text) + 20  # Allow for truncation marker


class TestCountTokens:
    """Tests for count_tokens function."""

    def test_empty_string(self) -> None:
        """Empty string should have 0 tokens."""
        assert count_tokens("") == 0

    def test_simple_text(self) -> None:
        """Simple text should have expected token count."""
        # "hello world" should be 2-3 tokens depending on tokenizer
        count = count_tokens("hello world")
        assert count >= 2
        assert count <= 4

    def test_code_snippet(self) -> None:
        """Code should have reasonable token count."""
        code = "def foo():\n    return 1"
        count = count_tokens(code)
        assert count >= 5  # At least a few tokens
        assert count <= 20  # Should not be too many