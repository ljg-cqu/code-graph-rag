from __future__ import annotations

import json
from functools import cache
from typing import TYPE_CHECKING, Literal

import tiktoken
from loguru import logger

from .. import constants as cs
from .. import logs as ls
from ..models import TruncationResult
from ..types_defs import ResultRow

if TYPE_CHECKING:
    pass


@cache
def _get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(cs.TIKTOKEN_ENCODING)


def count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))


def truncate_results_by_tokens(
    results: list[ResultRow],
    max_tokens: int,
    original_total: int | None = None,
) -> tuple[list[ResultRow], int, bool]:
    if not results:
        return results, 0, False

    kept: list[ResultRow] = []
    total_tokens = 0
    total_for_log = original_total if original_total is not None else len(results)

    for row in results:
        row_text = json.dumps(row, default=str)
        row_tokens = count_tokens(row_text)

        if total_tokens + row_tokens > max_tokens and kept:
            logger.warning(
                ls.QUERY_RESULTS_TRUNCATED.format(
                    kept=len(kept),
                    total=total_for_log,
                    tokens=total_tokens,
                    max_tokens=max_tokens,
                )
            )
            return kept, total_tokens, True

        kept.append(row)
        total_tokens += row_tokens

    return kept, total_tokens, False


def truncate_results_smart(
    results: list[ResultRow],
    max_tokens: int,
    row_cap: int | None = None,
    strategy: Literal["fifo", "relevance", "balanced"] = "balanced",
    max_row_tokens: int | None = None,
    min_rows: int = 5,
    diversity_budget_pct: float = 0.15,
) -> TruncationResult:
    """Intelligently truncate query results based on token budget.

    Args:
        results: Query results to truncate.
        max_tokens: Maximum total tokens allowed.
        row_cap: Maximum number of rows (applied before token counting).
        strategy: Truncation strategy.
            - "fifo": First-in-first-out (legacy behavior).
            - "relevance": Prioritize by relevance_score field if present.
            - "balanced": Fair distribution, cap individual row sizes.
        max_row_tokens: Maximum tokens per individual row.
        min_rows: Minimum rows to include in balanced strategy.
        diversity_budget_pct: Percentage of budget reserved for diversity.

    Returns:
        TruncationResult with metadata about what was dropped.
    """
    if not results:
        return TruncationResult(
            results=[],
            tokens_used=0,
            was_truncated=False,
            dropped_count=0,
            truncation_reason="none",
        )

    total_count = len(results)
    original_results = results.copy()  # Keep original for dropped_rows
    row_cap_applied = False
    dropped_by_row_cap: list[ResultRow] = []

    # Stage 1: Apply row cap
    if row_cap and total_count > row_cap:
        dropped_by_row_cap = results[row_cap:]
        results = results[:row_cap]
        row_cap_applied = True

    # Stage 2: Calculate token counts for all rows
    row_token_counts: list[tuple[ResultRow, int]] = []
    for row in results:
        row_text = json.dumps(row, default=str)
        row_tokens = count_tokens(row_text)
        row_token_counts.append((row, row_tokens))

    # Stage 3: Apply per-row token cap if configured
    if max_row_tokens:
        capped_rows: list[tuple[ResultRow, int]] = []
        for row, tokens in row_token_counts:
            if tokens > max_row_tokens:
                # Truncate large row's text fields
                row = truncate_row_fields(row, max_row_tokens)
                tokens = count_tokens(json.dumps(row, default=str))
            capped_rows.append((row, tokens))
        row_token_counts = capped_rows

    # Stage 4: Apply truncation strategy
    if strategy == "fifo":
        kept, tokens_used = _truncate_fifo(row_token_counts, max_tokens)
    elif strategy == "relevance":
        kept, tokens_used = _truncate_by_relevance(row_token_counts, max_tokens)
    else:  # balanced
        kept, tokens_used = _truncate_balanced(
            row_token_counts, max_tokens, min_rows, max_row_tokens or 2000, diversity_budget_pct
        )

    # Collect dropped rows from token truncation
    dropped_by_tokens = [r for r in results if r not in kept]
    all_dropped = dropped_by_row_cap + dropped_by_tokens
    dropped_count = total_count - len(kept)
    was_truncated = dropped_count > 0

    # Determine truncation reason: row_cap takes precedence if applied
    if not was_truncated:
        truncation_reason = "none"
    elif row_cap_applied and not dropped_by_tokens:
        truncation_reason = "row_cap"
    else:
        truncation_reason = "token_limit"

    return TruncationResult(
        results=kept,
        tokens_used=tokens_used,
        was_truncated=was_truncated,
        dropped_count=dropped_count,
        truncation_reason=truncation_reason,
        dropped_rows=all_dropped,
    )


def _truncate_fifo(
    row_token_counts: list[tuple[ResultRow, int]],
    max_tokens: int,
) -> tuple[list[ResultRow], int]:
    """Legacy FIFO truncation."""
    kept: list[ResultRow] = []
    total_tokens = 0

    for row, tokens in row_token_counts:
        if total_tokens + tokens > max_tokens and kept:
            break
        kept.append(row)
        total_tokens += tokens

    return kept, total_tokens


def _truncate_by_relevance(
    row_token_counts: list[tuple[ResultRow, int]],
    max_tokens: int,
) -> tuple[list[ResultRow], int]:
    """Prioritize by relevance_score field. O(n log n) complexity."""
    if not row_token_counts:
        return [], 0

    # Keep original indices for order restoration
    indexed_rows = [
        (idx, row, tokens, row.get("relevance_score", 0.5) or 0.5)
        for idx, (row, tokens) in enumerate(row_token_counts)
    ]

    # Sort by relevance descending (stable sort preserves order for equal scores)
    indexed_rows.sort(key=lambda x: x[3], reverse=True)

    kept_with_idx: list[tuple[int, ResultRow, int]] = []
    total_tokens = 0

    for idx, row, tokens, _ in indexed_rows:
        if total_tokens + tokens > max_tokens and kept_with_idx:
            break
        kept_with_idx.append((idx, row, tokens))
        total_tokens += tokens

    # Restore original order (O(n log n))
    kept_with_idx.sort(key=lambda x: x[0])
    return [row for _, row, _ in kept_with_idx], total_tokens


def _truncate_balanced(
    row_token_counts: list[tuple[ResultRow, int]],
    max_tokens: int,
    min_rows: int = 5,
    max_row_tokens: int = 2000,
    diversity_budget_pct: float = 0.15,
) -> tuple[list[ResultRow], int]:
    """Balanced truncation with fair distribution and minimum representation guarantee.

    Algorithm:
    1. Apply simple per-row cap (12% of budget or max_row_tokens)
    2. Sort by relevance, select rows until budget exhausted
    3. Guarantee min_rows by allowing full content when few rows
    4. Reserve diversity budget for lower-relevance rows
    """
    if not row_token_counts:
        return [], 0

    n = len(row_token_counts)

    # Simple max-per-row cap: 12% of budget per row (allows ~8 rows to fill 16k tokens)
    effective_cap = min(max_row_tokens, int(max_tokens * 0.12))
    effective_cap = max(effective_cap, 200)  # Absolute floor

    # Step 1: Apply capping and collect with indices
    capped_rows: list[tuple[int, ResultRow, int, float]] = []  # (original_idx, row, tokens, relevance)

    for idx, (row, tokens) in enumerate(row_token_counts):
        relevance = row.get("relevance_score", 0.5) or 0.5

        # For few rows, allow full content
        if n <= min_rows:
            row_cap = max_tokens
        else:
            # Adjust cap for relevance: 0.7x to 1.3x for 0-1 relevance
            relevance_multiplier = 0.7 + 0.6 * min(max(relevance, 0), 1)
            row_cap = int(effective_cap * relevance_multiplier)

        # Truncate row if needed
        if tokens > row_cap:
            row = truncate_row_fields(row, row_cap)
            tokens = count_tokens(json.dumps(row, default=str))

        capped_rows.append((idx, row, tokens, relevance))

    # Step 2: Sort by relevance for selection
    capped_rows.sort(key=lambda x: x[3], reverse=True)

    # Step 3: Fill with diversity budget
    main_budget = int(max_tokens * (1 - diversity_budget_pct))

    kept: list[tuple[int, ResultRow, int]] = []
    total_tokens = 0
    unselected: list[tuple[int, ResultRow, int]] = []

    # First pass: fill main budget with highest relevance
    for idx, row, tokens, _ in capped_rows:
        if total_tokens + tokens <= main_budget:
            kept.append((idx, row, tokens))
            total_tokens += tokens
        else:
            unselected.append((idx, row, tokens))

    # Second pass: fill diversity budget
    for idx, row, tokens in unselected:
        if total_tokens + tokens <= max_tokens:
            kept.append((idx, row, tokens))
            total_tokens += tokens
        else:
            break

    # Third pass: enforce min_rows guarantee by force-adding remaining rows
    # This ensures we always return at least min_rows (or all rows if fewer exist)
    remaining = unselected[len(kept) - len(row_token_counts):]  # Rows not yet processed
    remaining_unselected = [r for r in unselected if r not in kept]

    while len(kept) < min_rows and remaining_unselected and len(kept) < n:
        idx, row, tokens = remaining_unselected.pop(0)
        # Force add with aggressive truncation if needed
        if tokens > effective_cap:
            row = truncate_row_fields(row, effective_cap)
            tokens = count_tokens(json.dumps(row, default=str))
        kept.append((idx, row, tokens))
        total_tokens += tokens

    # Restore original order
    kept.sort(key=lambda x: x[0])
    return [row for _, row, _ in kept], total_tokens


def truncate_row_fields(row: ResultRow, max_tokens: int) -> ResultRow:
    """Truncate large text fields in a row to fit token budget.

    Uses statement-boundary-aware truncation for code fields.
    """
    row_copy = dict(row)
    row_text = json.dumps(row_copy, default=str)

    if count_tokens(row_text) <= max_tokens:
        return row_copy

    # Prioritize fields to truncate by size (largest first)
    fields_by_size = sorted(
        [(k, v) for k, v in row_copy.items() if isinstance(v, str)],
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for key, value in fields_by_size:
        field_tokens = count_tokens(value)
        if field_tokens <= max_tokens // 4:
            continue  # Skip small fields

        # Try statement-boundary-aware truncation for code
        if key in ("content", "code", "source", "body", "snippet"):
            truncated = _truncate_at_statement_boundary(value, max_tokens // 2)
        else:
            # For non-code fields, truncate at sentence/word boundary
            truncated = _truncate_at_word_boundary(value, max_tokens // 2)

        row_copy[key] = truncated

        row_text = json.dumps(row_copy, default=str)
        if count_tokens(row_text) <= max_tokens:
            break

    return row_copy


def _truncate_at_statement_boundary(text: str, max_chars: int) -> str:
    """Truncate code at statement boundaries (semicolons, newlines, braces)."""
    if len(text) <= max_chars:
        return text

    # Find last complete statement before max_chars
    truncated = text[:max_chars]

    # Look for statement boundaries
    boundaries = [";\n", "}\n", "\n\n", ";\r\n", "}\r\n"]
    best_pos = -1
    best_boundary = ""

    for boundary in boundaries:
        pos = truncated.rfind(boundary)
        if pos > best_pos:
            best_pos = pos
            best_boundary = boundary

    if best_pos > max_chars // 2:
        return truncated[: best_pos + len(best_boundary)] + "... [truncated]"
    else:
        # Fallback: truncate at word boundary
        return _truncate_at_word_boundary(text, max_chars)


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text at word boundary."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Find last space
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space] + "... [truncated]"
    return truncated + "... [truncated]"
