# Context Compression Feature Fix — Design Specification

## 1. Problem Statement

The context compression feature is **non-functional**. Despite reporting "✅ Context compressed successfully!", the metrics show:

| Metric | Value | Expected |
|--------|-------|----------|
| Original tokens | 3,549 | — |
| Compressed tokens | 3,549 | **< 3,549** |
| Reduction | 0.0% | **> 0%** |
| Strategy used | `no-op` | A real strategy (S1–S5) |
| Execution time | 0ms | **> 0ms** |

The `no-op` strategy indicates the compressor aborts before evaluating any compression strategy. The root cause is a **threshold mismatch**: the compressor checks if `compressible_content < 3` messages, but typical chat contexts have fewer than 3 compressible messages after system prompts are preserved. Additionally, the `max_context` parameter is never used to enforce a token budget — strategies only check message counts, not token counts.

---

## 2. Root Cause Analysis

### 2.1 Early `no-op` Exit (Primary)

**Location:** `codebase_rag/context_compressor.py:compress_sync()` lines 477–491

```python
preserved_content, compressible_content = self._preserve_matching_content(self.context)

if len(compressible_content) < 3:
    return CompressionResult(
        ...,
        strategy_used="no-op",
        ...
    )
```

**Issue:** A typical context has:
- 1 system message (preserved)
- 1–2 user/assistant messages (compressible)

This falls below the `< 3` threshold, causing immediate `no-op` return **without any token budget check**.

### 2.2 `max_context` Parameter Is Ignored

**Location:** `ContextCompressor.__init__()` and all strategy methods

The `max_context` parameter is stored but **never referenced** in:
- `compress_sync()` — no check against token budget
- Any strategy (`_hierarchical_summarization`, `_stale_context_pruning`, etc.) — strategies only look at message counts, not token counts

**Expected behavior:** Compression should trigger when `original_tokens > max_context`, regardless of message count.

### 2.3 Token Counting Inflates Preserved Content

**Location:** `_count_context_tokens()`

```python
def _count_context_tokens(self, context):
    return count_tokens(json.dumps(context))
```

The `json.dumps()` wrapper adds structural characters (`[`, `]`, `{`, `}`, quotes, commas) that inflate the token count. The `max_context` comparison (when fixed) would trigger too early.

### 2.4 Parallel Worker Pool Mismatch

**Location:** `compress_sync()` lines 493–508

The compressor uses `ParallelWorkerPool` (designed for Memgraph Cypher queries) to run strategy evaluations. The `_evaluate_strategy` method signature expects `(worker, params)` but the worker parameter is unused. This is harmless but architecturally wrong — compression strategies are CPU-bound, not I/O-bound, and don't need Memgraph connections.

### 2.5 Missing Tests

**Finding:** No test file exists for `ContextCompressor`. Searching `codebase_rag/tests/` found zero matches for compression-related tests.

---

## 3. Design Goals

| Goal | Priority | Description |
|------|----------|-------------|
| G1 | P0 | Compression must trigger based on **token budget**, not message count |
| G2 | P0 | `max_context` must be enforced as the hard token limit |
| G3 | P0 | Strategies must produce measurable token reduction |
| G4 | P1 | Fix semantic retention scoring to be meaningful |
| G5 | P1 | Add comprehensive unit tests |
| G6 | P2 | Replace Memgraph worker pool with simple ThreadPoolExecutor for CPU-bound work |

---

## 4. Proposed Fixes

### 4.1 Fix Early Exit Logic (G1)

**Change:** Replace message-count threshold with token-budget threshold.

**In `compress_sync()`, replace:**
```python
if len(compressible_content) < 3:
    return CompressionResult(..., strategy_used="no-op", ...)
```

**With:**
```python
# Always compress if we're over budget, even with few messages
if original_tokens <= self.max_context and len(compressible_content) < 3:
    return CompressionResult(..., strategy_used="no-op", ...)
```

This preserves the fast-path for small contexts that are already under budget, but forces compression when over budget.

### 4.2 Enforce `max_context` in Strategy Selection (G2)

**Change:** After selecting the best strategy, verify the result fits within `max_context`. If not, apply fallback truncation.

**Add to `compress_sync()` after strategy selection:**
```python
# Ensure compressed result fits within max_context budget
if final_tokens > self.max_context:
    logger.warning(f"Best strategy exceeded max_context ({final_tokens} > {self.max_context}), applying hard truncation")
    final_compressed = self._hard_truncate_to_budget(preserved_content + best_result.compressed_context, self.max_context)
    final_tokens = self._count_context_tokens(final_compressed)
```

**New method `_hard_truncate_to_budget()`:**
```python
def _hard_truncate_to_budget(
    self, context: list[dict[str, Any]], budget: int
) -> list[dict[str, Any]]:
    """Truncate context to fit within token budget, preserving system messages and recent context."""
    system_msgs = [m for m in context if m.get("role") == "system"]
    non_system = [m for m in context if m.get("role") != "system"]
    
    # Keep system messages
    result = list(system_msgs)
    result_tokens = self._count_context_tokens(result)
    
    # Add recent non-system messages from the end until budget exhausted
    for msg in reversed(non_system):
        msg_tokens = self._count_context_tokens([msg])
        if result_tokens + msg_tokens <= budget:
            result.insert(len(system_msgs), msg)  # Insert after system messages
            result_tokens += msg_tokens
        else:
            break
    
    return result
```

### 4.3 Fix Token Counting (G2)

**Change:** Count tokens from message content directly, not JSON wrapper.

**Replace `_count_context_tokens()`:**
```python
def _count_context_tokens(self, context: list[dict[str, Any]]) -> int:
    """Count tokens in message content, excluding JSON structural overhead."""
    total = 0
    for msg in context:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        else:
            total += count_tokens(json.dumps(content))
    # Add overhead per message (role tokens, formatting)
    total += len(context) * 4  # Approximate overhead: "role", "content", punctuation
    return total
```

### 4.4 Fix Semantic Retention Scoring (G4)

**Current issue:** `_calculate_semantic_retention()` only checks regex-extracted entities (function names, class names, error keywords). It ignores:
- Actual semantic meaning of content
- Whether the compressed context preserves the *intent* of user queries
- Tool results and structured data

**Proposed improvement:** Use a hybrid approach:
1. Keep entity-based scoring for code contexts
2. Add keyword overlap scoring between original and compressed for user queries
3. Add length-normalized content similarity

```python
def _calculate_semantic_retention(
    self, original: list[dict[str, Any]], compressed: list[dict[str, Any]]
) -> float:
    # Entity-based score (existing)
    entity_score = self._entity_retention_score(original, compressed)
    
    # Keyword overlap score for user content
    keyword_score = self._keyword_overlap_score(original, compressed)
    
    # Weighted combination
    return 0.6 * entity_score + 0.4 * keyword_score
```

### 4.5 Replace Worker Pool for Compression (G6)

**Change:** Use `concurrent.futures.ThreadPoolExecutor` directly instead of `ParallelWorkerPool`.

**Rationale:** `ParallelWorkerPool` is designed for Memgraph I/O with connection management, retry logic, and rate limiting. Compression strategies are pure CPU/Python operations — they don't need database connections.

**In `_get_worker_pool()`:**
```python
def _get_worker_pool(self):
    if self._worker_pool is None:
        # Use simple ThreadPoolExecutor for CPU-bound compression
        self._worker_pool = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix="compression-worker"
        )
    return self._worker_pool
```

**Note:** This requires importing `ThreadPoolExecutor` and adjusting task submission. Alternatively, keep using `ParallelWorkerPool` but document that the worker parameter is unused — this is lower risk.

### 4.6 Add Comprehensive Tests (G5)

**New file:** `codebase_rag/tests/test_context_compressor.py`

**Test cases:**
1. `test_compression_triggers_when_over_budget` — verify compression runs when tokens > max_context
2. `test_no_op_when_under_budget_and_few_messages` — verify fast path
3. `test_strategies_produce_different_results` — verify strategies are actually different
4. `test_semantic_retention_score_range` — verify score is 0.0–1.0
5. `test_rollback_on_low_retention` — verify rollback works
6. `test_hard_truncate_respects_budget` — verify _hard_truncate_to_budget
7. `test_preserve_pattern_works` — verify regex preservation
8. `test_archive_store_and_retrieve` — verify ContextArchive TTL

---

## 5. Implementation Plan

| Step | File | Change | Risk |
|------|------|--------|------|
| 1 | `context_compressor.py` | Fix early exit logic (4.1) | Low |
| 2 | `context_compressor.py` | Fix token counting (4.3) | Medium — affects all token metrics |
| 3 | `context_compressor.py` | Add max_context enforcement (4.2) | Medium — new code path |
| 4 | `context_compressor.py` | Improve semantic retention (4.4) | Low — scoring only |
| 5 | `context_compressor.py` | Document worker pool usage (4.5) | Low — no functional change |
| 6 | `tests/test_context_compressor.py` | Add tests (4.6) | Low — new file |
| 7 | `main.py` | Verify integration points | Low — no changes needed |

---

## 5.1 Implementation Notes

### Token Counting Performance

The proposed `_count_context_tokens()` change (4.3) iterates over messages instead of a single `json.dumps()` call. This is called multiple times per compression cycle:

- Once in `__init__` for `original_tokens`
- Once per strategy evaluation in `_evaluate_strategy`
- Once for `final_tokens` after strategy selection
- Potentially multiple times in `_hard_truncate_to_budget`

**Recommendation:** The per-message iteration overhead is negligible for typical context sizes (< 50 messages). For very large contexts (100+ messages), consider caching the result when the context hasn't changed. The `count_tokens()` call using tiktoken dominates runtime regardless.

### Variable Scope Clarification

In `compress_sync()`, `original_tokens` is a **local variable** computed at line 463, not `self.original_tokens` (which is set in `__init__`). The fix in 4.1 correctly uses the local `original_tokens` to compare against `self.max_context`.

---

## 6. Verification Criteria

After fixes, running `/compress` should produce:

| Metric | Expected |
|--------|----------|
| Original tokens | > 0 |
| Compressed tokens | < Original tokens (when over budget) |
| Reduction | > 0% (when over budget) |
| Semantic retention | ≥ min_retention_score (default 70%) |
| Strategy used | One of S1–S5, not `no-op` |
| Execution time | > 0ms |

---

## 7. Files to Modify

1. `codebase_rag/context_compressor.py` — Core fixes
2. `codebase_rag/tests/test_context_compressor.py` — New test file (create)

---

## 8. Configuration Alignment

The following settings in `codebase_rag/config.py` are relevant and correctly defined:

| Setting | Value | Usage Status |
|---------|-------|-------------|
| `CONTEXT_COMPRESSION_AUTO_TRIGGER_PCT` | 85.0 | ✅ Used in `main.py` |
| `CONTEXT_COMPRESSION_MIN_RETENTION_SCORE` | 70.0 | ✅ Used in `ContextCompressor` |
| `CONTEXT_COMPRESSION_PARALLEL_WORKERS` | 10 | ✅ Used in `ContextCompressor` |
| `CONTEXT_COMPRESSION_AGGRESSIVE_RETENTION_THRESHOLD` | 50.0 | ✅ Used in `ContextCompressor` |
| `CONTEXT_COMPRESSION_ENABLED` | True | ✅ Used in `main.py` |

No config changes needed.

---

## 9. Backward Compatibility

- The `CompressionResult` dataclass fields remain unchanged
- The `ContextCompressor` public API (`compress_sync()`) signature remains unchanged
- The `ContextArchive` API remains unchanged
- Existing `/compress` command in `main.py` requires no changes

---

## 10. Edge Cases

| Case | Handling |
|------|----------|
| Empty context | Returns `no-op` with 0 tokens |
| All messages preserved (system only) | Returns `no-op` if under budget |
| Single large message over budget | Hard truncate to budget |
| All strategies fail | Fallback to hybrid + hard truncate |
| Retention score below threshold | Rollback to original |
| `max_context` < system message tokens | Keep system messages, truncate everything else |

---

*Specification version: 1.1*
*Last updated: 2026-04-18*
*Target: codebase_rag v0.x*
*Author: AI Assistant*
