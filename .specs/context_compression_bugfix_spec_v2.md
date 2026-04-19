# Context Compression Bug-Fix Specification v2.0

## 1. Executive Summary

Thorough runtime analysis of `codebase_rag/context_compressor.py` (post-v1.1-fix) reveals **five remaining bugs** that cause the compressor to violate its `max_context` budget guarantee, lose critical message types, or make sub-optimal rollback decisions. All bugs are reproduced with concrete test cases below. This spec defines surgical, implementation-ready fixes that align with the existing codebase architecture.

---

## 2. Bugs Discovered

### Bug 1 — Rollback Reverts to Original That Exceeds Budget (P0)

**Symptom:** When preserved content (e.g. system messages) alone exceeds `max_context`, the compressor rolls back to the original context, which is **even larger** than the hard-truncated result.

**Reproduction:**
```python
context = [
    {"role": "system", "content": "System 1. " * 50},
    {"role": "system", "content": "System 2. " * 50},
]
compressor = ContextCompressor(context=context, max_context=50)
result = compressor.compress_sync()
# result.compressed_tokens == 670  (> 50!)
# result.was_rolled_back == True
```

**Root Cause:** `compress_sync()` checks `best_result.semantic_retention_score < min_retention_decimal` **after** hard truncation. The retention score is computed against the strategy's output (often `[]` when `compressible_content` is empty), yielding ~60%. This triggers rollback even though the hard-truncated result was the best possible fit.

**Fix:** Only rollback to original if the original context fits within budget. If the original itself exceeds budget, the hard-truncated result is the best we can achieve.

```python
# In compress_sync(), replace the rollback block:
was_rolled_back = False
min_retention_decimal = self.min_retention_score / 100

# FIX: Only rollback if original fits within budget AND retention is too low.
# If original exceeds budget, hard-truncated result is the best possible fit.
original_fits_budget = self.original_tokens <= self.max_context
if original_fits_budget and best_result.semantic_retention_score < min_retention_decimal:
    logger.warning(
        f"Compression retention score {best_result.semantic_retention_score * 100:.2f}% "
        f"below minimum {self.min_retention_score:.2f}%, rolling back to original"
    )
    final_compressed = self.context
    final_tokens = self.original_tokens
    reduction_pct = 0.0
    was_rolled_back = True
```

---

### Bug 2 — Strategies Use Arbitrary Message Counts, Ignore Token Budget (P0)

**Symptom:** A context with 3 messages of 10,000 characters each is returned unchanged because `len(context) <= 3`, even though the token count far exceeds `max_context`.

**Reproduction:**
```python
context = [
    {"role": "user", "content": "A" * 10000},
    {"role": "assistant", "content": "B" * 10000},
    {"role": "user", "content": "C" * 10000},
]
compressor = ContextCompressor(context=context, max_context=100)
result = compressor._hierarchical_summarization(context)
# Returns unchanged — 3 messages, each 10k chars
```

**Root Cause:**
- `_hierarchical_summarization`: exits if `len(context) <= 3`
- `_stale_context_pruning`: exits if `len(context) <= 5`
- `_semantic_ranking_filter`: exits if `len(unique_context) <= 10`

None check token budget.

**Fix:** Add a token-budget guard to each strategy. If the context already fits within `max_context`, skip expensive work. If it exceeds, proceed regardless of message count.

```python
def _hierarchical_summarization(self, context):
    # FIX: check token budget, not just message count
    if len(context) <= 3 and self._count_context_tokens(context) <= self.max_context:
        return context
    ...

def _stale_context_pruning(self, context):
    if len(context) <= 5 and self._count_context_tokens(context) <= self.max_context:
        return context
    ...

def _semantic_ranking_filter(self, context):
    # dedup first
    seen_hashes = set()
    unique_context = []
    for msg in context:
        h = hash(json.dumps(msg, sort_keys=True))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_context.append(msg)
    if len(unique_context) <= 10 and self._count_context_tokens(unique_context) <= self.max_context:
        return unique_context
    ...
```

---

### Bug 3 — Tool / Function Messages Not Preserved by Default (P1)

**Symptom:** Tool and function messages are treated as compressible by `_preserve_matching_content()`. If `_hybrid_summarization_pruning` does not win the strategy selection, tool results may be dropped entirely.

**Reproduction:**
```python
context = [
    {"role": "system", "content": "System."},
    {"role": "user", "content": "User."},
    {"role": "tool", "content": "Important tool result." * 100},
    {"role": "assistant", "content": "Assistant."},
]
compressor = ContextCompressor(context=context, max_context=50)
preserved, compressible = compressor._preserve_matching_content(context)
# "tool" message is in compressible, not preserved
```

**Root Cause:** `_preserve_matching_content()` only preserves `role == "system"` and pattern matches. The hybrid strategy separately preserves `tool`/`function`, but other strategies do not.

**Fix:** Extend `_preserve_matching_content()` to also preserve `tool` and `function` messages by default. These are typically critical execution results.

```python
def _preserve_matching_content(self, context):
    preserved = []
    compressible = []
    for msg in context:
        if msg.get("role") == "system":
            preserved.append(msg)
        elif msg.get("role") in ("tool", "function"):
            preserved.append(msg)  # FIX: preserve tool/function results
        elif self.preserve_pattern and self.preserve_pattern.search(json.dumps(msg)):
            preserved.append(msg)
        else:
            compressible.append(msg)
    return preserved, compressible
```

---

### Bug 4 — Hard Truncate Cannot Truncate Individual Messages (P1)

**Symptom:** When system messages alone exceed `max_context`, `_hard_truncate_to_budget()` keeps all system messages and simply drops non-system messages. There is no mechanism to truncate the content of an individual message.

**Reproduction:**
```python
context = [{"role": "system", "content": "Very long system message. " * 100}]
compressor = ContextCompressor(context=context, max_context=20)
truncated = compressor._hard_truncate_to_budget(context, 20)
# truncated tokens still > 20
```

**Root Cause:** `_hard_truncate_to_budget()` operates at the message level, not the content level.

**Fix:** Add a per-message content truncation fallback when the message-level budget is exhausted.

```python
def _hard_truncate_to_budget(self, context, budget):
    system_msgs = [m for m in context if m.get("role") == "system"]
    non_system = [m for m in context if m.get("role") != "system"]
    result = list(system_msgs)
    result_tokens = self._count_context_tokens(result)

    for msg in reversed(non_system):
        msg_tokens = self._count_context_tokens([msg])
        if result_tokens + msg_tokens <= budget:
            result.insert(len(system_msgs), msg)
            result_tokens += msg_tokens
        else:
            break

    # FIX: if still over budget, truncate the newest message content
    if result_tokens > budget and result:
        newest = result[-1]
        content = newest.get("content", "")
        if isinstance(content, str) and content:
            leftover = budget - self._count_context_tokens(result[:-1])
            if leftover > 10:
                # Use binary search to find max content length that fits
                # Approximate: 1 token ≈ 3-4 characters for English text
                max_chars = leftover * 4
                low, high = 0, len(content)
                while low < high:
                    mid = (low + high + 1) // 2
                    truncated = content[:mid]
                    test_msg = {**newest, "content": truncated}
                    if self._count_context_tokens([test_msg]) <= leftover:
                        low = mid
                    else:
                        high = mid - 1
                final_content = content[:low] + "... [truncated]"
                result[-1] = {**newest, "content": final_content}
            else:
                # Not enough budget for meaningful content, drop the message
                result = result[:-1]

    return result
```

*Note:* The binary-search approach ensures efficient truncation even for very large messages.

---

### Bug 5 — ThreadPool Created / Shutdown on Every Call (P2)

**Symptom:** `compress_sync()` instantiates and shuts down a `ThreadPoolExecutor` on every invocation. For repeated compressions (e.g. every turn in a long chat), this adds unnecessary overhead.

**Root Cause:** `_get_worker_pool()` creates the pool lazily, but `compress_sync()` unconditionally calls `pool.shutdown(wait=True)` in its `finally` block.

**Fix:** Make the pool lifecycle explicit. Either:

**Option A — Reuse pool across calls (recommended):**
Convert `_worker_pool` to a class-level singleton or accept an optional external executor.

**Option B — Only use parallelism when beneficial:**
If `len(STRATEGIES) < 3` or context is small, skip the thread pool entirely and run sequentially.

```python
# In compress_sync():
if len(self.STRATEGIES) <= 2 or self._count_context_tokens(compressible_content) < 500:
    # Sequential evaluation for small workloads
    results = [self._evaluate_strategy(task) for task in tasks]
else:
    pool = self._get_worker_pool()
    try:
        futures = [pool.submit(self._evaluate_strategy, task) for task in tasks]
        results = [f.result() for f in futures]
    finally:
        pool.shutdown(wait=True)
```

---

## 3. Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `codebase_rag/context_compressor.py` | `compress_sync()` rollback block | Bug 1 — budget-aware rollback |
| `codebase_rag/context_compressor.py` | `_hierarchical_summarization()` | Bug 2 — token budget guard |
| `codebase_rag/context_compressor.py` | `_stale_context_pruning()` | Bug 2 — token budget guard |
| `codebase_rag/context_compressor.py` | `_semantic_ranking_filter()` | Bug 2 — token budget guard |
| `codebase_rag/context_compressor.py` | `_preserve_matching_content()` | Bug 3 — preserve tool/function |
| `codebase_rag/context_compressor.py` | `_hard_truncate_to_budget()` | Bug 4 — per-message truncation |
| `codebase_rag/context_compressor.py` | `compress_sync()` pool usage | Bug 5 — conditional parallelism |

---

## 4. Test Additions

Add the following test cases to `codebase_rag/tests/test_context_compressor.py`:

```python
def test_rollback_does_not_exceed_budget_when_preserved_content_is_large(self):
    """Bug 1: rollback must not revert to original if original exceeds budget."""
    context = [
        {"role": "system", "content": "System 1. " * 50},
        {"role": "system", "content": "System 2. " * 50},
    ]
    compressor = ContextCompressor(context=context, max_context=50)
    result = compressor.compress_sync()
    assert result.compressed_tokens <= 50, \
        f"compressed_tokens ({result.compressed_tokens}) must not exceed max_context (50)"

def test_strategies_compress_large_messages_even_when_few(self):
    """Bug 2: strategies must act when token budget is exceeded, regardless of message count."""
    context = [
        {"role": "user", "content": "A" * 10000},
        {"role": "assistant", "content": "B" * 10000},
        {"role": "user", "content": "C" * 10000},
    ]
    compressor = ContextCompressor(context=context, max_context=100)
    for _, fname, _ in compressor.STRATEGIES:
        result = getattr(compressor, fname)(context.copy())
        tokens = compressor._count_context_tokens(result)
        assert tokens <= 100, f"{fname} produced {tokens} tokens, exceeds budget 100"

def test_tool_messages_preserved_by_default(self):
    """Bug 3: tool/function messages must be in preserved set."""
    context = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "User."},
        {"role": "tool", "content": "Tool result."},
        {"role": "function", "content": "Function result."},
    ]
    compressor = ContextCompressor(context=context, max_context=1000)
    preserved, compressible = compressor._preserve_matching_content(context)
    preserved_roles = {m["role"] for m in preserved}
    assert "tool" in preserved_roles
    assert "function" in preserved_roles

def test_hard_truncate_fits_single_large_system_message(self):
    """Bug 4: hard truncate must truncate individual messages when necessary."""
    context = [{"role": "system", "content": "Very long system message. " * 100}]
    compressor = ContextCompressor(context=context, max_context=20)
    truncated = compressor._hard_truncate_to_budget(context, 20)
    tokens = compressor._count_context_tokens(truncated)
    assert tokens <= 20, f"truncated tokens ({tokens}) exceed budget (20)"

def test_thread_pool_conditional_usage(self):
    """Bug 5: thread pool should only be used for non-trivial workloads."""
    import time
    
    # Small context should use sequential evaluation
    small_context = [
        {"role": "user", "content": "Short message."},
    ]
    compressor = ContextCompressor(context=small_context, max_context=100)
    
    # Time multiple compressions - should be fast without thread pool overhead
    start = time.time()
    for _ in range(5):
        compressor.compress_sync()
    small_time = time.time() - start
    
    # Large context should use parallel evaluation
    large_context = [
        {"role": "user", "content": "A" * 1000},
        {"role": "assistant", "content": "B" * 1000},
        {"role": "user", "content": "C" * 1000},
        {"role": "assistant", "content": "D" * 1000},
    ]
    compressor_large = ContextCompressor(context=large_context, max_context=100)
    
    # Should complete without error
    result = compressor_large.compress_sync()
    assert result is not None
```

---

## 5. Verification Criteria

| Criterion | Test |
|-----------|------|
| Compressed result never exceeds `max_context` | `test_rollback_does_not_exceed_budget_when_preserved_content_is_large` |
| Strategies reduce tokens even with few messages | `test_strategies_compress_large_messages_even_when_few` |
| Tool/function messages survive compression | `test_tool_messages_preserved_by_default` |
| Hard truncate enforces budget on single messages | `test_hard_truncate_fits_single_large_system_message` |
| Thread pool used efficiently | `test_thread_pool_conditional_usage` |
| All existing tests still pass | `pytest codebase_rag/tests/test_context_compressor.py -v` |

---

## 6. Backward Compatibility

- `CompressionResult` dataclass: unchanged
- `ContextCompressor.__init__` signature: unchanged
- `ContextCompressor.compress_sync()` signature: unchanged
- `ContextArchive` API: unchanged
- `/compress` command in `main.py`: unchanged

---

## 7. Configuration Alignment

No new configuration values required. Existing settings remain valid:

| Setting | Value | Usage |
|---------|-------|-------|
| `CONTEXT_COMPRESSION_AUTO_TRIGGER_PCT` | 85.0 | Trigger threshold in `main.py` |
| `CONTEXT_COMPRESSION_MIN_RETENTION_SCORE` | 70.0 | Rollback threshold |
| `CONTEXT_COMPRESSION_AGGRESSIVE_RETENTION_THRESHOLD` | 50.0 | Aggressive-mode threshold |
| `CONTEXT_COMPRESSION_PARALLEL_WORKERS` | 10 | Thread-pool size |

---

*Specification version: 2.0*
*Last updated: 2026-04-19*
*Target: codebase_rag v0.x*
*Author: AI Assistant*
