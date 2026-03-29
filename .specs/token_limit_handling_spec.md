# Token Limit Handling Specification

## Overview

This document describes how the Code-Graph-RAG project handles content that exceeds various token limits throughout the system. The project employs multiple strategies including **truncation**, **batch chunking**, and **result capping** depending on the context.

## Key Token Limits

### 1. Embedding Input Limit: 512 Tokens

**Configuration:** `EMBEDDING_MAX_LENGTH = 512` (default)

**Location:** `codebase_rag/config.py`

**Purpose:** Maximum token length for individual text inputs to embedding models.

#### Handling Strategy: **Truncation (Not Chunking)**

When a single text snippet exceeds the 512-token limit, the system **truncates** the input rather than splitting it into multiple chunks.

**Implementation Details:**

- **Local Provider (UniXcoder):** In `codebase_rag/unixcoder.py`, the `tokenize()` method truncates input tokens:
  ```python
  tokens = tokens[: max_length - 4]  # Reserves 4 tokens for special tokens
  ```
  The `-4` accounts for special tokens: `[CLS]`, mode token, and separator tokens.

- **External Providers:** Google GLA/Vertex and OpenAI providers accept the full text but rely on their respective APIs to handle truncation internally.

**Important:** There is **no automatic text chunking** for oversized inputs. If you need to embed content longer than 512 tokens, you must manually split it before calling the embedding functions.

---

### 2. OpenAI Batch Size Limit: 2048 Texts

**Configuration:** `MAX_BATCH_SIZE = 2048`

**Location:** `codebase_rag/embeddings/openai.py`

**Purpose:** Maximum number of **texts per batch** for OpenAI's embedding API (not a token limit).

#### Handling Strategy: **Automatic Batch Chunking**

When `embed_batch()` receives more than 2048 texts, the system automatically chunks them into smaller batches:

```python
for start in range(0, len(texts), batch_size):
    batch = texts[start : start + batch_size]
    # Process each batch separately
```

**Provider-Specific Batch Limits:**
- **OpenAI:** 2048 texts per batch
- **Google GLA:** 100 texts per batch
- **Google Vertex:** 250 texts per batch
- **Ollama:** No hard limit (varies by model)
- **Local:** Configurable via `batch_size` parameter (default: 32)

**Error Handling:** If batch size exceeds the provider limit, a `BatchSizeExceededError` is raised with a user-friendly message suggesting the recommended batch size.

---

### 3. Query Result Limit: 16000 Tokens

**Configuration:** `QUERY_RESULT_MAX_TOKENS = 16000` (default)

**Location:** `codebase_rag/config.py`

**Purpose:** Maximum total tokens for query results returned to the LLM.

#### Handling Strategy: **Token-Based Truncation**

The `truncate_results_by_tokens()` function in `codebase_rag/utils/token_utils.py` implements a two-stage filtering process:

**Stage 1: Row Cap**
```python
if total_count > settings.QUERY_RESULT_ROW_CAP:  # Default: 500 rows
    results = results[: settings.QUERY_RESULT_ROW_CAP]
```

**Stage 2: Token Truncation**
```python
for row in results:
    row_text = json.dumps(row, default=str)
    row_tokens = count_tokens(row_text)
    
    if total_tokens + row_tokens > max_tokens and kept:
        # Stop adding rows when limit is reached
        return kept, total_tokens, True
    
    kept.append(row)
    total_tokens += row_tokens
```

**Key Behaviors:**
- Uses tiktoken with `cl100k_base` encoding for accurate token counting
- Preserves row order (first rows are kept, later rows are dropped)
- Logs a warning when truncation occurs
- Returns a tuple: `(results, tokens_used, was_truncated)`
- Single large rows are always included (even if they exceed the limit alone)

**User Feedback:** When truncation occurs, the summary message informs the user:
```
Results truncated: showing {kept} of {total} items (~{tokens} tokens, limit {max_tokens}).
Refine your query for more specific results.
```

---

### 4. UniXcoder Model Context: 1024 Tokens

**Configuration:** `UNIXCODER_MAX_CONTEXT = 1024`

**Location:** `codebase_rag/constants.py`

**Purpose:** Maximum context window for the UniXcoder model's attention mechanism.

**Constraint:** The `tokenize()` method asserts that `max_length < UNIXCODER_MAX_CONTEXT`, ensuring the 512-token embedding limit stays well within the model's capacity.

---

## Token Counting Implementation

**Location:** `codebase_rag/utils/token_utils.py`

The project uses OpenAI's tiktoken library for accurate token counting:

```python
@cache
def _get_encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(cs.TIKTOKEN_ENCODING)  # "cl100k_base"

def count_tokens(text: str) -> int:
    return len(_get_encoding().encode(text))
```

This ensures consistency with OpenAI's token counting, which is critical for:
- Query result truncation
- Token limit enforcement
- User-facing token estimates

---

## Design Decisions

### Why Truncation Instead of Chunking for Embeddings?

1. **Semantic Integrity:** Code snippets (functions, methods, classes) are meant to be embedded as coherent units. Splitting them would lose contextual meaning.

2. **Vector Space Consistency:** Each embedding represents a single semantic unit. Chunking would create multiple vectors for one logical entity, complicating similarity search.

3. **Simplicity:** Truncation is deterministic and predictable. Users know exactly what to expect (first N tokens are kept).

4. **Retrieval Strategy:** The system retrieves many small code snippets rather than few large documents, making truncation acceptable.

### Why Two-Stage Query Result Filtering?

1. **Row Cap (500):** Prevents memory issues from extremely large result sets before token counting.

2. **Token Limit (16000):** Ensures the LLM context window isn't exceeded, accounting for variable row sizes.

3. **Graceful Degradation:** Returns as much data as possible within limits rather than failing completely.

---

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MAX_LENGTH` | 512 | Max tokens per embedding input |
| `QUERY_RESULT_MAX_TOKENS` | 16000 | Max tokens in query results |
| `QUERY_RESULT_ROW_CAP` | 500 | Max rows before token truncation |
| `EMBEDDING_DEFAULT_BATCH_SIZE` | 32 | Default batch size for embeddings |

### Provider-Specific Limits

These are hardcoded based on API specifications:

```python
# OpenAI
MAX_BATCH_SIZE = 2048  # texts per batch

# Google
BATCH_LIMITS = {
    "gla": {"max_texts": 100, "max_tokens_per_text": 20000},
    "vertex": {"max_texts": 250, "max_tokens_per_text": 30000},
}
```

---

## Error Messages

### Batch Size Exceeded

When a batch exceeds provider limits, users see:

```
Batch size {batch_size} exceeds maximum {max_batch_size} for {provider}.

Provider limits:
  - OpenAI: 2048 texts per batch
  - Google GLA: 100 texts per batch
  - Google Vertex: 250 texts per batch
  - Ollama: No hard limit (varies by model)

Solutions:
  1. Reduce batch size to {recommended_size}
  2. Let the provider auto-chunk: set batch_size=0
```

### Query Truncation Warning

Logged when query results are truncated:

```
Query results truncated: kept={kept}, total={total}, tokens={tokens}, max_tokens={max_tokens}
```

---

## Best Practices

### For Users

1. **Keep code snippets under 512 tokens** for optimal embedding quality.
2. **Refine queries** to return fewer, more relevant results.
3. **Use specific queries** rather than broad ones to avoid truncation.
4. **Monitor token usage** in query result summaries.

### For Developers

1. **Always use `truncate_results_by_tokens()`** when returning query results to LLMs.
2. **Respect provider batch limits** when implementing new embedding providers.
3. **Test with large inputs** to verify truncation behavior.
4. **Log truncation events** for debugging and monitoring.

---

## Future Enhancements

Potential improvements to consider:

1. **Smart Chunking:** Implement semantic-aware chunking for long documents (e.g., split at function boundaries).

2. **Hierarchical Embeddings:** Embed chunks separately and create a summary embedding.

3. **Configurable Truncation Strategy:** Allow users to choose between truncation, chunking, or error on overflow.

4. **Token Budget Allocation:** Distribute token budget across multiple query results intelligently.

5. **Streaming Results:** Return results in pages rather than truncating.

---

## Related Files

- `codebase_rag/config.py` - Configuration settings
- `codebase_rag/utils/token_utils.py` - Token counting and truncation
- `codebase_rag/embedder.py` - Embedding generation
- `codebase_rag/unixcoder.py` - UniXcoder model wrapper
- `codebase_rag/embeddings/openai.py` - OpenAI embedding provider
- `codebase_rag/embeddings/google.py` - Google embedding provider
- `codebase_rag/embeddings/base.py` - Base embedding provider class
- `codebase_rag/tools/codebase_query.py` - Query tool with result truncation
- `codebase_rag/exceptions.py` - BatchSizeExceededError

---

## Testing

Token limit handling is tested in:

- `codebase_rag/tests/test_token_utils.py` - Token counting and truncation tests
- `codebase_rag/tests/test_query_truncation.py` - Query result truncation tests
- `codebase_rag/tests/test_embedder.py` - Embedding max length tests

Key test cases:
- `test_token_count_accuracy` - Verifies token counting
- `test_results_exceed_limit` - Verifies truncation behavior
- `test_single_large_row_still_included` - Ensures large rows aren't dropped
- `test_embed_code_uses_default_max_length` - Verifies 512-token limit
