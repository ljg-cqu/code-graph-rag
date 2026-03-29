# Token Limit Handling Specification

## Overview

This document describes the token limit handling system in Code-Graph-RAG, implementing semantic preservation, intelligent truncation, and user control through:

1. **Semantic-Aware Chunking** - Split code at class/function/block boundaries instead of truncating
2. **Smart Truncation** - Relevance-aware token budget allocation for query results
3. **Token-Aware Batching** - Split embedding batches by token count, not text count
4. **Configuration** - User control with best-practice defaults

## Current Issues Addressed

### Critical Issues (Now Resolved)

1. **Semantic Loss in Embedding Truncation**
   - Before: Truncated code at 512 tokens by cutting from the END
   - Now: Splits at semantic boundaries (functions, classes, methods)
   - Result: Preserved semantic integrity

2. **Naive Query Result Truncation**
   - Before: First-come-first-served row retention
   - Now: Balanced strategy with relevance prioritization
   - Result: Important results prioritized, fair token distribution

3. **No Semantic-Aware Chunking**
   - Before: No chunking, only truncation
   - Now: Tree-sitter AST parsing for natural code boundaries
   - Result: Preserved semantic units, better embedding quality

4. **Count-Based Batch Chunking**
   - Before: Split batches by text count (2048), not token size
   - Now: Token-aware batching respecting provider limits
   - Result: Efficient API usage, no limit violations

5. **No User Control**
   - Before: Fixed strategies, no configuration
   - Now: Configurable strategies with visibility
   - Result: User control and debugging capability

---

## 1. Semantic-Aware Chunking for Embeddings

### Strategy: Hierarchical Chunking with Natural Boundaries

Instead of truncating code snippets, implement intelligent chunking that respects code structure:

**Chunking Rules (Priority Order):**
- **Priority 1:** Split at class/module boundaries (preserve class context first)
- **Priority 2:** Split at function/method boundaries
- **Priority 3:** Split at logical blocks (if/else, loops, try/catch)
- **Priority 4:** Split at statement boundaries
- **Priority 5:** Truncate at token limit (last resort)

> **Rationale:** Class context must be preserved before inner function boundaries. A method's semantics depend on its class context.

### Implementation

```python
# codebase_rag/utils/code_chunker.py

from dataclasses import dataclass, field
from typing import Literal
from loguru import logger
from tree_sitter import Language, Parser, Node

from codebase_rag.utils.token_utils import count_tokens
from codebase_rag.constants import UNIXCODER_MAX_CONTEXT

@dataclass
class CodeChunk:
    content: str
    start_line: int
    end_line: int
    chunk_type: Literal["class", "function", "method", "block", "statement", "truncated"]
    parent_scope: str | None = None
    parent_fqn: str | None = None  # Links to parent node in graph
    chunk_index: int = 0  # Position within parent's chunks
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

# Tree-sitter language module mapping (uses existing CGR infrastructure)
LANGUAGE_MODULES = {
    "python": "tree_sitter_python",           # ✅ PyPI: tree-sitter-python
    "javascript": "tree_sitter_javascript",   # ✅ PyPI: tree-sitter-javascript
    "typescript": "tree_sitter_typescript",   # ✅ PyPI: tree-sitter-typescript
    "java": "tree_sitter_java",               # ✅ PyPI: tree-sitter-java
    "rust": "tree_sitter_rust",               # ✅ PyPI: tree-sitter-rust
    "cpp": "tree_sitter_cpp",                 # ✅ PyPI: tree-sitter-cpp
    "c": "tree_sitter_c",                     # ✅ PyPI: tree-sitter-c
    "go": "tree_sitter_go",                   # ✅ PyPI: tree-sitter-go
    "php": "tree_sitter_php",                 # ✅ PyPI: tree-sitter-php
    "lua": "tree_sitter_lua",                 # ⚠️ Requires: pip install tree-sitter-lua
    "solidity": "tree_sitter_solidity",       # ⚠️ Requires: pip install tree-sitter-solidity
}

# Language-specific boundary node types (for splitting)
BOUNDARY_NODE_TYPES = {
    "python": {
        "class": ["class_definition"],
        "function": ["function_definition"],
        "method": ["function_definition"],  # Functions inside classes
        "block": ["if_statement", "for_statement", "while_statement", "try_statement", "with_statement"],
    },
    "java": {
        "class": ["class_declaration", "interface_declaration", "enum_declaration", "record_declaration"],
        "function": ["method_declaration", "constructor_declaration"],
        "method": ["method_declaration"],
        "block": ["if_statement", "for_statement", "while_statement", "try_statement", "switch_statement"],
    },
    "javascript": {
        "class": ["class_declaration"],
        "function": ["function_declaration", "function_expression", "arrow_function"],
        "method": ["method_definition"],
        "block": ["if_statement", "for_statement", "while_statement", "try_statement", "switch_statement"],
    },
    "typescript": {
        "class": ["class_declaration", "interface_declaration", "type_alias_declaration"],
        "function": ["function_declaration", "function_expression", "arrow_function"],
        "method": ["method_definition"],
        "block": ["if_statement", "for_statement", "while_statement", "try_statement", "switch_statement"],
    },
    "rust": {
        "class": ["struct_item", "enum_item", "trait_item"],
        "function": ["function_item", "closure_expression"],
        "method": ["function_item"],  # Inside impl blocks
        "block": ["if_expression", "for_expression", "while_expression", "match_expression"],
    },
    "cpp": {
        "class": ["class_specifier", "struct_specifier", "enum_specifier"],
        "function": ["function_definition", "declaration"],
        "method": ["function_definition"],
        "block": ["if_statement", "for_statement", "while_statement", "try_statement"],
    },
    "c": {
        "class": ["struct_specifier", "enum_specifier"],
        "function": ["function_definition"],
        "method": [],  # C has no methods
        "block": ["if_statement", "for_statement", "while_statement"],
    },
    "go": {
        "class": ["type_declaration"],
        "function": ["function_declaration", "method_declaration"],
        "method": ["method_declaration"],
        "block": ["if_statement", "for_statement", "switch_statement"],
    },
    "php": {
        "class": ["class_declaration", "interface_declaration", "trait_declaration"],
        "function": ["function_definition", "method_declaration"],
        "method": ["method_declaration"],
        "block": ["if_statement", "for_statement", "while_statement", "try_statement"],
    },
    "lua": {
        "class": [],  # Lua has no native classes
        "function": ["function_declaration", "function_definition"],
        "method": [],  # Methods are just functions with : syntax
        "block": ["if_statement", "for_statement", "while_statement", "repeat_statement"],
    },
    "solidity": {
        "class": ["contract_declaration", "interface_declaration", "library_declaration"],
        "function": ["function_definition", "modifier_definition", "constructor_definition"],
        "method": ["function_definition"],
        "block": ["if_statement", "for_statement", "while_statement", "try_statement"],
    },
}

class SemanticCodeChunker:
    """Split code at semantic boundaries while respecting token limits.

    Uses Tree-sitter AST parsing for accurate boundary detection.
    Falls back to regex/line-based splitting when AST parsing fails.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        language: str = "python",
        overlap_tokens: int = 50,
        max_chunks_per_node: int = 10,
    ):
        self.max_tokens = min(max_tokens, UNIXCODER_MAX_CONTEXT - 4)  # Reserve for special tokens
        self.language = language
        self.overlap_tokens = overlap_tokens
        self.max_chunks_per_node = max_chunks_per_node
        self._parser = self._init_parser()

    def _init_parser(self) -> Parser | None:
        """Initialize Tree-sitter parser for the language.

        Handles missing grammars gracefully with fallback to line-based chunking.
        """
        try:
            module_name = LANGUAGE_MODULES.get(self.language)
            if not module_name:
                logger.warning(f"No Tree-sitter grammar mapping for language: {self.language}")
                return None

            import importlib
            lang_module = importlib.import_module(module_name)

            # Verify grammar has required attributes
            if not hasattr(lang_module, 'language'):
                logger.warning(f"Tree-sitter grammar for {self.language} missing 'language' function")
                return None

            lang = Language(lang_module.language())
            parser = Parser(lang)
            logger.debug(f"Tree-sitter parser initialized for {self.language}")
            return parser
        except ImportError as e:
            logger.warning(
                f"Tree-sitter grammar not installed for {self.language}: {e}. "
                f"Install with: pip install tree-sitter-{self.language}"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize Tree-sitter for {self.language}: {e}")
            return None

    def _is_binary(self, content: str) -> bool:
        """Detect binary content by checking for null bytes and high ratio of non-printable chars."""
        if '\x00' in content:
            return True
        # Check for high ratio of non-printable characters
        non_printable = sum(1 for c in content[:1000] if ord(c) < 32 and c not in '\n\r\t')
        if len(content[:1000]) > 0 and non_printable / len(content[:1000]) > 0.3:
            return True
        return False

    def chunk(self, code: str) -> list[CodeChunk]:
        """Split code into semantic chunks under max_tokens.

        Returns empty list for empty input or binary content.
        Falls back to line-based splitting on AST parse failure.
        """
        # Edge case: empty input
        if not code or not code.strip():
            return []

        # Edge case: binary content detection
        if self._is_binary(code):
            logger.warning("Binary content detected, skipping chunking")
            return []

        token_count = count_tokens(code)

        # Fast path: already under limit
        if token_count <= self.max_tokens:
            return [CodeChunk(
                content=code,
                start_line=1,
                end_line=code.count('\n') + 1,
                chunk_type="function",  # Assume single unit
                token_count=token_count,
            )]

        # Try AST-based chunking
        if self._parser:
            try:
                return self._chunk_with_ast(code)
            except Exception as e:
                logger.warning(f"AST chunking failed for {self.language}: {e}, falling back to line-based")

        # Fallback: line-based chunking
        return self._chunk_at_line_boundaries(code)

    def _chunk_with_ast(self, code: str) -> list[CodeChunk]:
        """Chunk using Tree-sitter AST parsing."""
        tree = self._parser.parse(bytes(code, "utf-8"))
        root = tree.root_node

        # Check for parse errors
        if root.has_error:
            logger.debug(f"Parse errors in code, attempting partial chunking")

        boundary_types = BOUNDARY_NODE_TYPES.get(self.language, {})
        if not boundary_types:
            return self._chunk_at_line_boundaries(code)

        chunks: list[CodeChunk] = []

        # Priority 1: Find class-level boundaries
        class_nodes = self._find_nodes_by_types(root, boundary_types.get("class", []))
        for node in class_nodes:
            chunks.extend(self._process_node(node, code, "class"))

        # Priority 2: Find function/method boundaries (not already in classes)
        if not chunks or sum(c.token_count for c in chunks) < count_tokens(code) * 0.8:
            func_nodes = self._find_nodes_by_types(root, boundary_types.get("function", []))
            for node in func_nodes:
                if self._is_nested_in(node, [n for n, _ in class_nodes]):
                    continue
                chunks.extend(self._process_node(node, code, "function"))

        # If still no chunks, use the whole file as one chunk
        if not chunks:
            return self._chunk_at_line_boundaries(code)

        # Handle overlap between chunks
        if self.overlap_tokens > 0:
            chunks = self._add_overlap(chunks, code)

        return chunks[:self.max_chunks_per_node]

    def _process_node(self, node: Node, code: str, chunk_type: str) -> list[CodeChunk]:
        """Process a single AST node, potentially splitting if too large."""
        node_text = code[node.start_byte:node.end_byte]
        node_tokens = count_tokens(node_text)

        if node_tokens <= self.max_tokens:
            return [CodeChunk(
                content=node_text,
                start_line=node.start_point.row + 1,
                end_line=node.end_point.row + 1,
                chunk_type=chunk_type,
                token_count=node_tokens,
            )]

        # Node is too large - try splitting at nested boundaries
        boundary_types = BOUNDARY_NODE_TYPES.get(self.language, {})
        nested_types = (
            boundary_types.get("method", []) +
            boundary_types.get("function", []) +
            boundary_types.get("block", [])
        )

        nested_nodes = self._find_nodes_by_types(node, nested_types)

        if nested_nodes:
            # Split at nested boundaries
            chunks: list[CodeChunk] = []
            for nested in nested_nodes:
                chunks.extend(self._process_node(nested[0], code, "method"))
            return chunks

        # No nested boundaries - split at line boundaries
        return self._chunk_text_at_lines(node_text, node.start_point.row + 1, chunk_type)

    def _chunk_at_line_boundaries(self, code: str) -> list[CodeChunk]:
        """Fallback: chunk at line boundaries when AST parsing fails."""
        lines = code.split('\n')
        chunks: list[CodeChunk] = []
        current_chunk_lines: list[str] = []
        current_tokens = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_tokens = count_tokens(line + '\n')

            if line_tokens > self.max_tokens:
                # Flush current chunk
                if current_chunk_lines:
                    content = '\n'.join(current_chunk_lines)
                    chunks.append(CodeChunk(
                        content=content,
                        start_line=start_line,
                        end_line=i - 1,
                        chunk_type="truncated",
                        token_count=current_tokens,
                    ))
                    current_chunk_lines = []
                    current_tokens = 0

                # Split the long line
                chunks.extend(self._split_long_line(line, i))
                start_line = i + 1
                continue

            if current_tokens + line_tokens > self.max_tokens and current_chunk_lines:
                content = '\n'.join(current_chunk_lines)
                chunks.append(CodeChunk(
                    content=content,
                    start_line=start_line,
                    end_line=i - 1,
                    chunk_type="block",
                    token_count=current_tokens,
                ))
                current_chunk_lines = [line]
                current_tokens = line_tokens
                start_line = i
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens

        # Final chunk
        if current_chunk_lines:
            content = '\n'.join(current_chunk_lines)
            chunks.append(CodeChunk(
                content=content,
                start_line=start_line,
                end_line=len(lines),
                chunk_type="block",
                token_count=current_tokens,
            ))

        return chunks[:self.max_chunks_per_node]

    def _split_long_line(self, line: str, line_num: int) -> list[CodeChunk]:
        """Split a very long single line (e.g., minified code, long string)."""
        import re
        parts = re.split(r'(?<=[,;])\s*', line)

        chunks: list[CodeChunk] = []
        current_parts: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = count_tokens(part)
            if current_tokens + part_tokens > self.max_tokens and current_parts:
                chunks.append(CodeChunk(
                    content=' '.join(current_parts),
                    start_line=line_num,
                    end_line=line_num,
                    chunk_type="truncated",
                    token_count=current_tokens,
                ))
                current_parts = [part]
                current_tokens = part_tokens
            else:
                current_parts.append(part)
                current_tokens += part_tokens

        if current_parts:
            chunks.append(CodeChunk(
                content=' '.join(current_parts),
                start_line=line_num,
                end_line=line_num,
                chunk_type="truncated",
                token_count=current_tokens,
            ))

        return chunks

    def _chunk_text_at_lines(self, text: str, start_line: int, chunk_type: str) -> list[CodeChunk]:
        """Chunk text at line boundaries with token awareness."""
        lines = text.split('\n')
        chunks: list[CodeChunk] = []
        current_lines: list[str] = []
        current_tokens = 0
        chunk_start = start_line

        for i, line in enumerate(lines):
            line_tokens = count_tokens(line + '\n')

            if current_tokens + line_tokens > self.max_tokens and current_lines:
                chunks.append(CodeChunk(
                    content='\n'.join(current_lines),
                    start_line=chunk_start,
                    end_line=chunk_start + len(current_lines) - 1,
                    chunk_type=chunk_type,
                    token_count=current_tokens,
                ))
                current_lines = [line]
                current_tokens = line_tokens
                chunk_start = start_line + i
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            chunks.append(CodeChunk(
                content='\n'.join(current_lines),
                start_line=chunk_start,
                end_line=chunk_start + len(current_lines) - 1,
                chunk_type=chunk_type,
                token_count=current_tokens,
            ))

        return chunks[:self.max_chunks_per_node]

    def _find_nodes_by_types(self, root: Node, node_types: list[str]) -> list[tuple[Node, str]]:
        """Find all nodes of specified types using DFS."""
        results = []
        def dfs(node: Node):
            if node.type in node_types:
                results.append((node, node.type))
            for child in node.children:
                dfs(child)
        dfs(root)
        return results

    def _is_nested_in(self, node: Node, parents: list[Node]) -> bool:
        """Check if node is nested inside any of the parent nodes."""
        for parent in parents:
            if (node.start_byte >= parent.start_byte and
                node.end_byte <= parent.end_byte):
                return True
        return False

    def _add_overlap(self, chunks: list[CodeChunk], code: str) -> list[CodeChunk]:
        """Add overlapping context between chunks with token validation."""
        if len(chunks) <= 1:
            return chunks

        overlapped: list[CodeChunk] = []
        lines = code.split('\n')

        for i, chunk in enumerate(chunks):
            chunk_tokens = chunk.token_count
            overlap_budget = self.max_tokens - chunk_tokens
            if overlap_budget <= 0:
                overlapped.append(chunk)
                continue

            overlap_lines = []
            if i > 0 and self.overlap_tokens > 0:
                prev_chunk = chunks[i - 1]
                overlap_start = prev_chunk.end_line
                overlap_tokens_used = 0
                for line_idx in range(overlap_start - 1, max(overlap_start - 20, chunk.start_line - 1)):
                    if overlap_tokens_used >= overlap_budget:
                        break
                    line = lines[line_idx]
                    line_tokens = count_tokens(line + '\n')
                    if overlap_tokens_used + line_tokens <= overlap_budget:
                        overlap_lines.insert(0, line)
                        overlap_tokens_used += line_tokens

            content_lines = lines[chunk.start_line - 1:chunk.end_line]
            final_content = '\n'.join(overlap_lines + content_lines)
            final_tokens = count_tokens(final_content)

            if final_tokens > self.max_tokens:
                while overlap_lines and final_tokens > self.max_tokens:
                    overlap_lines.pop(0)
                    final_content = '\n'.join(overlap_lines + content_lines)
                    final_tokens = count_tokens(final_content)

            overlapped.append(CodeChunk(
                content=final_content,
                start_line=chunk.start_line - len(overlap_lines),
                end_line=chunk.end_line,
                chunk_type=chunk.chunk_type,
                parent_fqn=chunk.parent_fqn,
                chunk_index=i,
                token_count=final_tokens,
            ))

        return overlapped
```

### Embedding Strategy

```python
# codebase_rag/embedder.py

def generate_code_summary(code: str, max_tokens: int = 256) -> str:
    """Generate a static summary of code for hierarchical embedding strategy.

    Extracts: function/class names, signatures, docstrings, and key structure.
    No LLM calls - uses AST parsing for efficiency.
    """
    import re
    from codebase_rag.utils.token_utils import count_tokens

    lines = code.split('\n')
    summary_parts = []

    # Extract module/class docstrings
    docstring_pattern = r'"""(.+?)"""|\'\'\'(.+?)\'\'\''
    for i, line in enumerate(lines[:10]):
        match = re.search(docstring_pattern, line)
        if match:
            doc = match.group(1) or match.group(2)
            summary_parts.append(f"Module: {doc[:100]}")
            break

    # Extract function/method signatures
    func_pattern = r'^(async\s+)?def\s+(\w+)\s*\(([^)]*)\)(\s*->\s*[^:]+)?'
    for line in lines:
        match = re.search(func_pattern, line)
        if match:
            is_async, name, params, return_type = match.groups()
            async_prefix = "async " if is_async else ""
            ret = return_type.strip() if return_type else ""
            summary_parts.append(f"{async_prefix}def {name}({params}){ret}")

    # Extract class definitions
    class_pattern = r'^class\s+(\w+)(\s*\([^)]*\))?'
    for line in lines:
        match = re.search(class_pattern, line)
        if match:
            name, bases = match.groups()
            bases_str = bases if bases else ""
            summary_parts.append(f"class {name}{bases_str}")

    summary = '\n'.join(summary_parts)
    if count_tokens(summary) > max_tokens:
        words = summary.split()
        truncated = []
        token_count = 0
        for word in words:
            if token_count + 1 > max_tokens:
                break
            truncated.append(word)
            token_count += 1
        summary = ' '.join(truncated) + '...'

    return summary

def embed_code_smart(code: str, strategy: str = "chunk") -> list[EmbeddingResult]:
    """Generate embeddings with semantic-aware chunking."""
    token_count = count_tokens(code)

    if token_count <= settings.EMBEDDING_MAX_LENGTH:
        return [EmbeddingResult(embedding=embed_code(code), chunk=None)]

    match strategy:
        case "truncate":
            return [EmbeddingResult(embedding=embed_code(code), truncated=True)]

        case "chunk":
            chunker = SemanticCodeChunker(max_tokens=settings.EMBEDDING_MAX_LENGTH)
            chunks = chunker.chunk(code)
            return [
                EmbeddingResult(embedding=embed_code(chunk.content), chunk=chunk)
                for chunk in chunks
            ]

        case "hierarchical":
            chunker = SemanticCodeChunker(max_tokens=settings.EMBEDDING_MAX_LENGTH)
            chunks = chunker.chunk(code)
            chunk_embeddings = [embed_code(chunk.content) for chunk in chunks]
            summary = generate_code_summary(code)
            summary_embedding = embed_code(summary)
            return [
                EmbeddingResult(
                    embedding=summary_embedding,
                    is_summary=True,
                    child_embeddings=chunk_embeddings
                )
            ]

        case "error":
            raise EmbeddingLengthExceededError(
                f"Code exceeds {settings.EMBEDDING_MAX_LENGTH} tokens ({token_count} tokens). "
                f"Use strategy='chunk' to auto-split."
            )
```

---

## 2. Smart Query Result Truncation

### Strategy: Relevance-Aware Token Budget Allocation

Implement intelligent truncation that prioritizes important results and distributes token budget fairly:

```python
# codebase_rag/utils/token_utils.py

@dataclass
class TruncationResult:
    results: list[ResultRow]
    tokens_used: int
    was_truncated: bool
    dropped_count: int
    truncation_reason: Literal["token_limit", "row_cap", "none"]
    dropped_rows: list[ResultRow]

def truncate_results_smart(
    results: list[ResultRow],
    max_tokens: int,
    row_cap: int | None = None,
    strategy: Literal["fifo", "relevance", "balanced"] = "balanced",
    max_row_tokens: int | None = None,
    min_rows: int = 5,
    diversity_budget_pct: float = 0.15,
) -> TruncationResult:
    """Intelligently truncate query results based on token budget."""
    if not results:
        return TruncationResult([], 0, False, 0, "none", [])

    total_count = len(results)
    row_cap_applied = False
    dropped_by_row_cap: list[ResultRow] = []

    # Stage 1: Apply row cap
    if row_cap and total_count > row_cap:
        dropped_by_row_cap = results[row_cap:]
        results = results[:row_cap]
        row_cap_applied = True

    # Stage 2: Calculate token counts
    row_token_counts: list[tuple[ResultRow, int]] = []
    for row in results:
        row_text = json.dumps(row, default=str)
        row_tokens = count_tokens(row_text)
        row_token_counts.append((row, row_tokens))

    # Stage 3: Apply per-row token cap
    if max_row_tokens:
        capped_rows: list[tuple[ResultRow, int]] = []
        for row, tokens in row_token_counts:
            if tokens > max_row_tokens:
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

    dropped_by_tokens = [r for r in results if r not in kept]
    all_dropped = dropped_by_row_cap + dropped_by_tokens
    dropped_count = total_count - len(kept)
    was_truncated = dropped_count > 0

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

    indexed_rows = [
        (idx, row, tokens, row.get("relevance_score", 0.5) or 0.5)
        for idx, (row, tokens) in enumerate(row_token_counts)
    ]

    indexed_rows.sort(key=lambda x: x[3], reverse=True)

    kept_with_idx: list[tuple[int, ResultRow, int]] = []
    total_tokens = 0

    for idx, row, tokens, _ in indexed_rows:
        if total_tokens + tokens > max_tokens and kept_with_idx:
            break
        kept_with_idx.append((idx, row, tokens))
        total_tokens += tokens

    kept_with_idx.sort(key=lambda x: x[0])
    return [row for _, row, _ in kept_with_idx], total_tokens

def _truncate_balanced(
    row_token_counts: list[tuple[ResultRow, int]],
    max_tokens: int,
    min_rows: int = 5,
    max_row_tokens: int = 2000,
    diversity_budget_pct: float = 0.15,
) -> tuple[list[ResultRow], int]:
    """Balanced truncation with fair distribution and minimum representation.

    Algorithm:
    1. Apply simple per-row cap (12% of budget or max_row_tokens)
    2. Sort by relevance, select rows until budget exhausted
    3. Guarantee min_rows by allowing full content when few rows
    4. Reserve diversity budget for lower-relevance rows
    """
    if not row_token_counts:
        return [], 0

    n = len(row_token_counts)
    effective_cap = min(max_row_tokens, int(max_tokens * 0.12))
    effective_cap = max(effective_cap, 200)

    capped_rows: list[tuple[int, ResultRow, int, float]] = []

    for idx, (row, tokens) in enumerate(row_token_counts):
        relevance = row.get("relevance_score", 0.5) or 0.5

        if n <= min_rows:
            row_cap = max_tokens
        else:
            relevance_multiplier = 0.7 + 0.6 * min(max(relevance, 0), 1)
            row_cap = int(effective_cap * relevance_multiplier)

        if tokens > row_cap:
            row = truncate_row_fields(row, row_cap)
            tokens = count_tokens(json.dumps(row, default=str))

        capped_rows.append((idx, row, tokens, relevance))

    capped_rows.sort(key=lambda x: x[3], reverse=True)

    main_budget = int(max_tokens * (1 - diversity_budget_pct))

    kept: list[tuple[int, ResultRow, int]] = []
    total_tokens = 0
    unselected: list[tuple[int, ResultRow, int]] = []

    for idx, row, tokens, _ in capped_rows:
        if total_tokens + tokens <= main_budget:
            kept.append((idx, row, tokens))
            total_tokens += tokens
        else:
            unselected.append((idx, row, tokens))

    for idx, row, tokens in unselected:
        if total_tokens + tokens <= max_tokens:
            kept.append((idx, row, tokens))
            total_tokens += tokens
        else:
            break

    # Guarantee min_rows
    remaining_unselected = [r for r in unselected if r not in kept]
    while len(kept) < min_rows and remaining_unselected and len(kept) < n:
        idx, row, tokens = remaining_unselected.pop(0)
        if tokens > effective_cap:
            row = truncate_row_fields(row, effective_cap)
            tokens = count_tokens(json.dumps(row, default=str))
        kept.append((idx, row, tokens))
        total_tokens += tokens

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

    fields_by_size = sorted(
        [(k, v) for k, v in row_copy.items() if isinstance(v, str)],
        key=lambda x: len(x[1]),
        reverse=True,
    )

    for key, value in fields_by_size:
        field_tokens = count_tokens(value)
        if field_tokens <= max_tokens // 4:
            continue

        if key in ("content", "code", "source", "body", "snippet"):
            truncated = _truncate_at_statement_boundary(value, max_tokens // 2)
        else:
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

    truncated = text[:max_chars]
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
        return _truncate_at_word_boundary(text, max_chars)

def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Truncate text at word boundary."""
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        return truncated[:last_space] + "... [truncated]"
    return truncated + "... [truncated]"
```

---

## 3. Token-Aware Batch Chunking

### Strategy: Dynamic Batch Sizing by Token Count

Instead of splitting by text count, split by estimated total tokens:

```python
# codebase_rag/embeddings/base.py

class EmbeddingProvider(ABC):
    def embed_batch_with_token_limit(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_batch_tokens: int | None = None,
    ) -> list[list[float]]:
        """Generate embeddings with token-aware batching."""
        if not texts:
            return []

        if max_batch_tokens is None:
            max_batch_tokens = self._default_max_batch_tokens()

        batches = self._create_token_aware_batches(
            texts,
            max_texts=batch_size,
            max_tokens=max_batch_tokens,
        )

        all_embeddings: list[list[float]] = []
        for batch in batches:
            batch_embeddings = self.embed_batch(batch, batch_size=len(batch))
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def _create_token_aware_batches(
        self,
        texts: list[str],
        max_texts: int,
        max_tokens: int,
    ) -> list[list[str]]:
        """Create batches that respect both text count and token limits."""
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            text_tokens = count_tokens(text)

            would_exceed_count = len(current_batch) >= max_texts
            would_exceed_tokens = current_tokens + text_tokens > max_tokens

            if would_exceed_count or would_exceed_tokens:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    def _default_max_batch_tokens(self) -> int:
        """Default max tokens per batch based on provider."""
        provider = self.provider_name.value if hasattr(self.provider_name, "value") else str(self.provider_name)

        match provider:
            case "openai":
                return 500_000  # OpenAI allows ~500k tokens per batch
            case "google":
                provider_type = self.get_config("provider_type", "gla")
                if provider_type == "vertex":
                    return 7_500_000  # Vertex AI: 250 texts * 30k tokens each
                return 2_000_000  # GLA: 100 texts * 20k tokens each
            case "ollama":
                return 100_000
            case "local":
                return 100_000
            case _:
                return 100_000
```

---

## 4. Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_CHUNKING_STRATEGY` | `"chunk"` | How to handle oversized embeddings: `truncate`, `chunk`, `hierarchical`, `error` |
| `EMBEDDING_CHUNK_OVERLAP_TOKENS` | `32` | Overlapping tokens between chunks |
| `EMBEDDING_MAX_CHUNKS_PER_NODE` | `5` | Maximum chunks per code unit |
| `EMBEDDING_SKIP_BINARY_FILES` | `true` | Skip binary content during chunking |
| `QUERY_RESULT_TRUNCATION_STRATEGY` | `"balanced"` | Truncation strategy: `fifo`, `relevance`, `balanced` |
| `QUERY_RESULT_MAX_ROW_TOKENS` | `2000` | Max tokens per individual row |
| `QUERY_RESULT_MIN_ROWS` | `5` | Minimum rows to include |
| `QUERY_RESULT_DIVERSITY_BUDGET_PCT` | `0.15` | Diversity budget percentage |
| `LOG_TRUNCATION_DETAILS` | `true` | Log detailed truncation information |
| `RETURN_TRUNCATION_METADATA` | `true` | Include truncation metadata in responses |

### Pydantic Validators

```python
# codebase_rag/config.py

class Settings(BaseSettings):
    EMBEDDING_MAX_LENGTH: int = 512
    EMBEDDING_CHUNK_OVERLAP_TOKENS: int = 32
    QUERY_RESULT_MAX_TOKENS: int = 16000
    QUERY_RESULT_MAX_ROW_TOKENS: int = 2000

    @validator('EMBEDDING_MAX_LENGTH')
    def validate_embedding_max_length(cls, v):
        if v > UNIXCODER_MAX_CONTEXT - 4:
            raise ValueError(
                f"EMBEDDING_MAX_LENGTH ({v}) must be <= {UNIXCODER_MAX_CONTEXT - 4}"
            )
        if v < 64:
            raise ValueError(f"EMBEDDING_MAX_LENGTH must be >= 64")
        return v

    @validator('EMBEDDING_CHUNK_OVERLAP_TOKENS')
    def validate_overlap(cls, v, values):
        max_length = values.get('EMBEDDING_MAX_LENGTH', 512)
        if v >= max_length:
            raise ValueError(
                f"EMBEDDING_CHUNK_OVERLAP_TOKENS must be < EMBEDDING_MAX_LENGTH"
            )
        return v

    @validator('QUERY_RESULT_MAX_ROW_TOKENS')
    def validate_max_row_tokens(cls, v, values):
        max_tokens = values.get('QUERY_RESULT_MAX_TOKENS', 16000)
        if v >= max_tokens:
            raise ValueError(
                f"QUERY_RESULT_MAX_ROW_TOKENS must be < QUERY_RESULT_MAX_TOKENS"
            )
        return v
```

### Provider-Specific Batch Token Limits

| Provider | Max Texts | Max Tokens (Total) | Max Tokens/Text |
|----------|-----------|-------------------|-----------------|
| OpenAI | 2048 | 500,000 | N/A |
| Google GLA | 100 | 2,000,000 | 20,000 |
| Google Vertex | 250 | 7,500,000 | 30,000 |
| Ollama | Configurable | 100,000 | Varies |
| Local | Configurable | 100,000 | Varies |

---

## 5. Graph Database Schema for Chunked Embeddings

### Node Types

```cypher
// CodeChunk node type
CREATE CONSTRAINT chunk_unique_id IF NOT EXISTS
FOR (c:CodeChunk) REQUIRE c.chunk_id IS UNIQUE;

// CodeChunk properties:
// - chunk_id: string (parent_fqn + "_" + chunk_index)
// - content: string
// - start_line: int
// - end_line: int
// - chunk_type: string
// - token_count: int
// - embedding: list<float> (stored in vector backend)
```

### Relationships

```cypher
// Single relationship: HAS_CHUNK
// Links parent code unit to its chunks
(f:Function)-[:HAS_CHUNK {
    index: 0,
    start_line: 1,
    end_line: 50,
    is_summary: false
}]->(c:CodeChunk)
```

---

## 6. Edge Case Handling

### Error Handling Matrix

| Scenario | Handling | Result |
|----------|----------|--------|
| Empty input | Return `[]` | No chunks |
| Single unit < limit | Return single chunk | Normal path |
| Single unit > limit, has nested | Split at nested boundaries | Multiple chunks |
| Single unit > limit, no nested | Split at line boundaries | Multiple chunks |
| Single statement > limit | Split at token boundary | Multiple truncated chunks |
| AST parse failure | Fall back to line-based | Chunks created |
| Binary content | Return `[]` | No chunks |
| Unicode/encoding issue | Replace invalid chars | Graceful degradation |

### Binary Content Detection

```python
def _is_binary(content: str) -> bool:
    """Detect binary content by checking for null bytes."""
    if '\x00' in content:
        return True
    non_printable = sum(1 for c in content[:1000] if ord(c) < 32 and c not in '\n\r\t')
    if len(content[:1000]) > 0 and non_printable / len(content[:1000]) > 0.3:
        return True
    return False
```

---

## 7. Testing

### Key Test Cases

- `test_chunk_empty_code` - Empty input returns empty list
- `test_chunk_under_limit` - Code under limit returns single chunk
- `test_semantic_chunking_respects_class_boundaries` - Class boundaries respected
- `test_semantic_chunking_respects_function_boundaries` - Function boundaries respected
- `test_chunk_ast_parse_failure_fallback` - Invalid code falls back to line-based
- `test_chunk_very_long_single_line` - Very long lines are split
- `test_chunk_respects_unixcoder_context_limit` - Chunks stay under context limit
- `test_balanced_truncation_prevents_budget_hogging` - Large rows are capped
- `test_balanced_truncation_min_rows_guarantee` - Minimum rows are included
- `test_relevance_truncation_prioritizes_high_scores` - High scores are prioritized
- `test_relevance_truncation_preserves_order` - Original order is restored

### Test Files

- `codebase_rag/tests/test_code_chunker.py` - SemanticCodeChunker tests
- `codebase_rag/tests/test_smart_truncation.py` - Smart truncation tests
- `codebase_rag/tests/test_token_utils.py` - Token counting tests

---

## Related Files

- `codebase_rag/utils/code_chunker.py` - SemanticCodeChunker implementation
- `codebase_rag/utils/token_utils.py` - Token counting and smart truncation
- `codebase_rag/embedder.py` - Embedding generation with chunking strategies
- `codebase_rag/embeddings/base.py` - Token-aware batching
- `codebase_rag/config.py` - Configuration settings with validators
- `codebase_rag/models.py` - CodeChunk, ChunkedEmbeddingMetadata, TruncationResult
- `codebase_rag/constants.py` - NodeLabel.CODE_CHUNK, RelationshipType.HAS_CHUNK, UNIXCODER_MAX_CONTEXT
- `codebase_rag/exceptions.py` - EmbeddingLengthExceededError with token_count, max_tokens

---

## Performance Impact

### Expected Improvements

1. **Retrieval Quality:** +20-40% improvement for large code snippets
2. **Token Efficiency:** +15-30% better token utilization
3. **User Satisfaction:** Better visibility and control

### Trade-offs

1. **Embedding Time:** 5-10% overhead for large files
2. **Index Size:** 20-50% increase for hierarchical chunking
3. **Complexity:** More configuration options