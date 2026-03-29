"""Semantic code chunking with Tree-sitter AST parsing.

This module provides intelligent code chunking that respects semantic boundaries
(functions, classes, blocks) instead of simple truncation.

Usage:
    from codebase_rag.utils.code_chunker import SemanticCodeChunker

    chunker = SemanticCodeChunker(max_tokens=512, language="python")
    chunks = chunker.chunk(large_function_code)
"""

from __future__ import annotations

import importlib
import re
from typing import TYPE_CHECKING, Literal

from loguru import logger

from .. import constants as cs
from ..models import CodeChunk
from .token_utils import count_tokens

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Parser


# Tree-sitter language module mapping
# NOTE: Some languages require alternative packages or have limited availability
LANGUAGE_MODULES: dict[str, str] = {
    "python": "tree_sitter_python",  # PyPI: tree-sitter-python
    "javascript": "tree_sitter_javascript",  # PyPI: tree-sitter-javascript
    "typescript": "tree_sitter_typescript",  # PyPI: tree-sitter-typescript
    "java": "tree_sitter_java",  # PyPI: tree-sitter-java
    "rust": "tree_sitter_rust",  # PyPI: tree-sitter-rust
    "cpp": "tree_sitter_cpp",  # PyPI: tree-sitter-cpp
    "c": "tree_sitter_c",  # PyPI: tree-sitter-c
    "go": "tree_sitter_go",  # PyPI: tree-sitter-go
    "php": "tree_sitter_php",  # PyPI: tree-sitter-php
    "lua": "tree_sitter_lua",  # PyPI: tree-sitter-lua
    "solidity": "tree_sitter_solidity",  # PyPI: tree-sitter-solidity
}

# Language-specific boundary node types for splitting
BOUNDARY_NODE_TYPES: dict[str, dict[str, list[str]]] = {
    "python": {
        "class": ["class_definition"],
        "function": ["function_definition"],
        "method": ["function_definition"],  # Functions inside classes
        "block": [
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "with_statement",
        ],
    },
    "java": {
        "class": [
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "record_declaration",
        ],
        "function": ["method_declaration", "constructor_declaration"],
        "method": ["method_declaration"],
        "block": [
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "switch_statement",
        ],
    },
    "javascript": {
        "class": ["class_declaration"],
        "function": [
            "function_declaration",
            "function_expression",
            "arrow_function",
        ],
        "method": ["method_definition"],
        "block": [
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "switch_statement",
        ],
    },
    "typescript": {
        "class": ["class_declaration", "interface_declaration", "type_alias_declaration"],
        "function": [
            "function_declaration",
            "function_expression",
            "arrow_function",
        ],
        "method": ["method_definition"],
        "block": [
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "switch_statement",
        ],
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
        "class": ["struct_specifier", "enum_specifier", "union_specifier"],
        "function": ["function_definition"],
        "method": [],
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
        "function": [
            "function_definition",
            "modifier_definition",
            "constructor_definition",
        ],
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
        overlap_tokens: int = 32,
        max_chunks_per_node: int = 10,
    ) -> None:
        """Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk.
            language: Programming language for AST parsing.
            overlap_tokens: Overlapping tokens between chunks.
            max_chunks_per_node: Maximum chunks to create per node.
        """
        self.max_tokens = min(max_tokens, cs.UNIXCODER_MAX_CONTEXT - 4)
        self.language = language.lower()
        self.overlap_tokens = overlap_tokens
        self.max_chunks_per_node = max_chunks_per_node
        self._parser: Parser | None = self._init_parser()

    def _init_parser(self) -> Parser | None:
        """Initialize Tree-sitter parser for the language.

        Handles missing grammars gracefully with fallback to line-based chunking.
        """
        try:
            module_name = LANGUAGE_MODULES.get(self.language)
            if not module_name:
                logger.warning(
                    f"No Tree-sitter grammar mapping for language: {self.language}"
                )
                return None

            lang_module = importlib.import_module(module_name)

            # Verify grammar has required attributes
            if not hasattr(lang_module, "language"):
                logger.warning(
                    f"Tree-sitter grammar for {self.language} missing 'language' function"
                )
                return None

            from tree_sitter import Language as TSLanguage
            from tree_sitter import Parser as TSParser

            lang = TSLanguage(lang_module.language())
            parser = TSParser(lang)
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
        if "\x00" in content:
            return True
        # Check for high ratio of non-printable characters
        non_printable = sum(
            1 for c in content[:1000] if ord(c) < 32 and c not in "\n\r\t"
        )
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
            return [
                CodeChunk(
                    content=code,
                    start_line=1,
                    end_line=code.count("\n") + 1,
                    chunk_type="function",  # Assume single unit
                    token_count=token_count,
                )
            ]

        # Try AST-based chunking
        if self._parser:
            try:
                return self._chunk_with_ast(code)
            except Exception as e:
                logger.warning(
                    f"AST chunking failed for {self.language}: {e}, falling back to line-based"
                )

        # Fallback: line-based chunking with overlap
        chunks = self._chunk_at_line_boundaries(code)
        # Only apply overlap if chunks are semantic (not truncated from single line splits)
        if (
            self.overlap_tokens > 0
            and len(chunks) > 1
            and not all(c.chunk_type == "truncated" for c in chunks)
        ):
            chunks = self._add_overlap(chunks, code)
        return chunks

    def _chunk_with_ast(self, code: str) -> list[CodeChunk]:
        """Chunk using Tree-sitter AST parsing."""
        tree = self._parser.parse(bytes(code, "utf-8"))
        root = tree.root_node

        # Check for parse errors
        if root.has_error:
            logger.debug("Parse errors in code, attempting partial chunking")

        boundary_types = BOUNDARY_NODE_TYPES.get(self.language, {})
        if not boundary_types:
            chunks = self._chunk_at_line_boundaries(code)
            # Only apply overlap if chunks are semantic (not truncated from single line splits)
            if (
                self.overlap_tokens > 0
                and len(chunks) > 1
                and not all(c.chunk_type == "truncated" for c in chunks)
            ):
                chunks = self._add_overlap(chunks, code)
            return chunks

        chunks: list[CodeChunk] = []
        class_node_types = boundary_types.get("class", [])
        func_node_types = boundary_types.get("function", [])

        # Priority 1: Find class-level boundaries
        class_nodes = self._find_nodes_by_types(root, class_node_types)
        for node, _ in class_nodes:
            chunks.extend(self._process_node(node, code, "class"))

        # Priority 2: Find function/method boundaries (not already in classes)
        if not chunks or sum(c.token_count for c in chunks) < count_tokens(code) * 0.8:
            func_nodes = self._find_nodes_by_types(root, func_node_types)
            for node, _ in func_nodes:
                # Skip if already covered by a class chunk
                if self._is_nested_in(node, [n for n, _ in class_nodes]):
                    continue
                chunks.extend(self._process_node(node, code, "function"))

        # If still no chunks, use the whole file as one chunk
        if not chunks:
            chunks = self._chunk_at_line_boundaries(code)
            # Only apply overlap if chunks are semantic (not truncated from single line splits)
            if (
                self.overlap_tokens > 0
                and len(chunks) > 1
                and not all(c.chunk_type == "truncated" for c in chunks)
            ):
                chunks = self._add_overlap(chunks, code)
            return chunks

        # Handle overlap between chunks
        if self.overlap_tokens > 0:
            chunks = self._add_overlap(chunks, code)

        return chunks[: self.max_chunks_per_node]

    def _process_node(self, node: Node, code: str, chunk_type: str) -> list[CodeChunk]:
        """Process a single AST node, potentially splitting if too large."""
        node_text = code[node.start_byte : node.end_byte]
        node_tokens = count_tokens(node_text)

        if node_tokens <= self.max_tokens:
            return [
                CodeChunk(
                    content=node_text,
                    start_line=node.start_point.row + 1,
                    end_line=node.end_point.row + 1,
                    chunk_type=chunk_type,
                    token_count=node_tokens,
                )
            ]

        # Node is too large - try splitting at nested boundaries
        boundary_types = BOUNDARY_NODE_TYPES.get(self.language, {})
        nested_types = (
            boundary_types.get("method", [])
            + boundary_types.get("function", [])
            + boundary_types.get("block", [])
        )

        nested_nodes = self._find_nodes_by_types(node, nested_types)

        if nested_nodes:
            # Split at nested boundaries
            result_chunks: list[CodeChunk] = []
            for nested, _ in nested_nodes:
                result_chunks.extend(self._process_node(nested, code, "method"))
            return result_chunks[: self.max_chunks_per_node]

        # No nested boundaries - split at line boundaries
        return self._chunk_text_at_lines(
            node_text, node.start_point.row + 1, chunk_type
        )

    def _chunk_at_line_boundaries(self, code: str) -> list[CodeChunk]:
        """Fallback: chunk at line boundaries when AST parsing fails."""
        lines = code.split("\n")
        chunks: list[CodeChunk] = []
        current_chunk_lines: list[str] = []
        current_tokens = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_tokens = count_tokens(line + "\n")

            # Check if single line exceeds limit
            if line_tokens > self.max_tokens:
                # Flush current chunk
                if current_chunk_lines:
                    content = "\n".join(current_chunk_lines)
                    chunks.append(
                        CodeChunk(
                            content=content,
                            start_line=start_line,
                            end_line=i - 1,
                            chunk_type="truncated",
                            token_count=current_tokens,
                        )
                    )
                    current_chunk_lines = []
                    current_tokens = 0

                # Split the long line
                chunks.extend(self._split_long_line(line, i))
                start_line = i + 1
                continue

            if (
                current_tokens + line_tokens > self.max_tokens
                and current_chunk_lines
            ):
                # Flush current chunk
                content = "\n".join(current_chunk_lines)
                chunks.append(
                    CodeChunk(
                        content=content,
                        start_line=start_line,
                        end_line=i - 1,
                        chunk_type="block",
                        token_count=current_tokens,
                    )
                )
                current_chunk_lines = [line]
                current_tokens = line_tokens
                start_line = i
            else:
                current_chunk_lines.append(line)
                current_tokens += line_tokens

        # Final chunk
        if current_chunk_lines:
            content = "\n".join(current_chunk_lines)
            chunks.append(
                CodeChunk(
                    content=content,
                    start_line=start_line,
                    end_line=len(lines),
                    chunk_type="block",
                    token_count=current_tokens,
                )
            )

        return chunks[: self.max_chunks_per_node]

    def _split_long_line(self, line: str, line_num: int) -> list[CodeChunk]:
        """Split a very long single line (e.g., minified code, long string)."""
        # Try to split at natural boundaries (commas, operators)
        parts = re.split(r"(?<=[,;])\s*", line)

        chunks: list[CodeChunk] = []
        current_parts: list[str] = []
        current_tokens = 0

        for part in parts:
            part_tokens = count_tokens(part)
            if (
                current_tokens + part_tokens > self.max_tokens
                and current_parts
            ):
                chunks.append(
                    CodeChunk(
                        content=" ".join(current_parts),
                        start_line=line_num,
                        end_line=line_num,
                        chunk_type="truncated",
                        token_count=current_tokens,
                    )
                )
                current_parts = [part]
                current_tokens = part_tokens
            else:
                current_parts.append(part)
                current_tokens += part_tokens

        if current_parts:
            chunks.append(
                CodeChunk(
                    content=" ".join(current_parts),
                    start_line=line_num,
                    end_line=line_num,
                    chunk_type="truncated",
                    token_count=current_tokens,
                )
            )

        return chunks

    def _chunk_text_at_lines(
        self, text: str, start_line: int, chunk_type: str
    ) -> list[CodeChunk]:
        """Chunk text at line boundaries with token awareness."""
        lines = text.split("\n")
        chunks: list[CodeChunk] = []
        current_lines: list[str] = []
        current_tokens = 0
        chunk_start = start_line

        for i, line in enumerate(lines):
            line_tokens = count_tokens(line + "\n")

            if (
                current_tokens + line_tokens > self.max_tokens
                and current_lines
            ):
                chunks.append(
                    CodeChunk(
                        content="\n".join(current_lines),
                        start_line=chunk_start,
                        end_line=chunk_start + len(current_lines) - 1,
                        chunk_type=chunk_type,
                        token_count=current_tokens,
                    )
                )
                current_lines = [line]
                current_tokens = line_tokens
                chunk_start = start_line + i
            else:
                current_lines.append(line)
                current_tokens += line_tokens

        if current_lines:
            chunks.append(
                CodeChunk(
                    content="\n".join(current_lines),
                    start_line=chunk_start,
                    end_line=chunk_start + len(current_lines) - 1,
                    chunk_type=chunk_type,
                    token_count=current_tokens,
                )
            )

        return chunks[: self.max_chunks_per_node]

    def _find_nodes_by_types(
        self, root: Node, node_types: list[str]
    ) -> list[tuple[Node, str]]:
        """Find all nodes of specified types using DFS."""
        results: list[tuple[Node, str]] = []

        def dfs(node: Node) -> None:
            if node.type in node_types:
                results.append((node, node.type))
            for child in node.children:
                dfs(child)

        dfs(root)
        return results

    def _is_nested_in(self, node: Node, parents: list[Node]) -> bool:
        """Check if node is nested inside any of the parent nodes."""
        for parent in parents:
            if (
                node.start_byte >= parent.start_byte
                and node.end_byte <= parent.end_byte
            ):
                return True
        return False

    def _add_overlap(self, chunks: list[CodeChunk], code: str) -> list[CodeChunk]:
        """Add overlapping context between chunks for better coherence.

        IMPORTANT: Validates final token count stays under max_tokens.
        Uses token-aware overlap instead of fixed line count.
        """
        if len(chunks) <= 1:
            return chunks

        overlapped: list[CodeChunk] = []
        lines = code.split("\n")

        for i, chunk in enumerate(chunks):
            # Calculate token budget for overlap
            chunk_tokens = chunk.token_count
            overlap_budget = self.max_tokens - chunk_tokens
            if overlap_budget <= 0:
                # No room for overlap, keep original
                overlapped.append(chunk)
                continue

            # Get overlap from previous chunk (token-aware)
            overlap_lines: list[str] = []
            if i > 0 and self.overlap_tokens > 0:
                prev_chunk = chunks[i - 1]
                # Start from previous chunk's end and add lines until token budget exhausted
                overlap_start = prev_chunk.end_line
                overlap_tokens_used = 0
                for line_idx in range(
                    overlap_start - 1,
                    max(overlap_start - 20, chunk.start_line - 1),
                    -1,
                ):
                    if line_idx < 0 or line_idx >= len(lines):
                        break
                    if overlap_tokens_used >= overlap_budget:
                        break
                    line = lines[line_idx]
                    line_tokens = count_tokens(line + "\n")
                    if overlap_tokens_used + line_tokens <= overlap_budget:
                        overlap_lines.insert(0, line)
                        overlap_tokens_used += line_tokens

            # Combine with overlap (deduplicate lines)
            content_lines = lines[chunk.start_line - 1 : chunk.end_line]
            final_content = "\n".join(overlap_lines + content_lines)
            final_tokens = count_tokens(final_content)

            # Validate final token count
            if final_tokens > self.max_tokens:
                # Trim overlap if still over limit
                while overlap_lines and final_tokens > self.max_tokens:
                    overlap_lines.pop(0)
                    final_content = "\n".join(overlap_lines + content_lines)
                    final_tokens = count_tokens(final_content)

            overlapped.append(
                CodeChunk(
                    content=final_content,
                    start_line=chunk.start_line - len(overlap_lines),
                    end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type,
                    parent_fqn=chunk.parent_fqn,
                    chunk_index=i,
                    token_count=final_tokens,
                )
            )

        return overlapped


def generate_code_summary(code: str, max_tokens: int = 256) -> str:
    """Generate a static summary of code for hierarchical embedding strategy.

    PYTHON-SPECIFIC: This function uses Python-specific regex patterns (def, class).
    For other languages, implement language-specific summary generators or use
    the Tree-sitter AST to extract semantic information.

    Extracts: function/class names, signatures, docstrings, and key structure.
    No LLM calls - uses regex-based pattern matching for efficiency.

    Args:
        code: Source code to summarize (assumes Python syntax).
        max_tokens: Maximum tokens for summary.

    Returns:
        Summary string containing names, signatures, and docstrings.
    """
    lines = code.split("\n")
    summary_parts: list[str] = []

    # Extract module/class docstrings (triple-quoted strings at start)
    docstring_pattern = r'"""(.+?)"""|\'\'\'(.+?)\'\'\''
    for i, line in enumerate(lines[:10]):  # Check first 10 lines
        match = re.search(docstring_pattern, line)
        if match:
            doc = match.group(1) or match.group(2)
            summary_parts.append(f"Module: {doc[:100]}")
            break

    # Extract function/method signatures
    func_pattern = r"^(async\s+)?def\s+(\w+)\s*\(([^)]*)\)(\s*->\s*[^:]+)?"
    for line in lines:
        match = re.search(func_pattern, line)
        if match:
            is_async, name, params, return_type = match.groups()
            async_prefix = "async " if is_async else ""
            ret = return_type.strip() if return_type else ""
            summary_parts.append(f"{async_prefix}def {name}({params}){ret}")

    # Extract class definitions
    class_pattern = r"^class\s+(\w+)(\s*\([^)]*\))?"
    for line in lines:
        match = re.search(class_pattern, line)
        if match:
            name, bases = match.groups()
            bases_str = bases if bases else ""
            summary_parts.append(f"class {name}{bases_str}")

    # Combine and truncate to max_tokens
    summary = "\n".join(summary_parts)
    if count_tokens(summary) > max_tokens:
        # Truncate to max_tokens at word boundary
        words = summary.split()
        truncated: list[str] = []
        token_count = 0
        for word in words:
            if token_count + 1 > max_tokens:
                break
            truncated.append(word)
            token_count += 1
        summary = " ".join(truncated) + "..."

    return summary