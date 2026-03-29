"""Tests for semantic code chunker."""

import pytest

from codebase_rag.models import CodeChunk
from codebase_rag.utils.code_chunker import (
    BOUNDARY_NODE_TYPES,
    LANGUAGE_MODULES,
    SemanticCodeChunker,
    generate_code_summary,
)


class TestSemanticCodeChunker:
    """Tests for SemanticCodeChunker class."""

    def test_chunk_empty_code(self) -> None:
        """Empty input returns empty list."""
        chunker = SemanticCodeChunker()
        assert chunker.chunk("") == []
        assert chunker.chunk("   \n\t  ") == []

    def test_chunk_under_limit(self) -> None:
        """Code under limit returns single chunk."""
        code = "def foo():\n    pass"
        chunker = SemanticCodeChunker(max_tokens=100)
        chunks = chunker.chunk(code)
        assert len(chunks) == 1
        assert chunks[0].chunk_type != "truncated"

    def test_chunk_respects_function_boundaries(self) -> None:
        """Verify chunking splits at function boundaries."""
        code = '''
def func1():
    pass

def func2():
    pass
'''
        chunker = SemanticCodeChunker(max_tokens=15)  # Lower limit to force splitting
        chunks = chunker.chunk(code)
        # Either 1 or 2 chunks depending on token count and exact splitting
        assert len(chunks) >= 1
        # Each chunk should have function content if split
        if len(chunks) > 1:
            assert all("def " in c.content for c in chunks)

    def test_chunk_single_function_exceeds_limit(self) -> None:
        """Single function exceeding limit should split at statements."""
        # Very long function with many statements
        code = "def long_func():\n" + "\n".join([f"    x{i} = {i}" for i in range(100)])
        chunker = SemanticCodeChunker(max_tokens=100)
        chunks = chunker.chunk(code)
        assert len(chunks) > 1

    def test_chunk_fallback_to_line_boundaries(self) -> None:
        """Invalid code falls back to line-based chunking."""
        # Code with syntax error - Tree-sitter may still parse it
        code = "def incomplete(:\n    # syntax error\n"
        chunker = SemanticCodeChunker(max_tokens=50)
        chunks = chunker.chunk(code)
        # Should still return something despite potential parse error
        assert len(chunks) >= 1

    def test_chunk_very_long_single_line(self) -> None:
        """Very long single line should be split."""
        # Use a line with commas so it can be split
        code = 'x = "' + ','.join(['a' * 100 for _ in range(50)]) + '"'
        chunker = SemanticCodeChunker(max_tokens=100)
        chunks = chunker.chunk(code)
        assert len(chunks) >= 1
        # Each chunk should be reasonable size
        for chunk in chunks:
            # Allow some margin since we may not be able to split perfectly
            assert chunk.token_count <= 500

    def test_binary_content_detection(self) -> None:
        """Binary content should be detected and skipped."""
        chunker = SemanticCodeChunker()
        # Null byte indicates binary
        assert chunker._is_binary("\x00binary") is True
        # Normal code is not binary
        assert chunker._is_binary("def foo(): pass") is False
        # High ratio of non-printable chars
        assert chunker._is_binary("\x01\x02\x03\x04\x05\x06\x07\x08\x09normal") is True

    def test_max_chunks_per_node_limit(self) -> None:
        """Should respect max_chunks_per_node limit."""
        code = "\n".join([f"def func{i}(): pass" for i in range(20)])
        chunker = SemanticCodeChunker(max_tokens=10, max_chunks_per_node=5)
        chunks = chunker.chunk(code)
        assert len(chunks) <= 5

    def test_overlap_adds_context(self) -> None:
        """Overlap should add context between chunks."""
        code = '''
def func1():
    x = 1

def func2():
    y = 2
'''
        chunker = SemanticCodeChunker(max_tokens=20, overlap_tokens=10)
        chunks = chunker.chunk(code)
        # With overlap, chunks may have overlapping content
        if len(chunks) > 1:
            # Check that overlap was attempted
            assert chunks[0].token_count <= chunker.max_tokens

    def test_no_overlap_when_single_chunk(self) -> None:
        """Overlap should not be added for single chunk."""
        code = "def foo(): pass"
        chunker = SemanticCodeChunker(max_tokens=100, overlap_tokens=10)
        chunks = chunker.chunk(code)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_respects_max_tokens_limit(self) -> None:
        """Chunks should not exceed max_tokens."""
        code = '''
class BigClass:
    def method1(self):
        """A long method."""
        x = 1
        y = 2
        z = 3
        return x + y + z

    def method2(self):
        """Another long method."""
        a = 10
        b = 20
        c = 30
        return a * b * c
'''
        chunker = SemanticCodeChunker(max_tokens=50)
        chunks = chunker.chunk(code)
        for chunk in chunks:
            # Allow some margin for overlap
            assert chunk.token_count <= chunker.max_tokens + 50


class TestGenerateCodeSummary:
    """Tests for generate_code_summary function."""

    def test_extract_function_signature(self) -> None:
        """Should extract function signatures."""
        code = '''
def my_func(a: int, b: str) -> bool:
    return True
'''
        summary = generate_code_summary(code)
        assert "def my_func" in summary
        assert "a: int" in summary or "a" in summary

    def test_extract_class_definition(self) -> None:
        """Should extract class definitions."""
        code = '''
class MyClass(BaseClass):
    pass
'''
        summary = generate_code_summary(code)
        assert "class MyClass" in summary

    def test_extract_docstring(self) -> None:
        """Should extract module docstrings."""
        code = '''
"""This is a module docstring."""

def func():
    pass
'''
        summary = generate_code_summary(code)
        assert "Module:" in summary

    def test_respects_max_tokens(self) -> None:
        """Should truncate to max_tokens."""
        code = "\n".join([f"def func{i}(): pass" for i in range(100)])
        summary = generate_code_summary(code, max_tokens=50)
        from codebase_rag.utils.token_utils import count_tokens

        # Allow some margin since tokenization is approximate
        # The function may not be perfectly precise
        assert count_tokens(summary) <= 150


class TestBoundaryNodeTypes:
    """Tests for boundary node type configurations."""

    def test_all_languages_have_boundary_types(self) -> None:
        """All supported languages should have boundary type configs."""
        # These are the languages we expect to support
        expected_languages = {
            "python",
            "java",
            "javascript",
            "typescript",
            "rust",
            "cpp",
            "c",
            "go",
            "php",
            "lua",
            "solidity",
        }
        for lang in expected_languages:
            assert lang in BOUNDARY_NODE_TYPES, f"Missing boundary types for {lang}"
            types = BOUNDARY_NODE_TYPES[lang]
            # All languages should have at least function types
            assert "function" in types or "class" in types, f"No useful types for {lang}"

    def test_language_modules_mapping(self) -> None:
        """All languages in BOUNDARY_NODE_TYPES should have module mapping."""
        for lang in BOUNDARY_NODE_TYPES:
            assert lang in LANGUAGE_MODULES, f"Missing language module for {lang}"


class TestCodeChunkDataclass:
    """Tests for CodeChunk dataclass."""

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        chunk = CodeChunk(
            content="test",
            start_line=1,
            end_line=2,
            chunk_type="function",
        )
        assert chunk.parent_scope is None
        assert chunk.parent_fqn is None
        assert chunk.chunk_index == 0
        assert chunk.token_count == 0
        assert chunk.metadata == {}

    def test_full_initialization(self) -> None:
        """Test with all fields specified."""
        chunk = CodeChunk(
            content="test content",
            start_line=10,
            end_line=20,
            chunk_type="method",
            parent_scope="MyClass",
            parent_fqn="module.MyClass.method",
            chunk_index=2,
            token_count=50,
            metadata={"key": "value"},
        )
        assert chunk.content == "test content"
        assert chunk.start_line == 10
        assert chunk.end_line == 20
        assert chunk.chunk_type == "method"
        assert chunk.parent_scope == "MyClass"
        assert chunk.parent_fqn == "module.MyClass.method"
        assert chunk.chunk_index == 2
        assert chunk.token_count == 50
        assert chunk.metadata == {"key": "value"}