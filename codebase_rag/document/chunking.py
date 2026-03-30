"""Semantic document chunking for embeddings.

Documents should be chunked semantically (by section/topic) rather than by
character count for better retrieval quality.

CRITICAL: Don't use hardcoded character limits like 8000.
Use token-aware chunking with section boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .extractors.base import ExtractedDocument, ExtractedSection


@dataclass
class DocumentChunk:
    """A semantically coherent document chunk."""

    content: str
    section_title: str
    start_line: int
    end_line: int
    token_count: int
    document_path: str
    chunk_index: int = 0

    @property
    def qualified_name(self) -> str:
        """Generate unique qualified name for this chunk."""
        return f"{self.document_path}#chunk_{self.chunk_index}"


class SemanticDocumentChunker:
    """
    Chunks documents by semantic boundaries (headers/sections).

    Strategy:
    1. Each section is a potential chunk boundary
    2. If section exceeds max_tokens, split by paragraph
    3. Maintain overlap between chunks

    Example:
        chunker = SemanticDocumentChunker(max_tokens=512)
        chunks = list(chunker.chunk_document(doc))
    """

    MAX_CHUNK_TOKENS = 512  # Per chunk limit
    OVERLAP_TOKENS = 50  # Overlap for context continuity
    ENCODING_MODEL = "cl100k_base"  # GPT-4 encoding

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._encoder = None

    def _get_encoder(self):
        """Lazy-load tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken

                self._encoder = tiktoken.get_encoding(self.ENCODING_MODEL)
            except ImportError:
                # Fallback: approximate tokens as words / 0.75
                self._encoder = None
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        # Fallback: approximate
        return int(len(text.split()) / 0.75)

    def _count_lines(self, text: str) -> int:
        """Count number of lines in text (newline characters + 1)."""
        if not text:
            return 0
        return text.count("\n") + 1

    def _find_line_offset(self, content: str, position: int) -> int:
        """Find the line number at a given character position in content.

        Returns 0-indexed line offset (0 = first line).
        Use section.start_line + offset for absolute line number.
        """
        if position <= 0:
            return 0
        # Count newlines before the position (0-indexed)
        return content[:position].count("\n")

    def chunk_document(self, doc: ExtractedDocument) -> Iterator[DocumentChunk]:
        """
        Chunk document by semantic boundaries.

        Args:
            doc: ExtractedDocument to chunk

        Yields:
            DocumentChunk instances
        """
        chunk_index = 0

        if not doc.sections:
            # No sections - chunk by paragraphs
            yield from self._chunk_plain_text(doc.content, doc.path, chunk_index)
            return

        for section in doc.sections:
            for chunk in self._chunk_section(section, doc.path, chunk_index):
                chunk_index += 1
                yield chunk

    def _chunk_section(
        self, section: ExtractedSection, doc_path: str, start_index: int
    ) -> Iterator[DocumentChunk]:
        """Chunk a single section, respecting token limits."""
        tokens = self.count_tokens(section.content)

        if tokens <= self.max_tokens:
            # Section fits in one chunk
            yield DocumentChunk(
                content=section.content,
                section_title=section.title,
                start_line=section.start_line,
                end_line=section.end_line,
                token_count=tokens,
                document_path=doc_path,
                chunk_index=start_index,
            )
        else:
            # Split by paragraphs within section
            yield from self._split_by_paragraphs(section, doc_path, start_index)

    def _split_by_paragraphs(
        self, section: ExtractedSection, doc_path: str, start_index: int
    ) -> Iterator[DocumentChunk]:
        """Split large sections by paragraph boundaries with line tracking."""
        paragraphs = section.content.split("\n\n")
        current_chunk: list[str] = []
        current_tokens = 0
        chunk_index = start_index
        current_line_start = section.start_line

        # Track position in original content for line number calculation
        content_position = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # Calculate line offset for this paragraph in original content
            para_start_line = section.start_line + self._find_line_offset(
                section.content, content_position
            )

            # Handle very long paragraphs
            if para_tokens > self.max_tokens:
                # Flush current chunk first
                if current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    yield DocumentChunk(
                        content=chunk_content,
                        section_title=section.title,
                        start_line=current_line_start,
                        end_line=current_line_start + self._count_lines(chunk_content) - 1,
                        token_count=current_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long paragraph by sentences - pass start line for tracking
                yielded_count = 0
                for chunk in self._split_long_paragraph(
                    para, section.title, doc_path, chunk_index, para_start_line
                ):
                    yielded_count += 1
                    yield chunk
                chunk_index += yielded_count  # Track actual number of chunks yielded

                # Update content position and line tracking
                content_position += len(para) + 2  # +2 for "\n\n" separator
                continue

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                # Flush current chunk
                chunk_content = "\n\n".join(current_chunk)
                yield DocumentChunk(
                    content=chunk_content,
                    section_title=section.title,
                    start_line=current_line_start,
                    end_line=current_line_start + self._count_lines(chunk_content) - 1,
                    token_count=current_tokens,
                    document_path=doc_path,
                    chunk_index=chunk_index,
                )
                chunk_index += 1

                # Start new chunk with overlap
                if self.overlap_tokens > 0 and current_chunk:
                    # Include last paragraph as overlap
                    overlap_para = current_chunk[-1]
                    overlap_tokens = self.count_tokens(overlap_para)
                    if overlap_tokens <= self.overlap_tokens:
                        current_chunk = [overlap_para, para]
                        current_tokens = overlap_tokens + para_tokens
                        # Adjust line start for overlap
                        overlap_len = len(overlap_para) + 2
                        content_pos_for_overlap = content_position - overlap_len
                        current_line_start = section.start_line + self._find_line_offset(
                            section.content, max(0, content_pos_for_overlap)
                        )
                    else:
                        current_chunk = [para]
                        current_tokens = para_tokens
                        current_line_start = para_start_line
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
                    current_line_start = para_start_line
            else:
                if not current_chunk:
                    # First paragraph in chunk - set line start
                    current_line_start = para_start_line
                current_chunk.append(para)
                current_tokens += para_tokens

            # Update content position for next paragraph
            content_position += len(para) + 2  # +2 for "\n\n" separator

        # Flush remaining
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            yield DocumentChunk(
                content=chunk_content,
                section_title=section.title,
                start_line=current_line_start,
                end_line=section.end_line,
                token_count=current_tokens,
                document_path=doc_path,
                chunk_index=chunk_index,
            )

    def _split_long_paragraph(
        self, para: str, section_title: str, doc_path: str, start_index: int, para_start_line: int = 0
    ) -> Iterator[DocumentChunk]:
        """Split a very long paragraph by sentences with line tracking."""
        # Simple sentence split on punctuation followed by space
        import re

        sentences = re.split(r"(?<=[.!?])\s+", para)
        current_chunk: list[str] = []
        current_tokens = 0
        chunk_index = start_index
        para_position = 0  # Track position within paragraph for line calculation

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            # Calculate line offset for this sentence in the paragraph
            sent_start_line = para_start_line + self._find_line_offset(para, para_position)
            sent_end_line = sent_start_line + self._count_lines(sentence) - 1

            if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                # Use the start line of the first sentence in the chunk
                first_sent_pos = para_position - len(" ".join(current_chunk))
                chunk_start_line = para_start_line + self._find_line_offset(para, max(0, first_sent_pos))
                yield DocumentChunk(
                    content=" ".join(current_chunk),
                    section_title=section_title,
                    start_line=chunk_start_line,
                    end_line=chunk_start_line + self._count_lines(" ".join(current_chunk)) - 1,
                    token_count=current_tokens,
                    document_path=doc_path,
                    chunk_index=chunk_index,
                )
                chunk_index += 1
                current_chunk = [sentence]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens

            # Update position for next sentence
            para_position += len(sentence) + 1  # +1 for space separator

        if current_chunk:
            # Calculate start line for the final chunk
            first_sent_pos = para_position - len(" ".join(current_chunk))
            chunk_start_line = para_start_line + self._find_line_offset(para, max(0, first_sent_pos))
            yield DocumentChunk(
                content=" ".join(current_chunk),
                section_title=section_title,
                start_line=chunk_start_line,
                end_line=chunk_start_line + self._count_lines(" ".join(current_chunk)) - 1,
                token_count=current_tokens,
                document_path=doc_path,
                chunk_index=chunk_index,
            )

    def _chunk_plain_text(
        self, content: str, doc_path: str, start_index: int
    ) -> Iterator[DocumentChunk]:
        """Chunk plain text without sections with line tracking."""
        paragraphs = content.split("\n\n")
        current_chunk: list[str] = []
        current_tokens = 0
        chunk_index = start_index
        current_line_start = 1  # Documents start from line 1
        content_position = 0  # Track position in original content

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # Calculate line offset for this paragraph
            para_start_line = 1 + self._find_line_offset(content, content_position)

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                # Flush current chunk
                chunk_content = "\n\n".join(current_chunk)
                yield DocumentChunk(
                    content=chunk_content,
                    section_title="",
                    start_line=current_line_start,
                    end_line=current_line_start + self._count_lines(chunk_content) - 1,
                    token_count=current_tokens,
                    document_path=doc_path,
                    chunk_index=chunk_index,
                )
                chunk_index += 1
                current_chunk = [para]
                current_tokens = para_tokens
                current_line_start = para_start_line
            else:
                if not current_chunk:
                    # First paragraph - set line start
                    current_line_start = para_start_line
                current_chunk.append(para)
                current_tokens += para_tokens

            # Update content position for next paragraph
            content_position += len(para) + 2  # +2 for "\n\n" separator

        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            total_lines = self._count_lines(content)
            yield DocumentChunk(
                content=chunk_content,
                section_title="",
                start_line=current_line_start,
                end_line=min(current_line_start + self._count_lines(chunk_content) - 1, total_lines),
                token_count=current_tokens,
                document_path=doc_path,
                chunk_index=chunk_index,
            )


__all__ = ["DocumentChunk", "SemanticDocumentChunker"]