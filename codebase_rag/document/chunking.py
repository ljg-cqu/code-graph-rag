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
        """Split large sections by paragraph boundaries."""
        paragraphs = section.content.split("\n\n")
        current_chunk: list[str] = []
        current_tokens = 0
        chunk_index = start_index
        start_line = section.start_line

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # Handle very long paragraphs
            if para_tokens > self.max_tokens:
                # Flush current chunk first
                if current_chunk:
                    yield DocumentChunk(
                        content="\n\n".join(current_chunk),
                        section_title=section.title,
                        start_line=start_line,
                        end_line=start_line + len("\n\n".join(current_chunk).split("\n")),
                        token_count=current_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_index,
                    )
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long paragraph by sentences
                yield from self._split_long_paragraph(
                    para, section.title, doc_path, chunk_index
                )
                chunk_index += 1  # Account for yielded chunks
                continue

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                # Flush current chunk
                yield DocumentChunk(
                    content="\n\n".join(current_chunk),
                    section_title=section.title,
                    start_line=start_line,
                    end_line=start_line + len("\n\n".join(current_chunk).split("\n")),
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
                    else:
                        current_chunk = [para]
                        current_tokens = para_tokens
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Flush remaining
        if current_chunk:
            yield DocumentChunk(
                content="\n\n".join(current_chunk),
                section_title=section.title,
                start_line=start_line,
                end_line=section.end_line,
                token_count=current_tokens,
                document_path=doc_path,
                chunk_index=chunk_index,
            )

    def _split_long_paragraph(
        self, para: str, section_title: str, doc_path: str, start_index: int
    ) -> Iterator[DocumentChunk]:
        """Split a very long paragraph by sentences."""
        # Simple sentence split on punctuation followed by space
        import re

        sentences = re.split(r"(?<=[.!?])\s+", para)
        current_chunk: list[str] = []
        current_tokens = 0
        chunk_index = start_index

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                yield DocumentChunk(
                    content=" ".join(current_chunk),
                    section_title=section_title,
                    start_line=0,
                    end_line=0,
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

        if current_chunk:
            yield DocumentChunk(
                content=" ".join(current_chunk),
                section_title=section_title,
                start_line=0,
                end_line=0,
                token_count=current_tokens,
                document_path=doc_path,
                chunk_index=chunk_index,
            )

    def _chunk_plain_text(
        self, content: str, doc_path: str, start_index: int
    ) -> Iterator[DocumentChunk]:
        """Chunk plain text without sections."""
        paragraphs = content.split("\n\n")
        current_chunk: list[str] = []
        current_tokens = 0
        chunk_index = start_index

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            if current_tokens + para_tokens > self.max_tokens and current_chunk:
                yield DocumentChunk(
                    content="\n\n".join(current_chunk),
                    section_title="",
                    start_line=0,
                    end_line=0,
                    token_count=current_tokens,
                    document_path=doc_path,
                    chunk_index=chunk_index,
                )
                chunk_index += 1
                current_chunk = [para]
                current_tokens = para_tokens
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        if current_chunk:
            yield DocumentChunk(
                content="\n\n".join(current_chunk),
                section_title="",
                start_line=0,
                end_line=0,
                token_count=current_tokens,
                document_path=doc_path,
                chunk_index=chunk_index,
            )


__all__ = ["DocumentChunk", "SemanticDocumentChunker"]