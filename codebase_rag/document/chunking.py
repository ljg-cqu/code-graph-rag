"""Semantic document chunking for embeddings.

Documents should be chunked semantically (by section/topic) rather than by
character count for better retrieval quality.

CRITICAL: Don't use hardcoded character limits like 8000.
Use token-aware chunking with section boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from loguru import logger

if TYPE_CHECKING:
    from .extractors.base import ExtractedDocument, ExtractedSection


# Regex to split sentences, keeping punctuation with the sentence
# We split AFTER punctuation, then find the actual whitespace boundary
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])")


@dataclass
class DocumentChunk:
    """A semantically coherent document chunk.

    Represents a portion of a document split by semantic boundaries.
    All line numbers use 0-indexed convention (0 = first line).

    Attributes:
        content: The text content of this chunk.
        section_title: Title of the containing section (empty string for plain text).
        start_line: Starting line number (0-indexed, relative to document).
        end_line: Ending line number (0-indexed, inclusive).
        token_count: Number of tokens in this chunk (via tiktoken cl100k_base).
        document_path: Path to the source document.
        chunk_index: Sequential index within the document (0, 1, 2, ...).
            Auto-assigned by SemanticDocumentChunker to ensure unique qualified_name.
            Default 0 is placeholder; chunker always assigns explicitly.
    """

    content: str
    section_title: str
    start_line: int
    end_line: int
    token_count: int
    document_path: str
    chunk_index: int = 0

    @property
    def qualified_name(self) -> str:
        """Generate unique qualified name for this chunk.

        Format: {document_path}#chunk_{chunk_index}

        Used as unique key in document graph storage.
        """
        return f"{self.document_path}#chunk_{self.chunk_index}"


class SemanticDocumentChunker:
    """
    Chunks documents by semantic boundaries (headers/sections).

    Strategy:
    1. Each section is a potential chunk boundary
    2. If section exceeds max_tokens, split by paragraph
    3. Maintain overlap between chunks

    Line Number Convention:
        All line numbers are 0-indexed (0 = first line).

    Example:
        chunker = SemanticDocumentChunker(max_tokens=512)
        chunks = list(chunker.chunk_document(doc))
    """

    MAX_CHUNK_TOKENS = 512  # Per chunk limit
    OVERLAP_TOKENS = 50  # Overlap for context continuity
    ENCODING_MODEL = "cl100k_base"  # GPT-4 encoding
    MAX_REASONABLE_TOKENS = 8192  # Upper bound for max_tokens (most embedding models)
    MAX_CHUNKS_PER_DOCUMENT = 1000  # Prevent memory exhaustion on pathological inputs

    # Character-based segment reduction constants
    APPROX_CHARS_PER_TOKEN = 4  # Average characters per token
    SEGMENT_REDUCTION_FACTOR = 0.75  # Reduce segment by ~25% per iteration when oversized
    MAX_REDUCTION_ITERATIONS = 20  # Safety limit to prevent unbounded loops

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50):
        """Initialize the chunker.

        Args:
            max_tokens: Maximum tokens per chunk. Must be positive and not exceed
                MAX_REASONABLE_TOKENS (8192 by default).
            overlap_tokens: Overlapping tokens between chunks. Must be non-negative
                and less than max_tokens.

        Raises:
            ValueError: If parameters are invalid.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        if max_tokens > self.MAX_REASONABLE_TOKENS:
            raise ValueError(
                f"max_tokens ({max_tokens}) exceeds reasonable limit ({self.MAX_REASONABLE_TOKENS})"
            )
        if overlap_tokens < 0:
            raise ValueError(f"overlap_tokens must be non-negative, got {overlap_tokens}")
        if overlap_tokens >= max_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be less than max_tokens ({max_tokens})"
            )
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

    def _reduce_segment_to_fit(self, segment: str) -> tuple[str, int]:
        """Reduce a segment until it fits within max_tokens.

        Uses iterative 25% reduction with a safety limit to prevent
        unbounded loops on pathological inputs.

        GUARANTEES: returned segment_tokens <= self.max_tokens (or empty string).

        Args:
            segment: Text segment to reduce

        Returns:
            Tuple of (reduced_segment, token_count) - guaranteed to fit within max_tokens.
        """
        segment_tokens = self.count_tokens(segment)
        iterations = 0

        while (
            segment_tokens > self.max_tokens
            and len(segment) > 1
            and iterations < self.MAX_REDUCTION_ITERATIONS
        ):
            # Use max(1, ...) to prevent integer truncation producing empty string
            new_len = max(1, int(len(segment) * self.SEGMENT_REDUCTION_FACTOR))
            segment = segment[:new_len]
            segment_tokens = self.count_tokens(segment)
            iterations += 1

        # GUARANTEE: Ensure we never return an oversized segment
        if segment_tokens > self.max_tokens and len(segment) > 0:
            # Force truncate to a size guaranteed to fit based on chars-per-token estimate
            target_chars = self.max_tokens * self.APPROX_CHARS_PER_TOKEN
            segment = segment[:max(target_chars, 1)]
            segment_tokens = self.count_tokens(segment)

            # Character-by-character fallback for pathological cases
            while segment_tokens > self.max_tokens and len(segment) > 0:
                segment = segment[:-1]
                segment_tokens = self.count_tokens(segment)

        return segment, segment_tokens

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

        Note:
            Stops yielding after MAX_CHUNKS_PER_DOCUMENT to prevent memory exhaustion.
        """
        # Use a mutable counter to track chunk index across recursive calls
        chunk_counter = [0]  # Use list for mutable reference

        if not doc.sections:
            # No sections - chunk by paragraphs
            for chunk in self._chunk_plain_text(doc.content, doc.path, chunk_counter):
                if chunk_counter[0] >= self.MAX_CHUNKS_PER_DOCUMENT:
                    logger.warning(
                        f"Document {doc.path} reached MAX_CHUNKS_PER_DOCUMENT ({self.MAX_CHUNKS_PER_DOCUMENT}) "
                        "limit. Some content may not be indexed."
                    )
                    return
                yield chunk
            return

        for section in doc.sections:
            for chunk in self._chunk_section_recursive(section, doc.path, chunk_counter):
                if chunk_counter[0] >= self.MAX_CHUNKS_PER_DOCUMENT:
                    logger.warning(
                        f"Document {doc.path} reached MAX_CHUNKS_PER_DOCUMENT ({self.MAX_CHUNKS_PER_DOCUMENT}) "
                        "limit. Some content may not be indexed."
                    )
                    return
                yield chunk

    def _chunk_section_recursive(
        self, section: ExtractedSection, doc_path: str, chunk_counter: list[int]
    ) -> Iterator[DocumentChunk]:
        """Chunk a section and its subsections recursively.

        Each section only chunks its "own" content - the content between
        its header and the next subsection's header (or its end_line).

        Line tracking note: section.start_line is the header line.
        section.content starts at section.start_line + 1.

        Args:
            section: Section to chunk
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
        """
        # Skip sections with empty content
        if not section.content or not section.content.strip():
            # Still process subsections even if this section is empty
            for subsection in section.subsections:
                yield from self._chunk_section_recursive(subsection, doc_path, chunk_counter)
            return

        # Calculate the content that belongs to THIS section only
        # (excluding subsection headers and their content)
        content_start = section.start_line + 1  # Content starts after header

        # Find where subsections start
        subsection_starts = [s.start_line for s in section.subsections]
        if subsection_starts:
            # This section's "own" content ends at the first subsection
            own_content_end = min(subsection_starts) - 1
        else:
            # No subsections - use the section's end_line
            own_content_end = section.end_line

        # Extract only the content specific to this section
        if own_content_end >= content_start:
            own_content_lines = own_content_end - content_start + 1
            # Split content by lines and take only the "own" portion
            all_lines = section.content.split("\n")
            own_lines = all_lines[:own_content_lines]
            own_content = "\n".join(own_lines)

            if own_content.strip():
                tokens = self.count_tokens(own_content)

                if tokens <= self.max_tokens:
                    yield DocumentChunk(
                        content=own_content,
                        section_title=section.title,
                        start_line=content_start,
                        end_line=own_content_end,
                        token_count=tokens,
                        document_path=doc_path,
                        chunk_index=chunk_counter[0],
                    )
                    chunk_counter[0] += 1
                else:
                    # Split large sections - helper updates counter
                    yield from self._split_section_content(
                        own_content, section.title, content_start, own_content_end, doc_path, chunk_counter
                    )

        # Recursively process subsections with updated counter
        for subsection in section.subsections:
            yield from self._chunk_section_recursive(subsection, doc_path, chunk_counter)

    def _split_section_content(
        self,
        content: str,
        section_title: str,
        start_line: int,
        end_line: int,
        doc_path: str,
        chunk_counter: list[int],
    ) -> Iterator[DocumentChunk]:
        """Split section content by paragraphs with line tracking.

        Args:
            content: Section content to split
            section_title: Title of the section
            start_line: Starting line (0-indexed)
            end_line: Ending line (0-indexed)
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
        """
        paragraphs = content.split("\n\n")
        current_chunk: list[str] = []
        current_tokens = 0
        current_line = start_line

        # Track line positions within content
        line_positions = []
        line_idx = 0
        for i, char in enumerate(content):
            if char == '\n':
                line_idx += 1
            line_positions.append(line_idx)

        for para in paragraphs:
            if not para.strip():
                continue

            para_tokens = self.count_tokens(para)
            sep_tokens = 1 if current_chunk else 0  # \n\n = ~1 token

            if current_tokens + sep_tokens + para_tokens <= self.max_tokens:
                current_chunk.append(para)
                current_tokens += sep_tokens + para_tokens
            else:
                # Flush current chunk first
                if current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    chunk_lines = chunk_content.count("\n")
                    yield DocumentChunk(
                        content=chunk_content,
                        section_title=section_title,
                        start_line=current_line,
                        end_line=current_line + chunk_lines,
                        token_count=current_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_counter[0],
                    )
                    chunk_counter[0] += 1
                    current_line += chunk_lines + 2  # +2 for paragraph separator
                    current_chunk = []
                    current_tokens = 0

                # Handle oversized paragraph
                if para_tokens > self.max_tokens:
                    # Split oversized paragraph - helper updates counter
                    yield from self._split_oversized_paragraph_in_section(
                        para, section_title, current_line, doc_path, chunk_counter
                    )
                    # Update line position after the oversized paragraph
                    para_lines = para.count("\n")
                    current_line += para_lines + 1
                else:
                    # Start new chunk with this paragraph
                    current_chunk.append(para)
                    current_tokens = para_tokens

        # Flush remaining chunk
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            yield DocumentChunk(
                content=chunk_content,
                section_title=section_title,
                start_line=current_line,
                end_line=end_line,
                token_count=current_tokens,
                document_path=doc_path,
                chunk_index=chunk_counter[0],
            )
            chunk_counter[0] += 1

    def _split_oversized_paragraph_in_section(
        self,
        paragraph: str,
        section_title: str,
        start_line: int,
        doc_path: str,
        chunk_counter: list[int],
    ) -> Iterator[DocumentChunk]:
        """Split an oversized paragraph within a section.

        Uses sentence boundaries and eventually hard splits if needed.

        Args:
            paragraph: Paragraph text to split
            section_title: Title of containing section
            start_line: Starting line (0-indexed)
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
        """
        lines = paragraph.split("\n")
        current_chunk_lines: list[str] = []
        current_tokens = 0
        current_line = start_line

        for line in lines:
            line_tokens = self.count_tokens(line)
            sep_tokens = 1 if current_chunk_lines else 0

            if current_tokens + sep_tokens + line_tokens <= self.max_tokens:
                current_chunk_lines.append(line)
                current_tokens += sep_tokens + line_tokens
            else:
                # Flush current chunk
                if current_chunk_lines:
                    chunk_content = "\n".join(current_chunk_lines)
                    chunk_end_line = current_line + len(current_chunk_lines) - 1
                    yield DocumentChunk(
                        content=chunk_content,
                        section_title=section_title,
                        start_line=current_line,
                        end_line=chunk_end_line,
                        token_count=current_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_counter[0],
                    )
                    chunk_counter[0] += 1
                    current_line = chunk_end_line + 1
                    current_chunk_lines = []
                    current_tokens = 0

                # Handle single line that exceeds limit
                if line_tokens > self.max_tokens:
                    # Hard split the line - helper updates counter
                    yield from self._split_long_line(
                        line, section_title, current_line, doc_path, chunk_counter
                    )
                    current_line += 1
                else:
                    current_chunk_lines.append(line)
                    current_tokens = line_tokens

        # Flush remaining lines
        if current_chunk_lines:
            chunk_content = "\n".join(current_chunk_lines)
            yield DocumentChunk(
                content=chunk_content,
                section_title=section_title,
                start_line=current_line,
                end_line=current_line + len(current_chunk_lines) - 1,
                token_count=current_tokens,
                document_path=doc_path,
                chunk_index=chunk_counter[0],
            )
            chunk_counter[0] += 1

    def _split_long_line(
        self,
        line: str,
        section_title: str,
        start_line: int,
        doc_path: str,
        chunk_counter: list[int],
    ) -> Iterator[DocumentChunk]:
        """Split a single line that exceeds token limit.

        Args:
            line: Line text to split
            section_title: Title of containing section
            start_line: Starting line (0-indexed)
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
        """
        # Split by sentences first
        sentences = self._split_by_sentences(line)

        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)

            if current_tokens + sent_tokens <= self.max_tokens:
                current_chunk += sentence
                current_tokens += sent_tokens
            else:
                if current_chunk:
                    yield DocumentChunk(
                        content=current_chunk,
                        section_title=section_title,
                        start_line=start_line,
                        end_line=start_line,
                        token_count=current_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_counter[0],
                    )
                    chunk_counter[0] += 1

                # Handle sentence that exceeds limit
                if sent_tokens > self.max_tokens:
                    # Hard split by characters - helper updates counter
                    yield from self._hard_split(
                        sentence, section_title, start_line, doc_path, chunk_counter
                    )
                else:
                    current_chunk = sentence
                    current_tokens = sent_tokens

        if current_chunk:
            yield DocumentChunk(
                content=current_chunk,
                section_title=section_title,
                start_line=start_line,
                end_line=start_line,
                token_count=current_tokens,
                document_path=doc_path,
                chunk_index=chunk_counter[0],
            )
            chunk_counter[0] += 1

    def _split_by_sentences(self, text: str) -> list[str]:
        """Split text by sentence boundaries."""
        # Split by sentence-ending punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s for s in sentences if s.strip()]

    def _hard_split(
        self,
        text: str,
        section_title: str,
        start_line: int,
        doc_path: str,
        chunk_counter: list[int],
    ) -> Iterator[DocumentChunk]:
        """Hard split text by character count when other methods fail.

        Args:
            text: Text to split
            section_title: Title of containing section
            start_line: Starting line (0-indexed)
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
        """
        # Estimate characters per token (rough approximation)
        chars_per_token = 4
        max_chars = self.max_tokens * chars_per_token

        for i in range(0, len(text), max_chars):
            chunk_text = text[i:i + max_chars]
            yield DocumentChunk(
                content=chunk_text,
                section_title=section_title,
                start_line=start_line,
                end_line=start_line,
                token_count=self.count_tokens(chunk_text),
                document_path=doc_path,
                chunk_index=chunk_counter[0],
            )
            chunk_counter[0] += 1

    def _chunk_section(
        self, section: ExtractedSection, doc_path: str, chunk_counter: list[int]
    ) -> Iterator[DocumentChunk]:
        """Chunk a single section, respecting token limits.

        Line tracking note: section.start_line is the header line.
        section.content starts at section.start_line + 1.
        Chunk line numbers must account for this +1 offset.

        Args:
            section: Section to chunk
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
        """
        # Skip sections with empty content
        if not section.content or not section.content.strip():
            return

        tokens = self.count_tokens(section.content)

        if tokens <= self.max_tokens:
            # Section fits in one chunk
            # Content starts at header line + 1
            content_start_line = section.start_line + 1
            yield DocumentChunk(
                content=section.content,
                section_title=section.title,
                start_line=content_start_line,
                end_line=section.end_line,
                token_count=tokens,
                document_path=doc_path,
                chunk_index=chunk_counter[0],
            )
            chunk_counter[0] += 1
        else:
            # Split by paragraphs within section
            yield from self._split_by_paragraphs(section, doc_path, chunk_counter)

    def _split_by_paragraphs(
        self, section: ExtractedSection, doc_path: str, chunk_counter: list[int]
    ) -> Iterator[DocumentChunk]:
        """Split large sections by paragraph boundaries with line tracking.

        Line tracking note: section.start_line is the header line.
        section.content starts at section.start_line + 1.
        All chunk line numbers are relative to content, so we add +1 offset.

        Args:
            section: Section to split
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
        """
        paragraphs = section.content.split("\n\n")
        current_chunk: list[str] = []
        current_tokens = 0  # Tokens including separator tokens
        # Content starts at header line + 1
        content_start_offset = section.start_line + 1
        current_line_start = content_start_offset

        # Track position in original content for line number calculation
        content_position = 0

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # Calculate line offset for this paragraph in original content
            # Use content_start_offset (header+1) as base
            para_start_line = content_start_offset + self._find_line_offset(
                section.content, content_position
            )

            # Handle very long paragraphs
            if para_tokens > self.max_tokens:
                # Flush current chunk first with validation
                if current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    actual_tokens = self.count_tokens(chunk_content)

                    # Validate chunk doesn't exceed max_tokens
                    if actual_tokens > self.max_tokens:
                        yield from self._split_oversized_block(
                            chunk_content, section.title, doc_path, chunk_counter,
                            current_line_start, 0
                        )
                    else:
                        yield DocumentChunk(
                            content=chunk_content,
                            section_title=section.title,
                            start_line=current_line_start,
                            end_line=current_line_start + self._count_lines(chunk_content) - 1,
                            token_count=actual_tokens,
                            document_path=doc_path,
                            chunk_index=chunk_counter[0],
                        )
                        chunk_counter[0] += 1
                    current_chunk = []
                    current_tokens = 0

                # Split long paragraph by sentences - pass counter for tracking
                yield from self._split_long_paragraph(
                    para, section.title, doc_path, chunk_counter, para_start_line, section.content, content_position
                )

                # Update content position and line tracking
                content_position += len(para) + 2  # +2 for "\n\n" separator
                continue

            # Account for separator tokens in the split decision
            # If current_chunk is not empty, adding para will add separator
            separator_tokens = 2 if current_chunk else 0  # "\n\n" tokens
            effective_tokens = current_tokens + separator_tokens + para_tokens

            if effective_tokens > self.max_tokens and current_chunk:
                # Flush current chunk with validation
                chunk_content = "\n\n".join(current_chunk)
                actual_tokens = self.count_tokens(chunk_content)

                # Validate chunk doesn't exceed max_tokens
                if actual_tokens > self.max_tokens:
                    yield from self._split_oversized_block(
                        chunk_content, section.title, doc_path, chunk_counter,
                        current_line_start, 0
                    )
                else:
                    yield DocumentChunk(
                        content=chunk_content,
                        section_title=section.title,
                        start_line=current_line_start,
                        end_line=current_line_start + self._count_lines(chunk_content) - 1,
                        token_count=actual_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_counter[0],
                    )
                    chunk_counter[0] += 1

                # Start new chunk with overlap
                if self.overlap_tokens > 0:  # current_chunk guaranteed non-empty here
                    # Include last paragraph as overlap
                    overlap_para = current_chunk[-1]
                    overlap_tok = self.count_tokens(overlap_para)
                    # Account for separator tokens
                    sep_tokens = 2  # "\n\n" typically tokenizes as ~2 tokens
                    combined_tokens = overlap_tok + sep_tokens + para_tokens

                    # Only add overlap if it doesn't exceed max_tokens
                    if overlap_tok <= self.overlap_tokens and combined_tokens <= self.max_tokens:
                        current_chunk = [overlap_para, para]
                        current_tokens = overlap_tok + sep_tokens + para_tokens
                        # Adjust line start for overlap using content_start_offset
                        overlap_len = len(overlap_para) + 2
                        content_pos_for_overlap = content_position - overlap_len
                        current_line_start = content_start_offset + self._find_line_offset(
                            section.content, max(0, content_pos_for_overlap)
                        )
                    else:
                        # Overlap would exceed limits, start fresh
                        current_chunk = [para]
                        current_tokens = para_tokens
                        current_line_start = para_start_line
                else:
                    current_chunk = [para]
                    current_tokens = para_tokens
                    current_line_start = para_start_line
            else:
                if not current_chunk:
                    # First paragraph - set line start
                    current_line_start = para_start_line
                current_chunk.append(para)
                current_tokens += separator_tokens + para_tokens

            # Update content position for next paragraph
            content_position += len(para) + 2  # +2 for "\n\n" separator

        # Flush remaining
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            actual_tokens = self.count_tokens(chunk_content)

            # P0 FIX: Validate final chunk doesn't exceed max_tokens
            if actual_tokens > self.max_tokens:
                # Use _split_oversized_block for any oversized final chunk
                yield from self._split_oversized_block(
                    chunk_content, section.title, doc_path, chunk_counter,
                    current_line_start, 0
                )
            else:
                yield DocumentChunk(
                    content=chunk_content,
                    section_title=section.title,
                    start_line=current_line_start,
                    end_line=current_line_start + self._count_lines(chunk_content) - 1,
                    token_count=actual_tokens,
                    document_path=doc_path,
                    chunk_index=chunk_counter[0],
                )
                chunk_counter[0] += 1

    def _split_long_paragraph(
        self, para: str, section_title: str, doc_path: str, chunk_counter: list[int],
        para_start_line: int, original_content: str = "", content_position: int = 0
    ) -> Iterator[DocumentChunk]:
        """Split a very long paragraph by sentences with accurate line tracking.

        Uses regex that preserves position accuracy by finding sentence boundaries
        without consuming the whitespace that follows.

        Args:
            para: Paragraph text to split
            section_title: Title of the containing section
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
            para_start_line: Starting line number (0-indexed)
            original_content: Original section content for line tracking (optional)
            content_position: Position of paragraph in original content (optional)
        """
        # Split on sentence boundaries, keeping punctuation with the sentence
        raw_splits = _SENTENCE_BOUNDARY_RE.split(para)

        # Process splits to get actual sentences (strip leading whitespace)
        # and track the actual position where each sentence starts in the original para
        sentences_with_positions: list[tuple[str, int]] = []
        position = 0
        for raw_split in raw_splits:
            if not raw_split:
                continue
            # Find where the actual content starts (skip leading whitespace)
            content_start = len(raw_split) - len(raw_split.lstrip())
            sentence = raw_split.lstrip()
            if sentence:
                sentences_with_positions.append((sentence, position + content_start))
            # Move position forward for next split
            position += len(raw_split)

        current_chunk: list[tuple[str, int]] = []  # (sentence, start_position in para)
        current_tokens = 0

        for sentence, sent_start_pos in sentences_with_positions:
            sent_tokens = self.count_tokens(sentence)

            # CRITICAL: If single sentence exceeds max_tokens, split it further
            # This handles code blocks, diagrams, etc. with no sentence-ending punctuation
            if sent_tokens > self.max_tokens:
                # First flush any accumulated chunk with validation
                if current_chunk:
                    first_sent_pos = current_chunk[0][1]
                    last_sent_pos = current_chunk[-1][1] + len(current_chunk[-1][0])
                    chunk_content = para[first_sent_pos:last_sent_pos].strip()
                    chunk_start_line = para_start_line + self._find_line_offset(para, first_sent_pos)
                    chunk_end_line = para_start_line + self._find_line_offset(para, last_sent_pos)
                    actual_tokens = self.count_tokens(chunk_content)

                    # P0 FIX: Validate chunk before yielding oversized sentence split
                    if actual_tokens > self.max_tokens:
                        # Split this chunk using _split_oversized_block
                        yield from self._split_oversized_block(
                            chunk_content, section_title, doc_path, chunk_counter,
                            chunk_start_line, first_sent_pos
                        )
                    else:
                        yield DocumentChunk(
                            content=chunk_content,
                            section_title=section_title,
                            start_line=chunk_start_line,
                            end_line=chunk_end_line,
                            token_count=actual_tokens,
                            document_path=doc_path,
                            chunk_index=chunk_counter[0],
                        )
                        chunk_counter[0] += 1
                    current_chunk = []
                    current_tokens = 0

                # Split oversized sentence by newlines (for code/diagrams)
                yield from self._split_oversized_block(
                    sentence, section_title, doc_path, chunk_counter, para_start_line, sent_start_pos
                )
                continue

            # Account for separator tokens (space) between sentences
            # When sentences are joined in the paragraph, they have spaces between them
            separator_tokens = 1 if current_chunk else 0  # Space between sentences
            effective_tokens = current_tokens + separator_tokens + sent_tokens

            if effective_tokens > self.max_tokens and current_chunk:
                # Use the position of the first and last sentence for accurate line calculation
                first_sent_pos = current_chunk[0][1]
                last_sent_pos = current_chunk[-1][1] + len(current_chunk[-1][0])
                chunk_content = para[first_sent_pos:last_sent_pos].strip()
                chunk_start_line = para_start_line + self._find_line_offset(para, first_sent_pos)
                chunk_end_line = para_start_line + self._find_line_offset(para, last_sent_pos)
                actual_tokens = self.count_tokens(chunk_content)

                # P0 FIX: Validate chunk doesn't exceed max_tokens
                # If it does, split the chunk by removing sentences from the end
                if actual_tokens > self.max_tokens:
                    # Try removing sentences one by one until it fits
                    while actual_tokens > self.max_tokens and len(current_chunk) > 1:
                        current_chunk.pop()  # Remove last sentence
                        last_sent_pos = current_chunk[-1][1] + len(current_chunk[-1][0])
                        chunk_content = para[first_sent_pos:last_sent_pos].strip()
                        actual_tokens = self.count_tokens(chunk_content)

                    # If single sentence still exceeds, use _split_oversized_block
                    if actual_tokens > self.max_tokens:
                        single_sentence = current_chunk[0][0]
                        single_pos = current_chunk[0][1]
                        yield from self._split_oversized_block(
                            single_sentence, section_title, doc_path, chunk_counter,
                            para_start_line + self._find_line_offset(para, single_pos),
                            single_pos
                        )
                        # Add current sentence as new chunk
                        current_chunk = [(sentence, sent_start_pos)]
                        current_tokens = sent_tokens
                        continue

                yield DocumentChunk(
                    content=chunk_content,
                    section_title=section_title,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    token_count=actual_tokens,
                    document_path=doc_path,
                    chunk_index=chunk_counter[0],
                )
                chunk_counter[0] += 1
                current_chunk = [(sentence, sent_start_pos)]
                current_tokens = sent_tokens
            else:
                current_chunk.append((sentence, sent_start_pos))
                current_tokens += separator_tokens + sent_tokens

        if current_chunk:
            # Calculate start and end line from original positions
            first_sent_pos = current_chunk[0][1]
            last_sent_pos = current_chunk[-1][1] + len(current_chunk[-1][0])
            chunk_content = para[first_sent_pos:last_sent_pos].strip()
            chunk_start_line = para_start_line + self._find_line_offset(para, first_sent_pos)
            chunk_end_line = para_start_line + self._find_line_offset(para, last_sent_pos)
            actual_tokens = self.count_tokens(chunk_content)

            # P0 FIX: Validate final chunk doesn't exceed max_tokens
            if actual_tokens > self.max_tokens:
                # Use _split_oversized_block for any oversized final chunk
                yield from self._split_oversized_block(
                    chunk_content, section_title, doc_path, chunk_counter,
                    chunk_start_line, first_sent_pos
                )
            else:
                yield DocumentChunk(
                    content=chunk_content,
                    section_title=section_title,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    token_count=actual_tokens,
                    document_path=doc_path,
                    chunk_index=chunk_counter[0],
                )
                chunk_counter[0] += 1

    def _split_oversized_block(
        self,
        block: str,
        section_title: str,
        doc_path: str,
        chunk_counter: list[int],
        block_start_line: int,
        block_start_pos: int,
    ) -> Iterator[DocumentChunk]:
        """Split an oversized block (no punctuation) by newlines.

        Handles code blocks, ASCII diagrams, data structures, etc. that don't
        have sentence-ending punctuation (.!?). Splits by newline boundaries
        to respect max_tokens.

        Args:
            block: Block text to split
            section_title: Title of containing section
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
            block_start_line: Starting line in parent paragraph
            block_start_pos: Starting position in parent paragraph (for line tracking)
        """
        lines = block.split("\n")
        current_lines: list[str] = []
        current_tokens = 0
        current_line_offset = 0

        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line)

            # Even single lines can exceed max_tokens (very long code lines)
            if line_tokens > self.max_tokens:
                # Flush current chunk first with validation
                if current_lines:
                    chunk_content = "\n".join(current_lines)
                    actual_tokens = self.count_tokens(chunk_content)
                    chunk_start_line = block_start_line + current_line_offset

                    # Validate chunk doesn't exceed max_tokens
                    if actual_tokens > self.max_tokens:
                        # Remove lines one by one until it fits
                        while actual_tokens > self.max_tokens and len(current_lines) > 1:
                            current_lines.pop()
                            chunk_content = "\n".join(current_lines)
                            actual_tokens = self.count_tokens(chunk_content)
                            chunk_start_line = block_start_line + current_line_offset

                        if actual_tokens > self.max_tokens and current_lines:
                            # Single line still exceeds, use character split
                            single_line = current_lines[0]
                            target_chars = self.max_tokens * self.APPROX_CHARS_PER_TOKEN
                            for start in range(0, len(single_line), target_chars):
                                segment = single_line[start:start + target_chars]
                                segment, segment_tokens = self._reduce_segment_to_fit(segment)
                                yield DocumentChunk(
                                    content=segment,
                                    section_title=section_title,
                                    start_line=chunk_start_line,
                                    end_line=chunk_start_line,
                                    token_count=segment_tokens,
                                    document_path=doc_path,
                                    chunk_index=chunk_counter[0],
                                )
                                chunk_counter[0] += 1
                        else:
                            yield DocumentChunk(
                                content=chunk_content,
                                section_title=section_title,
                                start_line=chunk_start_line,
                                end_line=chunk_start_line + len(current_lines) - 1,
                                token_count=actual_tokens,
                                document_path=doc_path,
                                chunk_index=chunk_counter[0],
                            )
                            chunk_counter[0] += 1
                    else:
                        yield DocumentChunk(
                            content=chunk_content,
                            section_title=section_title,
                            start_line=chunk_start_line,
                            end_line=chunk_start_line + len(current_lines) - 1,
                            token_count=actual_tokens,
                            document_path=doc_path,
                            chunk_index=chunk_counter[0],
                        )
                        chunk_counter[0] += 1
                    current_lines = []
                    current_tokens = 0

                # Single oversized line: split by character approximation with validation
                target_chars = self.max_tokens * self.APPROX_CHARS_PER_TOKEN

                for start in range(0, len(line), target_chars):
                    segment = line[start:start + target_chars]
                    # P1 FIX: Use helper method for segment reduction
                    segment, segment_tokens = self._reduce_segment_to_fit(segment)

                    chunk_start_line = block_start_line + i
                    yield DocumentChunk(
                        content=segment,
                        section_title=section_title,
                        start_line=chunk_start_line,
                        end_line=chunk_start_line,
                        token_count=segment_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_counter[0],
                    )
                    chunk_counter[0] += 1

                current_line_offset = i + 1
                continue

            # Account for separator tokens (newline) between lines
            # When lines are joined, each newline adds ~1 token
            separator_tokens = 1 if current_lines else 0  # Newline between lines
            effective_tokens = current_tokens + separator_tokens + line_tokens

            if effective_tokens > self.max_tokens and current_lines:
                # Flush current chunk
                chunk_content = "\n".join(current_lines)
                actual_tokens = self.count_tokens(chunk_content)
                chunk_start_line = block_start_line + current_line_offset

                # P0 FIX: Validate chunk doesn't exceed max_tokens
                if actual_tokens > self.max_tokens:
                    # Remove lines one by one until it fits
                    while actual_tokens > self.max_tokens and len(current_lines) > 1:
                        current_lines.pop()
                        chunk_content = "\n".join(current_lines)
                        actual_tokens = self.count_tokens(chunk_content)
                    # If still exceeds, use character-based split
                    if actual_tokens > self.max_tokens:
                        single_line = current_lines[0]
                        target_chars = self.max_tokens * self.APPROX_CHARS_PER_TOKEN
                        for start in range(0, len(single_line), target_chars):
                            segment = single_line[start:start + target_chars]
                            # P1 FIX: Use helper method for segment reduction
                            segment, segment_tokens = self._reduce_segment_to_fit(segment)
                            yield DocumentChunk(
                                content=segment,
                                section_title=section_title,
                                start_line=chunk_start_line,
                                end_line=chunk_start_line,
                                token_count=segment_tokens,
                                document_path=doc_path,
                                chunk_index=chunk_counter[0],
                            )
                            chunk_counter[0] += 1
                    else:
                        yield DocumentChunk(
                            content=chunk_content,
                            section_title=section_title,
                            start_line=chunk_start_line,
                            end_line=chunk_start_line + len(current_lines) - 1,
                            token_count=actual_tokens,
                            document_path=doc_path,
                            chunk_index=chunk_counter[0],
                        )
                        chunk_counter[0] += 1
                else:
                    yield DocumentChunk(
                        content=chunk_content,
                        section_title=section_title,
                        start_line=chunk_start_line,
                        end_line=chunk_start_line + len(current_lines) - 1,
                        token_count=actual_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_counter[0],
                    )
                    chunk_counter[0] += 1

                current_lines = [line]
                current_tokens = line_tokens
                current_line_offset = i
            else:
                if not current_lines:
                    current_line_offset = i
                current_lines.append(line)
                current_tokens += separator_tokens + line_tokens

        # Flush remaining
        if current_lines:
            chunk_content = "\n".join(current_lines)
            actual_tokens = self.count_tokens(chunk_content)
            chunk_start_line = block_start_line + current_line_offset
            chunk_end_line = chunk_start_line + len(current_lines) - 1

            # P0 FIX: Validate final chunk doesn't exceed max_tokens
            if actual_tokens > self.max_tokens:
                # Remove lines one by one until it fits
                while actual_tokens > self.max_tokens and len(current_lines) > 1:
                    current_lines.pop()
                    chunk_content = "\n".join(current_lines)
                    actual_tokens = self.count_tokens(chunk_content)
                    chunk_end_line = chunk_start_line + len(current_lines) - 1

                # If still exceeds, use character-based split
                if actual_tokens > self.max_tokens and current_lines:
                    single_line = current_lines[0]
                    target_chars = self.max_tokens * self.APPROX_CHARS_PER_TOKEN
                    for start in range(0, len(single_line), target_chars):
                        segment = single_line[start:start + target_chars]
                        # P1 FIX: Use helper method for segment reduction
                        segment, segment_tokens = self._reduce_segment_to_fit(segment)
                        yield DocumentChunk(
                            content=segment,
                            section_title=section_title,
                            start_line=chunk_start_line,
                            end_line=chunk_start_line,
                            token_count=segment_tokens,
                            document_path=doc_path,
                            chunk_index=chunk_counter[0],
                        )
                        chunk_counter[0] += 1
                else:
                    yield DocumentChunk(
                        content=chunk_content,
                        section_title=section_title,
                        start_line=chunk_start_line,
                        end_line=chunk_end_line,
                        token_count=actual_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_counter[0],
                    )
                    chunk_counter[0] += 1
            else:
                yield DocumentChunk(
                    content=chunk_content,
                    section_title=section_title,
                    start_line=chunk_start_line,
                    end_line=chunk_end_line,
                    token_count=actual_tokens,
                    document_path=doc_path,
                    chunk_index=chunk_counter[0],
                )
                chunk_counter[0] += 1

    def _chunk_plain_text(
        self, content: str, doc_path: str, chunk_counter: list[int]
    ) -> Iterator[DocumentChunk]:
        """Chunk plain text without sections with line tracking.

        Uses 0-indexed line numbers for consistency with _split_by_paragraphs().

        Args:
            content: Document content to chunk
            doc_path: Document path
            chunk_counter: Mutable counter [current_index] for tracking unique chunk indices
        """
        # P1 FIX: Check for whitespace-only content (consistent with _chunk_section)
        if not content or not content.strip():
            return

        paragraphs = content.split("\n\n")
        current_chunk: list[str] = []
        current_tokens = 0  # Tokens including separator tokens
        current_line_start = 0  # P0 FIX: 0-indexed (was 1-indexed)
        content_position = 0  # Track position in original content

        for para in paragraphs:
            # Skip empty paragraphs
            if not para or not para.strip():
                content_position += len(para) + 2
                continue

            para_tokens = self.count_tokens(para)

            # P0 FIX: 0-indexed line calculation (removed +1)
            para_start_line = self._find_line_offset(content, content_position)

            # Handle oversized paragraphs (same as _split_by_paragraphs)
            if para_tokens > self.max_tokens:
                # Flush current chunk first with validation
                if current_chunk:
                    chunk_content = "\n\n".join(current_chunk)
                    actual_tokens = self.count_tokens(chunk_content)

                    # P0 FIX: Validate before yielding
                    if actual_tokens > self.max_tokens:
                        yield from self._split_oversized_block(
                            chunk_content, "", doc_path, chunk_counter,
                            current_line_start, 0
                        )
                    else:
                        yield DocumentChunk(
                            content=chunk_content,
                            section_title="",
                            start_line=current_line_start,
                            end_line=current_line_start + self._count_lines(chunk_content) - 1,
                            token_count=actual_tokens,
                            document_path=doc_path,
                            chunk_index=chunk_counter[0],
                        )
                        chunk_counter[0] += 1
                    current_chunk = []
                    current_tokens = 0

                # Split oversized paragraph by sentences
                yield from self._split_long_paragraph(
                    para, "", doc_path, chunk_counter, para_start_line, content, content_position
                )

                # Update position and continue
                content_position += len(para) + 2
                continue

            # Account for separator tokens in the split decision
            separator_tokens = 2 if current_chunk else 0  # "\n\n" tokens
            effective_tokens = current_tokens + separator_tokens + para_tokens

            if effective_tokens > self.max_tokens and current_chunk:
                # Flush current chunk with validation
                chunk_content = "\n\n".join(current_chunk)
                actual_tokens = self.count_tokens(chunk_content)

                # P0 FIX: Validate before yielding
                if actual_tokens > self.max_tokens:
                    yield from self._split_oversized_block(
                        chunk_content, "", doc_path, chunk_counter,
                        current_line_start, 0
                    )
                else:
                    yield DocumentChunk(
                        content=chunk_content,
                        section_title="",
                        start_line=current_line_start,
                        end_line=current_line_start + self._count_lines(chunk_content) - 1,
                        token_count=actual_tokens,
                        document_path=doc_path,
                        chunk_index=chunk_counter[0],
                    )
                    chunk_counter[0] += 1

                # P1 FIX: Add overlap handling (consistent with _split_by_paragraphs)
                if self.overlap_tokens > 0 and current_chunk:
                    overlap_para = current_chunk[-1]
                    overlap_tok = self.count_tokens(overlap_para)
                    sep_tokens = 2  # "\n\n" separator
                    combined_tokens = overlap_tok + sep_tokens + para_tokens

                    if overlap_tok <= self.overlap_tokens and combined_tokens <= self.max_tokens:
                        current_chunk = [overlap_para, para]
                        current_tokens = overlap_tok + sep_tokens + para_tokens
                        overlap_len = len(overlap_para) + 2
                        content_pos_for_overlap = content_position - overlap_len
                        current_line_start = self._find_line_offset(
                            content, max(0, content_pos_for_overlap)
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
                    current_line_start = para_start_line
                current_chunk.append(para)
                current_tokens += separator_tokens + para_tokens

            content_position += len(para) + 2  # +2 for "\n\n" separator

        # P0 FIX: Final flush with validation
        if current_chunk:
            chunk_content = "\n\n".join(current_chunk)
            actual_tokens = self.count_tokens(chunk_content)

            if actual_tokens > self.max_tokens:
                yield from self._split_oversized_block(
                    chunk_content, "", doc_path, chunk_counter,
                    current_line_start, 0
                )
            else:
                total_lines = self._count_lines(content)
                yield DocumentChunk(
                    content=chunk_content,
                    section_title="",
                    start_line=current_line_start,
                    end_line=min(current_line_start + self._count_lines(chunk_content) - 1, total_lines - 1),
                    token_count=actual_tokens,
                    document_path=doc_path,
                    chunk_index=chunk_counter[0],
                )
                chunk_counter[0] += 1


__all__ = ["DocumentChunk", "SemanticDocumentChunker"]