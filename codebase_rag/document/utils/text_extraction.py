"""Text extraction utilities.

Helper functions for extracting text content from various sources.
"""

from __future__ import annotations

import re
from pathlib import Path


def extract_text_content(content: str, max_length: int | None = None) -> str:
    """
    Extract and clean text content.

    Args:
        content: Raw content string
        max_length: Optional maximum length to return

    Returns:
        Cleaned text content
    """
    # Remove excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", content)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    # Strip leading/trailing whitespace
    cleaned = cleaned.strip()

    # Apply length limit if specified
    if max_length and len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip() + "..."

    return cleaned


def extract_title_from_content(content: str, file_type: str) -> str | None:
    """
    Extract document title from content.

    Args:
        content: Document content
        file_type: File extension (.md, .rst, .txt, etc.)

    Returns:
        Title string or None if not found
    """
    if file_type in {".md", ".markdown"}:
        # Look for first heading
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

    elif file_type in {".rst", ".rest"}:
        # Look for underline-style header
        match = re.search(r"^([=\-~^\"']+)\n(.+)\n\1$", content, re.MULTILINE)
        if match:
            return match.group(2).strip()

    elif file_type in {".txt", ".text"}:
        # Use first non-empty line as title
        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if line and len(line) < 100:  # Reasonable title length
                return line

    return None


def count_words(content: str) -> int:
    """Count words in content."""
    # Split on whitespace and filter empty strings
    words = [w for w in content.split() if w]
    return len(words)


def extract_summary(content: str, max_sentences: int = 3) -> str:
    """
    Extract a brief summary from content.

    Uses first N sentences as summary.

    Args:
        content: Document content
        max_sentences: Maximum sentences to include

    Returns:
        Summary text
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", content)

    # Take first N non-empty sentences
    summary_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            summary_sentences.append(sentence)
            if len(summary_sentences) >= max_sentences:
                break

    return " ".join(summary_sentences)


def is_binary_content(content: str) -> bool:
    """
    Check if content appears to be binary/non-text.

    Args:
        content: Content to check

    Returns:
        True if content appears binary
    """
    # Check for high ratio of non-printable characters
    printable_count = sum(1 for c in content if c.isprintable() or c in "\n\r\t")
    total_count = len(content)

    if total_count == 0:
        return False

    # If less than 90% printable, likely binary
    return printable_count / total_count < 0.9


def normalize_whitespace(content: str) -> str:
    """Normalize whitespace in content."""
    # Replace tabs with spaces
    content = content.replace("\t", "    ")

    # Normalize line endings
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing whitespace on lines
    lines = [line.rstrip() for line in content.split("\n")]
    content = "\n".join(lines)

    # Collapse multiple blank lines to max 2
    content = re.sub(r"\n{3,}", "\n\n", content)

    return content.strip()


__all__ = [
    "extract_text_content",
    "extract_title_from_content",
    "count_words",
    "extract_summary",
    "is_binary_content",
    "normalize_whitespace",
]