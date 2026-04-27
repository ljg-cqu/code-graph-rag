from __future__ import annotations

import re

from ..config import settings


def _split_chunk(content: str, max_chars: int | None = None) -> list[str]:
    """Split chunk into sub-chunks by heading, paragraph, or sentence."""
    if max_chars is None:
        max_chars = max(1000, settings.DOC_CHUNK_SIZE // 2)

    min_size = settings.DOC_CONCEPT_MIN_CHUNK_SIZE

    # Prefer Markdown heading boundaries
    headings = list(re.finditer(r"(?:^|\n)#{1,6}\s", content))
    if len(headings) > 1:
        splits = []
        for i, m in enumerate(headings):
            start = m.start()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            splits.append(content[start:end].strip())
        return [s for s in splits if len(s) > min_size]

    # Fallback 1: paragraph boundaries
    paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > min_size]
    batches = []
    current = ""
    for p in paragraphs:
        if len(p) >= max_chars:
            if current:
                batches.append(current)
                current = ""
            batches.append(p[:max_chars])
        elif len(current) + len(p) < max_chars:
            current += "\n\n" + p if current else p
        else:
            if current:
                batches.append(current)
            current = p
    if current:
        batches.append(current)
    if batches:
        return batches

    # Fallback 2: sentence boundaries (avoid mid-sentence truncation)
    sentences = re.split(r"(?<=[.!?])\s+", content)
    batches = []
    current = ""
    for s in sentences:
        if len(s) >= max_chars:
            if current:
                batches.append(current)
                current = ""
            batches.append(s[:max_chars])
        elif len(current) + len(s) < max_chars:
            current += " " + s if current else s
        else:
            if current:
                batches.append(current)
            current = s
    if current:
        batches.append(current)
    return batches or [content[:max_chars]]


__all__ = ["_split_chunk"]
