"""Concept consolidation for cross-chunk deduplication.

Provides ConceptConsolidator which accumulates concepts from multiple
chunks/documents and resolves conflicts via confidence-weighted plurality voting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConsolidatedConcept:
    """A concept after cross-source consolidation."""

    qualified_name: str
    workspace: str
    name: str
    aliases: list[str]
    type: str
    definition: str
    confidence: float
    entity_category: str
    entity_subtype: str | None
    entity_emoji: str
    context: str
    source_chunks: list[str] = field(default_factory=list)


class ConceptConsolidator:
    """Accumulate concepts from multiple chunks and resolve conflicts.

    Conflict resolution uses confidence-weighted plurality voting for
    entity_category, with highest-confidence source winning for definition,
    context, and entity_subtype.
    """

    def __init__(self) -> None:
        self._concepts: dict[str, list[dict]] = {}

    def add(self, node: dict, chunk_qn: str) -> None:
        """Add a concept node from a single extraction source."""
        qn = str(node["qualified_name"])
        self._concepts.setdefault(qn, []).append({**node, "_chunk_qn": chunk_qn})

    def consolidate(self) -> list[dict]:
        """Resolve conflicts and return deduplicated concept nodes.

        Returns:
            List of concept node dicts sorted by qualified_name.
        """
        resolved: list[dict] = []
        for qn, sources in self._concepts.items():
            resolved.append(self._resolve_one(qn, sources))
        return sorted(resolved, key=lambda n: str(n["qualified_name"]))

    def _resolve_one(self, qn: str, sources: list[dict]) -> dict:
        """Resolve a single concept from multiple sources."""
        workspaces = {str(s["workspace"]) for s in sources}
        if len(workspaces) > 1:
            raise ValueError(
                f"Concept {qn} has conflicting workspaces: {workspaces}"
            )
        workspace = workspaces.pop()

        names = {str(s["name"]) for s in sources}
        if len(names) > 1:
            raise ValueError(
                f"Concept {qn} has conflicting names: {names}"
            )
        name = names.pop()

        # Union of aliases
        all_aliases: set[str] = set()
        for s in sources:
            aliases = s.get("aliases") or []
            all_aliases.update(str(a) for a in aliases)
        aliases = sorted(all_aliases)

        # Confidence-weighted plurality vote for entity_category
        category_votes: dict[str, float] = {}
        for s in sources:
            cat = str(s.get("entity_category") or s.get("type") or "ABSTRACT_CONCEPT")
            conf = float(s.get("confidence") or 0.0)
            category_votes[cat] = category_votes.get(cat, 0.0) + conf

        # Tie-breaker: highest single-source confidence for the tied category
        best_category = max(
            category_votes.keys(),
            key=lambda cat: (
                category_votes[cat],
                max(
                    float(s.get("confidence") or 0.0)
                    for s in sources
                    if str(s.get("entity_category") or s.get("type") or "ABSTRACT_CONCEPT") == cat
                ),
            ),
        )

        # Pick the winning source (highest confidence among sources with winning category)
        winning_sources = [
            s for s in sources
            if str(s.get("entity_category") or s.get("type") or "ABSTRACT_CONCEPT") == best_category
        ]
        best_source = max(
            winning_sources,
            key=lambda s: (
                float(s.get("confidence") or 0.0),
                len(str(s.get("definition") or "")),
            ),
        )

        # Confidence: max across all sources
        max_confidence = max(float(s.get("confidence") or 0.0) for s in sources)

        # Definition: from best_source (highest confidence for winning category; tie -> longest)
        definition = str(best_source.get("definition") or "")

        # Context: from best_source (same rule as definition)
        context = str(best_source.get("context") or "")

        # Subtype: from best_source
        entity_subtype = best_source.get("entity_subtype")
        if entity_subtype is not None:
            entity_subtype = str(entity_subtype) if entity_subtype else None

        # Emoji: server-resolved from winning category
        from codebase_rag.constants import ENTITY_CATEGORY_EMOJI_MAP

        entity_emoji = ENTITY_CATEGORY_EMOJI_MAP.get(best_category, "💡")

        # Source chunks
        source_chunks = sorted({str(s["_chunk_qn"]) for s in sources})

        return {
            "qualified_name": qn,
            "workspace": workspace,
            "name": name,
            "aliases": aliases,
            "type": best_category,
            "definition": definition,
            "confidence": max_confidence,
            "entity_category": best_category,
            "entity_subtype": entity_subtype,
            "entity_emoji": entity_emoji,
            "context": context,
            "source_chunks": source_chunks,
        }


__all__ = ["ConsolidatedConcept", "ConceptConsolidator"]
