"""Cross-graph reference resolution for document-to-code references.

Provides robust resolution of code references from the document graph
in the code graph, with partial matching fallback and error handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from ..services import QueryProtocol


def _coerce_int(value: object, default: int = 0) -> int:
    return value if isinstance(value, int) else default


@dataclass
class ResolvedReference:
    """A successfully resolved code reference."""

    qualified_name: str
    node_type: str
    file_path: str
    line_range: tuple[int, int] | None
    found: bool = True


@dataclass
class UnresolvedReference:
    """A reference that could not be resolved."""

    qualified_name: str
    reason: str


@dataclass
class ReferenceResolutionResult:
    """Result of reference resolution."""

    resolved: list[ResolvedReference]
    unresolved: list[UnresolvedReference]
    warnings: list[str] = field(default_factory=list)


class CrossGraphReferenceResolver:
    """Resolves code references from document graph in code graph."""

    def __init__(self, code_graph: QueryProtocol | None) -> None:
        self.code_graph = code_graph

    def resolve_references(
        self,
        references: list[str],
        max_references: int = 20,
    ) -> ReferenceResolutionResult:
        """Resolve code references with robust error handling."""
        if not self.code_graph:
            return ReferenceResolutionResult(
                resolved=[],
                unresolved=[
                    UnresolvedReference(ref, "graph_unavailable")
                    for ref in references
                ],
                warnings=["Code graph not available for reference resolution"],
            )

        if not references:
            return ReferenceResolutionResult(resolved=[], unresolved=[])

        refs_to_resolve = references[:max_references]
        resolved: list[ResolvedReference] = []
        unresolved: list[UnresolvedReference] = []

        try:
            rows = self.code_graph.fetch_all(
                """
                MATCH (n)
                WHERE n.qualified_name IN $qualified_names
                RETURN n.qualified_name AS qualified_name,
                       coalesce(labels(n)[0], 'Unknown') AS node_type,
                       coalesce(n.path, 'unknown') AS file_path,
                       n.start_line AS start_line,
                       n.end_line AS end_line
                """,
                {"qualified_names": refs_to_resolve},
            )

            found_names = {row["qualified_name"] for row in rows}

            for row in rows:
                start_line = _coerce_int(row.get("start_line"), 0)
                end_line = _coerce_int(row.get("end_line"), start_line)
                line_range = None
                if start_line > 0 and end_line > 0:
                    line_range = (start_line, end_line)

                resolved.append(
                    ResolvedReference(
                        qualified_name=str(row.get("qualified_name", "")),
                        node_type=str(row.get("node_type", "Unknown")),
                        file_path=str(row.get("file_path") or "unknown"),
                        line_range=line_range,
                    )
                )

            for ref in refs_to_resolve:
                if ref not in found_names:
                    partial_match = self._try_partial_match(ref)
                    if partial_match:
                        resolved.append(partial_match)
                    else:
                        unresolved.append(UnresolvedReference(ref, "not_found"))

        except Exception as e:
            logger.warning(f"Reference resolution failed: {e}")
            return ReferenceResolutionResult(
                resolved=[],
                unresolved=[
                    UnresolvedReference(ref, "resolution_error")
                    for ref in refs_to_resolve
                ],
                warnings=[f"Reference resolution error: {e}"],
            )

        return ReferenceResolutionResult(
            resolved=resolved,
            unresolved=unresolved,
        )

    def _try_partial_match(self, qualified_name: str) -> ResolvedReference | None:
        """Try to find a partial match for a qualified name."""
        parts = qualified_name.split(".")
        if len(parts) < 2:
            return None

        short_name = parts[-1]

        try:
            if self.code_graph is None:
                return None

            rows = self.code_graph.fetch_all(
                """
                MATCH (n)
                WHERE n.name = $short_name
                RETURN n.qualified_name AS qualified_name,
                       coalesce(labels(n)[0], 'Unknown') AS node_type,
                       coalesce(n.path, 'unknown') AS file_path,
                       n.start_line AS start_line,
                       n.end_line AS end_line
                LIMIT 1
                """,
                {"short_name": short_name},
            )

            if rows:
                row = rows[0]
                start_line = _coerce_int(row.get("start_line"), 0)
                end_line = _coerce_int(row.get("end_line"), start_line)
                line_range = None
                if start_line > 0 and end_line > 0:
                    line_range = (start_line, end_line)

                return ResolvedReference(
                    qualified_name=str(row.get("qualified_name", "")),
                    node_type=str(row.get("node_type", "Unknown")),
                    file_path=str(row.get("file_path") or "unknown"),
                    line_range=line_range,
                )
        except Exception:
            pass

        return None
