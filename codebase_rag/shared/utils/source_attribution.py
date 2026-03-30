"""Source attribution utility.

Labels query results with their source (code vs document).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SourceType(StrEnum):
    """Source of query result."""

    CODE_GRAPH = "code_graph"
    DOCUMENT_GRAPH = "document_graph"
    MERGED = "merged"  # Combined from both sources
    UNKNOWN = "unknown"


@dataclass
class SourceAttribution:
    """Attribution for a query result."""

    source: SourceType
    source_path: str | None  # File path if applicable
    source_node_type: str | None  # Node label (Function, Class, Section, etc.)
    source_node_name: str | None  # Node name/title
    confidence: float  # 0.0 to 1.0


def get_source_label(node: dict) -> SourceAttribution:
    """
    Determine source attribution for a graph node.

    Args:
        node: Node data from graph query

    Returns:
        SourceAttribution with source metadata
    """
    # Check node labels/type
    labels = node.get("_labels", [])
    if isinstance(labels, str):
        labels = [labels]

    # Document graph nodes
    doc_labels = {"Document", "Section", "Chunk"}
    if any(label in doc_labels for label in labels):
        return SourceAttribution(
            source=SourceType.DOCUMENT_GRAPH,
            source_path=node.get("path") or node.get("document_path"),
            source_node_type=_get_primary_label(labels, doc_labels),
            source_node_name=node.get("title") or node.get("path"),
            confidence=1.0,
        )

    # Code graph nodes
    code_labels = {"Function", "Class", "Method", "Module", "File"}
    if any(label in code_labels for label in labels):
        return SourceAttribution(
            source=SourceType.CODE_GRAPH,
            source_path=node.get("path"),
            source_node_type=_get_primary_label(labels, code_labels),
            source_node_name=node.get("name") or node.get("qualified_name"),
            confidence=1.0,
        )

    # Unknown
    return SourceAttribution(
        source=SourceType.UNKNOWN,
        source_path=node.get("path"),
        source_node_type=None,
        source_node_name=None,
        confidence=0.0,
    )


def _get_primary_label(labels: list[str], target_labels: set[str]) -> str | None:
    """Get the primary label from a set of target labels."""
    for label in labels:
        if label in target_labels:
            return label
    return None


def label_results_with_source(
    results: list[dict],
    source: SourceType,
) -> list[dict]:
    """
    Add source labels to query results.

    Args:
        results: Query results
        source: Known source type

    Returns:
        Results with source label added
    """
    labeled = []
    for result in results:
        labeled_result = dict(result)
        labeled_result["source"] = source.value
        labeled_result["source_type"] = source.value
        labeled.append(labeled_result)
    return labeled


def merge_results_with_attribution(
    code_results: list[dict],
    doc_results: list[dict],
) -> list[dict]:
    """
    Merge results from both graphs with attribution.

    Args:
        code_results: Results from code graph
        doc_results: Results from document graph

    Returns:
        Combined results with source labels
    """
    merged = []

    # Label code results
    for result in code_results:
        labeled = dict(result)
        labeled["source"] = SourceType.CODE_GRAPH.value
        merged.append(labeled)

    # Label document results
    for result in doc_results:
        labeled = dict(result)
        labeled["source"] = SourceType.DOCUMENT_GRAPH.value
        merged.append(labeled)

    return merged


def format_source_summary(attribution: SourceAttribution) -> str:
    """
    Format source attribution as human-readable summary.

    Args:
        attribution: Source attribution

    Returns:
        Human-readable summary string
    """
    source_map = {
        SourceType.CODE_GRAPH: "Code",
        SourceType.DOCUMENT_GRAPH: "Doc",
        SourceType.MERGED: "Both",
        SourceType.UNKNOWN: "Unknown",
    }

    source_str = source_map.get(attribution.source, "Unknown")

    if attribution.source_node_type:
        node_type = attribution.source_node_type
    else:
        node_type = "node"

    if attribution.source_node_name:
        return f"{source_str}: {node_type} '{attribution.source_node_name}'"
    else:
        return f"{source_str}: {node_type}"


__all__ = [
    "SourceType",
    "SourceAttribution",
    "get_source_label",
    "label_results_with_source",
    "merge_results_with_attribution",
    "format_source_summary",
]