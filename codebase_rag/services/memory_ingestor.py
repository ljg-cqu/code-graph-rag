from __future__ import annotations

from collections import defaultdict
from typing import Any

from codebase_rag import constants as cs
from codebase_rag.types_defs import PropertyDict, PropertyValue


class MemoryIngestor:
    __slots__ = ("nodes", "relationships", "call_edges", "batch_size")

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.nodes: list[tuple[str, PropertyDict]] = []
        self.relationships: defaultdict[
            tuple[str, str, str, str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        self.call_edges: list[dict[str, Any]] = []

    def ensure_node_batch(self, label: str, properties: PropertyDict) -> None:
        self.nodes.append((label, properties))

    def ensure_node(self, label: str, properties: PropertyDict) -> None:
        self.ensure_node_batch(label, properties)

    def ensure_edge(
        self,
        rel_type: str,
        from_identifier: str,
        to_identifier: str,
        properties: PropertyDict | None = None,
    ) -> None:
        # Simplified interface for graph_updater.py
        # Determine node label based on relationship type and identifier pattern
        def _get_label(identifier: str) -> str:
            # Simple heuristic: if identifier contains a dot and the part after
            # the last dot starts with lowercase, it might be a method
            # This is language-dependent but works for many cases
            if "." in identifier:
                # Check if it looks like a method (e.g., ClassName.methodName)
                parts = identifier.split(".")
                if len(parts) > 1 and parts[-1][0].islower():
                    return cs.NodeLabel.METHOD
            return cs.NodeLabel.FUNCTION

        if rel_type == cs.RelType.CALLS:
            # For CALLS relationships, try to determine if nodes are methods or functions
            from_label = _get_label(from_identifier)
            to_label = _get_label(to_identifier)
        else:
            from_label = cs.NodeLabel.FUNCTION
            to_label = cs.NodeLabel.FUNCTION

        self.ensure_relationship_batch(
            (from_label, cs.KEY_QUALIFIED_NAME, from_identifier),
            rel_type,
            (to_label, cs.KEY_QUALIFIED_NAME, to_identifier),
            properties,
        )

    def ensure_relationship_batch(
        self,
        from_spec: tuple[str, str, PropertyValue],
        rel_type: str,
        to_spec: tuple[str, str, PropertyValue],
        properties: PropertyDict | None = None,
    ) -> None:
        from_label, from_key, from_val = from_spec
        to_label, to_key, to_val = to_spec
        pattern = (from_label, from_key, rel_type, to_label, to_key)
        rel_data = {"from_val": from_val, "to_val": to_val, "props": properties or {}}
        self.relationships[pattern].append(rel_data)

        # Track call edges separately for easy retrieval
        if rel_type == cs.RelType.CALLS:
            self.call_edges.append(
                {
                    "from": from_val,
                    "to": to_val,
                    "line": properties.get("line") if properties else None,
                }
            )

    def get_call_edges(self) -> list[dict[str, Any]]:
        return self.call_edges

    def flush_all(self) -> None:
        pass

    def flush_nodes(self) -> None:
        pass

    def flush_relationships(self) -> None:
        pass

    def get_all_data(
        self,
    ) -> tuple[
        list[tuple[str, PropertyDict]],
        dict[tuple[str, str, str, str, str], list[dict[str, Any]]],
    ]:
        return self.nodes, dict(self.relationships)
