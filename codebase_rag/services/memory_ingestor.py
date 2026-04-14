from __future__ import annotations

from collections import defaultdict
from typing import Any

from codebase_rag import constants as cs
from codebase_rag.types_defs import PropertyValue


class MemoryIngestor:
    __slots__ = ("nodes", "relationships", "call_edges", "batch_size")

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.nodes: list[tuple[str, dict[str, PropertyValue]]] = []
        self.relationships: defaultdict[
            tuple[str, str, str, str, str], list[dict[str, Any]]
        ] = defaultdict(list)
        self.call_edges: list[dict[str, Any]] = []

    def ensure_node_batch(
        self, label: str, properties: dict[str, PropertyValue]
    ) -> None:
        self.nodes.append((label, properties))

    def ensure_relationship_batch(
        self,
        from_spec: tuple[str, str, PropertyValue],
        rel_type: str,
        to_spec: tuple[str, str, PropertyValue],
        properties: dict[str, PropertyValue] | None = None,
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
        list[tuple[str, dict[str, PropertyValue]]],
        dict[tuple[str, str, str, str, str], list[dict[str, Any]]],
    ]:
        return self.nodes, dict(self.relationships)
