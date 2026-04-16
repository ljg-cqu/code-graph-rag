from .constants import NodeLabel
from .types_defs import (
    NODE_SCHEMAS,
    RELATIONSHIP_SCHEMAS,
    NodeSchema,
    RelationshipSchema,
)


def _format_node_schema(schema: NodeSchema) -> str:
    return f"- {schema.label}: {schema.properties}"


def _format_relationship_schema(schema: RelationshipSchema) -> str:
    sources = "|".join(str(s) for s in schema.sources)
    targets = "|".join(str(t) for t in schema.targets)
    if len(schema.sources) > 1:
        sources = f"({sources})"
    if len(schema.targets) > 1:
        targets = f"({targets})"
    return f"- {sources} -[:{schema.rel_type}]-> {targets}"


_DOCUMENT_GRAPH_LABELS = frozenset(
    {
        NodeLabel.DOCUMENT,
        NodeLabel.SECTION,
        NodeLabel.CHUNK,
    }
)


def _build_node_labels_section(node_schemas: tuple[NodeSchema, ...]) -> str:
    lines = ["Node Labels and Their Key Properties:"]
    lines.extend(_format_node_schema(schema) for schema in node_schemas)
    return "\n".join(lines)


def _build_relationships_section(
    relationship_schemas: tuple[RelationshipSchema, ...],
) -> str:
    lines = ["Relationships (source)-[REL_TYPE]->(target):"]
    lines.extend(
        _format_relationship_schema(schema) for schema in relationship_schemas
    )
    return "\n".join(lines)


def _is_code_graph_relationship(schema: RelationshipSchema) -> bool:
    return not (
        _DOCUMENT_GRAPH_LABELS.intersection(schema.sources)
        or _DOCUMENT_GRAPH_LABELS.intersection(schema.targets)
    )


CODE_GRAPH_NODE_SCHEMAS: tuple[NodeSchema, ...] = tuple(
    schema for schema in NODE_SCHEMAS if schema.label not in _DOCUMENT_GRAPH_LABELS
)
CODE_GRAPH_RELATIONSHIP_SCHEMAS: tuple[RelationshipSchema, ...] = tuple(
    schema for schema in RELATIONSHIP_SCHEMAS if _is_code_graph_relationship(schema)
)


def build_graph_schema_text(
    node_schemas: tuple[NodeSchema, ...] = NODE_SCHEMAS,
    relationship_schemas: tuple[RelationshipSchema, ...] = RELATIONSHIP_SCHEMAS,
) -> str:
    return f"""{_build_node_labels_section(node_schemas)}

{_build_relationships_section(relationship_schemas)}"""


GRAPH_SCHEMA_DEFINITION = build_graph_schema_text()
CODE_GRAPH_SCHEMA_DEFINITION = build_graph_schema_text(
    CODE_GRAPH_NODE_SCHEMAS,
    CODE_GRAPH_RELATIONSHIP_SCHEMAS,
)
