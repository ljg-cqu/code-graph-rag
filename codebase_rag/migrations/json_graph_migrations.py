"""Data model migrations for JSON graph quality improvements.

This module contains migration functions for:
- REL-001: Relationship type normalization (emoji to canonical)
- LABEL-001: Label normalization (remove space-containing labels)
- PROP-001: Property consolidation (remove redundant properties)
"""
from __future__ import annotations

from loguru import logger
import mgclient

from .. import constants as cs
from .. import logs as ls
from ..config import settings


# Complete mapping: old_type -> (canonical_label, verb, emoji)
RELATIONSHIP_TYPE_MAPPING = {
    # HIERARCHICAL
    "🌳 is a": ("HIERARCHICAL", "is-a", "🌳"),
    "🌳 is a variant of": ("HIERARCHICAL", "is-a-variant-of", "🌳"),
    "🌳 classifies as": ("HIERARCHICAL", "classifies-as", "🌳"),
    # COMPOSITIONAL
    "🧩 part of": ("COMPOSITIONAL", "part-of", "🧩"),
    "🧩 comprises": ("COMPOSITIONAL", "comprises", "🧩"),
    "🧩 built upon": ("COMPOSITIONAL", "built-upon", "🧩"),
    # CAUSAL
    "⚡ enables": ("CAUSAL", "enables", "⚡"),
    "⚡ operationalizes": ("CAUSAL", "operationalizes", "⚡"),
    "⚡ operationalized by": ("CAUSAL", "operationalized-by", "⚡"),
    "⚡ produces": ("CAUSAL", "produces", "⚡"),
    "⚡ generates": ("CAUSAL", "generates", "⚡"),
    "⚡ develops": ("CAUSAL", "develops", "⚡"),
    "⚡ reinforces": ("CAUSAL", "reinforces", "⚡"),
    "⚡ facilitates": ("CAUSAL", "facilitates", "⚡"),
    "⚡ triggers": ("CAUSAL", "triggers", "⚡"),
    "⚡ leads to": ("CAUSAL", "leads-to", "⚡"),
    "⚡ affects": ("CAUSAL", "affects", "⚡"),
    "⚡ benefits from": ("CAUSAL", "benefits-from", "⚡"),
    # ATTRIBUTIVE
    "💭 supports": ("ATTRIBUTIVE", "supports", "💭"),
    "💭 implements": ("ATTRIBUTIVE", "implements", "💭"),
    # CAUSAL
    "⚡ prevents": ("CAUSAL", "prevents", "⚡"),
    "⚡ threatens": ("CAUSAL", "threatens", "⚡"),
    "⚡ influences": ("CAUSAL", "influences", "⚡"),
    "⚡ validates": ("CAUSAL", "validates", "⚡"),
    "⚡ causes": ("CAUSAL", "causes", "⚡"),
    "⚡ enforced by": ("CAUSAL", "enforced-by", "⚡"),
    "⚡ informs": ("CAUSAL", "informs", "⚡"),
    # CONTEXTUAL
    "🎯 interacts with": ("CONTEXTUAL", "interacts-with", "🎯"),
    "🎯 informs": ("CONTEXTUAL", "informs", "🎯"),
    "🎯 provides context for": ("CONTEXTUAL", "provides-context-for", "🎯"),
    "🎯 serves": ("CONTEXTUAL", "serves", "🎯"),
    "🎯 sets scope for": ("CONTEXTUAL", "sets-scope-for", "🎯"),
    "🎯 classifies": ("CONTEXTUAL", "classifies", "🎯"),
    "🎯 provides": ("CONTEXTUAL", "provides", "🎯"),
    "🎯 frames": ("CONTEXTUAL", "frames", "🎯"),
    "🎯 defines": ("CONTEXTUAL", "defines", "🎯"),
    "🎯 situated in": ("CONTEXTUAL", "situated-in", "🎯"),
    "🎯 measures": ("CONTEXTUAL", "measures", "🎯"),
    # ATTRIBUTIVE
    "💭 exhibits": ("ATTRIBUTIVE", "exhibits", "💭"),
    "💭 characterized by": ("ATTRIBUTIVE", "characterized-by", "💭"),
    "💭 characterizes": ("ATTRIBUTIVE", "characterizes", "💭"),
    "💭 possesses": ("ATTRIBUTIVE", "possesses", "💭"),
    # COMPARATIVE
    "⚖️ differs from": ("COMPARATIVE", "differs-from", "⚖️"),
    "⚖️ collaborates with": ("COMPARATIVE", "collaborates-with", "⚖️"),
    "⚖️ contrasts with": ("COMPARATIVE", "contrasts-with", "⚖️"),
    "⚖️ similar to": ("COMPARATIVE", "similar-to", "⚖️"),
    # SEQUENTIAL
    "⏩ precedes": ("SEQUENTIAL", "precedes", "⏩"),
    "⏩ follows": ("SEQUENTIAL", "follows", "⏩"),
    "⏩ validates": ("SEQUENTIAL", "validates", "⏩"),
    "⏩ leads to": ("SEQUENTIAL", "leads-to", "⏩"),
    # ANALOGICAL
    "🌉 maps to": ("ANALOGICAL", "maps-to", "🌉"),
    "🌉 analogous to": ("ANALOGICAL", "analogous-to", "🌉"),
    "🌉 corresponds to": ("ANALOGICAL", "corresponds-to", "🌉"),
}

# Types with spaces that need label normalization
LABELS_TO_NORMALIZE = {
    "Dynamic Construct": "DynamicConstruct",
    "Governance Construct": "GovernanceConstruct",
    "Abstract Class": "AbstractClass",
    "External Metaphor": "ExternalMetaphor",
    "Emergent State": "EmergentState",
    "Safety Boundary": "SafetyBoundary",
    "Progression Stage": "ProgressionStage",
    "Foundational Construct": "FoundationalConstruct",
    "Framework Component": "FrameworkComponent",
    "Regulatory Framework": "RegulatoryFramework",
    "Role Archetype": "RoleArchetype",
    "Planning Tool": "PlanningTool",
    "Risk Construct": "RiskConstruct",
    "Monitoring Construct": "MonitoringConstruct",
    "Classification Construct": "ClassificationConstruct",
    "Resource Construct": "ResourceConstruct",
    "Meta Construct": "MetaConstruct",
    "Decision Construct": "DecisionConstruct",
    "Documentation Construct": "DocumentationConstruct",
    "Methodological Framework": "MethodologicalFramework",
    "Governance Rule": "GovernanceRule",
    "Quality Standard": "QualityStandard",
    "Practice Exercise": "PracticeExercise",
    "Structural Construct": "StructuralConstruct",
    "Mental Model": "MentalModel",
}

VALID_CATEGORIES = frozenset({
    "HIERARCHICAL", "COMPOSITIONAL", "CAUSAL", "CONTEXTUAL",
    "ATTRIBUTIVE", "SEQUENTIAL", "COMPARATIVE", "ANALOGICAL", "RELATED_TO",
})


def run_json_graph_migrations(
    dry_run: bool = True,
    port: int | None = None,
) -> dict[str, int]:
    """Run all JSON graph quality migrations.

    Args:
        dry_run: If True, only report what would be changed.
        port: Memgraph port (defaults to JSON graph port 7689).

    Returns:
        Dict mapping migration name to affected count.
    """
    results = {}
    conn = mgclient.connect(
        host=settings.MEMGRAPH_HOST,
        port=port or settings.JSON_MEMGRAPH_PORT,
    )
    cursor = conn.cursor()

    try:
        # Run in order: relationships first (required for concept sync)
        results["relationship_types"] = _migrate_relationship_types(cursor, dry_run)
        results["entity_labels"] = _migrate_entity_labels(cursor, dry_run)
        results["relationship_properties"] = _migrate_relationship_properties(cursor, dry_run)
    finally:
        cursor.close()
        conn.close()

    return results


def _escape_cypher_identifier(name: str) -> str:
    """Escape a Cypher identifier (for edge types with special chars)."""
    return f"`{name.replace('`', '``')}`"


def _migrate_relationship_types(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Migrate emoji-prefixed relationship types to canonical labels.

    For each emoji-prefixed relationship type:
    1. Create new relationship with canonical type (HIERARCHICAL, CAUSAL, etc.)
    2. Copy all properties plus verb, emoji, category
    3. Delete old relationship
    """
    total_migrated = 0

    for old_type, (new_label, verb, emoji) in RELATIONSHIP_TYPE_MAPPING.items():
        escaped_old = _escape_cypher_identifier(old_type)

        # Count relationships of this type
        cursor.execute(f"""
            MATCH ()-[r:{escaped_old}]->()
            RETURN count(r)
        """)
        count = cursor.fetchone()[0]

        if count == 0:
            continue

        if dry_run:
            logger.info(ls.MIGRATION_DRY_RUN_REL_TYPES.format(
                count=count, old_type=old_type, new_type=new_label
            ))
            total_migrated += count
            continue

        # Get all relationships with their properties
        cursor.execute(f"""
            MATCH (a)-[r:{escaped_old}]->(b)
            RETURN a.unique_id AS source_id, b.unique_id AS target_id,
                   properties(r) AS props
        """)
        rows = cursor.fetchall()

        for row in rows:
            source_id, target_id, props = row
            props = dict(props) if props else {}

            # Set canonical properties
            props["verb"] = verb
            props["emoji"] = emoji
            props["category"] = new_label

            # Remove redundant properties
            props.pop("relationship_category", None)
            props.pop("category_with_emoji", None)
            props.pop("relationship_emoji", None)

            # Build SET clause
            set_parts = [f"r.{k} = ${k}" for k in props.keys()]
            set_clause = ", ".join(set_parts)

            # Create new relationship
            cursor.execute(f"""
                MATCH (a {{unique_id: $source_id}})
                MATCH (b {{unique_id: $target_id}})
                CREATE (a)-[r:{new_label}]->(b)
                SET {set_clause}
            """, {"source_id": source_id, "target_id": target_id, **props})

            # Delete old relationship
            cursor.execute(f"""
                MATCH (a {{unique_id: $source_id}})-[r:{escaped_old}]->(b {{unique_id: $target_id}})
                DELETE r
            """, {"source_id": source_id, "target_id": target_id})

        total_migrated += count
        logger.info(ls.MIGRATION_REL_TYPES_DONE.format(
            count=count, old_type=old_type, new_type=new_label
        ))

    return total_migrated


def _migrate_entity_labels(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Remove space-containing labels from JsonEntity nodes.

    For each node with a space-containing label:
    1. Remove the space-containing label
    2. Keep the PascalCase version (already exists as a label)
    """
    total_migrated = 0

    for old_label, new_label in LABELS_TO_NORMALIZE.items():
        escaped_old = _escape_cypher_identifier(old_label)

        # Count nodes with this label
        cursor.execute(f"""
            MATCH (n:{escaped_old})
            RETURN count(n)
        """)
        count = cursor.fetchone()[0]

        if count == 0:
            continue

        if dry_run:
            logger.info(ls.MIGRATION_DRY_RUN_LABELS.format(
                count=count, old_label=old_label
            ))
            total_migrated += count
            continue

        # Remove the space-containing label
        cursor.execute(f"""
            MATCH (n:{escaped_old})
            REMOVE n:{escaped_old}
        """)

        total_migrated += count
        logger.info(ls.MIGRATION_LABELS_DONE.format(
            count=count, old_label=old_label
        ))

    return total_migrated


def _migrate_relationship_properties(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Consolidate redundant relationship properties.

    Removes:
    - relationship_category (duplicate of category)
    - category_with_emoji (inconsistent format)
    - relationship_emoji (replaced by emoji)

    Adds:
    - emoji (clean single emoji derived from category)
    """
    # Count relationships with redundant properties
    cursor.execute("""
        MATCH ()-[r]->()
        WHERE r.relationship_category IS NOT NULL
           OR r.category_with_emoji IS NOT NULL
           OR r.relationship_emoji IS NOT NULL
        RETURN count(r)
    """)
    count = cursor.fetchone()[0]

    if count == 0:
        return 0

    if dry_run:
        logger.info(ls.MIGRATION_DRY_RUN_PROPS.format(count=count))
        return count

    # Get all relationships with properties
    cursor.execute("""
        MATCH ()-[r]->()
        WHERE r.relationship_category IS NOT NULL
           OR r.category_with_emoji IS NOT NULL
           OR r.relationship_emoji IS NOT NULL
        RETURN id(r) AS rel_id, properties(r) AS props
    """)
    rows = cursor.fetchall()

    for row in rows:
        rel_id, props = row
        props = dict(props) if props else {}

        # Determine category
        category = props.get("category") or props.get("relationship_category", "RELATED_TO")
        if category not in VALID_CATEGORIES:
            category = "RELATED_TO"

        # Determine emoji
        emoji = None
        if "emoji" in props:
            emoji = props["emoji"]
        elif "relationship_emoji" in props:
            emoji = props["relationship_emoji"]
        elif "category_with_emoji" in props:
            # Extract emoji from inconsistent format
            cwe = props["category_with_emoji"]
            if cwe and len(cwe) > 0 and cwe[0] in cs.CATEGORY_EMOJI_MAP.values():
                emoji = cwe[0]
            else:
                emoji = cs.CATEGORY_EMOJI_MAP.get(category, "🔗")
        else:
            emoji = cs.CATEGORY_EMOJI_MAP.get(category, "🔗")

        # Update properties
        cursor.execute("""
            MATCH ()-[r]-() WHERE id(r) = $rel_id
            SET r.category = $category
            SET r.emoji = $emoji
            REMOVE r.relationship_category, r.category_with_emoji, r.relationship_emoji
        """, {"rel_id": rel_id, "category": category, "emoji": emoji})

    logger.info(ls.MIGRATION_PROPS_DONE.format(count=count))
    return count

