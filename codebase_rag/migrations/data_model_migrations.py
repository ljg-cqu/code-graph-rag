"""Data model migration utilities for graph database."""
from __future__ import annotations

from loguru import logger
import mgclient

from .. import logs as ls
from ..config import settings


def run_migrations(dry_run: bool = True) -> dict[str, int]:
    """Run all data model migrations.

    Args:
        dry_run: If True, only report what would be changed.

    Returns:
        Dict mapping migration name to affected node count.
    """
    results = {}
    conn = mgclient.connect(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
    )
    cursor = conn.cursor()

    try:
        results["orphaned_builtins"] = _migrate_orphaned_builtins(cursor, dry_run)
        results["external_module_paths"] = _migrate_external_module_paths(cursor, dry_run)
        results["json_node_names"] = _migrate_json_node_names(cursor, dry_run)
        results["incomplete_test_nodes"] = _cleanup_incomplete_test_nodes(cursor, dry_run)
    finally:
        cursor.close()
        conn.close()

    return results


def _migrate_orphaned_builtins(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Create DEFINES relationships for orphaned builtin functions."""
    cursor.execute("""
        MATCH (f:Function)
        WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-()
        RETURN count(f)
    """)
    count = cursor.fetchone()[0]

    if count == 0:
        return 0

    if dry_run:
        logger.info(ls.MIGRATION_DRY_RUN_BUILTINS.format(count=count))
        return count

    cursor.execute("""
        MERGE (b:Module {
            qualified_name: 'builtin',
            name: '__builtins__',
            is_virtual: true
        })
    """)

    cursor.execute("""
        MATCH (f:Function)
        WHERE f.qualified_name STARTS WITH 'builtin.' AND NOT (f)<-[:DEFINES]-()
        MATCH (b:Module {qualified_name: 'builtin'})
        MERGE (b)-[:DEFINES]->(f)
    """)

    logger.info(ls.MIGRATION_BUILTINS_DONE.format(count=count))
    return count


def _migrate_external_module_paths(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Move path to import_path for external modules."""
    cursor.execute("""
        MATCH (m:Module)
        WHERE m.is_external = true AND m.path IS NOT NULL
        RETURN count(m)
    """)
    count = cursor.fetchone()[0]

    if count == 0:
        return 0

    if dry_run:
        logger.info(ls.MIGRATION_DRY_RUN_EXTERNAL_PATHS.format(count=count))
        return count

    cursor.execute("""
        MATCH (m:Module)
        WHERE m.is_external = true AND m.path IS NOT NULL
        SET m.import_path = m.path,
            m.path = null
    """)

    logger.info(ls.MIGRATION_EXTERNAL_PATHS_DONE.format(count=count))
    return count


def _migrate_json_node_names(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Add name property to JSON nodes."""
    total = 0

    cursor.execute("MATCH (n:JsonObject) WHERE n.name IS NULL RETURN count(n)")
    json_object_count = cursor.fetchone()[0]
    total += json_object_count

    cursor.execute("MATCH (n:JsonArray) WHERE n.name IS NULL RETURN count(n)")
    json_array_count = cursor.fetchone()[0]
    total += json_array_count

    cursor.execute("MATCH (n:JsonField) WHERE n.name IS NULL RETURN count(n)")
    json_field_count = cursor.fetchone()[0]
    total += json_field_count

    cursor.execute("MATCH (n:JsonValue) WHERE n.name IS NULL RETURN count(n)")
    json_value_count = cursor.fetchone()[0]
    total += json_value_count

    if total == 0:
        return 0

    if dry_run:
        logger.info(ls.MIGRATION_DRY_RUN_JSON_NAMES.format(count=total))
        return total

    cursor.execute("""
        MATCH (n:JsonObject) WHERE n.name IS NULL
        SET n.name = split(n.qualified_name, '.')[-1]
    """)

    cursor.execute("""
        MATCH (n:JsonArray) WHERE n.name IS NULL
        SET n.name = split(n.qualified_name, '.')[-1]
    """)

    cursor.execute("""
        MATCH (n:JsonField) WHERE n.name IS NULL AND n.key IS NOT NULL
        SET n.name = n.key
    """)

    cursor.execute("""
        MATCH (n:JsonValue) WHERE n.name IS NULL AND n.value IS NOT NULL
        WITH n, toString(n.value) AS val
        SET n.name = CASE
            WHEN size(val) > 50 THEN left(val, 50) + '...'
            ELSE val
        END
    """)

    logger.info(ls.MIGRATION_JSON_NAMES_DONE.format(count=total))
    return total


def _cleanup_incomplete_test_nodes(cursor: mgclient.Cursor, dry_run: bool) -> int:
    """Delete incomplete Test nodes."""
    cursor.execute("""
        MATCH (t:Test)
        WHERE t.name IS NULL AND t.qualified_name IS NULL
        RETURN count(t)
    """)
    count = cursor.fetchone()[0]

    if count == 0:
        return 0

    if dry_run:
        logger.info(ls.MIGRATION_DRY_RUN_TEST_NODES.format(count=count))
        return count

    cursor.execute("""
        MATCH (t:Test)
        WHERE t.name IS NULL AND t.qualified_name IS NULL
        DETACH DELETE t
    """)

    logger.info(ls.MIGRATION_TEST_NODES_DONE.format(count=count))
    return count
