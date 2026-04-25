"""One-time migration to clean stale Concept/Topic nodes, edges, and indexes from doc instance.

Run against the doc instance (port 7688) after deploying the concept instance (port 7690).

Usage:
    python -m codebase_rag.migrations.cleanup_doc_concepts [--yes]
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from codebase_rag.config import settings
from codebase_rag.services.graph_service import MemgraphIngestor


def _count_stale_data(ingestor: MemgraphIngestor) -> dict[str, int]:
    """Count stale concept-related data in doc instance."""
    counts: dict[str, int] = {}
    queries = {
        "concept_nodes": "MATCH (c:Concept) RETURN count(c) as cnt",
        "topic_nodes": "MATCH (t:Topic) RETURN count(t) as cnt",
        "mention_edges": "MATCH (:Chunk)-[m:MENTIONS]->(:Concept) RETURN count(m) as cnt",
        "mention_topic_edges": "MATCH (:Chunk)-[m:MENTIONS]->(:Topic) RETURN count(m) as cnt",
    }
    for key, query in queries.items():
        try:
            result = ingestor.fetch_all(query)
            counts[key] = result[0]["cnt"] if result else 0
        except Exception as e:
            logger.warning(f"Could not count {key}: {e}")
            counts[key] = -1
    return counts


def _list_stale_indexes(ingestor: MemgraphIngestor) -> list[str]:
    """List concept-related indexes in doc instance."""
    concept_indexes = [
        "Concept(qualified_name)",
        "Concept(workspace)",
        "Concept(entity_category)",
        "Concept(entity_subtype)",
        "Topic(qualified_name)",
        "Topic(workspace)",
    ]
    edge_indexes = [
        "HIERARCHICAL(verb)",
        "COMPOSITIONAL(verb)",
        "CONTEXTUAL(verb)",
        "ATTRIBUTIVE(verb)",
        "COMPARATIVE(verb)",
        "SEQUENTIAL(verb)",
        "CAUSAL(verb)",
        "ANALOGICAL(verb)",
        "RELATED_TO(verb)",
    ]
    found: list[str] = []
    try:
        result = ingestor.fetch_all("SHOW INDEX INFO")
        for row in result:
            label = row.get("index label", "")
            prop = row.get("index property", "")
            if label and prop:
                key = f"{label}({prop})"
                if key in concept_indexes + edge_indexes:
                    found.append(key)
    except Exception as e:
        logger.warning(f"Could not list indexes: {e}")
    return found


def _delete_stale_data(ingestor: MemgraphIngestor) -> dict[str, int]:
    """Delete stale Concept/Topic data from doc instance."""
    deleted: dict[str, int] = {}
    operations = [
        ("mention_edges", "MATCH (:Chunk)-[m:MENTIONS]->(:Concept) DELETE m RETURN count(m) as cnt"),
        ("mention_topic_edges", "MATCH (:Chunk)-[m:MENTIONS]->(:Topic) DELETE m RETURN count(m) as cnt"),
    ]
    for key, query in operations:
        try:
            result = ingestor.fetch_all(query)
            deleted[key] = result[0]["cnt"] if result else 0
        except Exception as e:
            logger.warning(f"Could not delete {key}: {e}")
            deleted[key] = -1

    try:
        result = ingestor.fetch_all("MATCH (c:Concept) DETACH DELETE c RETURN count(c) as cnt")
        deleted["concept_nodes"] = result[0]["cnt"] if result else 0
    except Exception as e:
        logger.warning(f"Could not delete Concept nodes: {e}")
        deleted["concept_nodes"] = -1

    try:
        result = ingestor.fetch_all("MATCH (t:Topic) DETACH DELETE t RETURN count(t) as cnt")
        deleted["topic_nodes"] = result[0]["cnt"] if result else 0
    except Exception as e:
        logger.warning(f"Could not delete Topic nodes: {e}")
        deleted["topic_nodes"] = -1

    return deleted


def _drop_stale_indexes(ingestor: MemgraphIngestor) -> tuple[int, int]:
    """Drop concept-related indexes from doc instance. Gracefully skips missing indexes."""
    index_labels = [
        ("Concept", "qualified_name"),
        ("Concept", "workspace"),
        ("Concept", "entity_category"),
        ("Concept", "entity_subtype"),
        ("Topic", "qualified_name"),
        ("Topic", "workspace"),
    ]
    edge_labels = [
        "HIERARCHICAL",
        "COMPOSITIONAL",
        "CONTEXTUAL",
        "ATTRIBUTIVE",
        "COMPARATIVE",
        "SEQUENTIAL",
        "CAUSAL",
        "ANALOGICAL",
        "RELATED_TO",
    ]

    dropped = 0
    skipped = 0

    for label, prop in index_labels:
        cypher = f"DROP INDEX ON :{label}({prop})"
        try:
            ingestor.fetch_all(cypher)
            logger.info(f"Dropped index on :{label}({prop})")
            dropped += 1
        except Exception as e:
            logger.debug(f"Skipped index on :{label}({prop}): {e}")
            skipped += 1

    for label in edge_labels:
        cypher = f"DROP INDEX ON :{label}(verb)"
        try:
            ingestor.fetch_all(cypher)
            logger.info(f"Dropped edge index on :{label}(verb)")
            dropped += 1
        except Exception as e:
            logger.debug(f"Skipped edge index on :{label}(verb): {e}")
            skipped += 1

    return dropped, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean stale concept data from doc Memgraph instance"
    )
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    logger.info("Connecting to doc instance (port {})...", settings.DOC_MEMGRAPH_PORT)
    try:
        ingestor = MemgraphIngestor(
            host=settings.DOC_MEMGRAPH_HOST,
            port=settings.DOC_MEMGRAPH_PORT,
            batch_size=1000,
            connection_timeout=settings.DOC_MEMGRAPH_CONNECTION_TIMEOUT,
        ).__enter__()
    except Exception as e:
        logger.error(f"Could not connect to doc instance: {e}")
        sys.exit(1)

    try:
        # Phase 1: Count stale data
        logger.info("Counting stale data in doc instance...")
        counts = _count_stale_data(ingestor)
        logger.info(f"Stale data found: {counts}")

        # Phase 2: List stale indexes
        logger.info("Listing stale indexes in doc instance...")
        stale_indexes = _list_stale_indexes(ingestor)
        if stale_indexes:
            logger.info(f"Stale indexes found: {stale_indexes}")
        else:
            logger.info("No stale indexes found.")

        if not any(v > 0 for v in counts.values() if v != -1) and not stale_indexes:
            logger.info("No stale concept data found in doc instance. Nothing to clean.")
            return

        # Phase 3: Confirm
        if not args.yes:
            print("\nThis will permanently delete:")
            for key, cnt in counts.items():
                if cnt > 0:
                    print(f"  - {key}: {cnt}")
            if stale_indexes:
                print(f"  - indexes: {len(stale_indexes)}")
            response = input("\nProceed? [y/N]: ").strip().lower()
            if response not in ("y", "yes"):
                logger.info("Aborted by user.")
                return

        # Phase 4: Delete edges first, then nodes
        logger.info("Deleting stale concept data...")
        deleted = _delete_stale_data(ingestor)
        logger.info(f"Deleted: {deleted}")

        # Phase 5: Drop indexes
        if stale_indexes:
            logger.info("Dropping stale indexes...")
            dropped, skipped = _drop_stale_indexes(ingestor)
            logger.info(f"Indexes: {dropped} dropped, {skipped} skipped (already missing)")

        logger.info("Migration complete. Doc instance concept data cleaned.")
    finally:
        try:
            ingestor.__exit__(None, None, None)
        except Exception as e:
            logger.warning(f"Error closing ingestor: {e}")


if __name__ == "__main__":
    main()
