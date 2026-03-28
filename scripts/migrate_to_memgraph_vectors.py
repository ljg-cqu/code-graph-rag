#!/usr/bin/env python
"""Migration script: Qdrant to Memgraph native vector storage.

This script migrates embeddings from an existing Qdrant collection
to Memgraph's native vector storage.

Usage:
    python scripts/migrate_to_memgraph_vectors.py [--project PROJECT_NAME]

Requirements:
    - Both Qdrant and Memgraph must be accessible
    - Semantic dependencies must be installed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


def migrate_qdrant_to_memgraph(project_name: str | None = None) -> int:
    """Migrate embeddings from Qdrant to Memgraph native storage.

    Args:
        project_name: Optional project name prefix to filter embeddings

    Returns:
        Number of embeddings migrated
    """
    try:
        from codebase_rag.config import settings
        from codebase_rag.services.graph_service import MemgraphIngestor
        from codebase_rag.vector_store_qdrant import QdrantBackend
        from codebase_rag.vector_store_memgraph import MemgraphBackend
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install with: uv sync --extra semantic")
        return 0

    logger.info("Starting migration from Qdrant to Memgraph native vectors...")
    logger.info(f"Source: Qdrant collection '{settings.QDRANT_COLLECTION_NAME}'")
    logger.info(f"Target: Memgraph vector index '{settings.MEMGRAPH_VECTOR_INDEX_NAME}'")

    # Initialize backends
    qdrant_backend = QdrantBackend()
    memgraph_backend = MemgraphBackend()

    # Check source health
    if not qdrant_backend.health_check():
        logger.error("Qdrant is not healthy. Check QDRANT_URI or QDRANT_DB_PATH.")
        return 0

    # Initialize Memgraph vector index
    logger.info("Initializing Memgraph vector index...")
    memgraph_backend.initialize()

    # Get all points from Qdrant
    logger.info("Fetching embeddings from Qdrant...")
    qdrant_client = qdrant_backend._get_client()

    offset = None
    batch_size = 100
    total_migrated = 0

    while True:
        try:
            results, offset = qdrant_client.scroll(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                limit=batch_size,
                offset=offset,
                with_vectors=True,
                with_payload=True,
            )
        except Exception as e:
            logger.error(f"Failed to fetch from Qdrant: {e}")
            break

        if not results:
            break

        # Filter by project if specified
        points_to_migrate = []
        for point in results:
            if project_name:
                payload = point.payload or {}
                qn = payload.get("qualified_name", "")
                if not qn.startswith(project_name):
                    continue
            points_to_migrate.append((
                point.id,
                point.vector,
                point.payload.get("qualified_name", "") if point.payload else ""
            ))

        # Store in Memgraph
        if points_to_migrate:
            stored = memgraph_backend.store_batch(points_to_migrate)
            total_migrated += stored
            logger.info(f"Migrated {total_migrated} embeddings...")

        if offset is None:
            break

    logger.info(f"Migration complete! Total embeddings migrated: {total_migrated}")

    # Verify migration
    if total_migrated > 0:
        logger.info("Verifying migration...")
        stats = memgraph_backend.get_stats()
        logger.info(f"Memgraph stats: {stats}")

    # Cleanup
    qdrant_backend.close()
    memgraph_backend.close()

    return total_migrated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate embeddings from Qdrant to Memgraph native vector storage"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Project name prefix to filter embeddings (migrates all if not specified)",
    )
    args = parser.parse_args()

    migrated = migrate_qdrant_to_memgraph(args.project)
    if migrated > 0:
        logger.info(f"Successfully migrated {migrated} embeddings to Memgraph")
        logger.info("You can now set VECTOR_STORE_BACKEND=memgraph in your .env")
    else:
        logger.warning("No embeddings were migrated")


if __name__ == "__main__":
    main()