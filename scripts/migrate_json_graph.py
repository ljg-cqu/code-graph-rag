#!/usr/bin/env python3
"""migrate_json_graph.py - Migrate JSON graph to canonical schema.

This script runs all JSON graph quality migrations in the correct order:
1. Relationship type normalization (emoji-prefixed to canonical)
2. Entity label normalization (remove space-containing labels)
3. Relationship property consolidation (remove redundant properties)

Usage:
    python scripts/migrate_json_graph.py --dry-run
    python scripts/migrate_json_graph.py --port 7689

Run with --dry-run first to preview changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from codebase_rag.migrations.json_graph_migrations import run_json_graph_migrations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate JSON graph to canonical schema",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview changes
    python scripts/migrate_json_graph.py --dry-run

    # Run migrations on default JSON graph (port 7689)
    python scripts/migrate_json_graph.py

    # Run on specific port
    python scripts/migrate_json_graph.py --port 7689
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without making them",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Memgraph port (default: JSON_MEMGRAPH_PORT from config)",
    )
    args = parser.parse_args()

    logger.info("Starting JSON graph migrations...")
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")

    try:
        results = run_json_graph_migrations(
            dry_run=args.dry_run,
            port=args.port,
        )

        print("\n" + "=" * 60)
        print("Migration Results:")
        print("=" * 60)
        for name, count in results.items():
            print(f"  {name}: {count}")
        print("=" * 60)

        if args.dry_run:
            print("\nRun without --dry-run to apply changes.")

        return 0

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
