#!/usr/bin/env python3
"""Check that generated Pydantic models are in sync with JSON Schema.

This script verifies that the auto-generated Pydantic models are up-to-date
with the current JSON Schema. It checks modification times and optionally
verifies content matches.

Usage:
    python scripts/check_schema_sync.py

Exit codes:
    0: Models are in sync
    1: Models are out of sync or missing (regenerate with generate_schemas.py)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Check that schema and generated models are synchronized."""
    schema_path = Path("codebase_rag/schema.json")
    models_path = Path("codebase_rag/schemas/ingestion.py")

    if not schema_path.exists():
        print(f"Error: Schema file not found: {schema_path}", file=sys.stderr)
        return 1

    if not models_path.exists():
        print(f"Error: Generated models not found: {models_path}", file=sys.stderr)
        print("Run: python scripts/generate_schemas.py", file=sys.stderr)
        return 1

    # Get modification times
    schema_mtime = schema_path.stat().st_mtime
    models_mtime = models_path.stat().st_mtime

    # Check if schema is newer than models
    if schema_mtime > models_mtime:
        print("Error: Schema is newer than generated models", file=sys.stderr)
        print("Run: python scripts/generate_schemas.py", file=sys.stderr)
        return 1

    # Verify the models file contains the expected header
    content = models_path.read_text()
    if "Auto-generated Pydantic models from JSON Schema" not in content:
        print("Error: Generated models file appears invalid", file=sys.stderr)
        print("Run: python scripts/generate_schemas.py", file=sys.stderr)
        return 1

    print("Schema and models are in sync")
    return 0


if __name__ == "__main__":
    sys.exit(main())
